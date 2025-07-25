import os
import re
import time
import mimetypes
import secrets
import logging
import random
import requests
import json
import uuid
from datetime import datetime, timedelta
from pathlib import Path
import psutil
from sqlalchemy import text

from log_utils import setup_logging
from exceptions import (
    AuthenticationError, InvalidCredentialsError, AccountLockedError,
    SessionExpiredError, InsufficientPermissionsError,
    FileError, UnsupportedFileTypeError, FileTooBigError, InvalidFileFormatError,
    FileProcessingError, QuotaExceededError, RateLimitExceededError,
    PredictionError, ModelNotAvailableError, PredictionFailedError,
    DatabaseError, RecordNotFoundError, DuplicateRecordError,
    ConfigurationError, MissingConfigError, ExternalServiceError,
    ValidationError, InvalidInputError, RequiredFieldError,
    StorageError, convert_exception, CircuitBreakerOpenException
)

# Circuit breaker imports - using centralized configuration
from circuit_breaker_config import (
    get_database_circuit_breaker, get_cache_circuit_breaker,
    get_http_circuit_breaker, initialize_circuit_breakers
)

from flask import (
    Flask,
    request,
    render_template,
    flash,
    redirect,
    url_for,
    session,
    jsonify,
)
from flask_sqlalchemy import SQLAlchemy
from flask_wtf.csrf import CSRFProtect
from flask_login import (
    LoginManager,
    login_user,
    login_required,
    logout_user,
    current_user,
    UserMixin,
)
from flask_talisman import Talisman
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from flask_sqlalchemy import SQLAlchemy
from metrics import (track_request_metrics, record_failed_login, record_quota_usage, 
                    get_metrics, CONTENT_TYPE_LATEST)
from cache_utils import cache_health_check
from api_v1 import api_v1
from openapi_spec import create_openapi_endpoint
from config import get_config
from websocket_service import FlaskWebSocketIntegration, get_websocket_manager
from analytics_dashboard import analytics_bp
from cache_manager import invalidate_analytics_cache
from cache_monitoring import cache_monitor_bp
from notification_service import get_notification_service
from security.csp_nonce import CSPNonceManager, csp_nonce_required
from auth.auth_routes import auth_bp
from auth.session_jwt_bridge import init_session_jwt_bridge
from health_checks import health_bp

logger = logging.getLogger(__name__)
setup_logging()

# Load configuration
config = get_config()

def get_or_create_web_circuit_breakers():
    """Get circuit breakers for web application operations using centralized configuration."""
    # Get circuit breakers from centralized manager
    db_circuit = get_database_circuit_breaker("web_app", db_session=db.session)
    cache_circuit = get_cache_circuit_breaker("web_app")
    notification_circuit = get_http_circuit_breaker("web_app", service_type="notification")
    
    return db_circuit, cache_circuit, notification_circuit

app = Flask(__name__)
app.secret_key = config.security.secret_key
app.config["WTF_CSRF_SECRET_KEY"] = config.security.csrf_secret_key or config.security.secret_key
app.config["WTF_CSRF_ENABLED"] = True  # Enable CSRF protection
csrf = CSRFProtect(app)
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

# Initialize circuit breakers with configuration
initialize_circuit_breakers(config_file="config/circuit_breakers.json")

db = SQLAlchemy()

# Enhanced Content Security Policy
# Initialize CSP nonce manager
csp_manager = CSPNonceManager(app)

# Enhanced security headers without CSP (handled by CSP manager)
Talisman(app, 
    force_https=False,  # Disabled for local development 
    frame_options="DENY",
    content_security_policy=None,  # CSP handled by CSPNonceManager with nonces
    referrer_policy="strict-origin-when-cross-origin",
    feature_policy={
        "microphone": "'self'",
        "camera": "'self'",
        "geolocation": "'self'",
        "payment": "'none'",
        "usb": "'none'"
    }
)

MAX_FILE_SIZE = config.upload.max_file_size
MAX_TOTAL_SIZE = config.upload.max_total_size

login_manager = LoginManager(app)
login_manager.login_view = "login"

# Enhanced rate limiter with different limits from config
limiter = Limiter(
    app=app, 
    key_func=get_remote_address,
    default_limits=[config.rate_limit.default_limit]
) if config.rate_limit.enabled else None

# Track failed login attempts per IP - now persistent
LOCKOUT_THRESHOLD = config.security.lockout_threshold
LOCKOUT_WINDOW = config.security.lockout_window
LOCKOUT_FILE = config.security.lockout_file
CLEANUP_INTERVAL = 3600  # Clean old entries every hour

# Password validation based on config
min_length = config.security.password_min_length
PASSWORD_RE = re.compile(rf"^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[\W_]).{{{min_length},}}$")

# Username validation - alphanumeric and underscore only
USERNAME_RE = re.compile(r"^[a-zA-Z0-9_]{3,30}$")

PREDICT_API_URL = os.environ.get("PREDICT_API_URL", "http://localhost:8001/api/predict")


def load_failed_logins() -> dict:
    """Load failed login attempts from persistent storage."""
    try:
        if os.path.exists(LOCKOUT_FILE):
            with open(LOCKOUT_FILE, 'r') as f:
                data = json.load(f)
                # Clean old entries
                now = time.time()
                cleaned = {ip: (count, timestamp) for ip, (count, timestamp) in data.items() 
                          if now - timestamp < LOCKOUT_WINDOW}
                if len(cleaned) != len(data):
                    save_failed_logins(cleaned)
                return cleaned
    except (json.JSONDecodeError, ValueError, KeyError) as e:
        logger.warning(f"Failed to load lockout file: {e}")
    return {}

def save_failed_logins(failed_logins: dict) -> None:
    """Save failed login attempts to persistent storage."""
    try:
        with open(LOCKOUT_FILE, 'w') as f:
            json.dump(failed_logins, f)
    except (OSError, IOError, PermissionError) as e:
        logger.error(f"Failed to save lockout file: {e}")
        raise StorageError(operation='write', path=str(LOCKOUT_FILE), reason=str(e))
    except (TypeError, ValueError) as e:
        logger.error(f"Failed to serialize lockout data: {e}")
        raise ValidationError(f"Invalid lockout data format: {e}")

def cleanup_old_failed_logins() -> None:
    """Clean up old failed login entries."""
    failed_logins = load_failed_logins()
    now = time.time()
    cleaned = {ip: (count, timestamp) for ip, (count, timestamp) in failed_logins.items() 
              if now - timestamp < LOCKOUT_WINDOW}
    if len(cleaned) != len(failed_logins):
        save_failed_logins(cleaned)
        logger.info(f"Cleaned {len(failed_logins) - len(cleaned)} old lockout entries")

def is_ip_locked(ip: str) -> bool:
    """Check if IP is currently locked out."""
    failed_logins = load_failed_logins()
    data = failed_logins.get(ip)
    if data and data[0] >= LOCKOUT_THRESHOLD:
        return time.time() - data[1] < LOCKOUT_WINDOW
    return False

def record_failed_login(ip: str) -> None:
    """Record a failed login attempt."""
    failed_logins = load_failed_logins()
    attempts, first_attempt = failed_logins.get(ip, (0, time.time()))
    failed_logins[ip] = (attempts + 1, first_attempt)
    save_failed_logins(failed_logins)

def clear_failed_logins(ip: str) -> None:
    """Clear failed login attempts for IP."""
    failed_logins = load_failed_logins()
    if ip in failed_logins:
        del failed_logins[ip]
        save_failed_logins(failed_logins)

def validate_input(text: str, max_length: int = 255) -> str:
    """Validate and sanitize text input."""
    if not text:
        return ""
    
    # Remove null bytes and control characters
    text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
    
    # Limit length
    text = text[:max_length]
    
    # Strip whitespace
    text = text.strip()
    
    return text

def validate_filename(filename: str) -> str:
    """Validate and sanitize filename."""
    if not filename:
        return ""
    
    # Use werkzeug's secure_filename
    filename = secure_filename(filename)
    
    # Ensure it's not empty after sanitization
    if not filename:
        return "upload.wav"
    
    # Ensure .wav extension
    if not filename.lower().endswith('.wav'):
        filename += '.wav'
    
    return filename

def is_wav_header(file_obj) -> bool:
    """Check whether the file-like object has a valid RIFF/WAVE header."""
    pos = file_obj.tell()
    header = file_obj.read(44)  # Read full WAV header
    file_obj.seek(pos)
    
    if len(header) < 44:
        return False
    
    # Check RIFF signature
    if header[0:4] != b"RIFF":
        return False
    
    # Check WAVE signature
    if header[8:12] != b"WAVE":
        return False
    
    # Check fmt chunk
    if header[12:16] != b"fmt ":
        return False
    
    # Check data chunk signature (should be at offset 36 for standard WAV)
    if header[36:40] != b"data":
        return False
    
    return True


class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)

    def set_password(self, password: str) -> None:
        self.password_hash = generate_password_hash(password)

    def check_password(self, password: str) -> bool:
        return check_password_hash(self.password_hash, password)


class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    filename = db.Column(db.String(200))
    result = db.Column(db.Text)
    file_size = db.Column(db.Integer)
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)

    user = db.relationship("User", backref=db.backref("predictions", lazy=True))

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "filename": self.filename,
            "result": self.result,
            "file_size": self.file_size,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "user_id": self.user_id
        }


class Detection(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    species = db.Column(db.String(100), nullable=False)
    time = db.Column(db.DateTime, nullable=False, server_default=db.func.now())
    latitude = db.Column(db.Float)
    longitude = db.Column(db.Float)
    zone = db.Column(db.String(100))
    image_url = db.Column(db.String(200))
    confidence = db.Column(db.Float, default=0.0)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=True)
    description = db.Column(db.Text)

    user = db.relationship("User", backref=db.backref("detections", lazy=True))

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "species": self.species,
            "time": self.time.isoformat() if self.time else None,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "zone": self.zone,
            "image": self.image_url,
            "confidence": self.confidence,
            "description": self.description,
            "timestamp": self.time.isoformat() if self.time else None,
        }


class NotificationPreference(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False, unique=True)
    email_notifications = db.Column(db.Boolean, default=True)
    push_notifications = db.Column(db.Boolean, default=True)
    email_address = db.Column(db.String(200))
    min_priority = db.Column(db.String(20), default='normal')
    species_filter = db.Column(db.Text)  # JSON string
    zone_filter = db.Column(db.Text)     # JSON string
    quiet_hours_start = db.Column(db.String(5))  # HH:MM
    quiet_hours_end = db.Column(db.String(5))    # HH:MM
    slack_webhook = db.Column(db.String(500))
    discord_webhook = db.Column(db.String(500))

    user = db.relationship("User", backref=db.backref("notification_preferences", uselist=False))

    def to_dict(self) -> dict:
        import json
        return {
            "user_id": self.user_id,
            "email_notifications": self.email_notifications,
            "push_notifications": self.push_notifications,
            "email_address": self.email_address,
            "min_priority": self.min_priority,
            "species_filter": json.loads(self.species_filter) if self.species_filter else [],
            "zone_filter": json.loads(self.zone_filter) if self.zone_filter else [],
            "quiet_hours_start": self.quiet_hours_start,
            "quiet_hours_end": self.quiet_hours_end,
            "slack_webhook": self.slack_webhook,
            "discord_webhook": self.discord_webhook,
        }


# ===== QUOTA MANAGEMENT MODELS =====

class PlanFeatures(db.Model):
    """Plan features and pricing configuration"""
    __tablename__ = 'plan_features'
    
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    plan_type = db.Column(db.String(50), nullable=False, unique=True)
    plan_name = db.Column(db.String(100), nullable=False)
    monthly_quota = db.Column(db.Integer, nullable=False, default=100)
    max_file_size_mb = db.Column(db.Integer, nullable=False, default=50)
    max_concurrent_uploads = db.Column(db.Integer, nullable=False, default=1)
    priority_queue = db.Column(db.Boolean, nullable=False, default=False)
    advanced_analytics = db.Column(db.Boolean, nullable=False, default=False)
    api_access = db.Column(db.Boolean, nullable=False, default=False)
    email_support = db.Column(db.Boolean, nullable=False, default=False)
    phone_support = db.Column(db.Boolean, nullable=False, default=False)
    features_json = db.Column(db.Text)  # JSON string for additional features
    price_monthly_cents = db.Column(db.Integer, nullable=False, default=0)
    price_yearly_cents = db.Column(db.Integer)
    data_retention_days = db.Column(db.Integer, nullable=False, default=30)  # Data retention period
    is_active = db.Column(db.Boolean, nullable=False, default=True)
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)

    def to_dict(self) -> dict:
        return {
            "plan_type": self.plan_type,
            "plan_name": self.plan_name,
            "monthly_quota": self.monthly_quota,
            "data_retention_days": self.data_retention_days,
            "price_monthly": self.price_monthly_cents / 100 if self.price_monthly_cents else 0,
            "is_active": self.is_active
        }


class UserPlan(db.Model):
    """User subscription plan assignment"""
    __tablename__ = 'user_plans'
    
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False, unique=True)
    plan_type = db.Column(db.String(50), db.ForeignKey("plan_features.plan_type"), nullable=False)
    subscription_start = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    subscription_end = db.Column(db.DateTime)
    auto_renew = db.Column(db.Boolean, nullable=False, default=False)
    payment_method = db.Column(db.String(50))
    subscription_id = db.Column(db.String(200))  # For payment provider
    trial_end = db.Column(db.DateTime)
    is_trial = db.Column(db.Boolean, nullable=False, default=False)
    status = db.Column(db.String(20), nullable=False, default='active')  # active, cancelled, suspended, expired
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    user = db.relationship("User", backref=db.backref("user_plan", uselist=False))
    plan_features = db.relationship("PlanFeatures", backref="subscriptions")

    def to_dict(self) -> dict:
        return {
            "user_id": self.user_id,
            "plan_type": self.plan_type,
            "subscription_start": self.subscription_start.isoformat() if self.subscription_start else None,
            "subscription_end": self.subscription_end.isoformat() if self.subscription_end else None,
            "auto_renew": self.auto_renew,
            "is_trial": self.is_trial,
            "trial_end": self.trial_end.isoformat() if self.trial_end else None,
            "status": self.status
        }


class QuotaUsage(db.Model):
    """Monthly quota usage tracking"""
    __tablename__ = 'quota_usage'
    
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    month = db.Column(db.Integer, nullable=False)  # 1-12
    year = db.Column(db.Integer, nullable=False)
    prediction_count = db.Column(db.Integer, nullable=False, default=0)
    total_file_size_bytes = db.Column(db.BigInteger, nullable=False, default=0)
    successful_predictions = db.Column(db.Integer, nullable=False, default=0)
    failed_predictions = db.Column(db.Integer, nullable=False, default=0)
    premium_features_used = db.Column(db.Text)  # JSON string
    reset_date = db.Column(db.DateTime, nullable=False)
    last_prediction_at = db.Column(db.DateTime)
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    user = db.relationship("User", backref="quota_usage_records")
    
    # Unique constraint
    __table_args__ = (db.UniqueConstraint('user_id', 'month', 'year', name='unique_user_month_year'),)

    def to_dict(self) -> dict:
        import json
        return {
            "user_id": self.user_id,
            "month": self.month,
            "year": self.year,
            "prediction_count": self.prediction_count,
            "total_file_size_mb": round(self.total_file_size_bytes / (1024 * 1024), 2),
            "successful_predictions": self.successful_predictions,
            "failed_predictions": self.failed_predictions,
            "premium_features_used": json.loads(self.premium_features_used) if self.premium_features_used else {},
            "reset_date": self.reset_date.isoformat() if self.reset_date else None,
            "last_prediction_at": self.last_prediction_at.isoformat() if self.last_prediction_at else None
        }


class DailyUsageDetails(db.Model):
    """Daily usage details for analytics"""
    __tablename__ = 'daily_usage_details'
    
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    usage_date = db.Column(db.Date, nullable=False, default=datetime.utcnow().date)
    prediction_count = db.Column(db.Integer, nullable=False, default=0)
    total_file_size_bytes = db.Column(db.BigInteger, nullable=False, default=0)
    average_processing_time_ms = db.Column(db.Integer)
    peak_hour = db.Column(db.Integer)  # 0-23
    device_type = db.Column(db.String(50))
    app_version = db.Column(db.String(20))
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    
    # Relationships
    user = db.relationship("User", backref="daily_usage_details")
    
    # Unique constraint
    __table_args__ = (db.UniqueConstraint('user_id', 'usage_date', name='unique_user_date'),)

    def to_dict(self) -> dict:
        return {
            "user_id": self.user_id,
            "usage_date": self.usage_date.isoformat() if self.usage_date else None,
            "prediction_count": self.prediction_count,
            "total_file_size_mb": round(self.total_file_size_bytes / (1024 * 1024), 2),
            "average_processing_time_ms": self.average_processing_time_ms,
            "peak_hour": self.peak_hour,
            "device_type": self.device_type,
            "app_version": self.app_version
        }


class QuotaTransaction(db.Model):
    """Audit trail for quota changes"""
    __tablename__ = 'quota_transactions'
    
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    transaction_type = db.Column(db.String(50), nullable=False)  # usage, bonus, reset, adjustment
    amount = db.Column(db.Integer, nullable=False)  # Can be negative for usage
    reason = db.Column(db.String(200))
    metadata_ = db.Column('metadata', db.Text)  # JSON string - using metadata_ to avoid SQLAlchemy reserved word
    prediction_id = db.Column(db.Integer, db.ForeignKey("prediction.id"))
    admin_user_id = db.Column(db.Integer, db.ForeignKey("user.id"))
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    
    # Relationships
    user = db.relationship("User", foreign_keys=[user_id], backref="quota_transactions")
    admin_user = db.relationship("User", foreign_keys=[admin_user_id])
    prediction = db.relationship("Prediction", backref="quota_transactions")

    def to_dict(self) -> dict:
        import json
        return {
            "id": self.id,
            "user_id": self.user_id,
            "transaction_type": self.transaction_type,
            "amount": self.amount,
            "reason": self.reason,
            "metadata": json.loads(self.transaction_metadata) if self.transaction_metadata else {},
            "prediction_id": self.prediction_id,
            "admin_user_id": self.admin_user_id,
            "created_at": self.created_at.isoformat()
        }


class SubscriptionEvent(db.Model):
    """Subscription lifecycle events"""
    __tablename__ = 'subscription_events'
    
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    event_type = db.Column(db.String(50), nullable=False)  # created, upgraded, downgraded, cancelled, renewed, expired
    old_plan_type = db.Column(db.String(50))
    new_plan_type = db.Column(db.String(50))
    effective_date = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    metadata_ = db.Column('metadata', db.Text)  # JSON string - using metadata_ to avoid SQLAlchemy reserved word
    created_by = db.Column(db.Integer, db.ForeignKey("user.id"))
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    
    # Relationships
    user = db.relationship("User", foreign_keys=[user_id], backref="subscription_events")
    created_by_user = db.relationship("User", foreign_keys=[created_by])

    def to_dict(self) -> dict:
        import json
        return {
            "id": self.id,
            "user_id": self.user_id,
            "event_type": self.event_type,
            "old_plan_type": self.old_plan_type,
            "new_plan_type": self.new_plan_type,
            "effective_date": self.effective_date.isoformat(),
            "metadata": json.loads(self.transaction_metadata) if self.transaction_metadata else {},
            "created_by": self.created_by,
            "created_at": self.created_at.isoformat()
        }


# ===== DATA RETENTION MODELS =====

class DataRetentionLog(db.Model):
    """Log of data retention operations for audit purposes"""
    __tablename__ = 'data_retention_log'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    plan_type = db.Column(db.String(50), nullable=False)
    retention_days = db.Column(db.Integer, nullable=False)
    records_deleted = db.Column(db.Integer, nullable=False, default=0)
    total_size_deleted_bytes = db.Column(db.BigInteger, nullable=False, default=0)
    deletion_date = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    retention_policy_version = db.Column(db.String(50), default='1.0')
    admin_override = db.Column(db.Boolean, default=False)
    admin_reason = db.Column(db.Text)
    metadata_ = db.Column('metadata', db.Text)  # JSON string for additional info - using metadata_ to avoid SQLAlchemy reserved word
    
    # Relationships
    user = db.relationship("User", backref=db.backref("retention_logs", lazy=True))
    
    def to_dict(self) -> dict:
        import json
        return {
            "id": self.id,
            "user_id": self.user_id,
            "plan_type": self.plan_type,
            "retention_days": self.retention_days,
            "records_deleted": self.records_deleted,
            "total_size_deleted_bytes": self.total_size_deleted_bytes,
            "total_size_deleted_mb": round(self.total_size_deleted_bytes / 1024 / 1024, 2),
            "deletion_date": self.deletion_date.isoformat(),
            "admin_override": self.admin_override,
            "admin_reason": self.admin_reason,
            "metadata": json.loads(self.metadata) if self.metadata else {}
        }


class PredictionArchive(db.Model):
    """Archive of deleted predictions for soft delete functionality"""
    __tablename__ = 'prediction_archive'
    
    id = db.Column(db.Integer, primary_key=True)
    original_prediction_id = db.Column(db.Integer, nullable=False)
    user_id = db.Column(db.Integer, nullable=False)
    filename = db.Column(db.String(200))
    result = db.Column(db.Text)
    file_size = db.Column(db.Integer)
    created_at = db.Column(db.DateTime, nullable=False)
    archived_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    plan_type_at_archive = db.Column(db.String(50))
    retention_days = db.Column(db.Integer)
    archived_by = db.Column(db.String(50), default='system')
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "original_prediction_id": self.original_prediction_id,
            "user_id": self.user_id,
            "filename": self.filename,
            "file_size": self.file_size,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "archived_at": self.archived_at.isoformat(),
            "plan_type_at_archive": self.plan_type_at_archive,
            "retention_days": self.retention_days,
            "archived_by": self.archived_by
        }


@login_manager.user_loader
def load_user(user_id: str):
    return User.query.get(int(user_id))


@app.route("/register", methods=["GET", "POST"])
@limiter.limit(config.rate_limit.login_limit if limiter else None)
@track_request_metrics
@csp_nonce_required
def register():
    if current_user.is_authenticated:
        return redirect(url_for("index"))
    if request.method == "POST":
        username = validate_input(request.form.get("username", ""), 30)
        password = request.form.get("password", "")
        
        if not username or not password:
            flash("Please provide username and password")
        elif not USERNAME_RE.match(username):
            flash("Username must be 3-30 characters long and contain only letters, numbers and underscore")
        elif not PASSWORD_RE.match(password):
            flash(
                "Password must be at least 10 characters long and include uppercase, lowercase, a digit and symbol"
            )
        elif User.query.filter_by(username=username).first():
            flash("Username already exists")
        else:
            user = User(username=username)
            user.set_password(password)
            db.session.add(user)
            db.session.commit()
            flash("Registration successful. Please log in.")
            return redirect(url_for("login"))
    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
@limiter.limit(config.rate_limit.login_limit if limiter else None)
@track_request_metrics
@csp_nonce_required
def login():
    if current_user.is_authenticated:
        return redirect(url_for("index"))

    ip = get_remote_address()
    
    # Clean up old entries periodically
    if hasattr(app, '_last_cleanup'):
        if time.time() - app._last_cleanup > CLEANUP_INTERVAL:
            cleanup_old_failed_logins()
            app._last_cleanup = time.time()
    else:
        app._last_cleanup = time.time()
    
    if is_ip_locked(ip):
        return "Too many failed attempts. Try again later.", 429

    if request.method == "POST":
        username = validate_input(request.form.get("username", ""), 30)
        password = request.form.get("password", "")
        
        if not username or not password:
            record_failed_login(ip)
            flash("Please provide username and password")
            return render_template("login.html")
        
        # Protected user lookup with circuit breaker
        db_circuit, cache_circuit, notification_circuit = get_or_create_web_circuit_breakers()
        
        try:
            def get_user_by_username():
                return User.query.filter_by(username=username).first()
            
            user = db_circuit.execute_query(
                User.query.filter_by(username=username),
                read_only=True,
                timeout=2.0
            ).first()
            
            if user and user.check_password(password):
                login_user(user)
                clear_failed_logins(ip)
                logger.info(f"Successful login for user: {username}")
                return redirect(url_for("index"))
                
        except CircuitBreakerOpenException as e:
            logger.error(f"Database circuit breaker open during login for {username}: {e}")
            flash("Authentication service temporarily unavailable. Please try again later.")
            return render_template("login.html")
        except DatabaseError as e:
            logger.error(f"Database error during login for {username}: {e}")
            flash("Authentication service error. Please try again.")
            return render_template("login.html")
        except Exception as e:
            logger.error(f"Unexpected error during login for {username}: {e}")
            flash("Authentication error. Please try again.")
            return render_template("login.html")
        record_failed_login(ip)
        flash("Invalid credentials")

    return render_template("login.html")


@app.route("/logout")
@login_required
@track_request_metrics
def logout():
    logout_user()
    return redirect(url_for("login"))


@app.route("/api/v1/user/change-password", methods=["POST"])
@login_required
@limiter.limit("5 per hour")
@track_request_metrics
def change_password():
    """Handle password change for authenticated users."""
    data = request.get_json()
    
    if not data:
        return jsonify({"error": "No data provided"}), 400
        
    current_password = data.get("current_password")
    new_password = data.get("new_password")
    
    if not current_password or not new_password:
        return jsonify({"error": "Current and new passwords are required"}), 400
        
    # Verify current password
    if not current_user.check_password(current_password):
        return jsonify({"error": "Current password is incorrect"}), 401
        
    # Validate new password
    if not PASSWORD_RE.match(new_password):
        return jsonify({
            "error": "New password must be at least 10 characters long and include uppercase, lowercase, a digit and symbol"
        }), 400
        
    # Check if new password is different from current
    if current_user.check_password(new_password):
        return jsonify({"error": "New password must be different from current password"}), 400
        
    # Protected password update with circuit breaker
    db_circuit, cache_circuit, notification_circuit = get_or_create_web_circuit_breakers()
    
    try:
        with db_circuit.transaction(timeout=5.0):
            current_user.set_password(new_password)
            # Transaction is automatically committed by circuit breaker context manager
        
        logger.info(f"Password changed successfully for user: {current_user.username}")
        return jsonify({
            "message": "Password changed successfully",
            "status": "success"
        }), 200
        
    except CircuitBreakerOpenException as e:
        logger.error(f"Database circuit breaker open during password change for user {current_user.username}: {e}")
        return jsonify({
            "error": "Database service temporarily unavailable",
            "code": "DATABASE_SERVICE_UNAVAILABLE",
            "details": "Unable to update password at this time. Please try again later.",
            "circuit_open": True
        }), 503
    except DatabaseError as e:
        logger.error(f"Database error changing password for user {current_user.username}: {e}")
        return jsonify(e.to_dict()), 500
    except ValidationError as e:
        logger.error(f"Validation error changing password for user {current_user.username}: {e}")
        return jsonify(e.to_dict()), 400
    except Exception as e:
        logger.error(f"Unexpected error changing password for user {current_user.username}: {e}")
        error = convert_exception(e, operation='password_change', user=current_user.username)
        return jsonify(error.to_dict()), 500


@app.route("/", methods=["GET", "POST"])
@login_required
@limiter.limit(config.rate_limit.upload_limit, methods=["POST"] if limiter else None)
@track_request_metrics
@csp_nonce_required
def index():
    result = None
    db_circuit, cache_circuit, notification_circuit = get_or_create_web_circuit_breakers()
    
    # Protected total file size calculation with circuit breaker
    try:
        def get_user_file_total():
            return (
                db.session.query(db.func.coalesce(db.func.sum(Prediction.file_size), 0))
                .filter_by(user_id=current_user.id)
                .scalar()
                or 0
            )
        
        total = db_circuit.execute_query(
            db.session.query(db.func.coalesce(db.func.sum(Prediction.file_size), 0))
            .filter_by(user_id=current_user.id),
            read_only=True,
            timeout=2.0
        ).scalar() or 0
        
    except CircuitBreakerOpenException as e:
        logger.warning(f"Database circuit breaker open for file size calculation - user {current_user.id}: {e}")
        # Use fallback value
        total = 0
    except DatabaseError as e:
        logger.error(f"Database error calculating file size - user {current_user.id}: {e}")
        # Use fallback value
        total = 0
    except Exception as e:
        logger.error(f"Unexpected error calculating file size - user {current_user.id}: {e}")
        # Use fallback value
        total = 0
        
    remaining = MAX_TOTAL_SIZE - total
    if request.method == "POST":
        file = request.files.get("file")
        if not file or not file.filename:
            flash("Please upload a file.")
        else:
            # Validate and sanitize filename
            original_filename = validate_filename(file.filename)
            if not original_filename.lower().endswith(".wav"):
                flash("Please upload a WAV file.")
            else:
                mime_guess = mimetypes.guess_type(original_filename)[0]
                if (
                    file.mimetype not in ("audio/wav", "audio/x-wav")
                    and mime_guess != "audio/x-wav"
                ):
                    flash("Invalid file type. Only WAV files are accepted.")
                elif not is_wav_header(file.stream):
                    flash("Invalid WAV header.")
                else:
                    file_size = request.content_length
                    if file_size is None:
                        pos = file.stream.tell()
                        file.stream.seek(0, os.SEEK_END)
                        file_size = file.stream.tell()
                        file.stream.seek(pos)
                    if file_size > MAX_FILE_SIZE:
                        flash("File exceeds 100 MB limit.")
                    elif total + file_size > MAX_TOTAL_SIZE:
                        flash("Upload quota exceeded (10 GB total).")
                    else:
                        # Use streaming to save file temporarily
                        from streaming_utils import StreamingFileHandler
                        import tempfile
                        import uuid
                        
                        handler = StreamingFileHandler()
                        
                        # Create unique temp file name
                        temp_filename = f"upload_{uuid.uuid4().hex}_{secure_filename(original_filename)}"
                        temp_path = os.path.join(tempfile.gettempdir(), temp_filename)
                        
                        # Save file using streaming
                        try:
                            saved_size, file_hash = handler.save_file_streaming(
                                file,
                                Path(temp_path),
                                max_size=MAX_FILE_SIZE
                            )
                            
                            # Protected prediction insertion with circuit breaker
                            try:
                                with db_circuit.transaction(timeout=5.0):
                                    pred = Prediction(
                                        user_id=current_user.id,
                                        filename=original_filename,
                                        result="PENDING",
                                        file_size=saved_size,
                                    )
                                    db.session.add(pred)
                                    # Transaction is automatically committed by circuit breaker context manager
                                    
                            except CircuitBreakerOpenException as e:
                                logger.error(f"Database circuit breaker open during prediction creation - user {current_user.id}: {e}")
                                flash("Database service temporarily unavailable. Please try again later.")
                                return render_template(
                                    "index_secure.html",
                                    result=result,
                                    predictions=[],  # Empty fallback
                                    remaining_bytes=remaining,
                                )
                            except DatabaseError as e:
                                logger.error(f"Database error creating prediction - user {current_user.id}: {e}")
                                flash("Unable to create prediction record. Please try again.")
                                return render_template(
                                    "index_secure.html",
                                    result=result,
                                    predictions=[],  # Empty fallback
                                    remaining_bytes=remaining,
                                )
                            
                            from .tasks import run_prediction
                            
                            # Pass file path instead of content
                            run_prediction.delay(
                                pred.id,
                                original_filename,
                                temp_path,  # Pass path instead of data
                                PREDICT_API_URL,
                            )
                        except FileTooBigError as e:
                            flash(e.user_message)
                            logger.warning(f"File too big for user {current_user.username}: {e}")
                        except (OSError, IOError, PermissionError) as e:
                            flash("Unable to save file. Please try again.")
                            logger.error(f"File save error for user {current_user.username}: {e}")
                        except ValueError as e:
                            flash(str(e))
                            logger.warning(f"File validation error for user {current_user.username}: {e}")
                        except DatabaseError as e:
                            flash("Unable to create prediction record. Please try again.")
                            logger.error(f"Database error for user {current_user.username}: {e}")
                            db.session.rollback()
                        except Exception as e:
                            error = convert_exception(e, file_name=original_filename, operation='file_upload')
                            flash(error.user_message)
                            logger.error(f"Unexpected upload error for user {current_user.username}: {error}")
                            db.session.rollback()
                        else:
                            total += file_size
                        remaining = MAX_TOTAL_SIZE - total
                        
                        # Record quota usage metrics
                        usage_percent = (total / MAX_TOTAL_SIZE) * 100
                        record_quota_usage(usage_percent)
                        
                        flash("File queued for processing.")
                        
                        # Emit WebSocket notification for upload
                        import asyncio
                        try:
                            loop = asyncio.get_event_loop()
                        except RuntimeError:
                            loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(loop)
                        
                        if hasattr(application, 'websocket_manager'):
                            loop.create_task(application.websocket_manager.notify_prediction_complete({
                                'filename': original_filename,
                                'status': 'queued',
                                'user_id': current_user.id,
                                'file_size': file_size
                            }, current_user.id))
    # Protected predictions retrieval with circuit breaker
    try:
        predictions = db_circuit.execute_query(
            Prediction.query.filter_by(user_id=current_user.id)
            .order_by(Prediction.id.desc()),
            read_only=True,
            timeout=3.0
        ).all()
        
    except CircuitBreakerOpenException as e:
        logger.warning(f"Database circuit breaker open for predictions retrieval - user {current_user.id}: {e}")
        # Use empty fallback
        predictions = []
        flash("Unable to load prediction history at this time.")
    except DatabaseError as e:
        logger.error(f"Database error retrieving predictions - user {current_user.id}: {e}")
        # Use empty fallback
        predictions = []
        flash("Unable to load prediction history.")
    except Exception as e:
        logger.error(f"Unexpected error retrieving predictions - user {current_user.id}: {e}")
        # Use empty fallback
        predictions = []
        
    return render_template(
        "index_secure.html",
        result=result,
        predictions=predictions,
        remaining_bytes=remaining,
    )


@app.route("/metrics")
def metrics_endpoint():
    """Prometheus metrics endpoint."""
    return get_metrics(), 200, {'Content-Type': CONTENT_TYPE_LATEST}


@app.route("/circuit-breakers/metrics")
@track_request_metrics
def circuit_breaker_metrics():
    """Circuit breaker metrics endpoint."""
    from circuit_breaker_config import circuit_breaker_metrics_endpoint
    return jsonify(circuit_breaker_metrics_endpoint()), 200


@app.route("/circuit-breakers/health")
@track_request_metrics  
def circuit_breaker_health():
    """Circuit breaker health check endpoint."""
    from circuit_breaker_config import circuit_breaker_health_endpoint
    health_status = circuit_breaker_health_endpoint()
    
    status_code = 200
    if health_status['overall_status'] == 'unhealthy':
        status_code = 503
    elif health_status['overall_status'] == 'degraded':
        status_code = 206  # Partial Content
    
    return jsonify(health_status), status_code


@app.route("/circuit-breakers/reset", methods=["POST"])
@login_required
@track_request_metrics
def reset_circuit_breakers():
    """Reset all circuit breakers (admin only)."""
    # Check if user has admin permissions (implement based on your auth system)
    if not hasattr(current_user, 'is_admin') or not current_user.is_admin:
        return jsonify({"error": "Admin access required"}), 403
    
    try:
        from circuit_breaker_config import get_circuit_breaker_manager
        manager = get_circuit_breaker_manager()
        manager.reset_all_circuits()
        
        logger.info(f"All circuit breakers reset by admin user: {current_user.username}")
        return jsonify({
            "message": "All circuit breakers have been reset",
            "timestamp": datetime.utcnow().isoformat(),
            "reset_by": current_user.username
        }), 200
        
    except Exception as e:
        logger.error(f"Failed to reset circuit breakers: {e}")
        return jsonify({"error": "Failed to reset circuit breakers"}), 500


@app.route("/health")
@track_request_metrics
def health_check():
    """Basic health check endpoint."""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0"
    })


@app.route("/ready")
@track_request_metrics
def readiness_check():
    """Comprehensive readiness check for load balancers."""
    checks = {
        "database": False,
        "disk_space": False,
        "memory": False
    }
    
    # Check database connectivity
    try:
        db.session.execute(text("SELECT 1"))
        checks["database"] = True
    except DatabaseError as e:
        logger.error(f"Database check failed: {e}")
        checks["database"] = False
    except Exception as e:
        logger.error(f"Unexpected database check error: {e}")
        checks["database"] = False
    
    # Check disk space (> 1GB free)
    try:
        disk_usage = psutil.disk_usage('/')
        free_gb = disk_usage.free / (1024**3)
        checks["disk_space"] = free_gb > 1.0
    except (OSError, IOError, PermissionError) as e:
        logger.error(f"Disk space check failed - file system error: {e}")
        checks["disk_space"] = False
    except Exception as e:
        logger.error(f"Unexpected disk space check error: {e}")
        checks["disk_space"] = False
    
    # Check memory usage (< 90%)
    try:
        memory = psutil.virtual_memory()
        checks["memory"] = memory.percent < 90.0
    except (OSError, RuntimeError) as e:
        logger.error(f"Memory check failed - system error: {e}")
        checks["memory"] = False
    except Exception as e:
        logger.error(f"Unexpected memory check error: {e}")
        checks["memory"] = False
    
    all_healthy = all(checks.values())
    status_code = 200 if all_healthy else 503
    
    # Add cache health check
    cache_status = cache_health_check()
    checks["cache"] = cache_status["status"] in ["healthy", "disabled"]
    
    return jsonify({
        "status": "ready" if all_healthy else "not_ready",
        "checks": checks,
        "cache": cache_status,
        "timestamp": datetime.utcnow().isoformat()
    }), status_code


@app.route("/dashboard")
@login_required
@track_request_metrics
@csp_nonce_required
def dashboard():
    """Live dashboard with real-time notifications."""
    return render_template("dashboard.html")


@app.route("/data-retention")
@login_required
@track_request_metrics
@csp_nonce_required
def data_retention():
    """Data retention management page."""
    return render_template("data_retention.html")


@app.route("/api/csp-report", methods=["POST"])
@csrf.exempt  # CSP reports don't include CSRF tokens
def csp_report():
    """Log CSP violations for monitoring and debugging."""
    try:
        report = request.get_json(force=True)
        if report:
            logger.warning(f"CSP Violation: {json.dumps(report, indent=2)}")
            
            # Extract key information from the report
            violation = report.get('csp-report', {})
            blocked_uri = violation.get('blocked-uri', 'unknown')
            violated_directive = violation.get('violated-directive', 'unknown')
            document_uri = violation.get('document-uri', 'unknown')
            
            logger.warning(
                f"CSP blocked {blocked_uri} on {document_uri} "
                f"(violated directive: {violated_directive})"
            )
    except (ValueError, TypeError, KeyError) as e:
        logger.error(f"Invalid CSP report format: {e}")
    except Exception as e:
        logger.error(f"Unexpected error processing CSP report: {e}")
    
    return '', 204  # No content response


@app.route("/api/detections")
@login_required
@limiter.limit("30 per minute")
@track_request_metrics
def api_detections():
    # Add pagination support
    page = request.args.get('page', 1, type=int)
    per_page = min(request.args.get('per_page', 50, type=int), 100)  # Max 100 items
    
    detections_query = Detection.query.order_by(Detection.time.desc())
    detections = detections_query.paginate(
        page=page, per_page=per_page, error_out=False
    )
    
    return jsonify({
        "detections": [d.to_dict() for d in detections.items],
        "pagination": {
            "page": page,
            "per_page": per_page,
            "total": detections.total,
            "pages": detections.pages,
            "has_next": detections.has_next,
            "has_prev": detections.has_prev
        }
    })


@app.route("/api/notifications/preferences", methods=["GET", "POST"])
@login_required
@limiter.limit("20 per minute")
@track_request_metrics
def notification_preferences():
    """Get or update user notification preferences."""
    if request.method == "GET":
        prefs = NotificationPreference.query.filter_by(user_id=current_user.id).first()
        if prefs:
            return jsonify(prefs.to_dict())
        else:
            # Return default preferences
            return jsonify({
                "user_id": current_user.id,
                "email_notifications": True,
                "push_notifications": True,
                "email_address": None,
                "min_priority": "normal",
                "species_filter": [],
                "zone_filter": [],
                "quiet_hours_start": None,
                "quiet_hours_end": None,
                "slack_webhook": None,
                "discord_webhook": None
            })
    
    elif request.method == "POST":
        import json
        data = request.get_json()
        
        prefs = NotificationPreference.query.filter_by(user_id=current_user.id).first()
        if not prefs:
            prefs = NotificationPreference(user_id=current_user.id)
        
        # Update preferences
        prefs.email_notifications = data.get('email_notifications', True)
        prefs.push_notifications = data.get('push_notifications', True)
        prefs.email_address = validate_input(data.get('email_address', ''), 200)
        prefs.min_priority = data.get('min_priority', 'normal')
        prefs.species_filter = json.dumps(data.get('species_filter', []))
        prefs.zone_filter = json.dumps(data.get('zone_filter', []))
        prefs.quiet_hours_start = data.get('quiet_hours_start')
        prefs.quiet_hours_end = data.get('quiet_hours_end')
        prefs.slack_webhook = validate_input(data.get('slack_webhook', ''), 500)
        prefs.discord_webhook = validate_input(data.get('discord_webhook', ''), 500)
        
        db.session.add(prefs)
        db.session.commit()
        
        return jsonify({"status": "success", "message": "Preferences updated"})


@app.route("/api/simulate/detection", methods=["POST"])
@login_required
@limiter.limit("5 per minute")
@track_request_metrics  
def simulate_detection():
    """Simulate a new detection for testing (development only)."""
    if not app.debug:
        return jsonify({"error": "Only available in debug mode"}), 403
    
    data = request.get_json() or {}
    
    # Create a new detection
    detection = Detection(
        species=data.get('species', 'Great Horned Owl'),
        zone=data.get('zone', 'Test Zone'),
        latitude=data.get('latitude', 46.2044),
        longitude=data.get('longitude', 6.1432),
        confidence=data.get('confidence', 0.95),
        description=data.get('description', 'Simulated detection for testing'),
        user_id=current_user.id
    )
    
    db.session.add(detection)
    db.session.commit()
    
    # Invalidate analytics cache after new detection
    invalidate_analytics_cache()
    
    # Send real-time notification via WebSocket
    import asyncio
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    if hasattr(application, 'websocket_manager'):
        loop.create_task(application.websocket_manager.notify_new_detection(
            detection.to_dict(), current_user.id
        ))
    
    # Send notification via notification service
    notification_service = get_notification_service(db)
    loop.create_task(notification_service.send_detection_notification(
        detection.to_dict(), [current_user.id]
    ))
    
    return jsonify({
        "status": "success", 
        "message": "Detection simulated",
        "detection": detection.to_dict()
    })


def create_database_indexes():
    """Create database indexes for better performance."""
    with db.engine.connect() as conn:
        # Index on user_id for predictions (most common query)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_prediction_user_id ON prediction(user_id)")
        
        # Composite index for user predictions ordered by ID (for pagination)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_prediction_user_id_id ON prediction(user_id, id DESC)")
        
        # Index on detection time for chronological queries
        conn.execute("CREATE INDEX IF NOT EXISTS idx_detection_time ON detection(time DESC)")
        
        # Index on detection species for filtering
        conn.execute("CREATE INDEX IF NOT EXISTS idx_detection_species ON detection(species)")
        
        # Composite index for species and zone filtering
        conn.execute("CREATE INDEX IF NOT EXISTS idx_detection_species_zone ON detection(species, zone)")
        
        # Index on user username for login queries
        conn.execute("CREATE INDEX IF NOT EXISTS idx_user_username ON user(username)")
        
        # Index for geospatial queries if using lat/lng
        conn.execute("CREATE INDEX IF NOT EXISTS idx_detection_location ON detection(latitude, longitude)")
        
        conn.commit()
        logger.info("Database indexes created successfully")


def create_app():
    """Initialize the database and return the Flask app."""
    app.config["SQLALCHEMY_DATABASE_URI"] = config.database.uri
    app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
        "pool_size": config.database.pool_size,
        "max_overflow": config.database.max_overflow,
        "pool_timeout": config.database.pool_timeout,
        "pool_recycle": config.database.pool_recycle,
        "pool_pre_ping": config.database.pool_pre_ping,
        "echo": config.database.echo,
        "echo_pool": config.database.echo_pool
    }
    db.init_app(app)
    with app.app_context():
        # Database should be created using database/create_database.sql
        # This just verifies the connection and tables exist
        try:
            # Test database connection with a timeout
            from sqlalchemy import create_engine
            
            # Only add connect_timeout for PostgreSQL
            connect_args = {}
            if "postgresql" in config.database.uri or "postgres" in config.database.uri:
                connect_args = {"connect_timeout": 5}  # 5 second timeout
            
            engine = create_engine(
                config.database.uri,
                connect_args=connect_args
            )
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            logger.info("Database connection successful")
            
            # Create tables if they don't exist (for development)
            db.create_all()
            logger.info("Database tables ensured")
            
        except DatabaseError as e:
            logger.error(f"Database initialization error: {e}")
            logger.error("Please ensure PostgreSQL is running and the database exists")
            logger.error("To create the database schema, run:")
            logger.error("  1. export DB_HOST=localhost DB_PORT=5432 DB_NAME=nightscan DB_USER=nightscan DB_PASSWORD=your_password")
            logger.error("  2. cd database && ./init_database.sh")
            # Don't raise error in development to allow app to start
            if os.environ.get('FLASK_ENV') != 'development':
                raise ConfigurationError(
                    message=f"Database not properly configured: {e}",
                    config_key="SQLALCHEMY_DATABASE_URI",
                    user_message="Database configuration error. Please contact support."
                )
        except Exception as e:
            logger.error(f"Unexpected database initialization error: {e}")
            # Don't raise error in development to allow app to start
            if os.environ.get('FLASK_ENV') != 'development':
                error = convert_exception(e, operation='database_initialization')
                raise error
    return app


application = create_app()

# Register API v1 blueprint and OpenAPI docs
with application.app_context():
    application.register_blueprint(api_v1)
    application.register_blueprint(analytics_bp)
    application.register_blueprint(cache_monitor_bp)
    application.register_blueprint(auth_bp)
    application.register_blueprint(health_bp)
    create_openapi_endpoint(application)
    
    # Initialize Flask-Mail
    from web.app_extensions import init_mail
    init_mail(application)
    
    # Initialize Session-JWT Bridge
    init_session_jwt_bridge(application)
    
    # Register password reset blueprint
    try:
        from web.password_reset import password_reset_bp
        application.register_blueprint(password_reset_bp)
        logger.info("Password reset blueprint registered successfully")
    except ImportError as e:
        logger.warning(f"Could not import password reset blueprint: {e}")
    
    # Initialize WebSocket integration
    websocket_integration = FlaskWebSocketIntegration(application)


if __name__ == "__main__":
    application.run(host="0.0.0.0", port=8000)
