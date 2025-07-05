import os
import re
import time
import mimetypes
import secrets
import logging
import random
import requests
import json
from datetime import datetime, timedelta
import psutil

from log_utils import setup_logging

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

logger = logging.getLogger(__name__)
setup_logging()

# Load configuration
config = get_config()

app = Flask(__name__)
app.secret_key = config.security.secret_key
app.config["WTF_CSRF_SECRET_KEY"] = config.security.csrf_secret_key or config.security.secret_key
csrf = CSRFProtect(app)
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy()

# Enhanced Content Security Policy
csp = {
    "default-src": "'self'",
    "script-src": "'self' 'unsafe-inline'",  # Allow inline scripts for forms
    "style-src": "'self' 'unsafe-inline'",   # Allow inline styles
    "img-src": "'self' data:",              # Allow data URLs for images
    "font-src": "'self'",
    "connect-src": "'self'",
    "media-src": "'self'",
    "object-src": "'none'",
    "base-uri": "'self'",
    "form-action": "'self'",
    "frame-ancestors": "'none'",
    "upgrade-insecure-requests": True
}

# Enhanced security headers
Talisman(app, 
    force_https=True, 
    frame_options="DENY",
    content_security_policy=csp,
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


CAPTCHA_OPERATIONS = ["+", "-"]


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
    except Exception as e:
        logger.error(f"Failed to save lockout file: {e}")

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

def generate_captcha() -> str:
    """Create a simple math challenge and store the result in the session."""
    a = random.randint(1, 9)
    b = random.randint(1, 9)
    op = random.choice(CAPTCHA_OPERATIONS)
    answer = a + b if op == "+" else a - b
    session["captcha_answer"] = str(answer)
    return f"{a} {op} {b} = ?"


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

    user = db.relationship("User", backref=db.backref("predictions", lazy=True))


class Detection(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    species = db.Column(db.String(100), nullable=False)
    time = db.Column(db.DateTime, nullable=False, server_default=db.func.now())
    latitude = db.Column(db.Float)
    longitude = db.Column(db.Float)
    zone = db.Column(db.String(100))
    image_url = db.Column(db.String(200))

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "species": self.species,
            "time": self.time.isoformat() if self.time else None,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "zone": self.zone,
            "image": self.image_url,
        }


@login_manager.user_loader
def load_user(user_id: str):
    return User.query.get(int(user_id))


@app.route("/register", methods=["GET", "POST"])
@limiter.limit(config.rate_limit.login_limit if limiter else None)
@track_request_metrics
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
        if request.form.get("captcha") != session.get("captcha_answer"):
            record_failed_login(ip)
            record_failed_login('invalid_captcha')
            flash("Invalid captcha")
            question = generate_captcha()
            return render_template("login.html", captcha_question=question)

        username = validate_input(request.form.get("username", ""), 30)
        password = request.form.get("password", "")
        
        if not username or not password:
            record_failed_login(ip)
            record_failed_login('missing_credentials')
            flash("Please provide username and password")
            question = generate_captcha()
            return render_template("login.html", captcha_question=question)
        
        user = User.query.filter_by(username=username).first()
        if user and user.check_password(password):
            login_user(user)
            clear_failed_logins(ip)
            session.pop("captcha_answer", None)
            return redirect(url_for("index"))
        record_failed_login(ip)
        record_failed_login('invalid_credentials')
        flash("Invalid credentials")

    question = generate_captcha()
    return render_template("login.html", captcha_question=question)


@app.route("/logout")
@login_required
@track_request_metrics
def logout():
    logout_user()
    return redirect(url_for("login"))


@app.route("/", methods=["GET", "POST"])
@login_required
@limiter.limit(config.rate_limit.upload_limit, methods=["POST"] if limiter else None)
@track_request_metrics
def index():
    result = None
    total = (
        db.session.query(db.func.coalesce(db.func.sum(Prediction.file_size), 0))
        .filter_by(user_id=current_user.id)
        .scalar()
        or 0
    )
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
                        data = file.read()
                        pred = Prediction(
                            user_id=current_user.id,
                            filename=original_filename,
                            result="PENDING",
                            file_size=file_size,
                        )
                        db.session.add(pred)
                        db.session.commit()
                        from .tasks import run_prediction

                        run_prediction.delay(
                            pred.id,
                            original_filename,
                            data,
                            PREDICT_API_URL,
                        )
                        total += file_size
                        remaining = MAX_TOTAL_SIZE - total
                        
                        # Record quota usage metrics
                        usage_percent = (total / MAX_TOTAL_SIZE) * 100
                        record_quota_usage(usage_percent)
                        
                        flash("File queued for processing.")
    predictions = (
        Prediction.query.filter_by(user_id=current_user.id)
        .order_by(Prediction.id.desc())
        .all()
    )
    return render_template(
        "index.html",
        result=result,
        predictions=predictions,
        remaining_bytes=remaining,
    )


@app.route("/metrics")
def metrics_endpoint():
    """Prometheus metrics endpoint."""
    return get_metrics(), 200, {'Content-Type': CONTENT_TYPE_LATEST}


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
        db.session.execute("SELECT 1")
        checks["database"] = True
    except Exception as e:
        logger.error(f"Database check failed: {e}")
    
    # Check disk space (> 1GB free)
    try:
        disk_usage = psutil.disk_usage('/')
        free_gb = disk_usage.free / (1024**3)
        checks["disk_space"] = free_gb > 1.0
    except Exception as e:
        logger.error(f"Disk space check failed: {e}")
    
    # Check memory usage (< 90%)
    try:
        memory = psutil.virtual_memory()
        checks["memory"] = memory.percent < 90.0
    except Exception as e:
        logger.error(f"Memory check failed: {e}")
    
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
        "pool_timeout": config.database.pool_timeout,
        "pool_recycle": config.database.pool_recycle,
        "echo": config.database.echo
    }
    db.init_app(app)
    with app.app_context():
        db.create_all()
        try:
            create_database_indexes()
        except Exception as e:
            logger.warning(f"Failed to create some indexes: {e}")
    return app


application = create_app()

# Register API v1 blueprint and OpenAPI docs
with application.app_context():
    application.register_blueprint(api_v1)
    create_openapi_endpoint(application)


if __name__ == "__main__":
    create_app().run(host="0.0.0.0", port=8000)
