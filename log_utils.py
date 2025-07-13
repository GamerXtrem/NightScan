import logging
import logging.handlers
import json
import os
import time
import uuid
import gzip
import shutil
from pathlib import Path
from typing import Any, Optional, Dict, List, Union
from contextvars import ContextVar
from datetime import datetime

# Import secure logging components
try:
    from secure_logging_filters import SecureLoggingFilter, SecureJSONFormatter, get_secure_logging_manager
    SECURE_LOGGING_AVAILABLE = True
except ImportError:
    SECURE_LOGGING_AVAILABLE = False

try:
    from flask import request, g, has_request_context
except ImportError:
    # Flask not available (e.g., in standalone scripts)
    def has_request_context():
        return False
    
    class MockG:
        pass
    
    g = MockG()
    request = None


# Context variables for request tracing
request_id_var: ContextVar[Optional[str]] = ContextVar('request_id', default=None)
user_id_var: ContextVar[Optional[int]] = ContextVar('user_id', default=None)

# Global configuration instance
_log_config: Optional[LogRotationConfig] = None
_specialized_loggers: Dict[str, logging.Logger] = {}


class JSONFormatter(logging.Formatter):
    """Simple JSON log formatter."""

    def format(self, record: logging.LogRecord) -> str:
        log_record: dict[str, Any] = {
            "time": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            log_record["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(log_record)


class EnhancedJSONFormatter(logging.Formatter):
    """Enhanced JSON log formatter with request context and metrics."""

    def format(self, record: logging.LogRecord) -> str:
        # Base log record
        log_record: dict[str, Any] = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S.%fZ"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "process_id": os.getpid(),
            "thread_id": record.thread
        }
        
        # Add request context if available
        if has_request_context() and request:
            try:
                log_record["request"] = {
                    "method": request.method,
                    "path": request.path,
                    "remote_addr": request.remote_addr,
                    "user_agent": request.headers.get('User-Agent', '')[:200],  # Truncate
                    "request_id": getattr(g, 'request_id', None)
                }
                
                # Add user context if authenticated
                if hasattr(g, 'current_user') and g.current_user:
                    log_record["user_id"] = g.current_user.id
                    
            except Exception:
                # Don't fail logging if request context is unavailable
                pass
        
        # Add context variables
        if request_id_var.get():
            log_record["request_id"] = request_id_var.get()
        if user_id_var.get():
            log_record["user_id"] = user_id_var.get()
        
        # Add exception info
        if record.exc_info:
            log_record["exception"] = {
                "class": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "traceback": self.formatException(record.exc_info)
            }
        
        # Add custom fields from record
        for key, value in record.__dict__.items():
            if key.startswith('custom_') or key in ['duration', 'status_code', 'file_size']:
                log_record[key] = value
        
        return json.dumps(log_record, default=str)


class CompressedRotatingFileHandler(logging.handlers.RotatingFileHandler):
    """RotatingFileHandler that compresses old log files with gzip."""
    
    def doRollover(self):
        """Override to add gzip compression of rotated files."""
        if self.stream:
            self.stream.close()
            self.stream = None
        
        if self.backupCount > 0:
            for i in range(self.backupCount - 1, 0, -1):
                sfn = self.rotation_filename("%s.%d.gz" % (self.baseFilename, i))
                dfn = self.rotation_filename("%s.%d.gz" % (self.baseFilename, i + 1))
                if os.path.exists(sfn):
                    if os.path.exists(dfn):
                        os.remove(dfn)
                    os.rename(sfn, dfn)
            
            # Compress the current log file
            dfn = self.rotation_filename(self.baseFilename + ".1")
            if os.path.exists(dfn):
                os.remove(dfn)
            
            # Move current file to .1
            if os.path.exists(self.baseFilename):
                os.rename(self.baseFilename, dfn)
                
                # Compress the .1 file
                compressed_name = dfn + ".gz"
                with open(dfn, 'rb') as f_in:
                    with gzip.open(compressed_name, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                os.remove(dfn)
        
        if not self.delay:
            self.stream = self._open()


class LogRotationConfig:
    """Configuration for log rotation settings."""
    
    def __init__(self,
                 log_dir: str = "logs",
                 max_file_size: int = 10 * 1024 * 1024,  # 10MB
                 backup_count: int = 5,
                 compress_backups: bool = True,
                 use_json: bool = True,
                 level: str = "INFO",
                 enable_console: bool = True,
                 rotation_time: str = "midnight",  # for TimedRotatingFileHandler
                 retention_days: int = 30):
        self.log_dir = Path(log_dir)
        self.max_file_size = max_file_size
        self.backup_count = backup_count
        self.compress_backups = compress_backups
        self.use_json = use_json
        self.level = getattr(logging, level.upper(), logging.INFO)
        self.enable_console = enable_console
        self.rotation_time = rotation_time
        self.retention_days = retention_days
        
        # Ensure log directory exists
        self.log_dir.mkdir(parents=True, exist_ok=True)


def setup_logging(config: Optional[LogRotationConfig] = None, 
                  log_file: Optional[str] = None,
                  level: int = logging.INFO, 
                  stream=None, 
                  use_json: bool = True,
                  enable_secure_logging: bool = True) -> None:
    """Configure comprehensive logging with rotation and security support."""
    
    # Use provided config or create default
    if config is None:
        config = LogRotationConfig(
            level=logging.getLevelName(level),
            use_json=use_json
        )
    
    # Remove existing handlers to avoid duplicates
    root_logger = logging.getLogger()
    for existing_handler in root_logger.handlers[:]:
        root_logger.removeHandler(existing_handler)
    
    handlers = []
    
    # Console handler (if enabled)
    if config.enable_console:
        console_handler = logging.StreamHandler(stream)
        
        # Use secure formatter if available and enabled
        if config.use_json and SECURE_LOGGING_AVAILABLE and enable_secure_logging:
            console_handler.setFormatter(SecureJSONFormatter())
        elif config.use_json:
            console_handler.setFormatter(EnhancedJSONFormatter())
        else:
            console_handler.setFormatter(logging.Formatter(
                '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
            ))
        handlers.append(console_handler)
    
    # File handler with rotation (if log_file specified or from environment)
    log_file_path = log_file or os.environ.get('NIGHTSCAN_LOG_FILE')
    if log_file_path:
        log_path = Path(log_file_path)
        
        # Ensure directory exists
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        if config.compress_backups:
            file_handler = CompressedRotatingFileHandler(
                filename=str(log_path),
                maxBytes=config.max_file_size,
                backupCount=config.backup_count,
                encoding='utf-8'
            )
        else:
            file_handler = logging.handlers.RotatingFileHandler(
                filename=str(log_path),
                maxBytes=config.max_file_size,
                backupCount=config.backup_count,
                encoding='utf-8'
            )
        
        # Use secure formatter if available and enabled
        if config.use_json and SECURE_LOGGING_AVAILABLE and enable_secure_logging:
            file_handler.setFormatter(SecureJSONFormatter())
        elif config.use_json:
            file_handler.setFormatter(EnhancedJSONFormatter())
        else:
            file_handler.setFormatter(logging.Formatter(
                '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
            ))
        handlers.append(file_handler)
    
    # Add all handlers to root logger
    for handler in handlers:
        root_logger.addHandler(handler)
    
    # Add secure logging filter if available and enabled
    if SECURE_LOGGING_AVAILABLE and enable_secure_logging:
        secure_filter = SecureLoggingFilter()
        root_logger.addFilter(secure_filter)
    
    root_logger.setLevel(config.level)


def setup_specialized_loggers(config: LogRotationConfig) -> Dict[str, logging.Logger]:
    """Setup specialized loggers for different components."""
    loggers = {}
    
    specialized_logs = {
        'security': 'security.log',
        'audit': 'audit.log',
        'access': 'access.log',
        'performance': 'performance.log',
        'prediction': 'prediction.log',
        'circuit_breaker': 'circuit_breaker.log',
        'celery': 'celery.log',
        'api': 'api.log'
    }
    
    for logger_name, log_filename in specialized_logs.items():
        log_path = config.log_dir / log_filename
        
        # Create logger
        logger = logging.getLogger(f'nightscan.{logger_name}')
        logger.setLevel(config.level)
        
        # Remove existing handlers
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # Create rotating file handler
        if config.compress_backups:
            file_handler = CompressedRotatingFileHandler(
                filename=str(log_path),
                maxBytes=config.max_file_size,
                backupCount=config.backup_count,
                encoding='utf-8'
            )
        else:
            file_handler = logging.handlers.RotatingFileHandler(
                filename=str(log_path),
                maxBytes=config.max_file_size,
                backupCount=config.backup_count,
                encoding='utf-8'
            )
        
        # Set formatter
        if config.use_json:
            file_handler.setFormatter(EnhancedJSONFormatter())
        else:
            file_handler.setFormatter(logging.Formatter(
                '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
            ))
        
        logger.addHandler(file_handler)
        
        # Don't propagate to root logger to avoid duplication
        logger.propagate = False
        
        loggers[logger_name] = logger
    
    return loggers


def setup_environment_logging() -> None:
    """Setup logging based on environment variables and configuration."""
    environment = os.environ.get('NIGHTSCAN_ENV', 'development')
    
    if environment == 'development':
        config = LogRotationConfig(
            log_dir="logs",
            max_file_size=10 * 1024 * 1024,  # 10MB
            backup_count=5,
            level="DEBUG",
            enable_console=True,
            compress_backups=False,  # No compression in dev
            retention_days=7
        )
        log_file = "logs/dev.log"
        
    elif environment == 'staging':
        config = LogRotationConfig(
            log_dir="/var/log/nightscan",
            max_file_size=50 * 1024 * 1024,  # 50MB
            backup_count=10,
            level="INFO",
            enable_console=True,
            compress_backups=True,
            retention_days=30
        )
        log_file = "/var/log/nightscan/staging.log"
        
    elif environment == 'production':
        config = LogRotationConfig(
            log_dir="/var/log/nightscan",
            max_file_size=100 * 1024 * 1024,  # 100MB
            backup_count=20,
            level="INFO",
            enable_console=False,  # No console in production
            compress_backups=True,
            retention_days=90
        )
        log_file = "/var/log/nightscan/app.log"
        
    else:  # raspberry_pi or custom
        config = LogRotationConfig(
            log_dir=os.environ.get('NIGHTSCAN_LOG_DIR', '/home/pi/nightscan/logs'),
            max_file_size=20 * 1024 * 1024,  # 20MB
            backup_count=7,
            level="INFO",
            enable_console=True,
            compress_backups=True,
            retention_days=14
        )
        log_file = os.path.join(config.log_dir, "nightscan.log")
    
    # Override with environment variables if provided
    log_file = os.environ.get('NIGHTSCAN_LOG_FILE', log_file)
    
    # Setup main logging
    setup_logging(config=config, log_file=log_file)
    
    # Setup specialized loggers
    specialized_loggers = setup_specialized_loggers(config)
    
    # Log initialization
    logger = logging.getLogger('nightscan.init')
    logger.info(f"Logging initialized for environment: {environment}")
    logger.info(f"Main log file: {log_file}")
    logger.info(f"Log directory: {config.log_dir}")
    logger.info(f"Specialized loggers: {list(specialized_loggers.keys())}")
    
    return config, specialized_loggers


def get_request_id() -> str:
    """Get or create a request ID for tracing."""
    if has_request_context():
        if not hasattr(g, 'request_id'):
            g.request_id = str(uuid.uuid4())
        return g.request_id
    return str(uuid.uuid4())


def log_request_start(method: str, path: str, **kwargs) -> None:
    """Log the start of a request with timing."""
    request_id = get_request_id()
    logger = logging.getLogger('nightscan.request')
    logger.info(
        "Request started",
        extra={
            "custom_event": "request_start",
            "custom_method": method,
            "custom_path": path,
            "custom_start_time": time.time(),
            **kwargs
        }
    )


def log_request_end(method: str, path: str, status_code: int, duration: float, **kwargs) -> None:
    """Log the end of a request with metrics."""
    logger = logging.getLogger('nightscan.request')
    logger.info(
        f"Request completed: {method} {path} {status_code} ({duration:.3f}s)",
        extra={
            "custom_event": "request_end",
            "custom_method": method,
            "custom_path": path,
            "status_code": status_code,
            "duration": duration,
            **kwargs
        }
    )


def log_prediction(filename: str, duration: float, result_count: int, **kwargs) -> None:
    """Log prediction events with metrics."""
    logger = logging.getLogger('nightscan.prediction')
    logger.info(
        f"Prediction completed: {filename} ({duration:.3f}s, {result_count} results)",
        extra={
            "custom_event": "prediction",
            "custom_filename": filename,
            "duration": duration,
            "custom_result_count": result_count,
            **kwargs
        }
    )


def log_security_event(event_type: str, details: Dict[str, Any]) -> None:
    """Log security events to specialized security logger."""
    logger = logging.getLogger('nightscan.security')
    logger.warning(
        f"Security event: {event_type}",
        extra={
            "custom_event": "security",
            "custom_event_type": event_type,
            "custom_details": details,
            "timestamp": datetime.utcnow().isoformat()
        }
    )


def log_audit_event(action: str, user_id: Optional[int] = None, details: Dict[str, Any] = None) -> None:
    """Log audit events for compliance and tracking."""
    logger = logging.getLogger('nightscan.audit')
    logger.info(
        f"Audit: {action}",
        extra={
            "custom_event": "audit",
            "custom_action": action,
            "custom_user_id": user_id,
            "custom_details": details or {},
            "timestamp": datetime.utcnow().isoformat()
        }
    )


def log_performance_metric(metric_name: str, value: float, unit: str = "ms", **kwargs) -> None:
    """Log performance metrics for monitoring."""
    logger = logging.getLogger('nightscan.performance')
    logger.info(
        f"Performance: {metric_name} = {value}{unit}",
        extra={
            "custom_event": "performance",
            "custom_metric": metric_name,
            "custom_value": value,
            "custom_unit": unit,
            "timestamp": datetime.utcnow().isoformat(),
            **kwargs
        }
    )


def log_circuit_breaker_event(circuit_name: str, event_type: str, details: Dict[str, Any] = None) -> None:
    """Log circuit breaker state changes and events."""
    logger = logging.getLogger('nightscan.circuit_breaker')
    logger.info(
        f"Circuit breaker {circuit_name}: {event_type}",
        extra={
            "custom_event": "circuit_breaker",
            "custom_circuit": circuit_name,
            "custom_event_type": event_type,
            "custom_details": details or {},
            "timestamp": datetime.utcnow().isoformat()
        }
    )


def get_log_file_info(log_file_path: str) -> Dict[str, Any]:
    """Get information about a log file (size, modification time, etc.)."""
    try:
        log_path = Path(log_file_path)
        if not log_path.exists():
            return {"exists": False, "error": "File not found"}
        
        stat = log_path.stat()
        return {
            "exists": True,
            "size_bytes": stat.st_size,
            "size_mb": round(stat.st_size / (1024 * 1024), 2),
            "modified_time": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "permissions": oct(stat.st_mode)[-3:],
            "is_writable": os.access(log_file_path, os.W_OK)
        }
    except Exception as e:
        return {"exists": False, "error": str(e)}


def cleanup_old_logs(log_dir: str, retention_days: int = 30) -> Dict[str, Any]:
    """Clean up old log files beyond retention period."""
    try:
        log_path = Path(log_dir)
        if not log_path.exists():
            return {"error": f"Log directory {log_dir} does not exist"}
        
        cutoff_time = time.time() - (retention_days * 24 * 3600)
        removed_files = []
        total_size_freed = 0
        
        # Find old log files (including compressed ones)
        for log_file in log_path.glob("*.log*"):
            if log_file.stat().st_mtime < cutoff_time:
                size = log_file.stat().st_size
                removed_files.append({
                    "file": str(log_file),
                    "size_mb": round(size / (1024 * 1024), 2),
                    "age_days": round((time.time() - log_file.stat().st_mtime) / (24 * 3600), 1)
                })
                total_size_freed += size
                log_file.unlink()
        
        return {
            "success": True,
            "removed_files": len(removed_files),
            "files_details": removed_files,
            "total_size_freed_mb": round(total_size_freed / (1024 * 1024), 2),
            "retention_days": retention_days
        }
        
    except Exception as e:
        return {"error": str(e)}


def get_disk_usage(log_dir: str) -> Dict[str, Any]:
    """Get disk usage information for log directory."""
    try:
        log_path = Path(log_dir)
        if not log_path.exists():
            return {"error": f"Log directory {log_dir} does not exist"}
        
        # Get disk usage
        statvfs = os.statvfs(log_dir)
        total_bytes = statvfs.f_frsize * statvfs.f_blocks
        free_bytes = statvfs.f_frsize * statvfs.f_bavail
        used_bytes = total_bytes - free_bytes
        
        # Get log directory size
        log_dir_size = sum(f.stat().st_size for f in log_path.glob("**/*") if f.is_file())
        
        return {
            "disk_total_gb": round(total_bytes / (1024**3), 2),
            "disk_used_gb": round(used_bytes / (1024**3), 2),
            "disk_free_gb": round(free_bytes / (1024**3), 2),
            "disk_usage_percent": round((used_bytes / total_bytes) * 100, 1),
            "log_dir_size_mb": round(log_dir_size / (1024**2), 2),
            "log_dir_percent_of_disk": round((log_dir_size / total_bytes) * 100, 3)
        }
        
    except Exception as e:
        return {"error": str(e)}