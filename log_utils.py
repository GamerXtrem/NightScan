import logging
import json
import os
import time
import uuid
from typing import Any, Optional
from contextvars import ContextVar

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


def setup_logging(level: int = logging.INFO, stream=None, use_json: bool = True) -> None:
    """Configure root logger with enhanced JSON formatting."""
    handler = logging.StreamHandler(stream)
    
    if use_json:
        handler.setFormatter(EnhancedJSONFormatter())
    else:
        # Fallback to simple format for development
        formatter = logging.Formatter(
            '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        )
        handler.setFormatter(formatter)
    
    # Remove existing handlers to avoid duplicates
    root_logger = logging.getLogger()
    for existing_handler in root_logger.handlers[:]:
        root_logger.removeHandler(existing_handler)
    
    root_logger.addHandler(handler)
    root_logger.setLevel(level)


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