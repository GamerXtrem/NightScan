"""Prometheus metrics collection for NightScan."""

import time
from typing import Optional
from functools import wraps

try:
    from prometheus_client import Counter, Histogram, Gauge, Info, generate_latest, CONTENT_TYPE_LATEST
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    # Mock classes when prometheus_client is not available
    class Counter:
        def __init__(self, *args, **kwargs): pass
        def inc(self, *args, **kwargs): pass
        def labels(self, *args, **kwargs): return self
    
    class Histogram:
        def __init__(self, *args, **kwargs): pass
        def observe(self, *args, **kwargs): pass
        def time(self): return MockContextManager()
        def labels(self, *args, **kwargs): return self
    
    class Gauge:
        def __init__(self, *args, **kwargs): pass
        def set(self, *args, **kwargs): pass
        def inc(self, *args, **kwargs): pass
        def dec(self, *args, **kwargs): pass
        def labels(self, *args, **kwargs): return self
    
    class Info:
        def __init__(self, *args, **kwargs): pass
        def info(self, *args, **kwargs): pass
    
    class MockContextManager:
        def __enter__(self): return self
        def __exit__(self, *args): pass
    
    def generate_latest(): return ""
    CONTENT_TYPE_LATEST = "text/plain"


# Application info
app_info = Info('nightscan_info', 'Information about NightScan application')
app_info.info({
    'version': '1.0.0',
    'component': 'nightscan'
})

# Request metrics
http_requests_total = Counter(
    'nightscan_http_requests_total',
    'Total number of HTTP requests',
    ['method', 'endpoint', 'status']
)

http_request_duration = Histogram(
    'nightscan_http_request_duration_seconds',
    'HTTP request latency',
    ['method', 'endpoint']
)

# Prediction metrics
predictions_total = Counter(
    'nightscan_predictions_total',
    'Total number of predictions made',
    ['status']  # success, error
)

prediction_duration = Histogram(
    'nightscan_prediction_duration_seconds',
    'Time spent processing predictions',
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0]
)

audio_file_size = Histogram(
    'nightscan_audio_file_size_bytes',
    'Size of uploaded audio files',
    buckets=[1024, 10240, 102400, 1048576, 10485760, 104857600]  # 1KB to 100MB
)

audio_duration_seconds = Histogram(
    'nightscan_audio_duration_seconds',
    'Duration of processed audio files',
    buckets=[1, 5, 10, 30, 60, 120, 300, 600]  # 1s to 10min
)

# User activity metrics
active_users = Gauge(
    'nightscan_active_users',
    'Number of currently active users'
)

upload_quota_usage = Histogram(
    'nightscan_upload_quota_usage_percent',
    'User upload quota usage percentage',
    buckets=[10, 25, 50, 75, 90, 95, 99]
)

# System metrics
failed_logins = Counter(
    'nightscan_failed_logins_total',
    'Total number of failed login attempts',
    ['reason']  # invalid_credentials, invalid_captcha, locked_out
)

celery_queue_size = Gauge(
    'nightscan_celery_queue_size',
    'Number of tasks in Celery queue'
)

# Database metrics
database_connections = Gauge(
    'nightscan_database_connections',
    'Number of active database connections'
)

# Detection metrics
detections_total = Counter(
    'nightscan_detections_total',
    'Total number of wildlife detections',
    ['species', 'zone']
)


def track_request_metrics(func):
    """Decorator to track HTTP request metrics."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        if not PROMETHEUS_AVAILABLE:
            return func(*args, **kwargs)
        
        from flask import request
        method = request.method
        endpoint = request.endpoint or 'unknown'
        
        start_time = time.time()
        try:
            response = func(*args, **kwargs)
            status = getattr(response, 'status_code', 200)
            http_requests_total.labels(method=method, endpoint=endpoint, status=status).inc()
            return response
        except Exception as e:
            http_requests_total.labels(method=method, endpoint=endpoint, status=500).inc()
            raise
        finally:
            duration = time.time() - start_time
            http_request_duration.labels(method=method, endpoint=endpoint).observe(duration)
    
    return wrapper


def record_prediction_metrics(duration: float, success: bool, file_size: Optional[int] = None, 
                            audio_duration: Optional[float] = None):
    """Record metrics for a prediction operation."""
    if not PROMETHEUS_AVAILABLE:
        return
    
    status = 'success' if success else 'error'
    predictions_total.labels(status=status).inc()
    prediction_duration.observe(duration)
    
    if file_size is not None:
        audio_file_size.observe(file_size)
    
    if audio_duration is not None:
        audio_duration_seconds.observe(audio_duration)


def record_failed_login(reason: str):
    """Record a failed login attempt."""
    if not PROMETHEUS_AVAILABLE:
        return
    
    failed_logins.labels(reason=reason).inc()


def update_active_users(count: int):
    """Update the number of active users."""
    if not PROMETHEUS_AVAILABLE:
        return
    
    active_users.set(count)


def record_quota_usage(usage_percent: float):
    """Record user quota usage."""
    if not PROMETHEUS_AVAILABLE:
        return
    
    upload_quota_usage.observe(usage_percent)


def record_detection(species: str, zone: str = 'unknown'):
    """Record a wildlife detection."""
    if not PROMETHEUS_AVAILABLE:
        return
    
    detections_total.labels(species=species, zone=zone).inc()


def update_celery_queue_size(size: int):
    """Update Celery queue size metric."""
    if not PROMETHEUS_AVAILABLE:
        return
    
    celery_queue_size.set(size)


def get_metrics():
    """Get all metrics in Prometheus format."""
    if not PROMETHEUS_AVAILABLE:
        return "# Prometheus client not installed\n"
    
    return generate_latest()