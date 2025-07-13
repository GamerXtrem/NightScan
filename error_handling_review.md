# NightScan Error Handling and Logging Review

## Executive Summary
This review identifies critical issues in error handling and logging implementation across the NightScan project. Several patterns need immediate attention to improve system reliability, security, and maintainability.

## 1. Inconsistent Error Handling Patterns

### Issue 1.1: Generic Exception Catching
**Location**: Multiple files including `web/app.py`, `notification_service.py`
```python
except Exception as e:  # Too generic
    logger.error(f"Failed to send email to {email_address}: {e}")
```
**Recommendation**: Catch specific exceptions and handle them appropriately.

### Issue 1.2: Silent Exception Swallowing
**Location**: `log_utils.py:75-77`
```python
except Exception:
    # Don't fail logging if request context is unavailable
    pass
```
**Recommendation**: At minimum, log the exception at debug level to aid troubleshooting.

### Issue 1.3: Inconsistent Error Response Format
**Location**: Various API endpoints
- Some return `{"error": "message"}` 
- Others return `{"error": "message", "code": "ERROR_CODE"}`
- No standardized error schema across the application

**Recommendation**: Implement a standardized error response format:
```python
{
    "error": {
        "message": "Human-readable error message",
        "code": "MACHINE_READABLE_CODE",
        "details": {},  # Optional additional context
        "timestamp": "2024-01-10T12:00:00Z",
        "request_id": "uuid"
    }
}
```

## 2. Missing Try-Catch Blocks for Critical Operations

### Issue 2.1: Unprotected Database Operations
**Location**: `web/app.py:1037-1061` (create_database_indexes)
```python
# Only the entire function is wrapped, individual operations aren't protected
conn.execute("CREATE INDEX IF NOT EXISTS idx_prediction_user_id ON prediction(user_id)")
```
**Recommendation**: Wrap each index creation in try-except to continue on failure.

### Issue 2.2: Unprotected File Operations
**Location**: `NightScanPi/Program/main.py:47`
```python
audio_capture.record_segment(8, audio_dir / audio_filename)
```
**Recommendation**: Add specific exception handling for I/O errors, disk space issues.

### Issue 2.3: Missing Network Operation Protection
**Location**: API calls throughout the codebase
**Recommendation**: Implement retry logic with exponential backoff for network operations.

## 3. Generic Error Messages Exposing System Information

### Issue 3.1: Database Error Exposure
**Location**: `secure_database.py:30`
```python
logger.error(f"Database query failed: {e}")  # May expose schema/query details
```
**Recommendation**: Log full details internally, return sanitized messages to users.

### Issue 3.2: File Path Exposure
**Location**: Various file operations
**Recommendation**: Never expose absolute file paths in error messages to users.

## 4. Logging Configuration Issues

### Issue 4.1: No Log Rotation Configured
**Current State**: Logs can grow indefinitely
**Location**: `log_utils.py`, `config.py`
**Recommendation**: Implement rotating file handlers:
```python
from logging.handlers import RotatingFileHandler

handler = RotatingFileHandler(
    filename='nightscan.log',
    maxBytes=10*1024*1024,  # 10MB
    backupCount=5
)
```

### Issue 4.2: Inconsistent Log Levels
**Location**: Throughout the codebase
- Some files use `logging.info()` for errors
- Critical operations logged at INFO level
- No clear logging level strategy

**Recommendation**: 
- CRITICAL: System failures requiring immediate attention
- ERROR: Operation failures that need investigation
- WARNING: Recoverable issues or deprecated usage
- INFO: Normal operation events
- DEBUG: Detailed diagnostic information

### Issue 4.3: Missing Structured Logging Context
**Location**: Most logging statements
**Recommendation**: Always include relevant context:
```python
logger.error("Operation failed", extra={
    "user_id": user_id,
    "operation": "predict_audio",
    "file_size": file_size,
    "error_type": type(e).__name__
})
```

## 5. Missing Error Recovery Mechanisms

### Issue 5.1: No Circuit Breaker Pattern
**Location**: External service calls (API, database)
**Recommendation**: Implement circuit breaker for external dependencies.

### Issue 5.2: No Graceful Degradation
**Location**: Feature dependencies
**Recommendation**: Allow system to continue with reduced functionality when non-critical services fail.

### Issue 5.3: No Health Check Integration
**Location**: `web/app.py` health endpoints
**Recommendation**: Integrate error rates into health checks.

## 6. Unhandled Exceptions

### Issue 6.1: Missing Global Exception Handler
**Location**: Flask application
**Recommendation**: Add global error handler:
```python
@app.errorhandler(Exception)
def handle_exception(e):
    # Log the error
    logger.exception("Unhandled exception")
    # Return generic error to user
    return jsonify({"error": "Internal server error"}), 500
```

### Issue 6.2: Async Exception Handling
**Location**: WebSocket and async operations
**Recommendation**: Implement proper async exception handling.

## 7. Error Tracking and Monitoring

### Issue 7.1: No Error Aggregation
**Current State**: Errors only logged to files
**Recommendation**: Integrate error tracking service (Sentry, Rollbar).

### Issue 7.2: No Error Metrics
**Location**: `metrics.py`
**Recommendation**: Add error rate metrics:
```python
error_counter = Counter(
    'nightscan_errors_total',
    'Total number of errors',
    ['error_type', 'component']
)
```

## 8. Log Security Issues

### Issue 8.1: Sensitive Data in Logs
**Location**: `secure_logging.py` has patterns but not consistently used
**Recommendation**: 
- Use SecureLogger throughout the application
- Add more patterns for PII detection
- Implement log sanitization middleware

### Issue 8.2: Log Injection Vulnerabilities
**Location**: User input logged directly
**Recommendation**: Sanitize all user input before logging.

## Priority Recommendations

### Immediate (P0):
1. Implement global exception handlers
2. Fix generic exception catching in critical paths
3. Add log rotation to prevent disk space issues
4. Sanitize error messages exposed to users

### Short-term (P1):
1. Standardize error response format
2. Add structured logging context
3. Implement retry logic for network operations
4. Add error rate monitoring

### Medium-term (P2):
1. Implement circuit breaker pattern
2. Add comprehensive error tracking
3. Create error recovery strategies
4. Implement log aggregation

### Long-term (P3):
1. Full observability stack integration
2. Automated error analysis and alerting
3. Self-healing mechanisms
4. Chaos engineering tests

## Code Examples for Implementation

### 1. Standardized Error Handler
```python
class APIError(Exception):
    def __init__(self, message, code, status_code=500, details=None):
        self.message = message
        self.code = code
        self.status_code = status_code
        self.details = details or {}
        super().__init__(self.message)

@app.errorhandler(APIError)
def handle_api_error(error):
    response = {
        "error": {
            "message": error.message,
            "code": error.code,
            "details": error.details,
            "timestamp": datetime.utcnow().isoformat(),
            "request_id": g.get('request_id')
        }
    }
    return jsonify(response), error.status_code
```

### 2. Retry Decorator
```python
def retry_on_exception(max_retries=3, backoff_factor=2, exceptions=(Exception,)):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_retries - 1:
                        raise
                    wait_time = backoff_factor ** attempt
                    logger.warning(f"Attempt {attempt + 1} failed, retrying in {wait_time}s: {e}")
                    time.sleep(wait_time)
            return None
        return wrapper
    return decorator
```

### 3. Circuit Breaker Implementation
```python
class CircuitBreaker:
    def __init__(self, failure_threshold=5, recovery_timeout=60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'closed'  # closed, open, half-open
        
    def call(self, func, *args, **kwargs):
        if self.state == 'open':
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = 'half-open'
            else:
                raise Exception("Circuit breaker is open")
                
        try:
            result = func(*args, **kwargs)
            if self.state == 'half-open':
                self.state = 'closed'
                self.failure_count = 0
            return result
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            if self.failure_count >= self.failure_threshold:
                self.state = 'open'
            raise
```

## Testing Recommendations

1. Add unit tests for all error paths
2. Implement integration tests for error scenarios
3. Add chaos testing for resilience
4. Regular error injection testing

## Conclusion

The NightScan project has basic error handling but lacks consistency and robustness. Implementing these recommendations will significantly improve system reliability, security, and maintainability. Start with P0 items to address the most critical issues, then progressively implement other recommendations based on system requirements and resources.