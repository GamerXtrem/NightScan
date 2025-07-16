"""
Custom exception classes for NightScan application.

This module defines a hierarchical exception system that provides specific
exception types for different error scenarios, improving error handling,
debugging, and security throughout the application.
"""

from typing import Optional, Dict, Any


class NightScanError(Exception):
    """
    Base exception class for all NightScan application errors.
    
    Provides common functionality for error codes, user messages,
    and additional context information.
    """
    
    def __init__(self, message: str, code: str = None, details: Dict[str, Any] = None, user_message: str = None):
        super().__init__(message)
        self.message = message
        self.code = code or self.__class__.__name__.upper()
        self.details = details or {}
        self.user_message = user_message or message
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for API responses."""
        return {
            'error': self.user_message,
            'code': self.code,
            'details': self.details
        }


# Authentication and Authorization Exceptions
class AuthenticationError(NightScanError):
    """Base class for authentication-related errors."""
    pass


class InvalidCredentialsError(AuthenticationError):
    """Raised when user provides invalid username/password."""
    
    def __init__(self, username: str = None):
        super().__init__(
            message=f"Invalid credentials for user: {username}" if username else "Invalid credentials",
            code="INVALID_CREDENTIALS",
            user_message="Invalid username or password. Please try again."
        )


class AccountLockedError(AuthenticationError):
    """Raised when user account is locked due to too many failed attempts."""
    
    def __init__(self, username: str, unlock_time: str = None):
        super().__init__(
            message=f"Account locked for user: {username}",
            code="ACCOUNT_LOCKED",
            details={'username': username, 'unlock_time': unlock_time},
            user_message="Account temporarily locked due to multiple failed login attempts. Please try again later."
        )


class SessionExpiredError(AuthenticationError):
    """Raised when user session has expired."""
    
    def __init__(self):
        super().__init__(
            message="User session has expired",
            code="SESSION_EXPIRED",
            user_message="Your session has expired. Please log in again."
        )


class InsufficientPermissionsError(AuthenticationError):
    """Raised when user lacks required permissions for an action."""
    
    def __init__(self, required_permission: str = None, user_role: str = None):
        super().__init__(
            message=f"Insufficient permissions. Required: {required_permission}, User role: {user_role}",
            code="INSUFFICIENT_PERMISSIONS",
            details={'required_permission': required_permission, 'user_role': user_role},
            user_message="You don't have permission to perform this action."
        )


class CaptchaValidationError(AuthenticationError):
    """Raised when captcha validation fails."""
    
    def __init__(self):
        super().__init__(
            message="Captcha validation failed",
            code="INVALID_CAPTCHA",
            user_message="Invalid captcha. Please try again."
        )


# File Upload and Processing Exceptions
class FileError(NightScanError):
    """Base class for file-related errors."""
    pass


class UnsupportedFileTypeError(FileError):
    """Raised when uploaded file type is not supported."""
    
    def __init__(self, file_type: str, supported_types: list = None):
        supported_types = supported_types or ['wav', 'mp3', 'jpg', 'png']
        super().__init__(
            message=f"Unsupported file type: {file_type}",
            code="UNSUPPORTED_FILE_TYPE",
            details={'file_type': file_type, 'supported_types': supported_types},
            user_message=f"File type '{file_type}' is not supported. Supported types: {', '.join(supported_types)}"
        )


class FileTooBigError(FileError):
    """Raised when uploaded file exceeds size limit."""
    
    def __init__(self, file_size: int, max_size: int):
        super().__init__(
            message=f"File size {file_size} exceeds maximum {max_size} bytes",
            code="FILE_TOO_BIG",
            details={'file_size': file_size, 'max_size': max_size},
            user_message=f"File size exceeds the maximum allowed size of {max_size // (1024*1024)}MB"
        )


class InvalidFileFormatError(FileError):
    """Raised when file format is invalid or corrupted."""
    
    def __init__(self, file_name: str, expected_format: str = None, reason: str = None):
        super().__init__(
            message=f"Invalid file format for {file_name}: {reason}",
            code="INVALID_FILE_FORMAT",
            details={'file_name': file_name, 'expected_format': expected_format, 'reason': reason},
            user_message=f"The file '{file_name}' has an invalid format or is corrupted."
        )


class FileProcessingError(FileError):
    """Raised when file processing fails."""
    
    def __init__(self, file_name: str, operation: str, reason: str = None):
        super().__init__(
            message=f"Failed to {operation} file {file_name}: {reason}",
            code="FILE_PROCESSING_ERROR",
            details={'file_name': file_name, 'operation': operation, 'reason': reason},
            user_message=f"Unable to process file '{file_name}'. Please check the file and try again."
        )


# Quota and Rate Limiting Exceptions
class QuotaError(NightScanError):
    """Base class for quota-related errors."""
    pass


class QuotaExceededError(QuotaError):
    """Raised when user exceeds their quota limits."""
    
    def __init__(self, quota_type: str, current_usage: int, limit: int, reset_time: str = None):
        super().__init__(
            message=f"{quota_type} quota exceeded: {current_usage}/{limit}",
            code="QUOTA_EXCEEDED",
            details={
                'quota_type': quota_type,
                'current_usage': current_usage,
                'limit': limit,
                'reset_time': reset_time
            },
            user_message=f"You have exceeded your {quota_type.lower()} quota. Please upgrade your plan or wait until {reset_time}."
        )


class RateLimitExceededError(QuotaError):
    """Raised when rate limit is exceeded."""
    
    def __init__(self, limit: int, window: str, retry_after: int = None):
        super().__init__(
            message=f"Rate limit exceeded: {limit} requests per {window}",
            code="RATE_LIMIT_EXCEEDED",
            details={'limit': limit, 'window': window, 'retry_after': retry_after},
            user_message=f"Too many requests. Please wait {retry_after} seconds before trying again." if retry_after 
                        else "Too many requests. Please try again later."
        )


# Machine Learning and Prediction Exceptions
class PredictionError(NightScanError):
    """Base class for ML prediction-related errors."""
    pass


class ModelNotAvailableError(PredictionError):
    """Raised when ML model is not available or failed to load."""
    
    def __init__(self, model_name: str, reason: str = None):
        super().__init__(
            message=f"Model '{model_name}' is not available: {reason}",
            code="MODEL_NOT_AVAILABLE",
            details={'model_name': model_name, 'reason': reason},
            user_message="The prediction service is currently unavailable. Please try again later."
        )


class ModelLoadError(PredictionError):
    """Raised when model fails to load."""
    
    def __init__(self, model_path: str, reason: str = None):
        super().__init__(
            message=f"Failed to load model from {model_path}: {reason}",
            code="MODEL_LOAD_ERROR",
            details={'model_path': model_path, 'reason': reason},
            user_message="Unable to load the prediction model. Please contact support."
        )


class PredictionFailedError(PredictionError):
    """Raised when prediction process fails."""
    
    def __init__(self, file_name: str, model_name: str, reason: str = None):
        super().__init__(
            message=f"Prediction failed for {file_name} using {model_name}: {reason}",
            code="PREDICTION_FAILED",
            details={'file_name': file_name, 'model_name': model_name, 'reason': reason},
            user_message="Unable to process your file for prediction. Please check the file format and try again."
        )


class DataPreprocessingError(PredictionError):
    """Raised when data preprocessing fails."""
    
    def __init__(self, file_name: str, step: str, reason: str = None):
        super().__init__(
            message=f"Data preprocessing failed for {file_name} at step '{step}': {reason}",
            code="DATA_PREPROCESSING_ERROR",
            details={'file_name': file_name, 'step': step, 'reason': reason},
            user_message="Unable to prepare your file for analysis. Please check the file format."
        )


# Database Exceptions
class DatabaseError(NightScanError):
    """Base class for database-related errors."""
    pass


class RecordNotFoundError(DatabaseError):
    """Raised when requested database record is not found."""
    
    def __init__(self, model_name: str, identifier: Any):
        super().__init__(
            message=f"{model_name} with identifier {identifier} not found",
            code="RECORD_NOT_FOUND",
            details={'model': model_name, 'identifier': str(identifier)},
            user_message="The requested item was not found."
        )


class DuplicateRecordError(DatabaseError):
    """Raised when attempting to create a duplicate record."""
    
    def __init__(self, model_name: str, field: str, value: Any):
        super().__init__(
            message=f"Duplicate {model_name}: {field}={value} already exists",
            code="DUPLICATE_RECORD",
            details={'model': model_name, 'field': field, 'value': str(value)},
            user_message=f"A record with that {field} already exists."
        )


class DatabaseConnectionError(DatabaseError):
    """Raised when database connection fails."""
    
    def __init__(self, reason: str = None):
        super().__init__(
            message=f"Database connection failed: {reason}",
            code="DATABASE_CONNECTION_ERROR",
            details={'reason': reason},
            user_message="Database temporarily unavailable. Please try again later."
        )


class DatabaseTransactionError(DatabaseError):
    """Raised when database transaction fails."""
    
    def __init__(self, operation: str, reason: str = None):
        super().__init__(
            message=f"Database transaction failed during {operation}: {reason}",
            code="DATABASE_TRANSACTION_ERROR",
            details={'operation': operation, 'reason': reason},
            user_message="Unable to complete the operation. Please try again."
        )


# Configuration and External Service Exceptions
class ConfigurationError(NightScanError):
    """Base class for configuration-related errors."""
    pass


class MissingConfigError(ConfigurationError):
    """Raised when required configuration is missing."""
    
    def __init__(self, config_key: str, config_file: str = None):
        super().__init__(
            message=f"Missing required configuration: {config_key}",
            code="MISSING_CONFIG",
            details={'config_key': config_key, 'config_file': config_file},
            user_message="Application configuration error. Please contact support."
        )


class InvalidConfigError(ConfigurationError):
    """Raised when configuration value is invalid."""
    
    def __init__(self, config_key: str, value: Any, expected_type: str = None):
        super().__init__(
            message=f"Invalid configuration value for {config_key}: {value}",
            code="INVALID_CONFIG",
            details={'config_key': config_key, 'value': str(value), 'expected_type': expected_type},
            user_message="Application configuration error. Please contact support."
        )


class ExternalServiceError(NightScanError):
    """Base class for external service errors."""
    pass


class CircuitBreakerOpenException(ExternalServiceError):
    """Raised when a circuit breaker is in open state."""
    
    def __init__(self, circuit_name: str, reason: str = None):
        super().__init__(
            message=f"Circuit breaker '{circuit_name}' is open",
            code="CIRCUIT_BREAKER_OPEN",
            details={'circuit_name': circuit_name, 'reason': reason},
            user_message="Service temporarily unavailable. Please try again later."
        )


class CacheServiceError(ExternalServiceError):
    """Raised when cache service (Redis) is unavailable."""
    
    def __init__(self, operation: str, reason: str = None):
        super().__init__(
            message=f"Cache service error during {operation}: {reason}",
            code="CACHE_SERVICE_ERROR",
            details={'operation': operation, 'reason': reason},
            user_message="Temporary service issue. Please try again."
        )


class NotificationServiceError(ExternalServiceError):
    """Raised when notification service fails."""
    
    def __init__(self, service: str, reason: str = None):
        super().__init__(
            message=f"Notification service '{service}' failed: {reason}",
            code="NOTIFICATION_SERVICE_ERROR",
            details={'service': service, 'reason': reason},
            user_message="Unable to send notification. The action was completed but you may not receive updates."
        )


# Validation Exceptions
class ValidationError(NightScanError):
    """Base class for validation errors."""
    pass


class InvalidInputError(ValidationError):
    """Raised when input validation fails."""
    
    def __init__(self, field: str, value: Any, reason: str = None):
        super().__init__(
            message=f"Invalid input for field '{field}': {value} - {reason}",
            code="INVALID_INPUT",
            details={'field': field, 'value': str(value), 'reason': reason},
            user_message=f"Invalid value for {field}. {reason}" if reason else f"Invalid value for {field}."
        )


class RequiredFieldError(ValidationError):
    """Raised when required field is missing."""
    
    def __init__(self, field: str):
        super().__init__(
            message=f"Required field missing: {field}",
            code="REQUIRED_FIELD_MISSING",
            details={'field': field},
            user_message=f"The field '{field}' is required."
        )


# Hardware and System Exceptions (for Raspberry Pi components)
class HardwareError(NightScanError):
    """Base class for hardware-related errors."""
    pass


class CameraError(HardwareError):
    """Raised when camera operations fail."""
    
    def __init__(self, operation: str, reason: str = None):
        super().__init__(
            message=f"Camera {operation} failed: {reason}",
            code="CAMERA_ERROR",
            details={'operation': operation, 'reason': reason},
            user_message="Camera is currently unavailable. Please check the hardware connection."
        )


class GPIOError(HardwareError):
    """Raised when GPIO operations fail."""
    
    def __init__(self, pin: int, operation: str, reason: str = None):
        super().__init__(
            message=f"GPIO {operation} failed on pin {pin}: {reason}",
            code="GPIO_ERROR",
            details={'pin': pin, 'operation': operation, 'reason': reason},
            user_message="Hardware control error. Please check device connections."
        )


class StorageError(HardwareError):
    """Raised when storage operations fail."""
    
    def __init__(self, operation: str, path: str = None, reason: str = None):
        super().__init__(
            message=f"Storage {operation} failed{f' on {path}' if path else ''}: {reason}",
            code="STORAGE_ERROR",
            details={'operation': operation, 'path': path, 'reason': reason},
            user_message="Storage operation failed. Please check available disk space."
        )


def handle_exception_safely(func):
    """
    Decorator to safely handle exceptions and convert them to NightScanError.
    
    Usage:
        @handle_exception_safely
        def risky_operation():
            # This will catch any unhandled exceptions and wrap them
            pass
    """
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except NightScanError:
            # Re-raise our custom exceptions
            raise
        except Exception as e:
            # Convert unexpected exceptions to NightScanError
            raise NightScanError(
                message=f"Unexpected error in {func.__name__}: {str(e)}",
                code="UNEXPECTED_ERROR",
                details={'function': func.__name__, 'original_error': str(e)},
                user_message="An unexpected error occurred. Please try again or contact support."
            ) from e
    return wrapper


# Exception mapping for common standard library exceptions
EXCEPTION_MAPPING = {
    FileNotFoundError: lambda e, **ctx: FileProcessingError(
        file_name=ctx.get('file_name', 'unknown'), 
        operation='access', 
        reason='File not found'
    ),
    PermissionError: lambda e, **ctx: FileProcessingError(
        file_name=ctx.get('file_name', 'unknown'), 
        operation='access', 
        reason='Permission denied'
    ),
    ValueError: lambda e, **ctx: InvalidInputError(
        field=ctx.get('field', 'unknown'), 
        value=ctx.get('value', 'unknown'), 
        reason=str(e)
    ),
    TypeError: lambda e, **ctx: InvalidInputError(
        field=ctx.get('field', 'unknown'), 
        value=ctx.get('value', 'unknown'), 
        reason=f'Type error: {str(e)}'
    ),
}


def convert_exception(exception: Exception, **context) -> NightScanError:
    """
    Convert standard library exceptions to NightScan custom exceptions.
    
    Args:
        exception: The original exception
        **context: Additional context for creating specific error messages
        
    Returns:
        NightScanError: Converted custom exception
    """
    exception_type = type(exception)
    
    if exception_type in EXCEPTION_MAPPING:
        return EXCEPTION_MAPPING[exception_type](exception, **context)
    
    # Default conversion for unmapped exceptions
    return NightScanError(
        message=f"Converted from {exception_type.__name__}: {str(exception)}",
        code="CONVERTED_ERROR",
        details={'original_type': exception_type.__name__, 'original_message': str(exception)},
        user_message="An error occurred while processing your request."
    )