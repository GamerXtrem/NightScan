"""
NightScan Unified Security Module

This module consolidates all security features into a single, comprehensive interface.
It provides authentication, authorization, encryption, rate limiting, secure file handling,
and other security features in a unified API.
"""

from typing import Optional, Dict, Any, List
from flask import Flask
import logging

# Import all security components
from .auth import SecurityAuth
from .middleware import SecurityMiddleware
from .headers import SecurityHeaders
from .rate_limiting import RateLimiter
from .encryption import EncryptionManager
from .validation import SecurityValidator
from .uploads import SecureFileHandler
from .logging import SecureLogger
from .session import SessionManager
from .database import DatabaseSecurity
from .utils import SecurityUtils

__all__ = [
    'UnifiedSecurity',
    'SecurityAuth',
    'SecurityMiddleware',
    'SecurityHeaders',
    'RateLimiter',
    'EncryptionManager',
    'SecurityValidator',
    'SecureFileHandler',
    'SecureLogger',
    'SessionManager',
    'DatabaseSecurity',
    'SecurityUtils'
]

logger = logging.getLogger(__name__)


class UnifiedSecurity:
    """
    Unified security interface for NightScan.
    
    This class provides a single entry point for all security features,
    making it easy to initialize and configure security across the application.
    """
    
    def __init__(self, config: Optional[Any] = None):
        """
        Initialize the unified security system.
        
        Args:
            config: Configuration object (from config.py)
        """
        # Import config if not provided
        if config is None:
            from config import get_config
            config = get_config()
        
        self.config = config
        
        # Initialize all security components
        self.auth = SecurityAuth(config)
        self.middleware = SecurityMiddleware(config)
        self.headers = SecurityHeaders(config)
        self.rate_limiter = RateLimiter(config)
        self.encryption = EncryptionManager(config)
        self.validator = SecurityValidator(config)
        self.file_handler = SecureFileHandler(config)
        self.logger = SecureLogger(config)
        self.session_manager = SessionManager(config)
        self.db_security = DatabaseSecurity(config)
        self.utils = SecurityUtils(config)
        
        logger.info("Unified Security System initialized")
    
    def init_app(self, app: Flask) -> None:
        """
        Initialize all security features with a Flask application.
        
        Args:
            app: Flask application instance
        """
        logger.info("Initializing security features with Flask app")
        
        # Initialize components in the correct order
        # 1. Session management (needed by auth)
        self.session_manager.init_app(app)
        
        # 2. Authentication (needed by middleware)
        self.auth.init_app(app)
        
        # 3. Core middleware
        self.middleware.init_app(app)
        
        # 4. Security headers
        self.headers.init_app(app)
        
        # 5. Rate limiting
        self.rate_limiter.init_app(app)
        
        # 6. Other components
        self.file_handler.init_app(app)
        self.logger.init_app(app)
        self.db_security.init_app(app)
        
        # Register security context processor
        @app.context_processor
        def security_context():
            """Add security utilities to template context."""
            return {
                'csrf_token': self.auth.generate_csrf_token,
                'is_authenticated': self.auth.is_authenticated,
                'current_user': self.auth.get_current_user
            }
        
        # Register security error handlers
        self._register_error_handlers(app)
        
        logger.info("Security initialization complete")
    
    def _register_error_handlers(self, app: Flask) -> None:
        """Register security-related error handlers."""
        
        @app.errorhandler(401)
        def unauthorized(e):
            """Handle unauthorized access."""
            self.logger.log_security_event('unauthorized_access', {
                'ip': self.utils.get_client_ip(),
                'path': app.request.path
            })
            return {'error': 'Unauthorized'}, 401
        
        @app.errorhandler(403)
        def forbidden(e):
            """Handle forbidden access."""
            self.logger.log_security_event('forbidden_access', {
                'ip': self.utils.get_client_ip(),
                'path': app.request.path
            })
            return {'error': 'Forbidden'}, 403
        
        @app.errorhandler(429)
        def rate_limit_exceeded(e):
            """Handle rate limit exceeded."""
            self.logger.log_security_event('rate_limit_exceeded', {
                'ip': self.utils.get_client_ip(),
                'path': app.request.path
            })
            return {'error': 'Rate limit exceeded'}, 429
    
    def get_security_status(self) -> Dict[str, Any]:
        """
        Get current security system status.
        
        Returns:
            Dictionary with security component statuses
        """
        return {
            'auth': {
                'enabled': self.config.security.secret_key is not None,
                'lockout_enabled': self.config.security.lockout_threshold > 0,
                'password_min_length': self.config.security.password_min_length
            },
            'rate_limiting': {
                'enabled': self.config.rate_limit.enabled,
                'default_limit': self.config.rate_limit.default_limit
            },
            'encryption': {
                'enabled': hasattr(self.config, 'secrets') and 
                          getattr(self.config.secrets, 'encryption_enabled', False)
            },
            'https': {
                'enforced': self.config.security.force_https,
                'hsts_enabled': getattr(self.config.security, 'hsts_enabled', True)
            },
            'session': {
                'secure_cookies': getattr(self.config.security, 'session_cookie_secure', True),
                'lifetime': getattr(self.config.security, 'session_lifetime', 3600)
            },
            'file_upload': {
                'max_size': self.config.upload.max_file_size,
                'allowed_extensions': getattr(self.config.upload, 'allowed_extensions', [])
            }
        }
    
    def perform_security_check(self) -> List[Dict[str, Any]]:
        """
        Perform a comprehensive security check.
        
        Returns:
            List of security issues found
        """
        issues = []
        
        # Check configuration
        if not self.config.security.secret_key:
            issues.append({
                'severity': 'critical',
                'component': 'auth',
                'issue': 'No secret key configured'
            })
        
        if self.config.environment == 'production':
            if not self.config.security.force_https:
                issues.append({
                    'severity': 'high',
                    'component': 'transport',
                    'issue': 'HTTPS not enforced in production'
                })
            
            if self.config.security.password_min_length < 12:
                issues.append({
                    'severity': 'medium',
                    'component': 'auth',
                    'issue': 'Password minimum length below recommended (12 chars)'
                })
            
            if not self.config.rate_limit.enabled:
                issues.append({
                    'severity': 'medium',
                    'component': 'rate_limiting',
                    'issue': 'Rate limiting disabled in production'
                })
        
        return issues
    
    def apply_security_defaults(self, app: Flask) -> None:
        """
        Apply secure defaults to Flask application.
        
        Args:
            app: Flask application instance
        """
        # Secure session configuration
        app.config.update(
            SESSION_COOKIE_SECURE=self.config.security.force_https,
            SESSION_COOKIE_HTTPONLY=True,
            SESSION_COOKIE_SAMESITE='Lax',
            PERMANENT_SESSION_LIFETIME=self.config.security.get('session_lifetime', 3600),
            
            # Security settings
            SECRET_KEY=self.config.security.secret_key,
            WTF_CSRF_ENABLED=True,
            WTF_CSRF_TIME_LIMIT=None,
            
            # File upload settings
            MAX_CONTENT_LENGTH=self.config.upload.max_file_size,
            
            # JSON security
            JSON_SORT_KEYS=False,
            JSONIFY_PRETTYPRINT_REGULAR=False
        )


# Convenience functions for backward compatibility
_unified_security: Optional[UnifiedSecurity] = None


def get_security() -> UnifiedSecurity:
    """Get the global unified security instance."""
    global _unified_security
    if _unified_security is None:
        _unified_security = UnifiedSecurity()
    return _unified_security


def init_security(app: Flask, config: Optional[Any] = None) -> UnifiedSecurity:
    """
    Initialize security for a Flask application.
    
    Args:
        app: Flask application instance
        config: Optional configuration object
        
    Returns:
        UnifiedSecurity instance
    """
    global _unified_security
    _unified_security = UnifiedSecurity(config)
    _unified_security.init_app(app)
    return _unified_security