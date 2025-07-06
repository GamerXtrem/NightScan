"""
Comprehensive Security Middleware for NightScan
Integrates all security components into a unified middleware.
"""

from flask import Flask, request, g, session
from functools import wraps
import os

# Import our security modules
from secure_secrets import get_secrets_manager
from secure_auth import get_auth, require_auth
from security_headers import setup_security_headers
from rate_limiting import rate_limit
from secure_logging import get_security_logger

class SecurityMiddleware:
    """Comprehensive security middleware."""
    
    def __init__(self, app: Flask = None):
        self.app = app
        self.logger = get_security_logger()
        self.auth = get_auth()
        
        if app:
            self.init_app(app)
            
    def init_app(self, app: Flask):
        """Initialize security middleware with Flask app."""
        self.app = app
        
        # Setup security headers and CORS
        setup_security_headers(app)
        
        # Register security hooks
        app.before_request(self.before_request)
        app.after_request(self.after_request)
        app.teardown_appcontext(self.teardown)
        
        # Register error handlers
        app.errorhandler(401)(self.handle_unauthorized)
        app.errorhandler(403)(self.handle_forbidden)
        app.errorhandler(429)(self.handle_rate_limit)
        
    def before_request(self):
        """Execute before each request."""
        try:
            # Log incoming request
            self.logger.security_event('REQUEST_START', {
                'method': request.method,
                'path': request.path,
                'remote_addr': request.remote_addr
            })
            
            # Check for security headers
            if request.headers.get('X-Forwarded-For'):
                g.real_ip = request.headers['X-Forwarded-For'].split(',')[0].strip()
            else:
                g.real_ip = request.remote_addr
                
            # Validate session if present
            session_token = request.headers.get('Authorization')
            if session_token and session_token.startswith('Bearer '):
                token = session_token[7:]
                user_data = self.auth.verify_token(token)
                if user_data:
                    g.user_id = user_data['user_id']
                    g.user_data = user_data
                    
        except Exception as e:
            self.logger.security_event('REQUEST_ERROR', {
                'error': str(e),
                'path': request.path
            }, 'ERROR')
            
    def after_request(self, response):
        """Execute after each request."""
        try:
            # Log response
            self.logger.security_event('REQUEST_END', {
                'status': response.status_code,
                'path': request.path,
                'user_id': getattr(g, 'user_id', None)
            })
            
            # Add security context to response headers (for debugging)
            if self.app.debug:
                response.headers['X-Security-Version'] = '1.0'
                
        except Exception as e:
            self.logger.security_event('RESPONSE_ERROR', {
                'error': str(e)
            }, 'ERROR')
            
        return response
        
    def teardown(self, exception):
        """Cleanup after request."""
        if exception:
            self.logger.security_event('REQUEST_EXCEPTION', {
                'exception': str(exception),
                'path': request.path
            }, 'ERROR')
            
    def handle_unauthorized(self, error):
        """Handle 401 errors."""
        self.logger.security_event('UNAUTHORIZED_ACCESS', {
            'path': request.path,
            'user_id': getattr(g, 'user_id', None)
        }, 'WARNING')
        
        return {'error': 'Unauthorized access'}, 401
        
    def handle_forbidden(self, error):
        """Handle 403 errors."""
        self.logger.security_event('FORBIDDEN_ACCESS', {
            'path': request.path,
            'user_id': getattr(g, 'user_id', None)
        }, 'WARNING')
        
        return {'error': 'Access forbidden'}, 403
        
    def handle_rate_limit(self, error):
        """Handle 429 rate limit errors."""
        self.logger.security_event('RATE_LIMIT_EXCEEDED', {
            'path': request.path,
            'ip': request.remote_addr
        }, 'WARNING')
        
        return {'error': 'Rate limit exceeded'}, 429

def create_secure_app(config=None) -> Flask:
    """Create Flask app with all security features enabled."""
    app = Flask(__name__)
    
    # Load configuration
    if config:
        app.config.update(config)
    else:
        # Default secure configuration
        secrets_manager = get_secrets_manager()
        app.config.update({
            'SECRET_KEY': secrets_manager.get_flask_secret_key(),
            'SESSION_COOKIE_SECURE': True,
            'SESSION_COOKIE_HTTPONLY': True,
            'SESSION_COOKIE_SAMESITE': 'Lax',
            'PERMANENT_SESSION_LIFETIME': 3600,
            'WTF_CSRF_ENABLED': True,
            'WTF_CSRF_TIME_LIMIT': 3600
        })
    
    # Initialize security middleware
    SecurityMiddleware(app)
    
    return app
