"""
Security Headers Middleware for NightScan
Implements comprehensive security headers protection.
"""

from flask import Flask, Response
from typing import Dict, Optional

class SecurityHeaders:
    """Security headers middleware."""
    
    def __init__(self, app: Optional[Flask] = None):
        self.app = app
        if app is not None:
            self.init_app(app)
            
    def init_app(self, app: Flask):
        """Initialize security headers for Flask app."""
        app.after_request(self.add_security_headers)
        
    def add_security_headers(self, response: Response) -> Response:
        """Add security headers to response."""
        
        # Content Security Policy
        csp_policy = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' 'unsafe-eval'; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data: https:; "
            "font-src 'self'; "
            "connect-src 'self'; "
            "media-src 'self'; "
            "object-src 'none'; "
            "child-src 'none'; "
            "frame-ancestors 'none'; "
            "base-uri 'self'; "
            "form-action 'self'"
        )
        response.headers['Content-Security-Policy'] = csp_policy
        
        # HTTP Strict Transport Security
        response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains; preload'
        
        # X-Frame-Options
        response.headers['X-Frame-Options'] = 'DENY'
        
        # X-Content-Type-Options
        response.headers['X-Content-Type-Options'] = 'nosniff'
        
        # X-XSS-Protection
        response.headers['X-XSS-Protection'] = '1; mode=block'
        
        # Referrer Policy
        response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
        
        # Feature Policy / Permissions Policy
        permissions_policy = (
            "camera=(), "
            "microphone=(), "
            "geolocation=(), "
            "payment=(), "
            "usb=(), "
            "magnetometer=(), "
            "gyroscope=(), "
            "accelerometer=()"
        )
        response.headers['Permissions-Policy'] = permissions_policy
        
        # Remove server header
        response.headers.pop('Server', None)
        
        # X-Powered-By removal
        response.headers.pop('X-Powered-By', None)
        
        # Cache control for sensitive pages
        if response.status_code in [200, 201] and 'login' in str(response.location or ''):
            response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, private'
            response.headers['Pragma'] = 'no-cache'
            response.headers['Expires'] = '0'
            
        return response

class CORSHandler:
    """Secure CORS handler."""
    
    def __init__(self, allowed_origins: list = None):
        self.allowed_origins = allowed_origins or ['https://localhost:3000']
        
    def handle_cors(self, response: Response, origin: str = None) -> Response:
        """Handle CORS with security checks."""
        if origin and origin in self.allowed_origins:
            response.headers['Access-Control-Allow-Origin'] = origin
        else:
            # Don't set wildcard origin for security
            response.headers['Access-Control-Allow-Origin'] = self.allowed_origins[0]
            
        response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, DELETE, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = (
            'Content-Type, Authorization, X-Requested-With, X-CSRF-Token'
        )
        response.headers['Access-Control-Allow-Credentials'] = 'true'
        response.headers['Access-Control-Max-Age'] = '86400'  # 24 hours
        
        return response

def setup_security_headers(app: Flask, cors_origins: list = None) -> Flask:
    """Setup all security headers for Flask app."""
    # Initialize security headers
    SecurityHeaders(app)
    
    # Setup CORS if origins provided
    if cors_origins:
        cors_handler = CORSHandler(cors_origins)
        
        @app.after_request
        def handle_cors(response):
            from flask import request
            origin = request.headers.get('Origin')
            return cors_handler.handle_cors(response, origin)
            
    return app
