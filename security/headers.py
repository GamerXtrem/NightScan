"""
Security Headers Module

Manages security headers for all HTTP responses.
"""

import logging
from typing import Dict, List, Optional, Any
from flask import Flask, make_response
from functools import wraps

logger = logging.getLogger(__name__)


class SecurityHeaders:
    """Manages security headers for HTTP responses."""
    
    def __init__(self, config):
        self.config = config
        self.default_headers = self._build_default_headers()
    
    def _build_default_headers(self) -> Dict[str, str]:
        """Build default security headers based on configuration."""
        headers = {
            # Prevent MIME type sniffing
            'X-Content-Type-Options': 'nosniff',
            
            # Prevent clickjacking
            'X-Frame-Options': 'DENY',
            
            # XSS Protection (for older browsers)
            'X-XSS-Protection': '1; mode=block',
            
            # Referrer Policy
            'Referrer-Policy': 'strict-origin-when-cross-origin',
            
            # Permissions Policy (formerly Feature Policy)
            'Permissions-Policy': 'geolocation=(), microphone=(), camera=()',
            
            # Cache control for security
            'Cache-Control': 'no-store, no-cache, must-revalidate, private',
            'Pragma': 'no-cache',
            'Expires': '0'
        }
        
        # Add HSTS if HTTPS is enforced
        if self.config.security.force_https:
            headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains; preload'
        
        # Content Security Policy
        headers['Content-Security-Policy'] = self._build_csp()
        
        return headers
    
    def _build_csp(self) -> str:
        """Build Content Security Policy header."""
        # Base CSP directives
        csp_directives = [
            "default-src 'self'",
            "script-src 'self' 'unsafe-inline' 'unsafe-eval'",  # Will tighten in production
            "style-src 'self' 'unsafe-inline'",
            "img-src 'self' data: https:",
            "font-src 'self'",
            "connect-src 'self'",
            "media-src 'self'",
            "object-src 'none'",
            "frame-src 'none'",
            "base-uri 'self'",
            "form-action 'self'",
            "frame-ancestors 'none'",
            "upgrade-insecure-requests"
        ]
        
        # Tighten CSP for production
        if self.config.environment == 'production':
            # Remove unsafe-inline and unsafe-eval in production
            csp_directives[1] = "script-src 'self'"
            csp_directives[2] = "style-src 'self'"
        
        return "; ".join(csp_directives)
    
    def init_app(self, app: Flask) -> None:
        """Initialize security headers with Flask app."""
        self.app = app
        
        # Apply headers to all responses
        @app.after_request
        def add_security_headers(response):
            """Add security headers to response."""
            # Skip if response already has security headers
            if response.headers.get('X-Security-Headers-Applied'):
                return response
            
            # Apply default headers
            for header, value in self.default_headers.items():
                if header not in response.headers:
                    response.headers[header] = value
            
            # Mark as processed
            response.headers['X-Security-Headers-Applied'] = 'true'
            
            return response
        
        logger.info("Security headers initialized")
    
    def secure_response(self, response, additional_headers: Optional[Dict[str, str]] = None):
        """Apply security headers to a specific response."""
        # Apply default headers
        for header, value in self.default_headers.items():
            response.headers[header] = value
        
        # Apply additional headers if provided
        if additional_headers:
            for header, value in additional_headers.items():
                response.headers[header] = value
        
        return response
    
    def get_headers_for_file_type(self, file_type: str) -> Dict[str, str]:
        """Get appropriate headers for specific file types."""
        headers = self.default_headers.copy()
        
        # Adjust headers based on file type
        if file_type in ['html', 'htm']:
            # HTML files need standard security headers
            pass
        
        elif file_type in ['js', 'javascript']:
            # JavaScript files
            headers['Content-Type'] = 'application/javascript; charset=utf-8'
            headers['X-Content-Type-Options'] = 'nosniff'
        
        elif file_type in ['css']:
            # CSS files
            headers['Content-Type'] = 'text/css; charset=utf-8'
            headers['X-Content-Type-Options'] = 'nosniff'
        
        elif file_type in ['json']:
            # JSON responses
            headers['Content-Type'] = 'application/json; charset=utf-8'
            headers['X-Content-Type-Options'] = 'nosniff'
        
        elif file_type in ['jpg', 'jpeg', 'png', 'gif', 'webp', 'svg']:
            # Images
            headers['Cache-Control'] = 'public, max-age=31536000'
            del headers['Pragma']
            del headers['Expires']
        
        elif file_type in ['pdf']:
            # PDFs
            headers['Content-Type'] = 'application/pdf'
            headers['Content-Disposition'] = 'inline'
        
        elif file_type in ['zip', 'tar', 'gz']:
            # Archives
            headers['Content-Disposition'] = 'attachment'
        
        return headers
    
    def api_headers(self) -> Dict[str, str]:
        """Get headers specifically for API responses."""
        headers = {
            'Content-Type': 'application/json; charset=utf-8',
            'X-Content-Type-Options': 'nosniff',
            'X-Frame-Options': 'DENY',
            'Cache-Control': 'no-store'
        }
        
        # Add CORS headers if configured
        if hasattr(self.config, 'cors') and self.config.cors.enabled:
            headers['Access-Control-Allow-Origin'] = self.config.cors.allowed_origins or '*'
            headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, DELETE, OPTIONS'
            headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
            headers['Access-Control-Max-Age'] = '86400'
        
        return headers
    
    def download_headers(self, filename: str, inline: bool = False) -> Dict[str, str]:
        """Get headers for file downloads."""
        headers = {
            'X-Content-Type-Options': 'nosniff',
            'X-Download-Options': 'noopen',
            'X-Frame-Options': 'DENY'
        }
        
        # Set content disposition
        disposition = 'inline' if inline else 'attachment'
        headers['Content-Disposition'] = f'{disposition}; filename="{filename}"'
        
        return headers
    
    def error_headers(self, error_code: int) -> Dict[str, str]:
        """Get headers for error responses."""
        headers = {
            'Content-Type': 'application/json; charset=utf-8',
            'X-Content-Type-Options': 'nosniff',
            'Cache-Control': 'no-store, no-cache, must-revalidate',
            'Pragma': 'no-cache'
        }
        
        # Add additional headers for specific error codes
        if error_code == 401:
            headers['WWW-Authenticate'] = 'Bearer realm="NightScan"'
        
        return headers
    
    def validate_headers(self, headers: Dict[str, str]) -> List[str]:
        """Validate security headers and return any issues."""
        issues = []
        
        # Check for required security headers
        required_headers = [
            'X-Content-Type-Options',
            'X-Frame-Options',
            'Content-Security-Policy'
        ]
        
        for header in required_headers:
            if header not in headers:
                issues.append(f"Missing required security header: {header}")
        
        # Check for insecure header values
        if headers.get('X-Frame-Options') not in ['DENY', 'SAMEORIGIN']:
            issues.append("X-Frame-Options should be DENY or SAMEORIGIN")
        
        if headers.get('X-Content-Type-Options') != 'nosniff':
            issues.append("X-Content-Type-Options should be nosniff")
        
        # Check CSP
        csp = headers.get('Content-Security-Policy', '')
        if 'unsafe-inline' in csp and self.config.environment == 'production':
            issues.append("CSP contains unsafe-inline in production")
        
        if 'unsafe-eval' in csp and self.config.environment == 'production':
            issues.append("CSP contains unsafe-eval in production")
        
        # Check HSTS in production
        if self.config.environment == 'production' and self.config.security.force_https:
            if 'Strict-Transport-Security' not in headers:
                issues.append("Missing HSTS header in production with HTTPS")
        
        return issues
    
    # Decorators for specific header requirements
    def no_cache(self, f):
        """Decorator to ensure response is not cached."""
        @wraps(f)
        def decorated_function(*args, **kwargs):
            response = make_response(f(*args, **kwargs))
            response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, private'
            response.headers['Pragma'] = 'no-cache'
            response.headers['Expires'] = '0'
            return response
        return decorated_function
    
    def cache_for(self, seconds: int):
        """Decorator to cache response for specified seconds."""
        def decorator(f):
            @wraps(f)
            def decorated_function(*args, **kwargs):
                response = make_response(f(*args, **kwargs))
                response.headers['Cache-Control'] = f'public, max-age={seconds}'
                return response
            return decorated_function
        return decorator