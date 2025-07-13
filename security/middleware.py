"""
Security Middleware Module

Provides security middleware for request/response processing.
"""

import time
import hashlib
import logging
from typing import Optional, Dict, Any, Callable
from functools import wraps
from flask import Flask, request, g, jsonify, abort
from werkzeug.exceptions import BadRequest
import re

logger = logging.getLogger(__name__)


class SecurityMiddleware:
    """Handles security middleware for all requests."""
    
    def __init__(self, config):
        self.config = config
        self.request_stats = {}  # Track request statistics
        
        # Compile regex patterns for performance
        self.suspicious_patterns = [
            re.compile(r'<script[^>]*>', re.IGNORECASE),
            re.compile(r'javascript:', re.IGNORECASE),
            re.compile(r'on\w+\s*=', re.IGNORECASE),  # onclick, onload, etc.
            re.compile(r'\.\.[\\/]'),  # Path traversal
            re.compile(r'union.*select', re.IGNORECASE),  # SQL injection
            re.compile(r'exec\s*\(', re.IGNORECASE),
            re.compile(r'eval\s*\(', re.IGNORECASE),
            re.compile(r'<iframe', re.IGNORECASE),
            re.compile(r'base64_decode', re.IGNORECASE)
        ]
        
        # Content type validation
        self.allowed_content_types = {
            'application/json',
            'application/x-www-form-urlencoded',
            'multipart/form-data',
            'text/plain'
        }
    
    def init_app(self, app: Flask) -> None:
        """Initialize middleware with Flask app."""
        self.app = app
        
        # Register before_request handlers
        app.before_request(self.security_checks)
        app.before_request(self.request_validation)
        app.before_request(self.track_request)
        
        # Register after_request handlers
        app.after_request(self.security_response_headers)
        app.after_request(self.log_request)
        
        # Register teardown handlers
        app.teardown_request(self.cleanup_request)
    
    def security_checks(self) -> Optional[Any]:
        """Perform security checks on incoming requests."""
        # Store request start time
        g.request_start_time = time.time()
        
        # Check request size
        if request.content_length and request.content_length > self.config.upload.max_file_size:
            logger.warning(f"Request too large: {request.content_length} bytes")
            abort(413, "Request entity too large")
        
        # Check for suspicious patterns in URL
        if self._contains_suspicious_patterns(request.url):
            logger.warning(f"Suspicious URL pattern: {request.url}")
            abort(400, "Invalid request")
        
        # Check for suspicious headers
        for header_name, header_value in request.headers:
            if self._contains_suspicious_patterns(str(header_value)):
                logger.warning(f"Suspicious header: {header_name}")
                abort(400, "Invalid request headers")
        
        # Validate content type for POST/PUT requests
        if request.method in ['POST', 'PUT']:
            content_type = request.content_type
            if content_type:
                # Extract main content type (ignore charset, etc.)
                main_type = content_type.split(';')[0].strip().lower()
                if main_type not in self.allowed_content_types:
                    logger.warning(f"Invalid content type: {content_type}")
                    abort(415, "Unsupported media type")
        
        return None
    
    def request_validation(self) -> Optional[Any]:
        """Validate request data."""
        # Check for common attack patterns in request data
        if request.method in ['POST', 'PUT', 'PATCH']:
            # Check JSON data
            if request.is_json:
                try:
                    data = request.get_json()
                    if data and self._check_data_for_attacks(data):
                        logger.warning("Suspicious data in JSON request")
                        abort(400, "Invalid request data")
                except BadRequest:
                    logger.warning("Invalid JSON in request")
                    abort(400, "Invalid JSON")
            
            # Check form data
            elif request.form:
                for key, value in request.form.items():
                    if self._contains_suspicious_patterns(str(value)):
                        logger.warning(f"Suspicious form data in field: {key}")
                        abort(400, "Invalid form data")
        
        # Check query parameters
        for key, value in request.args.items():
            if self._contains_suspicious_patterns(str(value)):
                logger.warning(f"Suspicious query parameter: {key}")
                abort(400, "Invalid query parameters")
        
        return None
    
    def track_request(self) -> None:
        """Track request for monitoring and rate limiting."""
        # Get client IP
        client_ip = self._get_client_ip()
        g.client_ip = client_ip
        
        # Track request
        endpoint = request.endpoint or 'unknown'
        key = f"{client_ip}:{endpoint}"
        
        if key not in self.request_stats:
            self.request_stats[key] = {
                'count': 0,
                'first_request': time.time(),
                'last_request': time.time()
            }
        
        self.request_stats[key]['count'] += 1
        self.request_stats[key]['last_request'] = time.time()
    
    def security_response_headers(self, response):
        """Add security headers to response."""
        # These headers will be handled by the headers module
        # but we can add additional middleware-specific headers here
        
        # Add request ID for tracking
        if hasattr(g, 'request_id'):
            response.headers['X-Request-ID'] = g.request_id
        
        # Add response time
        if hasattr(g, 'request_start_time'):
            response_time = time.time() - g.request_start_time
            response.headers['X-Response-Time'] = f"{response_time:.3f}s"
        
        return response
    
    def log_request(self, response):
        """Log request details for security monitoring."""
        # Log request details
        client_ip = getattr(g, 'client_ip', 'unknown')
        response_time = 0
        
        if hasattr(g, 'request_start_time'):
            response_time = time.time() - g.request_start_time
        
        log_data = {
            'ip': client_ip,
            'method': request.method,
            'path': request.path,
            'status': response.status_code,
            'response_time': f"{response_time:.3f}s",
            'user_agent': request.headers.get('User-Agent', 'unknown')
        }
        
        # Log based on status code
        if response.status_code >= 500:
            logger.error(f"Server error: {log_data}")
        elif response.status_code >= 400:
            logger.warning(f"Client error: {log_data}")
        else:
            logger.info(f"Request: {log_data}")
        
        return response
    
    def cleanup_request(self, exception=None) -> None:
        """Clean up after request."""
        # Clean up old request stats (older than 1 hour)
        current_time = time.time()
        cutoff_time = current_time - 3600
        
        keys_to_remove = []
        for key, stats in self.request_stats.items():
            if stats['last_request'] < cutoff_time:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self.request_stats[key]
    
    def _contains_suspicious_patterns(self, text: str) -> bool:
        """Check if text contains suspicious patterns."""
        if not text:
            return False
        
        text_lower = text.lower()
        
        # Check compiled patterns
        for pattern in self.suspicious_patterns:
            if pattern.search(text):
                return True
        
        # Check for null bytes
        if '\x00' in text:
            return True
        
        # Check for excessive special characters
        special_char_count = sum(1 for c in text if not c.isalnum() and c not in ' .-_/@')
        if len(text) > 10 and special_char_count / len(text) > 0.5:
            return True
        
        return False
    
    def _check_data_for_attacks(self, data: Any, depth: int = 0) -> bool:
        """Recursively check data structure for attack patterns."""
        if depth > 10:  # Prevent deep recursion
            return True
        
        if isinstance(data, dict):
            for key, value in data.items():
                if self._contains_suspicious_patterns(str(key)):
                    return True
                if self._check_data_for_attacks(value, depth + 1):
                    return True
        
        elif isinstance(data, list):
            for item in data:
                if self._check_data_for_attacks(item, depth + 1):
                    return True
        
        elif isinstance(data, str):
            return self._contains_suspicious_patterns(data)
        
        return False
    
    def _get_client_ip(self) -> str:
        """Get client IP address, considering proxies."""
        # Check for proxy headers
        if self.config.security.trusted_proxies:
            # X-Forwarded-For
            x_forwarded_for = request.headers.get('X-Forwarded-For')
            if x_forwarded_for:
                # Get first IP in the chain
                ip = x_forwarded_for.split(',')[0].strip()
                if self._is_valid_ip(ip):
                    return ip
            
            # X-Real-IP
            x_real_ip = request.headers.get('X-Real-IP')
            if x_real_ip and self._is_valid_ip(x_real_ip):
                return x_real_ip
        
        # Fall back to remote_addr
        return request.remote_addr or 'unknown'
    
    def _is_valid_ip(self, ip: str) -> bool:
        """Check if string is a valid IP address."""
        import ipaddress
        try:
            ipaddress.ip_address(ip)
            return True
        except ValueError:
            return False
    
    def get_request_stats(self) -> Dict[str, Any]:
        """Get current request statistics."""
        total_requests = sum(stats['count'] for stats in self.request_stats.values())
        unique_ips = len(set(key.split(':')[0] for key in self.request_stats.keys()))
        
        return {
            'total_requests': total_requests,
            'unique_ips': unique_ips,
            'tracked_endpoints': len(self.request_stats),
            'stats': dict(self.request_stats)
        }
    
    # Decorator for endpoint-specific security
    def secure_endpoint(self, allowed_methods=None, require_json=False, max_size=None):
        """Decorator to add additional security to specific endpoints."""
        def decorator(f: Callable) -> Callable:
            @wraps(f)
            def decorated_function(*args, **kwargs):
                # Check allowed methods
                if allowed_methods and request.method not in allowed_methods:
                    abort(405, "Method not allowed")
                
                # Check JSON requirement
                if require_json and not request.is_json:
                    abort(415, "JSON required")
                
                # Check size limit
                if max_size and request.content_length and request.content_length > max_size:
                    abort(413, "Request too large")
                
                return f(*args, **kwargs)
            
            return decorated_function
        
        return decorator