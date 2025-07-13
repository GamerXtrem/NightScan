"""
Rate Limiting Module

Implements rate limiting to prevent abuse and DDoS attacks.
"""

import time
import logging
from typing import Optional, Dict, Any, Callable, Tuple
from functools import wraps
from collections import defaultdict, deque
from datetime import datetime, timedelta
from flask import Flask, request, jsonify, g
import hashlib
import json

logger = logging.getLogger(__name__)


class RateLimiter:
    """Implements various rate limiting strategies."""
    
    def __init__(self, config):
        self.config = config
        self.enabled = config.rate_limit.enabled
        
        # Storage for rate limit data
        self.request_history = defaultdict(lambda: deque(maxlen=1000))
        self.blocked_ips = {}  # IP -> block_until timestamp
        self.endpoint_limits = {}  # endpoint -> (requests, period)
        
        # Default limits
        self.default_limit = config.rate_limit.default_limit
        self.default_period = config.rate_limit.default_period
        
        # Burst limits
        self.burst_limit = getattr(config.rate_limit, 'burst_limit', self.default_limit * 2)
        self.burst_period = getattr(config.rate_limit, 'burst_period', 60)
        
        # Initialize endpoint-specific limits
        self._init_endpoint_limits()
    
    def _init_endpoint_limits(self):
        """Initialize endpoint-specific rate limits."""
        # Define limits for specific endpoints
        self.endpoint_limits = {
            # Authentication endpoints - stricter limits
            'auth.login': (10, 300),  # 10 requests per 5 minutes
            'auth.register': (5, 300),  # 5 requests per 5 minutes
            'auth.reset_password': (3, 300),  # 3 requests per 5 minutes
            
            # API endpoints - moderate limits
            'api.predict': (100, 60),  # 100 requests per minute
            'api.upload': (50, 60),  # 50 uploads per minute
            
            # Public endpoints - relaxed limits
            'static': (1000, 60),  # 1000 requests per minute
            'public.home': (200, 60),  # 200 requests per minute
        }
    
    def init_app(self, app: Flask) -> None:
        """Initialize rate limiter with Flask app."""
        if not self.enabled:
            logger.info("Rate limiting is disabled")
            return
        
        self.app = app
        
        # Register before_request handler
        @app.before_request
        def check_rate_limit():
            if not self.enabled:
                return None
            
            # Check if IP is blocked
            client_ip = self._get_client_ip()
            if self._is_ip_blocked(client_ip):
                return jsonify({'error': 'IP temporarily blocked due to rate limit violations'}), 429
            
            # Check rate limit
            endpoint = request.endpoint or 'unknown'
            is_allowed, retry_after = self.is_allowed(client_ip, endpoint)
            
            if not is_allowed:
                response = jsonify({
                    'error': 'Rate limit exceeded',
                    'retry_after': retry_after
                })
                response.headers['Retry-After'] = str(retry_after)
                response.headers['X-RateLimit-Limit'] = str(self._get_limit(endpoint))
                response.headers['X-RateLimit-Remaining'] = '0'
                response.headers['X-RateLimit-Reset'] = str(int(time.time() + retry_after))
                return response, 429
            
            return None
        
        # Register after_request handler
        @app.after_request
        def add_rate_limit_headers(response):
            if not self.enabled:
                return response
            
            client_ip = self._get_client_ip()
            endpoint = request.endpoint or 'unknown'
            
            limit, period = self._get_limit_and_period(endpoint)
            remaining = self._get_remaining_requests(client_ip, endpoint)
            reset_time = self._get_reset_time(client_ip, endpoint)
            
            response.headers['X-RateLimit-Limit'] = str(limit)
            response.headers['X-RateLimit-Remaining'] = str(max(0, remaining))
            response.headers['X-RateLimit-Reset'] = str(reset_time)
            
            return response
        
        logger.info("Rate limiting initialized")
    
    def is_allowed(self, identifier: str, endpoint: str = 'default') -> Tuple[bool, int]:
        """
        Check if request is allowed under rate limit.
        
        Args:
            identifier: Client identifier (IP address)
            endpoint: Endpoint being accessed
            
        Returns:
            Tuple of (is_allowed, retry_after_seconds)
        """
        current_time = time.time()
        
        # Get limits for endpoint
        limit, period = self._get_limit_and_period(endpoint)
        
        # Get request history
        key = f"{identifier}:{endpoint}"
        history = self.request_history[key]
        
        # Clean old entries
        cutoff_time = current_time - period
        while history and history[0] < cutoff_time:
            history.popleft()
        
        # Check burst limit
        burst_cutoff = current_time - self.burst_period
        burst_count = sum(1 for t in history if t > burst_cutoff)
        if burst_count >= self.burst_limit:
            # Calculate retry after
            oldest_burst = next((t for t in history if t > burst_cutoff), current_time)
            retry_after = int(self.burst_period - (current_time - oldest_burst)) + 1
            
            # Check if should block IP
            self._check_for_blocking(identifier, endpoint)
            
            return False, retry_after
        
        # Check regular limit
        if len(history) >= limit:
            # Calculate retry after
            retry_after = int(period - (current_time - history[0])) + 1
            return False, retry_after
        
        # Record request
        history.append(current_time)
        
        return True, 0
    
    def _check_for_blocking(self, identifier: str, endpoint: str):
        """Check if IP should be blocked due to repeated violations."""
        # Track violations
        violation_key = f"{identifier}:violations"
        violations = self.request_history.get(violation_key, deque(maxlen=100))
        
        current_time = time.time()
        violations.append(current_time)
        
        # Count recent violations (last 5 minutes)
        recent_violations = sum(1 for t in violations if t > current_time - 300)
        
        # Block if too many violations
        if recent_violations >= 10:
            block_duration = 3600  # 1 hour
            self.blocked_ips[identifier] = current_time + block_duration
            logger.warning(f"IP {identifier} blocked for {block_duration} seconds due to rate limit violations")
    
    def _is_ip_blocked(self, ip: str) -> bool:
        """Check if IP is currently blocked."""
        if ip in self.blocked_ips:
            if time.time() < self.blocked_ips[ip]:
                return True
            else:
                # Block expired
                del self.blocked_ips[ip]
        return False
    
    def _get_limit_and_period(self, endpoint: str) -> Tuple[int, int]:
        """Get rate limit and period for endpoint."""
        # Check for exact match
        if endpoint in self.endpoint_limits:
            return self.endpoint_limits[endpoint]
        
        # Check for prefix match (e.g., 'auth.' matches 'auth.login')
        for pattern, limits in self.endpoint_limits.items():
            if endpoint.startswith(pattern):
                return limits
        
        # Return default
        return (self.default_limit, self.default_period)
    
    def _get_limit(self, endpoint: str) -> int:
        """Get rate limit for endpoint."""
        limit, _ = self._get_limit_and_period(endpoint)
        return limit
    
    def _get_remaining_requests(self, identifier: str, endpoint: str) -> int:
        """Get remaining requests for identifier."""
        limit, period = self._get_limit_and_period(endpoint)
        key = f"{identifier}:{endpoint}"
        history = self.request_history.get(key, deque())
        
        # Count requests in current period
        current_time = time.time()
        cutoff_time = current_time - period
        recent_requests = sum(1 for t in history if t > cutoff_time)
        
        return max(0, limit - recent_requests)
    
    def _get_reset_time(self, identifier: str, endpoint: str) -> int:
        """Get timestamp when rate limit resets."""
        _, period = self._get_limit_and_period(endpoint)
        key = f"{identifier}:{endpoint}"
        history = self.request_history.get(key, deque())
        
        if history:
            # Reset time is oldest request + period
            return int(history[0] + period)
        else:
            # No requests yet, reset time is now + period
            return int(time.time() + period)
    
    def _get_client_ip(self) -> str:
        """Get client IP address."""
        # Try to get from g object first (set by middleware)
        if hasattr(g, 'client_ip'):
            return g.client_ip
        
        # Fall back to request
        if self.config.security.trusted_proxies:
            x_forwarded_for = request.headers.get('X-Forwarded-For')
            if x_forwarded_for:
                return x_forwarded_for.split(',')[0].strip()
            
            x_real_ip = request.headers.get('X-Real-IP')
            if x_real_ip:
                return x_real_ip
        
        return request.remote_addr or 'unknown'
    
    # Decorators for custom rate limits
    def limit(self, requests: int, period: int = 60, key_func: Optional[Callable] = None):
        """
        Decorator to apply custom rate limit to endpoint.
        
        Args:
            requests: Number of allowed requests
            period: Time period in seconds
            key_func: Optional function to generate rate limit key
        """
        def decorator(f: Callable) -> Callable:
            @wraps(f)
            def decorated_function(*args, **kwargs):
                if not self.enabled:
                    return f(*args, **kwargs)
                
                # Generate key
                if key_func:
                    key = key_func()
                else:
                    key = self._get_client_ip()
                
                # Create custom endpoint name
                endpoint = f"custom:{f.__name__}"
                
                # Set custom limit
                self.endpoint_limits[endpoint] = (requests, period)
                
                # Check limit
                is_allowed, retry_after = self.is_allowed(key, endpoint)
                
                if not is_allowed:
                    response = jsonify({
                        'error': 'Rate limit exceeded',
                        'retry_after': retry_after
                    })
                    response.headers['Retry-After'] = str(retry_after)
                    return response, 429
                
                return f(*args, **kwargs)
            
            return decorated_function
        
        return decorator
    
    def get_stats(self) -> Dict[str, Any]:
        """Get rate limiter statistics."""
        total_tracked = len(self.request_history)
        total_blocked = len(self.blocked_ips)
        
        # Calculate requests per endpoint
        endpoint_stats = defaultdict(int)
        for key in self.request_history.keys():
            if ':' in key:
                _, endpoint = key.rsplit(':', 1)
                endpoint_stats[endpoint] = len(self.request_history[key])
        
        return {
            'enabled': self.enabled,
            'total_tracked_keys': total_tracked,
            'blocked_ips': total_blocked,
            'endpoint_stats': dict(endpoint_stats),
            'default_limit': self.default_limit,
            'default_period': self.default_period,
            'burst_limit': self.burst_limit,
            'burst_period': self.burst_period
        }
    
    def reset_limits(self, identifier: Optional[str] = None, endpoint: Optional[str] = None):
        """Reset rate limits for identifier and/or endpoint."""
        if identifier and endpoint:
            key = f"{identifier}:{endpoint}"
            if key in self.request_history:
                del self.request_history[key]
        elif identifier:
            # Reset all endpoints for identifier
            keys_to_remove = [k for k in self.request_history.keys() if k.startswith(f"{identifier}:")]
            for key in keys_to_remove:
                del self.request_history[key]
        elif endpoint:
            # Reset endpoint for all identifiers
            keys_to_remove = [k for k in self.request_history.keys() if k.endswith(f":{endpoint}")]
            for key in keys_to_remove:
                del self.request_history[key]
        
        # Also remove from blocked IPs if identifier provided
        if identifier and identifier in self.blocked_ips:
            del self.blocked_ips[identifier]
    
    def unblock_ip(self, ip: str) -> bool:
        """Manually unblock an IP address."""
        if ip in self.blocked_ips:
            del self.blocked_ips[ip]
            logger.info(f"IP {ip} manually unblocked")
            return True
        return False