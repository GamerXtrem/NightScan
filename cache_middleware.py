"""HTTP cache headers middleware for NightScan"""

import hashlib
import json
from functools import wraps
from flask import request, make_response, Response
from typing import Callable, Optional, Dict, Any

def add_cache_headers(
    max_age: int = 0,
    private: bool = True,
    must_revalidate: bool = False,
    etag: bool = False
):
    """Decorator to add cache headers to responses
    
    Args:
        max_age: Cache duration in seconds (0 = no cache)
        private: Whether cache is private (user-specific) or public
        must_revalidate: Force revalidation when stale
        etag: Generate and check ETags for conditional requests
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            response = func(*args, **kwargs)
            
            # Handle different response types
            if isinstance(response, tuple):
                # Flask allows returning (response, status_code) or (response, status_code, headers)
                if len(response) == 2:
                    resp_data, status_code = response
                    headers = {}
                else:
                    resp_data, status_code, headers = response
            else:
                resp_data = response
                status_code = 200
                headers = {}
            
            # Create proper Response object
            if not isinstance(resp_data, Response):
                if isinstance(resp_data, dict):
                    resp = make_response(json.dumps(resp_data), status_code)
                    resp.headers['Content-Type'] = 'application/json'
                else:
                    resp = make_response(resp_data, status_code)
            else:
                resp = resp_data
            
            # Update headers
            for key, value in headers.items():
                resp.headers[key] = value
            
            # Only add cache headers for successful GET requests
            if request.method == 'GET' and status_code == 200:
                # Build Cache-Control header
                cache_parts = []
                
                if max_age > 0:
                    cache_parts.append(f"max-age={max_age}")
                    cache_parts.append("public" if not private else "private")
                    
                    if must_revalidate:
                        cache_parts.append("must-revalidate")
                else:
                    cache_parts.append("no-cache")
                    cache_parts.append("no-store")
                
                resp.headers['Cache-Control'] = ', '.join(cache_parts)
                
                # Generate ETag if requested
                if etag and max_age > 0:
                    # Generate ETag from response content
                    if hasattr(resp, 'get_data'):
                        content = resp.get_data(as_text=True)
                    else:
                        content = str(resp_data)
                    
                    etag_value = f'"{hashlib.md5(content.encode()).hexdigest()}"'
                    resp.headers['ETag'] = etag_value
                    
                    # Check If-None-Match header
                    if_none_match = request.headers.get('If-None-Match')
                    if if_none_match and if_none_match == etag_value:
                        # Return 304 Not Modified
                        resp = make_response('', 304)
                        resp.headers['ETag'] = etag_value
                        resp.headers['Cache-Control'] = ', '.join(cache_parts)
                
                # Add Vary header for proper caching with authentication
                if private:
                    resp.headers['Vary'] = 'Authorization, Cookie'
            
            return resp
        
        return wrapper
    return decorator


def cache_for_analytics(ttl: int = 300):
    """Cache decorator specifically for analytics endpoints
    
    Args:
        ttl: Time to live in seconds (default 5 minutes)
    """
    return add_cache_headers(
        max_age=ttl,
        private=True,
        must_revalidate=True,
        etag=True
    )


def cache_for_static_api(ttl: int = 3600):
    """Cache decorator for static API responses
    
    Args:
        ttl: Time to live in seconds (default 1 hour)
    """
    return add_cache_headers(
        max_age=ttl,
        private=False,
        must_revalidate=False,
        etag=True
    )


def cache_for_user_data(ttl: int = 60):
    """Cache decorator for user-specific data
    
    Args:
        ttl: Time to live in seconds (default 1 minute)
    """
    return add_cache_headers(
        max_age=ttl,
        private=True,
        must_revalidate=True,
        etag=False
    )


def no_cache():
    """Decorator to explicitly prevent caching"""
    return add_cache_headers(max_age=0)