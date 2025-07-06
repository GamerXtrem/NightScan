"""
Rate Limiting System for NightScan
Protects against brute force and DoS attacks.
"""

import time
import redis
from typing import Dict, Tuple, Optional
from functools import wraps
from flask import request, jsonify, g

class RateLimiter:
    """Redis-based rate limiting system."""
    
    def __init__(self, redis_client=None):
        self.redis_client = redis_client or redis.Redis(decode_responses=True)
        
    def is_allowed(self, key: str, limit: int, window: int) -> Tuple[bool, Dict[str, int]]:
        """Check if request is within rate limit."""
        current_time = int(time.time())
        window_start = current_time - window
        
        pipe = self.redis_client.pipeline()
        
        # Remove old entries
        pipe.zremrangebyscore(key, 0, window_start)
        
        # Count current requests
        pipe.zcard(key)
        
        # Add current request
        pipe.zadd(key, {str(current_time): current_time})
        
        # Set expiration
        pipe.expire(key, window)
        
        results = pipe.execute()
        current_requests = results[1]
        
        is_allowed = current_requests < limit
        remaining = max(0, limit - current_requests - 1) if is_allowed else 0
        reset_time = current_time + window
        
        return is_allowed, {
            'limit': limit,
            'remaining': remaining,
            'reset': reset_time,
            'current': current_requests + 1
        }

def rate_limit(limit: int = 100, window: int = 3600, per: str = 'ip'):
    """Rate limiting decorator."""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            try:
                limiter = RateLimiter()
                
                if per == 'ip':
                    key = f"rate_limit:ip:{request.remote_addr}"
                elif per == 'user':
                    user_id = getattr(g, 'user_id', None)
                    if not user_id:
                        key = f"rate_limit:ip:{request.remote_addr}"
                    else:
                        key = f"rate_limit:user:{user_id}"
                else:
                    key = f"rate_limit:global"
                    
                allowed, info = limiter.is_allowed(key, limit, window)
                
                if not allowed:
                    return jsonify({
                        'error': 'Rate limit exceeded',
                        'retry_after': info['reset'] - int(time.time())
                    }), 429
                    
                # Add rate limit headers
                response = f(*args, **kwargs)
                if hasattr(response, 'headers'):
                    response.headers['X-RateLimit-Limit'] = str(info['limit'])
                    response.headers['X-RateLimit-Remaining'] = str(info['remaining'])
                    response.headers['X-RateLimit-Reset'] = str(info['reset'])
                    
                return response
                
            except Exception as e:
                # If rate limiting fails, allow the request but log the error
                print(f"Rate limiting error: {e}")
                return f(*args, **kwargs)
                
        return decorated_function
    return decorator
