"""Enhanced caching manager for NightScan with multi-tier support and advanced features."""

import hashlib
import json
import logging
import time
from typing import Optional, Any, Dict, List, Callable, Union, Tuple
from functools import wraps
from datetime import datetime, timedelta
import threading

try:
    import redis
    from redis.exceptions import RedisError, ConnectionError as RedisConnectionError
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None
    RedisError = Exception
    RedisConnectionError = Exception

from cache_utils import PredictionCache, get_cache

logger = logging.getLogger(__name__)


class CacheManager:
    """Advanced cache manager with multi-tier support and pattern-based invalidation."""
    
    def __init__(self, redis_url: Optional[str] = None):
        """Initialize cache manager with Redis connection."""
        self.redis_client = None
        self.cache_enabled = False
        self.local_cache = {}  # In-memory cache for multi-tier
        self.cache_stats = {
            'hits': 0,
            'misses': 0,
            'errors': 0,
            'invalidations': 0
        }
        
        if REDIS_AVAILABLE and redis_url:
            try:
                self.redis_client = redis.from_url(redis_url, decode_responses=True)
                self.redis_client.ping()
                self.cache_enabled = True
                logger.info("Cache manager initialized with Redis")
            except (RedisConnectionError, RedisError) as e:
                logger.warning(f"Redis connection failed: {e}")
    
    def _generate_cache_key(self, prefix: str, *args, **kwargs) -> str:
        """Generate a unique cache key from prefix and arguments."""
        # Create a string representation of args and kwargs
        key_parts = [prefix]
        
        # Add positional arguments
        for arg in args:
            if isinstance(arg, (str, int, float, bool)):
                key_parts.append(str(arg))
            else:
                # For complex types, use hash
                key_parts.append(hashlib.md5(str(arg).encode()).hexdigest()[:8])
        
        # Add keyword arguments (sorted for consistency)
        for k, v in sorted(kwargs.items()):
            if isinstance(v, (str, int, float, bool)):
                key_parts.append(f"{k}:{v}")
            else:
                key_parts.append(f"{k}:{hashlib.md5(str(v).encode()).hexdigest()[:8]}")
        
        return ":".join(key_parts)
    
    def get(self, key: str, use_local: bool = True) -> Optional[Any]:
        """Get value from cache with multi-tier support."""
        # Try local cache first if enabled
        if use_local and key in self.local_cache:
            entry = self.local_cache[key]
            if entry['expires'] > time.time():
                self.cache_stats['hits'] += 1
                return entry['value']
            else:
                # Expired in local cache
                del self.local_cache[key]
        
        # Try Redis
        if self.cache_enabled:
            try:
                value = self.redis_client.get(key)
                if value:
                    self.cache_stats['hits'] += 1
                    result = json.loads(value)
                    
                    # Store in local cache with short TTL
                    if use_local:
                        self.local_cache[key] = {
                            'value': result,
                            'expires': time.time() + 30  # 30 seconds local cache
                        }
                    
                    return result
            except (RedisError, json.JSONDecodeError) as e:
                logger.warning(f"Cache get error for {key}: {e}")
                self.cache_stats['errors'] += 1
        
        self.cache_stats['misses'] += 1
        return None
    
    def set(self, key: str, value: Any, ttl: int = 300, use_local: bool = True) -> bool:
        """Set value in cache with TTL."""
        try:
            # Store in local cache if enabled
            if use_local:
                self.local_cache[key] = {
                    'value': value,
                    'expires': time.time() + min(ttl, 30)  # Max 30s for local
                }
            
            # Store in Redis
            if self.cache_enabled:
                serialized = json.dumps(value, default=str)
                return self.redis_client.setex(key, ttl, serialized)
            
            return True
        except (RedisError, json.JSONEncodeError) as e:
            logger.warning(f"Cache set error for {key}: {e}")
            self.cache_stats['errors'] += 1
            return False
    
    def delete(self, key: str) -> bool:
        """Delete a specific key from cache."""
        # Remove from local cache
        if key in self.local_cache:
            del self.local_cache[key]
        
        # Remove from Redis
        if self.cache_enabled:
            try:
                self.redis_client.delete(key)
                self.cache_stats['invalidations'] += 1
                return True
            except RedisError as e:
                logger.warning(f"Cache delete error for {key}: {e}")
                self.cache_stats['errors'] += 1
        
        return False
    
    def invalidate_pattern(self, pattern: str) -> int:
        """Invalidate all keys matching a pattern."""
        count = 0
        
        # Clear matching keys from local cache
        keys_to_delete = [k for k in self.local_cache.keys() if self._matches_pattern(k, pattern)]
        for key in keys_to_delete:
            del self.local_cache[key]
            count += 1
        
        # Clear from Redis
        if self.cache_enabled:
            try:
                # Use SCAN to avoid blocking
                cursor = 0
                while True:
                    cursor, keys = self.redis_client.scan(cursor, match=pattern, count=100)
                    if keys:
                        self.redis_client.delete(*keys)
                        count += len(keys)
                    if cursor == 0:
                        break
                
                self.cache_stats['invalidations'] += count
            except RedisError as e:
                logger.warning(f"Pattern invalidation error for {pattern}: {e}")
                self.cache_stats['errors'] += 1
        
        return count
    
    def _matches_pattern(self, key: str, pattern: str) -> bool:
        """Check if key matches pattern (simple wildcard support)."""
        import fnmatch
        return fnmatch.fnmatch(key, pattern)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        stats = self.cache_stats.copy()
        
        # Calculate hit rate
        total_requests = stats['hits'] + stats['misses']
        stats['hit_rate'] = (stats['hits'] / total_requests * 100) if total_requests > 0 else 0
        
        # Add cache sizes
        stats['local_cache_size'] = len(self.local_cache)
        
        if self.cache_enabled:
            try:
                info = self.redis_client.info()
                stats['redis_keys'] = self.redis_client.dbsize()
                stats['redis_memory'] = info.get('used_memory_human', 'N/A')
            except RedisError:
                pass
        
        return stats
    
    def clear_expired_local(self):
        """Clear expired entries from local cache."""
        current_time = time.time()
        expired_keys = [k for k, v in self.local_cache.items() if v['expires'] <= current_time]
        for key in expired_keys:
            del self.local_cache[key]


# Global cache manager instance
_cache_manager: Optional[CacheManager] = None


def get_cache_manager() -> CacheManager:
    """Get or create global cache manager instance."""
    global _cache_manager
    
    if _cache_manager is None:
        import os
        redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
        _cache_manager = CacheManager(redis_url)
    
    return _cache_manager


def cached_result(ttl: int = 300, key_prefix: str = "generic", use_local: bool = True):
    """Decorator to cache function results with custom TTL and key prefix."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            cache = get_cache_manager()
            
            # Generate cache key
            cache_key = cache._generate_cache_key(key_prefix, *args[1:], **kwargs)  # Skip 'self' if present
            
            # Try to get from cache
            cached_value = cache.get(cache_key, use_local=use_local)
            if cached_value is not None:
                return cached_value
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            if result is not None:
                cache.set(cache_key, result, ttl=ttl, use_local=use_local)
            
            return result
        
        return wrapper
    return decorator


def cache_analytics_result(ttl: int = 300):
    """Specialized decorator for analytics caching with automatic invalidation."""
    return cached_result(ttl=ttl, key_prefix="analytics", use_local=True)


def cache_user_data(ttl: int = 60):
    """Specialized decorator for user-specific data caching."""
    return cached_result(ttl=ttl, key_prefix="user", use_local=True)


def cache_query_result(ttl: int = 300, key_prefix: str = "query"):
    """Decorator for database query result caching."""
    return cached_result(ttl=ttl, key_prefix=key_prefix, use_local=False)


def invalidate_user_cache(user_id: int):
    """Invalidate all cache entries for a specific user."""
    cache = get_cache_manager()
    pattern = f"user:*:{user_id}:*"
    count = cache.invalidate_pattern(pattern)
    logger.info(f"Invalidated {count} cache entries for user {user_id}")


def invalidate_analytics_cache():
    """Invalidate all analytics cache entries."""
    cache = get_cache_manager()
    count = cache.invalidate_pattern("analytics:*")
    logger.info(f"Invalidated {count} analytics cache entries")


def warm_cache(warmup_functions: List[Callable]):
    """Warm up cache by calling specified functions."""
    logger.info("Starting cache warm-up...")
    
    for func in warmup_functions:
        try:
            func()
        except Exception as e:
            logger.error(f"Cache warm-up failed for {func.__name__}: {e}")
    
    logger.info("Cache warm-up completed")


class CacheMetrics:
    """Cache metrics collector for monitoring."""
    
    def __init__(self):
        self.metrics = []
        self.lock = threading.Lock()
    
    def record_operation(self, operation: str, key: str, duration: float, hit: bool = None):
        """Record a cache operation."""
        with self.lock:
            self.metrics.append({
                'timestamp': datetime.utcnow(),
                'operation': operation,
                'key': key,
                'duration': duration,
                'hit': hit
            })
            
            # Keep only last 1000 entries
            if len(self.metrics) > 1000:
                self.metrics = self.metrics[-1000:]
    
    def get_summary(self, minutes: int = 5) -> Dict[str, Any]:
        """Get metrics summary for the last N minutes."""
        cutoff = datetime.utcnow() - timedelta(minutes=minutes)
        
        with self.lock:
            recent_metrics = [m for m in self.metrics if m['timestamp'] > cutoff]
        
        if not recent_metrics:
            return {
                'operations': 0,
                'avg_duration': 0,
                'hit_rate': 0
            }
        
        total_ops = len(recent_metrics)
        avg_duration = sum(m['duration'] for m in recent_metrics) / total_ops
        
        # Calculate hit rate
        hit_ops = [m for m in recent_metrics if m.get('hit') is not None]
        if hit_ops:
            hit_rate = sum(1 for m in hit_ops if m['hit']) / len(hit_ops) * 100
        else:
            hit_rate = 0
        
        return {
            'operations': total_ops,
            'avg_duration': avg_duration,
            'hit_rate': hit_rate,
            'operations_per_second': total_ops / (minutes * 60)
        }


# Global metrics instance
_cache_metrics = CacheMetrics()


def get_cache_metrics() -> CacheMetrics:
    """Get global cache metrics instance."""
    return _cache_metrics