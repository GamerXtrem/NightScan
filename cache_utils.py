"""Redis caching utilities for NightScan predictions."""

import hashlib
import json
import logging
import os
from typing import Optional, Any, Dict, List
from functools import wraps

try:
    import redis
    from redis.exceptions import RedisError, ConnectionError as RedisConnectionError
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    # Mock Redis for graceful fallback
    class MockRedis:
        def get(self, key): return None
        def set(self, key, value, ex=None): return True
        def delete(self, key): return True
        def flushdb(self): return True
        def ping(self): raise Exception("Redis not available")
    
    redis = None
    RedisError = Exception
    RedisConnectionError = Exception

logger = logging.getLogger(__name__)


class PredictionCache:
    """Redis-based cache for audio predictions with graceful fallback."""
    
    def __init__(self, redis_url: Optional[str] = None, default_ttl: int = 3600):
        """Initialize cache with optional Redis connection."""
        self.default_ttl = default_ttl
        self.redis_client = None
        self.cache_enabled = False
        
        if REDIS_AVAILABLE and redis_url:
            try:
                self.redis_client = redis.from_url(redis_url, decode_responses=True)
                # Test connection
                self.redis_client.ping()
                self.cache_enabled = True
                logger.info("Redis cache initialized successfully")
            except (RedisConnectionError, RedisError) as e:
                logger.warning(f"Redis connection failed, running without cache: {e}")
                self.redis_client = MockRedis()
        else:
            logger.info("Redis not available or not configured, running without cache")
            self.redis_client = MockRedis()
    
    def _generate_key(self, audio_data: bytes) -> str:
        """Generate cache key from audio file content."""
        audio_hash = hashlib.sha256(audio_data).hexdigest()
        return f"prediction:{audio_hash}"
    
    def _serialize_result(self, result: List[Dict]) -> str:
        """Serialize prediction result for storage."""
        return json.dumps(result, sort_keys=True)
    
    def _deserialize_result(self, data: str) -> List[Dict]:
        """Deserialize prediction result from storage."""
        return json.loads(data)
    
    def get_prediction(self, audio_data: bytes) -> Optional[List[Dict]]:
        """Get cached prediction result for audio data."""
        if not self.cache_enabled:
            return None
        
        try:
            key = self._generate_key(audio_data)
            cached_data = self.redis_client.get(key)
            
            if cached_data:
                logger.info(f"Cache hit for audio hash {key}")
                return self._deserialize_result(cached_data)
            else:
                logger.debug(f"Cache miss for audio hash {key}")
                return None
                
        except (RedisError, json.JSONDecodeError) as e:
            logger.warning(f"Cache retrieval failed: {e}")
            return None
    
    def get_prediction_by_hash(self, audio_hash: str) -> Optional[List[Dict]]:
        """Get cached prediction result by pre-computed hash."""
        if not self.cache_enabled:
            return None
        
        try:
            key = f"prediction:{audio_hash}"
            cached_data = self.redis_client.get(key)
            
            if cached_data:
                logger.info(f"Cache hit for audio hash {key}")
                return self._deserialize_result(cached_data)
            else:
                logger.debug(f"Cache miss for audio hash {key}")
                return None
                
        except (RedisError, json.JSONDecodeError) as e:
            logger.warning(f"Cache retrieval failed: {e}")
            return None
    
    def cache_prediction(self, audio_data: bytes, result: List[Dict], ttl: Optional[int] = None) -> bool:
        """Cache prediction result for audio data."""
        if not self.cache_enabled:
            return False
        
        try:
            key = self._generate_key(audio_data)
            serialized_result = self._serialize_result(result)
            ttl = ttl or self.default_ttl
            
            success = self.redis_client.set(key, serialized_result, ex=ttl)
            if success:
                logger.info(f"Cached prediction for audio hash {key} (TTL: {ttl}s)")
            return success
        except Exception as e:
            logger.error(f"Error caching prediction: {e}")
            return False
    
    def cache_prediction_by_hash(self, audio_hash: str, result: List[Dict], ttl: Optional[int] = None) -> bool:
        """Cache prediction result by pre-computed hash."""
        if not self.cache_enabled:
            return False
        
        try:
            key = f"prediction:{audio_hash}"
            serialized_result = self._serialize_result(result)
            ttl = ttl or self.default_ttl
            
            success = self.redis_client.set(key, serialized_result, ex=ttl)
            if success:
                logger.info(f"Cached prediction for audio hash {key} (TTL: {ttl}s)")
            return success
            
        except (RedisError, json.JSONEncodeError) as e:
            logger.warning(f"Cache storage failed: {e}")
            return False
    
    def invalidate_prediction(self, audio_data: bytes) -> bool:
        """Invalidate cached prediction for audio data."""
        if not self.cache_enabled:
            return False
        
        try:
            key = self._generate_key(audio_data)
            deleted = self.redis_client.delete(key)
            if deleted:
                logger.info(f"Invalidated cache for audio hash {key}")
            return bool(deleted)
            
        except RedisError as e:
            logger.warning(f"Cache invalidation failed: {e}")
            return False
    
    def clear_cache(self) -> bool:
        """Clear all cached predictions."""
        if not self.cache_enabled:
            return False
        
        try:
            self.redis_client.flushdb()
            logger.info("Cleared all prediction cache")
            return True
            
        except RedisError as e:
            logger.warning(f"Cache clear failed: {e}")
            return False
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        stats = {
            "enabled": self.cache_enabled,
            "redis_available": REDIS_AVAILABLE,
            "default_ttl": self.default_ttl
        }
        
        if self.cache_enabled:
            try:
                # Get basic Redis info
                info = self.redis_client.info()
                stats.update({
                    "redis_version": info.get("redis_version"),
                    "used_memory": info.get("used_memory_human"),
                    "connected_clients": info.get("connected_clients"),
                    "keyspace_hits": info.get("keyspace_hits", 0),
                    "keyspace_misses": info.get("keyspace_misses", 0)
                })
                
                # Calculate hit rate
                hits = stats.get("keyspace_hits", 0)
                misses = stats.get("keyspace_misses", 0)
                total = hits + misses
                stats["hit_rate"] = (hits / total * 100) if total > 0 else 0
                
            except RedisError as e:
                logger.warning(f"Failed to get cache stats: {e}")
                stats["error"] = str(e)
        
        return stats


# Global cache instance
_cache_instance: Optional[PredictionCache] = None


def get_cache() -> PredictionCache:
    """Get or create global cache instance."""
    global _cache_instance
    
    if _cache_instance is None:
        redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
        cache_ttl = int(os.environ.get("CACHE_TTL", "3600"))  # 1 hour default
        _cache_instance = PredictionCache(redis_url, cache_ttl)
    
    return _cache_instance


def cached_prediction(ttl: Optional[int] = None):
    """Decorator to cache prediction results."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Extract audio data from arguments (assume it's the first arg or 'audio_data' kwarg)
            audio_data = None
            if args and isinstance(args[0], bytes):
                audio_data = args[0]
            elif 'audio_data' in kwargs:
                audio_data = kwargs['audio_data']
            
            if audio_data:
                cache = get_cache()
                
                # Try to get from cache first
                cached_result = cache.get_prediction(audio_data)
                if cached_result is not None:
                    return cached_result
            
            # Not in cache, execute function
            result = func(*args, **kwargs)
            
            # Cache the result if we have audio data
            if audio_data and result:
                cache = get_cache()
                cache.cache_prediction(audio_data, result, ttl)
            
            return result
        
        return wrapper
    return decorator


def cache_health_check() -> Dict[str, Any]:
    """Health check for cache system."""
    cache = get_cache()
    stats = cache.get_cache_stats()
    
    # Determine health status
    if cache.cache_enabled:
        try:
            cache.redis_client.ping()
            stats["status"] = "healthy"
        except Exception as e:
            stats["status"] = "unhealthy"
            stats["error"] = str(e)
    else:
        stats["status"] = "disabled"
    
    return stats