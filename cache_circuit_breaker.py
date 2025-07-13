"""
Cache Circuit Breaker for NightScan

Provides specialized circuit breaker protection for Redis/cache operations
with intelligent fallbacks to ensure core functionality continues when
cache services are unavailable.

Features:
- Redis connection monitoring and health checks
- Automatic fallback to in-memory cache or database
- Session storage fallback mechanisms
- Task queue fallback for Celery operations
- Cache warming and preloading strategies
- Memory-based cache for critical operations
"""

import time
import json
import pickle
import logging
import threading
from typing import Any, Callable, Dict, List, Optional, Union
from dataclasses import dataclass, field
from collections import OrderedDict
import tempfile
import os

try:
    import redis
    from redis.exceptions import RedisError, ConnectionError as RedisConnectionError
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None
    RedisError = Exception
    RedisConnectionError = Exception

from circuit_breaker import (
    CircuitBreaker, CircuitBreakerConfig, CircuitBreakerOpenException,
    register_circuit_breaker
)
from exceptions import CacheServiceError, ExternalServiceError

logger = logging.getLogger(__name__)


@dataclass 
class CacheCircuitBreakerConfig(CircuitBreakerConfig):
    """Extended configuration for cache circuit breaker."""
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: Optional[str] = None
    
    # Timeout configuration
    connection_timeout: float = 2.0      # Redis connection timeout
    socket_timeout: float = 2.0          # Redis socket timeout
    
    # Fallback configuration
    enable_memory_fallback: bool = True   # Use in-memory cache as fallback
    memory_cache_size: int = 1000        # Max items in memory cache
    enable_disk_fallback: bool = True     # Use disk-based fallback for sessions
    disk_cache_dir: Optional[str] = None  # Directory for disk cache
    
    # Cache behavior
    default_ttl: int = 3600              # Default TTL for cache entries (seconds)
    cache_key_prefix: str = "nightscan:" # Prefix for all cache keys
    
    # Health checking
    health_check_key: str = "health_check"  # Key used for health checks
    max_memory_usage: float = 0.8        # Max Redis memory usage before warnings
    
    def __post_init__(self):
        super().__post_init__()
        
        # Cache-specific exception handling
        if REDIS_AVAILABLE:
            self.expected_exception = (
                RedisError, RedisConnectionError, ConnectionError,
                CacheServiceError, ExternalServiceError
            )
        else:
            self.expected_exception = (CacheServiceError, ExternalServiceError)
        
        # Set default disk cache directory
        if self.disk_cache_dir is None:
            self.disk_cache_dir = os.path.join(tempfile.gettempdir(), "nightscan_cache")


class MemoryCache:
    """Simple LRU memory cache for fallback purposes."""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.cache = OrderedDict()
        self.lock = threading.RLock()
    
    def get(self, key: str) -> Any:
        """Get value from memory cache."""
        with self.lock:
            if key in self.cache:
                # Move to end (most recently used)
                value = self.cache.pop(key)
                self.cache[key] = value
                return value
            return None
    
    def set(self, key: str, value: Any, ttl: int = None) -> bool:
        """Set value in memory cache."""
        with self.lock:
            if key in self.cache:
                # Update existing
                self.cache.pop(key)
            elif len(self.cache) >= self.max_size:
                # Remove least recently used
                self.cache.popitem(last=False)
            
            # Add TTL handling (simplified)
            if ttl:
                value = {
                    'data': value,
                    'expires': time.time() + ttl
                }
            
            self.cache[key] = value
            return True
    
    def delete(self, key: str) -> bool:
        """Delete key from memory cache."""
        with self.lock:
            if key in self.cache:
                del self.cache[key]
                return True
            return False
    
    def clear(self):
        """Clear all entries from memory cache."""
        with self.lock:
            self.cache.clear()
    
    def size(self) -> int:
        """Get current cache size."""
        with self.lock:
            return len(self.cache)


class DiskCache:
    """Simple disk-based cache for persistent fallback."""
    
    def __init__(self, cache_dir: str):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.lock = threading.RLock()
    
    def _get_path(self, key: str) -> str:
        """Get file path for cache key."""
        # Simple hash to avoid filesystem issues
        key_hash = str(abs(hash(key)))
        return os.path.join(self.cache_dir, f"{key_hash}.cache")
    
    def get(self, key: str) -> Any:
        """Get value from disk cache."""
        try:
            path = self._get_path(key)
            if os.path.exists(path):
                with open(path, 'rb') as f:
                    data = pickle.load(f)
                    
                # Check TTL
                if 'expires' in data and time.time() > data['expires']:
                    self.delete(key)
                    return None
                
                return data.get('value')
        except Exception as e:
            logger.warning(f"Error reading from disk cache: {e}")
        return None
    
    def set(self, key: str, value: Any, ttl: int = None) -> bool:
        """Set value in disk cache."""
        try:
            with self.lock:
                path = self._get_path(key)
                data = {'value': value}
                
                if ttl:
                    data['expires'] = time.time() + ttl
                
                with open(path, 'wb') as f:
                    pickle.dump(data, f)
                return True
        except Exception as e:
            logger.warning(f"Error writing to disk cache: {e}")
        return False
    
    def delete(self, key: str) -> bool:
        """Delete key from disk cache."""
        try:
            path = self._get_path(key)
            if os.path.exists(path):
                os.remove(path)
                return True
        except Exception as e:
            logger.warning(f"Error deleting from disk cache: {e}")
        return False


class CacheCircuitBreaker(CircuitBreaker):
    """
    Specialized circuit breaker for cache operations.
    
    Provides protection for Redis operations with intelligent
    fallbacks to memory cache, disk cache, or direct database access.
    """
    
    def __init__(self, config: CacheCircuitBreakerConfig):
        super().__init__(config)
        self.cache_config = config
        
        # Redis connection
        self.redis_client = None
        self._connect_redis()
        
        # Fallback caches
        self.memory_cache = MemoryCache(config.memory_cache_size) if config.enable_memory_fallback else None
        self.disk_cache = DiskCache(config.disk_cache_dir) if config.enable_disk_fallback else None
        
        # Cache statistics
        self.cache_stats = {
            'redis_hits': 0,
            'redis_misses': 0,
            'memory_hits': 0,
            'memory_misses': 0,
            'disk_hits': 0,
            'disk_misses': 0,
            'fallback_operations': 0
        }
        
        logger.info(f"Cache circuit breaker '{config.name}' initialized")
    
    def _connect_redis(self):
        """Establish Redis connection."""
        if not REDIS_AVAILABLE:
            logger.warning("Redis not available, cache circuit breaker will use fallbacks only")
            return
        
        try:
            self.redis_client = redis.Redis(
                host=self.cache_config.redis_host,
                port=self.cache_config.redis_port,
                db=self.cache_config.redis_db,
                password=self.cache_config.redis_password,
                socket_timeout=self.cache_config.socket_timeout,
                socket_connect_timeout=self.cache_config.connection_timeout,
                decode_responses=False  # Keep binary for pickle compatibility
            )
            
            # Test connection
            self.redis_client.ping()
            logger.info(f"Connected to Redis at {self.cache_config.redis_host}:{self.cache_config.redis_port}")
            
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self.redis_client = None
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get value from cache with fallback hierarchy.
        
        Tries: Redis -> Memory Cache -> Disk Cache -> Default
        """
        full_key = f"{self.cache_config.cache_key_prefix}{key}"
        
        def redis_get():
            if not self.redis_client:
                raise CacheServiceError("Redis client not available")
            
            value = self.redis_client.get(full_key)
            if value is not None:
                self.cache_stats['redis_hits'] += 1
                try:
                    return pickle.loads(value)
                except (pickle.PickleError, TypeError):
                    # If pickle fails, try as string
                    return value.decode('utf-8') if isinstance(value, bytes) else value
            else:
                self.cache_stats['redis_misses'] += 1
                raise CacheServiceError("Key not found in Redis")
        
        try:
            return self.call(redis_get)
        except CircuitBreakerOpenException:
            # Circuit is open, try fallbacks
            return self._get_from_fallback(key, default)
        except Exception as e:
            logger.warning(f"Redis get failed for key '{key}': {e}")
            return self._get_from_fallback(key, default)
    
    def set(self, key: str, value: Any, ttl: int = None) -> bool:
        """
        Set value in cache with fallback hierarchy.
        
        Tries: Redis -> Memory Cache -> Disk Cache
        """
        full_key = f"{self.cache_config.cache_key_prefix}{key}"
        ttl = ttl or self.cache_config.default_ttl
        
        def redis_set():
            if not self.redis_client:
                raise CacheServiceError("Redis client not available")
            
            try:
                serialized_value = pickle.dumps(value)
            except (pickle.PickleError, TypeError):
                # If pickle fails, try as string
                serialized_value = str(value)
            
            result = self.redis_client.setex(full_key, ttl, serialized_value)
            if not result:
                raise CacheServiceError("Failed to set value in Redis")
            return True
        
        try:
            return self.call(redis_set)
        except CircuitBreakerOpenException:
            # Circuit is open, try fallbacks
            return self._set_to_fallback(key, value, ttl)
        except Exception as e:
            logger.warning(f"Redis set failed for key '{key}': {e}")
            return self._set_to_fallback(key, value, ttl)
    
    def delete(self, key: str) -> bool:
        """
        Delete key from cache and all fallbacks.
        """
        full_key = f"{self.cache_config.cache_key_prefix}{key}"
        
        def redis_delete():
            if not self.redis_client:
                raise CacheServiceError("Redis client not available")
            
            result = self.redis_client.delete(full_key)
            return bool(result)
        
        redis_success = False
        try:
            redis_success = self.call(redis_delete)
        except Exception as e:
            logger.warning(f"Redis delete failed for key '{key}': {e}")
        
        # Always try to delete from fallbacks
        memory_success = self.memory_cache.delete(key) if self.memory_cache else False
        disk_success = self.disk_cache.delete(key) if self.disk_cache else False
        
        return redis_success or memory_success or disk_success
    
    def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        full_key = f"{self.cache_config.cache_key_prefix}{key}"
        
        def redis_exists():
            if not self.redis_client:
                raise CacheServiceError("Redis client not available")
            return bool(self.redis_client.exists(full_key))
        
        try:
            return self.call(redis_exists)
        except Exception:
            # Check fallbacks
            if self.memory_cache and self.memory_cache.get(key) is not None:
                return True
            if self.disk_cache and self.disk_cache.get(key) is not None:
                return True
            return False
    
    def increment(self, key: str, amount: int = 1) -> int:
        """Increment counter with fallback support."""
        full_key = f"{self.cache_config.cache_key_prefix}{key}"
        
        def redis_incr():
            if not self.redis_client:
                raise CacheServiceError("Redis client not available")
            return self.redis_client.incr(full_key, amount)
        
        try:
            return self.call(redis_incr)
        except Exception as e:
            logger.warning(f"Redis increment failed for key '{key}': {e}")
            # Fallback: get current value, increment, set
            current = self.get(key, 0)
            try:
                current = int(current)
            except (ValueError, TypeError):
                current = 0
            new_value = current + amount
            self.set(key, new_value)
            return new_value
    
    def _get_from_fallback(self, key: str, default: Any) -> Any:
        """Get value from fallback caches."""
        self.cache_stats['fallback_operations'] += 1
        
        # Try memory cache first
        if self.memory_cache:
            value = self.memory_cache.get(key)
            if value is not None:
                self.cache_stats['memory_hits'] += 1
                # Handle TTL wrapped values
                if isinstance(value, dict) and 'expires' in value:
                    if time.time() > value['expires']:
                        self.memory_cache.delete(key)
                        value = None
                    else:
                        value = value['data']
                
                if value is not None:
                    return value
            else:
                self.cache_stats['memory_misses'] += 1
        
        # Try disk cache
        if self.disk_cache:
            value = self.disk_cache.get(key)
            if value is not None:
                self.cache_stats['disk_hits'] += 1
                # Also store in memory cache for faster access
                if self.memory_cache:
                    self.memory_cache.set(key, value)
                return value
            else:
                self.cache_stats['disk_misses'] += 1
        
        return default
    
    def _set_to_fallback(self, key: str, value: Any, ttl: int) -> bool:
        """Set value to fallback caches."""
        self.cache_stats['fallback_operations'] += 1
        success = False
        
        # Set in memory cache
        if self.memory_cache:
            success = self.memory_cache.set(key, value, ttl) or success
        
        # Set in disk cache
        if self.disk_cache:
            success = self.disk_cache.set(key, value, ttl) or success
        
        return success
    
    def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive cache health check."""
        health_status = {
            'redis_available': False,
            'redis_responsive': False,
            'memory_cache_available': self.memory_cache is not None,
            'disk_cache_available': self.disk_cache is not None,
            'circuit_state': self.get_state().value
        }
        
        # Test Redis connectivity
        if self.redis_client:
            try:
                health_key = f"{self.cache_config.cache_key_prefix}{self.cache_config.health_check_key}"
                test_value = f"health_check_{time.time()}"
                
                # Test set and get
                self.redis_client.setex(health_key, 10, test_value)
                retrieved = self.redis_client.get(health_key)
                
                if retrieved and retrieved.decode('utf-8') == test_value:
                    health_status['redis_available'] = True
                    health_status['redis_responsive'] = True
                    
                    # Clean up test key
                    self.redis_client.delete(health_key)
                
                # Get Redis info
                info = self.redis_client.info()
                health_status['redis_memory_usage'] = info.get('used_memory_human', 'unknown')
                health_status['redis_connected_clients'] = info.get('connected_clients', 0)
                
            except Exception as e:
                logger.warning(f"Redis health check failed: {e}")
                health_status['redis_error'] = str(e)
        
        # Memory cache stats
        if self.memory_cache:
            health_status['memory_cache_size'] = self.memory_cache.size()
            health_status['memory_cache_max_size'] = self.memory_cache.max_size
        
        # Add cache statistics
        health_status['cache_stats'] = self.cache_stats.copy()
        
        return health_status
    
    def warm_cache(self, warm_data: Dict[str, Any]):
        """Pre-warm cache with critical data."""
        logger.info(f"Warming cache with {len(warm_data)} entries")
        
        for key, value in warm_data.items():
            try:
                self.set(key, value)
            except Exception as e:
                logger.warning(f"Failed to warm cache key '{key}': {e}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get detailed cache statistics."""
        stats = self.cache_stats.copy()
        stats.update(self.get_metrics())
        
        # Calculate hit rates
        total_redis = stats['redis_hits'] + stats['redis_misses']
        if total_redis > 0:
            stats['redis_hit_rate'] = stats['redis_hits'] / total_redis
        
        total_memory = stats['memory_hits'] + stats['memory_misses']
        if total_memory > 0:
            stats['memory_hit_rate'] = stats['memory_hits'] / total_memory
        
        total_disk = stats['disk_hits'] + stats['disk_misses']
        if total_disk > 0:
            stats['disk_hit_rate'] = stats['disk_hits'] / total_disk
        
        return stats
    
    def clear_all_caches(self):
        """Clear all cache levels (use with caution)."""
        logger.warning("Clearing all caches")
        
        # Clear Redis (only keys with our prefix)
        if self.redis_client:
            try:
                pattern = f"{self.cache_config.cache_key_prefix}*"
                keys = self.redis_client.keys(pattern)
                if keys:
                    self.redis_client.delete(*keys)
                logger.info(f"Cleared {len(keys)} keys from Redis")
            except Exception as e:
                logger.error(f"Failed to clear Redis cache: {e}")
        
        # Clear memory cache
        if self.memory_cache:
            self.memory_cache.clear()
            logger.info("Cleared memory cache")
        
        # Clear disk cache (remove all files)
        if self.disk_cache:
            try:
                import shutil
                if os.path.exists(self.cache_config.disk_cache_dir):
                    shutil.rmtree(self.cache_config.disk_cache_dir)
                    os.makedirs(self.cache_config.disk_cache_dir, exist_ok=True)
                logger.info("Cleared disk cache")
            except Exception as e:
                logger.error(f"Failed to clear disk cache: {e}")
    
    def cleanup(self):
        """Cleanup cache circuit breaker resources."""
        # Close Redis connection
        if self.redis_client:
            try:
                self.redis_client.close()
            except Exception as e:
                logger.error(f"Error closing Redis connection: {e}")
        
        # Clear memory cache
        if self.memory_cache:
            self.memory_cache.clear()
        
        self._stop_health_check()
        logger.info(f"Cache circuit breaker '{self.config.name}' cleaned up")


# Convenience functions for cache circuit breaker setup

def create_cache_circuit_breaker(name: str, **kwargs) -> CacheCircuitBreaker:
    """
    Create and register a cache circuit breaker.
    
    Args:
        name: Circuit breaker name
        **kwargs: Additional configuration parameters
        
    Returns:
        Configured CacheCircuitBreaker instance
    """
    config = CacheCircuitBreakerConfig(name=name, **kwargs)
    circuit_breaker = CacheCircuitBreaker(config)
    register_circuit_breaker(circuit_breaker)
    return circuit_breaker


def cache_circuit_breaker(name: str, **kwargs):
    """
    Decorator for cache operations with circuit breaker protection.
    
    Usage:
        cache_cb = cache_circuit_breaker(name="predictions", redis_host="localhost")
        
        @cache_cb
        def get_cached_prediction(file_hash):
            return cache.get(f"prediction:{file_hash}")
    """
    circuit_breaker = create_cache_circuit_breaker(name, **kwargs)
    return circuit_breaker


# Session storage with cache circuit breaker fallback

class SessionFallbackManager:
    """Manages session storage with cache circuit breaker fallback."""
    
    def __init__(self, cache_circuit: CacheCircuitBreaker):
        self.cache_circuit = cache_circuit
        self.session_prefix = "session:"
    
    def get_session(self, session_id: str) -> Dict[str, Any]:
        """Get session data with fallback support."""
        key = f"{self.session_prefix}{session_id}"
        session_data = self.cache_circuit.get(key, {})
        return session_data if isinstance(session_data, dict) else {}
    
    def set_session(self, session_id: str, session_data: Dict[str, Any], ttl: int = 3600) -> bool:
        """Set session data with fallback support."""
        key = f"{self.session_prefix}{session_id}"
        return self.cache_circuit.set(key, session_data, ttl)
    
    def delete_session(self, session_id: str) -> bool:
        """Delete session data."""
        key = f"{self.session_prefix}{session_id}"
        return self.cache_circuit.delete(key)


# Task queue fallback for Celery

class TaskQueueFallback:
    """Simple in-memory task queue fallback when Redis is unavailable."""
    
    def __init__(self):
        self.tasks = []
        self.lock = threading.RLock()
    
    def add_task(self, task_name: str, args: list, kwargs: dict):
        """Add task to fallback queue."""
        with self.lock:
            task = {
                'name': task_name,
                'args': args,
                'kwargs': kwargs,
                'timestamp': time.time()
            }
            self.tasks.append(task)
            logger.info(f"Added task '{task_name}' to fallback queue")
    
    def get_pending_tasks(self) -> List[Dict]:
        """Get all pending tasks."""
        with self.lock:
            return self.tasks.copy()
    
    def clear_tasks(self):
        """Clear all tasks from fallback queue."""
        with self.lock:
            count = len(self.tasks)
            self.tasks.clear()
            logger.info(f"Cleared {count} tasks from fallback queue")