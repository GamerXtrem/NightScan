"""ML Model Serving Optimization with Advanced Connection Pooling.

This module provides optimized ML model serving with:
- Advanced connection pooling for database and external services
- Batch inference optimization
- Model instance pooling and load balancing
- Memory-efficient model management
- Request queuing and rate limiting
- Performance monitoring and auto-scaling
"""

import asyncio
import logging
import time
import threading
import queue
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
from contextlib import asynccontextmanager, contextmanager
from concurrent.futures import ThreadPoolExecutor, as_completed
import gc
import psutil

import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict, deque

try:
    import uvloop
    UVLOOP_AVAILABLE = True
except ImportError:
    UVLOOP_AVAILABLE = False

try:
    import redis.asyncio as aioredis
    ASYNC_REDIS_AVAILABLE = True
except ImportError:
    ASYNC_REDIS_AVAILABLE = False

try:
    import asyncpg
    ASYNCPG_AVAILABLE = True
except ImportError:
    ASYNCPG_AVAILABLE = False

from config import get_config
from model_deployment import get_deployment_manager

logger = logging.getLogger(__name__)


@dataclass
class ConnectionPoolConfig:
    """Configuration for connection pools."""
    # Database connections
    db_min_connections: int = 5
    db_max_connections: int = 20
    db_connection_timeout: float = 10.0
    db_command_timeout: float = 30.0
    
    # Redis connections
    redis_min_connections: int = 3
    redis_max_connections: int = 15
    redis_connection_timeout: float = 5.0
    
    # HTTP client connections
    http_max_connections: int = 100
    http_connection_timeout: float = 10.0
    
    # Model instance pooling
    model_pool_size: int = 3
    model_warmup_requests: int = 5
    
    # Performance settings
    batch_timeout_ms: float = 100.0
    max_batch_size: int = 8
    request_queue_size: int = 1000
    worker_threads: int = 4


@dataclass
class InferenceRequest:
    """Represents a single inference request."""
    request_id: str
    audio_data: np.ndarray
    callback: Callable
    received_at: float = field(default_factory=time.time)
    priority: int = 0  # Higher numbers = higher priority
    timeout: float = 30.0
    experiment_id: Optional[str] = None
    user_id: Optional[str] = None


class DatabaseConnectionPool:
    """Async database connection pool for PostgreSQL."""
    
    def __init__(self, config: ConnectionPoolConfig):
        self.config = config
        self.pool = None
        self._lock = asyncio.Lock()
        
    async def initialize(self, database_url: str) -> bool:
        """Initialize the connection pool."""
        if not ASYNCPG_AVAILABLE:
            logger.warning("asyncpg not available, using fallback database handling")
            return False
            
        try:
            async with self._lock:
                if self.pool is None:
                    self.pool = await asyncpg.create_pool(
                        database_url,
                        min_size=self.config.db_min_connections,
                        max_size=self.config.db_max_connections,
                        command_timeout=self.config.db_command_timeout,
                        server_settings={
                            'application_name': 'nightscan_ml_serving',
                            'jit': 'off'  # Disable JIT for consistent performance
                        }
                    )
                    logger.info(f"Database pool initialized with {self.config.db_min_connections}-{self.config.db_max_connections} connections")
                    return True
        except Exception as e:
            logger.error(f"Failed to initialize database pool: {e}")
            return False
    
    @asynccontextmanager
    async def acquire_connection(self):
        """Acquire a database connection from the pool."""
        if not self.pool:
            raise RuntimeError("Database pool not initialized")
            
        async with self.pool.acquire() as connection:
            try:
                yield connection
            except Exception as e:
                logger.error(f"Database operation failed: {e}")
                raise
    
    async def execute_query(self, query: str, *args) -> List[Dict]:
        """Execute a query and return results."""
        async with self.acquire_connection() as conn:
            rows = await conn.fetch(query, *args)
            return [dict(row) for row in rows]
    
    async def execute_command(self, command: str, *args) -> str:
        """Execute a command and return status."""
        async with self.acquire_connection() as conn:
            return await conn.execute(command, *args)
    
    async def close(self):
        """Close the connection pool."""
        if self.pool:
            await self.pool.close()
            self.pool = None
            logger.info("Database pool closed")


class RedisConnectionPool:
    """Async Redis connection pool."""
    
    def __init__(self, config: ConnectionPoolConfig):
        self.config = config
        self.pool = None
        self._lock = asyncio.Lock()
        
    async def initialize(self, redis_url: str) -> bool:
        """Initialize the Redis connection pool."""
        if not ASYNC_REDIS_AVAILABLE:
            logger.warning("aioredis not available, using fallback caching")
            return False
            
        try:
            async with self._lock:
                if self.pool is None:
                    self.pool = aioredis.ConnectionPool.from_url(
                        redis_url,
                        max_connections=self.config.redis_max_connections,
                        retry_on_timeout=True,
                        health_check_interval=30
                    )
                    
                    # Test connection
                    redis_client = aioredis.Redis(connection_pool=self.pool)
                    await redis_client.ping()
                    await redis_client.close()
                    
                    logger.info(f"Redis pool initialized with max {self.config.redis_max_connections} connections")
                    return True
        except Exception as e:
            logger.error(f"Failed to initialize Redis pool: {e}")
            return False
    
    @asynccontextmanager
    async def acquire_client(self):
        """Acquire a Redis client from the pool."""
        if not self.pool:
            raise RuntimeError("Redis pool not initialized")
            
        redis_client = aioredis.Redis(connection_pool=self.pool)
        try:
            yield redis_client
        finally:
            await redis_client.close()
    
    async def get(self, key: str) -> Optional[bytes]:
        """Get value from Redis."""
        async with self.acquire_client() as client:
            return await client.get(key)
    
    async def set(self, key: str, value: bytes, ttl: Optional[int] = None) -> bool:
        """Set value in Redis with optional TTL."""
        async with self.acquire_client() as client:
            return await client.set(key, value, ex=ttl)
    
    async def delete(self, key: str) -> int:
        """Delete key from Redis."""
        async with self.acquire_client() as client:
            return await client.delete(key)
    
    async def close(self):
        """Close the connection pool."""
        if self.pool:
            await self.pool.disconnect()
            self.pool = None
            logger.info("Redis pool closed")


class ModelInstancePool:
    """Pool of model instances for load balancing and concurrent inference."""
    
    def __init__(self, config: ConnectionPoolConfig):
        self.config = config
        self.model_pools = {}  # deployment_id -> list of model instances
        self.model_locks = {}  # deployment_id -> list of locks
        self.model_usage = {}  # deployment_id -> list of usage counters
        self.warmup_status = {}  # deployment_id -> warmup complete flag
        self._pool_lock = threading.Lock()
        
    def initialize_model_pool(self, deployment_id: str, model_factory: Callable) -> bool:
        """Initialize a pool of model instances for a deployment."""
        try:
            with self._pool_lock:
                if deployment_id in self.model_pools:
                    logger.warning(f"Model pool for {deployment_id} already exists")
                    return True
                
                # Create pool of model instances
                instances = []
                locks = []
                usage_counters = []
                
                for i in range(self.config.model_pool_size):
                    logger.info(f"Creating model instance {i+1}/{self.config.model_pool_size} for {deployment_id}")
                    
                    model_instance = model_factory()
                    if model_instance is None:
                        logger.error(f"Failed to create model instance {i+1} for {deployment_id}")
                        # Clean up partial pool
                        for instance in instances:
                            del instance
                        return False
                    
                    instances.append(model_instance)
                    locks.append(threading.Lock())
                    usage_counters.append(0)
                
                self.model_pools[deployment_id] = instances
                self.model_locks[deployment_id] = locks
                self.model_usage[deployment_id] = usage_counters
                self.warmup_status[deployment_id] = False
                
                logger.info(f"Created model pool with {len(instances)} instances for {deployment_id}")
                
                # Start warmup in background
                warmup_thread = threading.Thread(
                    target=self._warmup_model_pool,
                    args=(deployment_id,),
                    daemon=True
                )
                warmup_thread.start()
                
                return True
                
        except Exception as e:
            logger.error(f"Failed to initialize model pool for {deployment_id}: {e}")
            return False
    
    def _warmup_model_pool(self, deployment_id: str):
        """Warm up model instances with dummy inference requests."""
        try:
            instances = self.model_pools.get(deployment_id, [])
            locks = self.model_locks.get(deployment_id, [])
            
            if not instances or not locks:
                return
            
            logger.info(f"Starting warmup for model pool {deployment_id}")
            
            # Create dummy input tensor
            dummy_input = torch.randn(1, 1, 1000)
            
            for instance_idx, (model, lock) in enumerate(zip(instances, locks)):
                logger.debug(f"Warming up instance {instance_idx} for {deployment_id}")
                
                with lock:
                    for warmup_iter in range(self.config.model_warmup_requests):
                        try:
                            with torch.no_grad():
                                _ = model(dummy_input)
                        except Exception as e:
                            logger.warning(f"Warmup failed for instance {instance_idx}: {e}")
                            break
            
            self.warmup_status[deployment_id] = True
            logger.info(f"Warmup completed for model pool {deployment_id}")
            
        except Exception as e:
            logger.error(f"Model warmup failed for {deployment_id}: {e}")
    
    @contextmanager
    def acquire_model_instance(self, deployment_id: str):
        """Acquire a model instance from the pool using load balancing."""
        if deployment_id not in self.model_pools:
            raise RuntimeError(f"Model pool not found for {deployment_id}")
        
        instances = self.model_pools[deployment_id]
        locks = self.model_locks[deployment_id]
        usage_counters = self.model_usage[deployment_id]
        
        # Find least used available instance
        best_instance_idx = None
        min_usage = float('inf')
        
        for i, (lock, usage) in enumerate(zip(locks, usage_counters)):
            if lock.acquire(blocking=False):  # Non-blocking acquire
                if usage < min_usage:
                    if best_instance_idx is not None:
                        # Release previously acquired lock
                        locks[best_instance_idx].release()
                    
                    best_instance_idx = i
                    min_usage = usage
                else:
                    lock.release()  # Release if not the best choice
        
        if best_instance_idx is None:
            # All instances busy, wait for first available
            for i, lock in enumerate(locks):
                if lock.acquire(blocking=True, timeout=5.0):
                    best_instance_idx = i
                    break
        
        if best_instance_idx is None:
            raise RuntimeError(f"No model instance available for {deployment_id}")
        
        try:
            # Update usage counter
            usage_counters[best_instance_idx] += 1
            
            yield instances[best_instance_idx]
            
        finally:
            locks[best_instance_idx].release()
    
    def cleanup_model_pool(self, deployment_id: str) -> bool:
        """Clean up model pool for a deployment."""
        try:
            with self._pool_lock:
                if deployment_id in self.model_pools:
                    # Clean up instances
                    instances = self.model_pools[deployment_id]
                    for instance in instances:
                        del instance
                    
                    # Clean up tracking data
                    del self.model_pools[deployment_id]
                    del self.model_locks[deployment_id]
                    del self.model_usage[deployment_id]
                    del self.warmup_status[deployment_id]
                    
                    # Force garbage collection
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    logger.info(f"Cleaned up model pool for {deployment_id}")
                    return True
                
                return False
                
        except Exception as e:
            logger.error(f"Failed to cleanup model pool for {deployment_id}: {e}")
            return False
    
    def get_pool_stats(self, deployment_id: str) -> Dict[str, Any]:
        """Get statistics for a model pool."""
        if deployment_id not in self.model_pools:
            return {}
        
        instances = self.model_pools[deployment_id]
        usage_counters = self.model_usage[deployment_id]
        
        return {
            'deployment_id': deployment_id,
            'pool_size': len(instances),
            'total_usage': sum(usage_counters),
            'usage_per_instance': usage_counters,
            'warmup_complete': self.warmup_status.get(deployment_id, False),
            'average_usage': sum(usage_counters) / len(usage_counters) if usage_counters else 0
        }


class BatchInferenceProcessor:
    """Processes inference requests in optimized batches."""
    
    def __init__(self, config: ConnectionPoolConfig, model_pool: ModelInstancePool):
        self.config = config
        self.model_pool = model_pool
        
        # Request queues per deployment
        self.request_queues = {}  # deployment_id -> priority queue
        self.batch_processors = {}  # deployment_id -> processor thread
        self.active_batches = {}  # deployment_id -> current batch
        
        # Performance tracking
        self.batch_stats = defaultdict(lambda: {
            'total_batches': 0,
            'total_requests': 0,
            'average_batch_size': 0.0,
            'average_processing_time': 0.0
        })
        
        self._shutdown_flag = threading.Event()
    
    def start_batch_processor(self, deployment_id: str) -> bool:
        """Start batch processing for a deployment."""
        try:
            if deployment_id in self.batch_processors:
                logger.warning(f"Batch processor already running for {deployment_id}")
                return True
            
            # Create priority queue for requests
            self.request_queues[deployment_id] = queue.PriorityQueue(maxsize=self.config.request_queue_size)
            
            # Start batch processor thread
            processor_thread = threading.Thread(
                target=self._batch_processing_loop,
                args=(deployment_id,),
                daemon=True,
                name=f"BatchProcessor-{deployment_id}"
            )
            
            self.batch_processors[deployment_id] = processor_thread
            processor_thread.start()
            
            logger.info(f"Started batch processor for {deployment_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start batch processor for {deployment_id}: {e}")
            return False
    
    def submit_request(self, deployment_id: str, request: InferenceRequest) -> bool:
        """Submit an inference request for batch processing."""
        if deployment_id not in self.request_queues:
            logger.error(f"No batch processor for {deployment_id}")
            return False
        
        try:
            # Use negative priority for max-heap behavior (higher priority first)
            priority_item = (-request.priority, request.received_at, request)
            self.request_queues[deployment_id].put(priority_item, timeout=1.0)
            return True
            
        except queue.Full:
            logger.warning(f"Request queue full for {deployment_id}")
            return False
        except Exception as e:
            logger.error(f"Failed to submit request for {deployment_id}: {e}")
            return False
    
    def _batch_processing_loop(self, deployment_id: str):
        """Main batch processing loop for a deployment."""
        logger.info(f"Batch processing loop started for {deployment_id}")
        
        request_queue = self.request_queues[deployment_id]
        current_batch = []
        batch_deadline = 0
        
        while not self._shutdown_flag.is_set():
            try:
                # Calculate timeout for next request
                current_time = time.time() * 1000  # Convert to milliseconds
                
                if current_batch and current_time >= batch_deadline:
                    # Process current batch if deadline reached
                    self._process_batch(deployment_id, current_batch)
                    current_batch = []
                    batch_deadline = 0
                    continue
                
                # Calculate remaining timeout
                if current_batch:
                    timeout = max(0.001, (batch_deadline - current_time) / 1000)
                else:
                    timeout = 1.0  # Wait up to 1 second for first request
                
                try:
                    # Get next request
                    priority_item = request_queue.get(timeout=timeout)
                    _, _, request = priority_item
                    
                    # Add to current batch
                    current_batch.append(request)
                    
                    # Set batch deadline if this is first request
                    if len(current_batch) == 1:
                        batch_deadline = current_time + self.config.batch_timeout_ms
                    
                    # Process batch if it reaches max size
                    if len(current_batch) >= self.config.max_batch_size:
                        self._process_batch(deployment_id, current_batch)
                        current_batch = []
                        batch_deadline = 0
                        
                except queue.Empty:
                    # Timeout reached, process current batch if any
                    if current_batch:
                        self._process_batch(deployment_id, current_batch)
                        current_batch = []
                        batch_deadline = 0
                    continue
                    
            except Exception as e:
                logger.error(f"Error in batch processing loop for {deployment_id}: {e}")
                time.sleep(0.1)
        
        # Process remaining requests on shutdown
        if current_batch:
            self._process_batch(deployment_id, current_batch)
        
        logger.info(f"Batch processing loop stopped for {deployment_id}")
    
    def _process_batch(self, deployment_id: str, batch: List[InferenceRequest]):
        """Process a batch of inference requests."""
        if not batch:
            return
        
        batch_start_time = time.time()
        batch_size = len(batch)
        
        try:
            logger.debug(f"Processing batch of {batch_size} requests for {deployment_id}")
            
            # Check for expired requests
            current_time = time.time()
            valid_requests = []
            
            for request in batch:
                if current_time - request.received_at < request.timeout:
                    valid_requests.append(request)
                else:
                    logger.warning(f"Request {request.request_id} expired")
                    # Call callback with timeout error
                    try:
                        request.callback({
                            'error': 'Request timeout',
                            'request_id': request.request_id
                        })
                    except Exception as cb_error:
                        logger.error(f"Callback error for expired request: {cb_error}")
            
            if not valid_requests:
                return
            
            # Prepare batch input
            batch_audio_data = np.stack([req.audio_data for req in valid_requests])
            
            # Run batch inference
            with self.model_pool.acquire_model_instance(deployment_id) as model:
                batch_results = self._run_batch_inference(model, batch_audio_data)
            
            # Send results to callbacks
            for request, result in zip(valid_requests, batch_results):
                try:
                    result['request_id'] = request.request_id
                    result['batch_size'] = len(valid_requests)
                    request.callback(result)
                except Exception as cb_error:
                    logger.error(f"Callback error for request {request.request_id}: {cb_error}")
            
            # Update statistics
            processing_time = time.time() - batch_start_time
            stats = self.batch_stats[deployment_id]
            stats['total_batches'] += 1
            stats['total_requests'] += len(valid_requests)
            
            # Update running averages
            total_batches = stats['total_batches']
            stats['average_batch_size'] = (
                (stats['average_batch_size'] * (total_batches - 1) + len(valid_requests)) / total_batches
            )
            stats['average_processing_time'] = (
                (stats['average_processing_time'] * (total_batches - 1) + processing_time) / total_batches
            )
            
            logger.debug(f"Batch processed in {processing_time:.3f}s for {deployment_id}")
            
        except Exception as e:
            logger.error(f"Batch processing failed for {deployment_id}: {e}")
            
            # Send error to all request callbacks
            for request in batch:
                try:
                    request.callback({
                        'error': f'Batch processing failed: {str(e)}',
                        'request_id': request.request_id
                    })
                except Exception as cb_error:
                    logger.error(f"Error callback failed for request {request.request_id}: {cb_error}")
    
    def _run_batch_inference(self, model: nn.Module, batch_audio_data: np.ndarray) -> List[Dict[str, Any]]:
        """Run inference on a batch of audio data."""
        try:
            # Convert to tensor
            if len(batch_audio_data.shape) == 2:
                # Add channel dimension: (batch_size, length) -> (batch_size, 1, length)
                batch_tensor = torch.from_numpy(batch_audio_data).float().unsqueeze(1)
            else:
                batch_tensor = torch.from_numpy(batch_audio_data).float()
            
            # Move to device
            device = next(model.parameters()).device
            batch_tensor = batch_tensor.to(device)
            
            # Run inference
            with torch.no_grad():
                outputs = model(batch_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                
                # Get top predictions for each sample
                top_probs, top_indices = torch.topk(probabilities, k=min(5, outputs.size(1)), dim=1)
                
                # Convert to CPU and numpy
                top_probs = top_probs.cpu().numpy()
                top_indices = top_indices.cpu().numpy()
            
            # Format results for each sample in batch
            results = []
            for i in range(batch_audio_data.shape[0]):
                sample_results = {
                    'top_predictions': [
                        {
                            'class_index': int(top_indices[i][j]),
                            'probability': float(top_probs[i][j])
                        }
                        for j in range(len(top_indices[i]))
                    ],
                    'predicted_class': int(top_indices[i][0]),
                    'confidence': float(top_probs[i][0]),
                    'inference_method': 'batch'
                }
                results.append(sample_results)
            
            return results
            
        except Exception as e:
            logger.error(f"Batch inference failed: {e}")
            # Return error results for all samples
            return [
                {
                    'error': f'Inference failed: {str(e)}',
                    'inference_method': 'batch'
                }
                for _ in range(batch_audio_data.shape[0])
            ]
    
    def get_batch_stats(self, deployment_id: str) -> Dict[str, Any]:
        """Get batch processing statistics."""
        return dict(self.batch_stats.get(deployment_id, {}))
    
    def stop_batch_processor(self, deployment_id: str) -> bool:
        """Stop batch processing for a deployment."""
        try:
            if deployment_id in self.batch_processors:
                # Signal shutdown
                self._shutdown_flag.set()
                
                # Wait for processor to finish
                processor_thread = self.batch_processors[deployment_id]
                processor_thread.join(timeout=5.0)
                
                # Clean up
                del self.batch_processors[deployment_id]
                del self.request_queues[deployment_id]
                
                logger.info(f"Stopped batch processor for {deployment_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to stop batch processor for {deployment_id}: {e}")
            return False
    
    def shutdown(self):
        """Shutdown all batch processors."""
        self._shutdown_flag.set()
        
        for deployment_id in list(self.batch_processors.keys()):
            self.stop_batch_processor(deployment_id)


class OptimizedModelServingManager:
    """Main manager for optimized ML model serving with connection pooling."""
    
    def __init__(self, config: Optional[ConnectionPoolConfig] = None):
        self.config = config or ConnectionPoolConfig()
        self.app_config = get_config()
        
        # Connection pools
        self.db_pool = DatabaseConnectionPool(self.config)
        self.redis_pool = RedisConnectionPool(self.config)
        
        # Model serving components
        self.model_pool = ModelInstancePool(self.config)
        self.batch_processor = BatchInferenceProcessor(self.config, self.model_pool)
        
        # Integration with existing deployment manager
        self.deployment_manager = get_deployment_manager()
        
        # Performance monitoring
        self.performance_metrics = defaultdict(lambda: {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'average_latency': 0.0,
            'cache_hits': 0,
            'cache_misses': 0
        })
        
        self._initialized = False
        self._lock = asyncio.Lock()
    
    async def initialize(self) -> bool:
        """Initialize all connection pools and services."""
        if self._initialized:
            return True
        
        async with self._lock:
            if self._initialized:
                return True
            
            try:
                logger.info("Initializing optimized model serving manager")
                
                # Initialize database pool
                if self.app_config.database.uri:
                    await self.db_pool.initialize(self.app_config.database.uri)
                
                # Initialize Redis pool
                if self.app_config.redis.enabled and self.app_config.redis.url:
                    await self.redis_pool.initialize(self.app_config.redis.url)
                
                # Set event loop policy for better performance
                if UVLOOP_AVAILABLE:
                    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
                    logger.info("Using uvloop for improved async performance")
                
                self._initialized = True
                logger.info("Optimized model serving manager initialized successfully")
                return True
                
            except Exception as e:
                logger.error(f"Failed to initialize optimized serving manager: {e}")
                return False
    
    async def deploy_optimized_model(self, deployment_id: str, model_factory: Callable) -> bool:
        """Deploy a model with optimized serving capabilities."""
        try:
            # Initialize model instance pool
            if not self.model_pool.initialize_model_pool(deployment_id, model_factory):
                return False
            
            # Start batch processing
            if not self.batch_processor.start_batch_processor(deployment_id):
                self.model_pool.cleanup_model_pool(deployment_id)
                return False
            
            logger.info(f"Deployed optimized model serving for {deployment_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to deploy optimized model {deployment_id}: {e}")
            return False
    
    async def predict_async(self, deployment_id: str, audio_data: np.ndarray, 
                          request_id: str, user_id: Optional[str] = None,
                          priority: int = 0, timeout: float = 30.0) -> Dict[str, Any]:
        """Async prediction with optimized serving."""
        start_time = time.time()
        
        try:
            # Check cache first
            cache_key = self._generate_cache_key(audio_data)
            cached_result = await self._get_cached_result(cache_key)
            
            if cached_result:
                self.performance_metrics[deployment_id]['cache_hits'] += 1
                logger.debug(f"Cache hit for request {request_id}")
                return {
                    **cached_result,
                    'request_id': request_id,
                    'cache_hit': True,
                    'latency': time.time() - start_time
                }
            
            self.performance_metrics[deployment_id]['cache_misses'] += 1
            
            # Create async future for result
            result_future = asyncio.Future()
            
            def result_callback(result):
                if not result_future.done():
                    result_future.set_result(result)
            
            # Create inference request
            inference_request = InferenceRequest(
                request_id=request_id,
                audio_data=audio_data,
                callback=result_callback,
                priority=priority,
                timeout=timeout,
                user_id=user_id
            )
            
            # Submit to batch processor
            if not self.batch_processor.submit_request(deployment_id, inference_request):
                return {
                    'error': 'Failed to submit request for processing',
                    'request_id': request_id
                }
            
            # Wait for result with timeout
            try:
                result = await asyncio.wait_for(result_future, timeout=timeout)
                
                # Cache successful result
                if 'error' not in result:
                    await self._cache_result(cache_key, result)
                
                # Update metrics
                latency = time.time() - start_time
                self._update_metrics(deployment_id, latency, 'error' not in result)
                
                result['latency'] = latency
                result['cache_hit'] = False
                
                return result
                
            except asyncio.TimeoutError:
                return {
                    'error': 'Request timeout',
                    'request_id': request_id,
                    'timeout': timeout
                }
            
        except Exception as e:
            logger.error(f"Async prediction failed for {request_id}: {e}")
            latency = time.time() - start_time
            self._update_metrics(deployment_id, latency, False)
            
            return {
                'error': str(e),
                'request_id': request_id,
                'latency': latency
            }
    
    def _generate_cache_key(self, audio_data: np.ndarray) -> str:
        """Generate cache key for audio data."""
        audio_hash = hashlib.sha256(audio_data.tobytes()).hexdigest()
        return f"prediction:optimized:{audio_hash}"
    
    async def _get_cached_result(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get cached prediction result."""
        try:
            if self.redis_pool.pool:
                cached_data = await self.redis_pool.get(cache_key)
                if cached_data:
                    import json
                    return json.loads(cached_data.decode('utf-8'))
            return None
        except Exception as e:
            logger.warning(f"Cache retrieval failed: {e}")
            return None
    
    async def _cache_result(self, cache_key: str, result: Dict[str, Any]) -> bool:
        """Cache prediction result."""
        try:
            if self.redis_pool.pool:
                import json
                cached_data = json.dumps(result).encode('utf-8')
                return await self.redis_pool.set(cache_key, cached_data, ttl=3600)
            return False
        except Exception as e:
            logger.warning(f"Cache storage failed: {e}")
            return False
    
    def _update_metrics(self, deployment_id: str, latency: float, success: bool):
        """Update performance metrics."""
        metrics = self.performance_metrics[deployment_id]
        metrics['total_requests'] += 1
        
        if success:
            metrics['successful_requests'] += 1
        else:
            metrics['failed_requests'] += 1
        
        # Update running average latency
        total_requests = metrics['total_requests']
        metrics['average_latency'] = (
            (metrics['average_latency'] * (total_requests - 1) + latency) / total_requests
        )
    
    async def get_optimization_stats(self) -> Dict[str, Any]:
        """Get comprehensive optimization statistics."""
        stats = {
            'connection_pools': {
                'database': {
                    'initialized': self.db_pool.pool is not None,
                    'min_connections': self.config.db_min_connections,
                    'max_connections': self.config.db_max_connections
                },
                'redis': {
                    'initialized': self.redis_pool.pool is not None,
                    'max_connections': self.config.redis_max_connections
                }
            },
            'model_pools': {},
            'batch_processing': {},
            'performance_metrics': dict(self.performance_metrics),
            'system_resources': {
                'memory_usage_mb': psutil.virtual_memory().used / 1024 / 1024,
                'cpu_percent': psutil.cpu_percent(),
                'available_memory_mb': psutil.virtual_memory().available / 1024 / 1024
            }
        }
        
        # Add model pool stats
        for deployment_id in self.model_pool.model_pools.keys():
            stats['model_pools'][deployment_id] = self.model_pool.get_pool_stats(deployment_id)
        
        # Add batch processing stats
        for deployment_id in self.batch_processor.request_queues.keys():
            stats['batch_processing'][deployment_id] = self.batch_processor.get_batch_stats(deployment_id)
        
        return stats
    
    async def undeploy_optimized_model(self, deployment_id: str) -> bool:
        """Undeploy an optimized model."""
        try:
            # Stop batch processing
            self.batch_processor.stop_batch_processor(deployment_id)
            
            # Clean up model pool
            self.model_pool.cleanup_model_pool(deployment_id)
            
            # Clean up metrics
            if deployment_id in self.performance_metrics:
                del self.performance_metrics[deployment_id]
            
            logger.info(f"Undeployed optimized model {deployment_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to undeploy optimized model {deployment_id}: {e}")
            return False
    
    async def shutdown(self):
        """Shutdown the optimized serving manager."""
        try:
            logger.info("Shutting down optimized model serving manager")
            
            # Stop batch processing
            self.batch_processor.shutdown()
            
            # Clean up all model pools
            for deployment_id in list(self.model_pool.model_pools.keys()):
                self.model_pool.cleanup_model_pool(deployment_id)
            
            # Close connection pools
            await self.db_pool.close()
            await self.redis_pool.close()
            
            self._initialized = False
            logger.info("Optimized model serving manager shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")


# Global optimized serving manager instance
_optimized_serving_manager: Optional[OptimizedModelServingManager] = None


async def get_optimized_serving_manager() -> OptimizedModelServingManager:
    """Get or create global optimized serving manager instance."""
    global _optimized_serving_manager
    
    if _optimized_serving_manager is None:
        _optimized_serving_manager = OptimizedModelServingManager()
        await _optimized_serving_manager.initialize()
    
    return _optimized_serving_manager


# Example usage and testing
if __name__ == "__main__":
    async def main():
        # Example of how to use the optimized serving manager
        serving_manager = await get_optimized_serving_manager()
        
        # Example model factory
        def create_model():
            import torch.nn as nn
            return nn.Sequential(
                nn.Linear(1000, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 10)  # 10 classes
            )
        
        # Deploy optimized model
        deployment_id = "wildlife_detector_optimized"
        success = await serving_manager.deploy_optimized_model(deployment_id, create_model)
        
        if success:
            print(f"Deployed optimized model: {deployment_id}")
            
            # Example prediction
            dummy_audio = np.random.randn(1000)
            result = await serving_manager.predict_async(
                deployment_id, dummy_audio, "test_request_1"
            )
            
            print(f"Prediction result: {result}")
            
            # Get optimization statistics
            stats = await serving_manager.get_optimization_stats()
            print(f"Optimization stats: {stats}")
        
        # Shutdown
        await serving_manager.shutdown()
    
    # Run example
    if UVLOOP_AVAILABLE:
        uvloop.run(main())
    else:
        asyncio.run(main())
