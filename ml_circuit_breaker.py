"""
ML Circuit Breaker for NightScan

Provides specialized circuit breaker protection for machine learning operations
including model loading, inference, and prediction processing with intelligent
fallbacks and resource monitoring.

Features:
- Model instance health monitoring
- GPU/CPU resource tracking
- Prediction queue management
- Model fallback strategies (lightweight models)
- Memory usage monitoring
- Inference timeout protection
- Model warming and preloading
"""

import time
import logging
import threading
import psutil
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import tempfile
import pickle
import json

try:
    import torch
    import torch.cuda
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

from circuit_breaker import (
    CircuitBreaker, CircuitBreakerConfig, CircuitBreakerOpenException,
    register_circuit_breaker
)
from exceptions import (
    PredictionError, ModelNotAvailableError, PredictionFailedError,
    DataPreprocessingError
)

logger = logging.getLogger(__name__)


@dataclass
class MLCircuitBreakerConfig(CircuitBreakerConfig):
    """Extended configuration for ML circuit breaker."""
    # Model configuration
    model_path: Optional[str] = None           # Path to primary model
    fallback_model_path: Optional[str] = None  # Path to lightweight fallback model
    model_device: str = "auto"                 # "cpu", "cuda", or "auto"
    max_model_memory_mb: int = 2048            # Max memory usage for model
    
    # Inference configuration
    inference_timeout: float = 30.0            # Timeout for single inference
    batch_inference_timeout: float = 120.0     # Timeout for batch inference
    max_batch_size: int = 32                   # Maximum batch size
    
    # Resource monitoring
    max_cpu_usage: float = 90.0                # Max CPU usage percentage
    max_memory_usage: float = 85.0             # Max memory usage percentage
    max_gpu_memory_usage: float = 90.0         # Max GPU memory usage percentage
    
    # Queue management
    max_queue_size: int = 100                  # Maximum prediction queue size
    queue_timeout: float = 300.0               # Timeout for queued predictions
    
    # Health checking
    health_check_interval: float = 60.0        # Seconds between health checks
    warmup_on_start: bool = True               # Warm up model on initialization
    
    # Fallback behavior
    enable_lightweight_fallback: bool = True   # Use lightweight model as fallback
    enable_cached_fallback: bool = True        # Use cached predictions as fallback
    cache_predictions: bool = True             # Cache successful predictions
    
    def __post_init__(self):
        super().__post_init__()
        
        # ML-specific exception handling
        exceptions = [PredictionError, ModelNotAvailableError, PredictionFailedError, DataPreprocessingError]
        
        if TORCH_AVAILABLE:
            exceptions.extend([torch.cuda.OutOfMemoryError, RuntimeError])
        
        self.expected_exception = tuple(exceptions)


class ModelMetrics:
    """Metrics tracking for ML models."""
    
    def __init__(self):
        self.prediction_count = 0
        self.successful_predictions = 0
        self.failed_predictions = 0
        self.cached_predictions = 0
        self.fallback_predictions = 0
        
        # Performance metrics
        self.total_inference_time = 0.0
        self.total_preprocessing_time = 0.0
        self.min_inference_time = float('inf')
        self.max_inference_time = 0.0
        
        # Resource usage
        self.peak_memory_usage = 0
        self.peak_gpu_memory = 0
        self.peak_cpu_usage = 0.0
        
        # Model state
        self.model_loads = 0
        self.model_load_failures = 0
        self.last_successful_prediction = None
        self.last_failed_prediction = None
        
        # Queue metrics
        self.queue_overflows = 0
        self.queue_timeouts = 0
        
        self.lock = threading.RLock()
    
    def record_prediction(self, inference_time: float, preprocessing_time: float = 0.0,
                         success: bool = True, cached: bool = False, fallback: bool = False):
        """Record metrics for a prediction."""
        with self.lock:
            self.prediction_count += 1
            
            if success:
                self.successful_predictions += 1
                self.last_successful_prediction = time.time()
            else:
                self.failed_predictions += 1
                self.last_failed_prediction = time.time()
            
            if cached:
                self.cached_predictions += 1
            
            if fallback:
                self.fallback_predictions += 1
            
            # Performance tracking
            if inference_time > 0:
                self.total_inference_time += inference_time
                self.min_inference_time = min(self.min_inference_time, inference_time)
                self.max_inference_time = max(self.max_inference_time, inference_time)
            
            if preprocessing_time > 0:
                self.total_preprocessing_time += preprocessing_time
    
    def record_resource_usage(self, memory_mb: int, gpu_memory_mb: int = 0, cpu_percent: float = 0.0):
        """Record resource usage metrics."""
        with self.lock:
            self.peak_memory_usage = max(self.peak_memory_usage, memory_mb)
            self.peak_gpu_memory = max(self.peak_gpu_memory, gpu_memory_mb)
            self.peak_cpu_usage = max(self.peak_cpu_usage, cpu_percent)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics."""
        with self.lock:
            avg_inference_time = (self.total_inference_time / self.successful_predictions 
                                if self.successful_predictions > 0 else 0)
            
            success_rate = (self.successful_predictions / self.prediction_count 
                          if self.prediction_count > 0 else 0)
            
            cache_hit_rate = (self.cached_predictions / self.prediction_count 
                            if self.prediction_count > 0 else 0)
            
            return {
                'prediction_count': self.prediction_count,
                'successful_predictions': self.successful_predictions,
                'failed_predictions': self.failed_predictions,
                'cached_predictions': self.cached_predictions,
                'fallback_predictions': self.fallback_predictions,
                'success_rate': success_rate,
                'cache_hit_rate': cache_hit_rate,
                'avg_inference_time': avg_inference_time,
                'min_inference_time': self.min_inference_time if self.min_inference_time != float('inf') else 0,
                'max_inference_time': self.max_inference_time,
                'total_preprocessing_time': self.total_preprocessing_time,
                'peak_memory_usage_mb': self.peak_memory_usage,
                'peak_gpu_memory_mb': self.peak_gpu_memory,
                'peak_cpu_usage': self.peak_cpu_usage,
                'model_loads': self.model_loads,
                'model_load_failures': self.model_load_failures,
                'queue_overflows': self.queue_overflows,
                'queue_timeouts': self.queue_timeouts,
                'last_successful_prediction': self.last_successful_prediction,
                'last_failed_prediction': self.last_failed_prediction
            }


class PredictionQueue:
    """Thread-safe prediction queue with timeout handling."""
    
    def __init__(self, max_size: int = 100, timeout: float = 300.0):
        self.max_size = max_size
        self.timeout = timeout
        self.queue = []
        self.lock = threading.RLock()
        self.condition = threading.Condition(self.lock)
    
    def add_prediction(self, prediction_data: Dict[str, Any]) -> bool:
        """Add prediction to queue."""
        with self.condition:
            if len(self.queue) >= self.max_size:
                return False
            
            prediction_data['queued_at'] = time.time()
            self.queue.append(prediction_data)
            self.condition.notify()
            return True
    
    def get_prediction(self, timeout: float = None) -> Optional[Dict[str, Any]]:
        """Get next prediction from queue."""
        timeout = timeout or self.timeout
        with self.condition:
            end_time = time.time() + timeout
            
            while not self.queue:
                remaining = end_time - time.time()
                if remaining <= 0:
                    return None
                self.condition.wait(remaining)
            
            return self.queue.pop(0)
    
    def size(self) -> int:
        """Get current queue size."""
        with self.lock:
            return len(self.queue)
    
    def clear(self):
        """Clear all items from queue."""
        with self.lock:
            self.queue.clear()


class MLCircuitBreaker(CircuitBreaker):
    """
    Specialized circuit breaker for ML operations.
    
    Provides protection for model loading, inference, and prediction
    processing with intelligent fallbacks and resource monitoring.
    """
    
    def __init__(self, config: MLCircuitBreakerConfig):
        super().__init__(config)
        self.ml_config = config
        self.ml_metrics = ModelMetrics()
        
        # Model state
        self.primary_model = None
        self.fallback_model = None
        self.model_device = self._determine_device()
        self.model_loaded = False
        
        # Prediction queue
        self.prediction_queue = PredictionQueue(
            max_size=config.max_queue_size,
            timeout=config.queue_timeout
        )
        
        # Resource monitoring
        self._resource_monitor_thread = None
        self._monitoring_active = False
        
        # Prediction cache
        self.prediction_cache = {}
        self.cache_lock = threading.RLock()
        
        # Initialize model and monitoring
        self._load_models()
        self._start_resource_monitoring()
        
        if config.warmup_on_start:
            self._warmup_model()
        
        logger.info(f"ML circuit breaker '{config.name}' initialized on {self.model_device}")
    
    def _determine_device(self) -> str:
        """Determine the best device for model execution."""
        if self.ml_config.model_device != "auto":
            return self.ml_config.model_device
        
        if TORCH_AVAILABLE and torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"
    
    def _load_models(self):
        """Load primary and fallback models."""
        try:
            if self.ml_config.model_path and Path(self.ml_config.model_path).exists():
                self.primary_model = self._load_single_model(self.ml_config.model_path)
                self.model_loaded = True
                self.ml_metrics.model_loads += 1
                logger.info(f"Primary model loaded from {self.ml_config.model_path}")
            else:
                logger.warning(f"Primary model not found at {self.ml_config.model_path}")
        except Exception as e:
            logger.error(f"Failed to load primary model: {e}")
            self.ml_metrics.model_load_failures += 1
        
        # Load fallback model
        if self.ml_config.enable_lightweight_fallback and self.ml_config.fallback_model_path:
            try:
                if Path(self.ml_config.fallback_model_path).exists():
                    self.fallback_model = self._load_single_model(self.ml_config.fallback_model_path)
                    logger.info(f"Fallback model loaded from {self.ml_config.fallback_model_path}")
                else:
                    logger.warning(f"Fallback model not found at {self.ml_config.fallback_model_path}")
            except Exception as e:
                logger.error(f"Failed to load fallback model: {e}")
    
    def _load_single_model(self, model_path: str) -> Any:
        """Load a single model from path."""
        if not TORCH_AVAILABLE:
            raise ModelNotAvailableError("PyTorch not available for model loading")
        
        try:
            # Load model based on file extension
            if model_path.endswith('.pth') or model_path.endswith('.pt'):
                model = torch.load(model_path, map_location=self.model_device)
                if hasattr(model, 'eval'):
                    model.eval()
                return model
            elif model_path.endswith('.pkl'):
                with open(model_path, 'rb') as f:
                    return pickle.load(f)
            else:
                raise ModelNotAvailableError(f"Unsupported model format: {model_path}")
        
        except Exception as e:
            raise ModelNotAvailableError(f"Failed to load model from {model_path}: {str(e)}")
    
    def predict(self, input_data: Any, timeout: float = None, use_cache: bool = True) -> Any:
        """
        Make prediction with circuit breaker protection.
        
        Args:
            input_data: Input data for prediction
            timeout: Custom timeout for this prediction
            use_cache: Whether to use cached predictions
            
        Returns:
            Prediction result
            
        Raises:
            CircuitBreakerOpenException: When circuit is open
            PredictionError: When prediction fails
        """
        timeout = timeout or self.ml_config.inference_timeout
        
        # Check cache first
        if use_cache and self.ml_config.cache_predictions:
            cached_result = self._get_cached_prediction(input_data)
            if cached_result is not None:
                self.ml_metrics.record_prediction(0, 0, success=True, cached=True)
                return cached_result
        
        def make_prediction():
            return self._execute_prediction(input_data, timeout)
        
        try:
            result = self.call(make_prediction)
            
            # Cache successful prediction
            if self.ml_config.cache_predictions:
                self._cache_prediction(input_data, result)
            
            return result
            
        except CircuitBreakerOpenException as e:
            # Try fallback mechanisms
            return self._try_prediction_fallback(input_data, e)
    
    async def predict_async(self, input_data: Any, timeout: float = None, use_cache: bool = True) -> Any:
        """Make async prediction with circuit breaker protection."""
        timeout = timeout or self.ml_config.inference_timeout
        
        # Check cache first
        if use_cache and self.ml_config.cache_predictions:
            cached_result = self._get_cached_prediction(input_data)
            if cached_result is not None:
                self.ml_metrics.record_prediction(0, 0, success=True, cached=True)
                return cached_result
        
        async def make_async_prediction():
            # Run prediction in thread pool for CPU-bound operations
            import asyncio
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self._execute_prediction, input_data, timeout)
        
        try:
            result = await self.call_async(make_async_prediction)
            
            # Cache successful prediction
            if self.ml_config.cache_predictions:
                self._cache_prediction(input_data, result)
            
            return result
            
        except CircuitBreakerOpenException as e:
            # Try fallback mechanisms
            return self._try_prediction_fallback(input_data, e)
    
    def predict_batch(self, input_batch: List[Any], timeout: float = None) -> List[Any]:
        """Make batch prediction with circuit breaker protection."""
        timeout = timeout or self.ml_config.batch_inference_timeout
        
        if len(input_batch) > self.ml_config.max_batch_size:
            # Split into smaller batches
            results = []
            for i in range(0, len(input_batch), self.ml_config.max_batch_size):
                batch = input_batch[i:i + self.ml_config.max_batch_size]
                batch_results = self.predict_batch(batch, timeout)
                results.extend(batch_results)
            return results
        
        def make_batch_prediction():
            return self._execute_batch_prediction(input_batch, timeout)
        
        try:
            return self.call(make_batch_prediction)
        except CircuitBreakerOpenException as e:
            # Try individual predictions as fallback
            logger.warning(f"Batch prediction failed, trying individual predictions: {e}")
            results = []
            for input_data in input_batch:
                try:
                    result = self._try_prediction_fallback(input_data, e)
                    results.append(result)
                except Exception as individual_error:
                    logger.error(f"Individual prediction fallback failed: {individual_error}")
                    results.append(None)
            return results
    
    def _execute_prediction(self, input_data: Any, timeout: float) -> Any:
        """Execute single prediction with resource monitoring."""
        if not self.model_loaded or not self.primary_model:
            raise ModelNotAvailableError("Primary model not loaded")
        
        start_time = time.time()
        preprocessing_start = start_time
        
        try:
            # Monitor resource usage before prediction
            memory_before = self._get_memory_usage()
            
            # Preprocess input data
            processed_input = self._preprocess_input(input_data)
            preprocessing_time = time.time() - preprocessing_start
            
            # Check resource limits
            self._check_resource_limits()
            
            # Execute prediction with timeout
            inference_start = time.time()
            result = self._run_inference(processed_input, timeout)
            inference_time = time.time() - inference_start
            
            # Monitor resource usage after prediction
            memory_after = self._get_memory_usage()
            memory_used = memory_after - memory_before
            
            # Record metrics
            self.ml_metrics.record_prediction(inference_time, preprocessing_time, success=True)
            self.ml_metrics.record_resource_usage(memory_used, self._get_gpu_memory_usage())
            
            total_time = time.time() - start_time
            logger.debug(f"Prediction completed in {total_time:.2f}s (preprocessing: {preprocessing_time:.2f}s, inference: {inference_time:.2f}s)")
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.ml_metrics.record_prediction(execution_time, 0, success=False)
            
            if "memory" in str(e).lower() or "cuda" in str(e).lower():
                raise PredictionError(f"Resource error during prediction: {str(e)}")
            elif "timeout" in str(e).lower():
                raise PredictionError(f"Prediction timeout after {execution_time:.2f}s: {str(e)}")
            else:
                raise PredictionFailedError(f"Prediction failed: {str(e)}")
    
    def _execute_batch_prediction(self, input_batch: List[Any], timeout: float) -> List[Any]:
        """Execute batch prediction with resource monitoring."""
        if not self.model_loaded or not self.primary_model:
            raise ModelNotAvailableError("Primary model not loaded for batch prediction")
        
        start_time = time.time()
        
        try:
            # Preprocess all inputs
            processed_batch = [self._preprocess_input(data) for data in input_batch]
            
            # Check resource limits
            self._check_resource_limits()
            
            # Execute batch inference
            results = self._run_batch_inference(processed_batch, timeout)
            
            # Record metrics
            execution_time = time.time() - start_time
            for _ in input_batch:
                self.ml_metrics.record_prediction(execution_time / len(input_batch), 0, success=True)
            
            logger.debug(f"Batch prediction of {len(input_batch)} items completed in {execution_time:.2f}s")
            return results
            
        except Exception as e:
            execution_time = time.time() - start_time
            for _ in input_batch:
                self.ml_metrics.record_prediction(execution_time / len(input_batch), 0, success=False)
            raise PredictionFailedError(f"Batch prediction failed: {str(e)}")
    
    def _preprocess_input(self, input_data: Any) -> Any:
        """Preprocess input data for model."""
        try:
            # Basic preprocessing - can be overridden in subclasses
            if TORCH_AVAILABLE and isinstance(input_data, (list, tuple)):
                return torch.tensor(input_data, dtype=torch.float32, device=self.model_device)
            elif NUMPY_AVAILABLE and hasattr(input_data, 'shape'):
                return input_data
            else:
                return input_data
        except Exception as e:
            raise DataPreprocessingError(f"Failed to preprocess input: {str(e)}")
    
    def _run_inference(self, processed_input: Any, timeout: float) -> Any:
        """Run model inference with timeout."""
        if TORCH_AVAILABLE and hasattr(self.primary_model, '__call__'):
            with torch.no_grad():
                if torch.cuda.is_available() and self.model_device == "cuda":
                    torch.cuda.empty_cache()  # Clear GPU cache
                
                result = self.primary_model(processed_input)
                
                # Convert result to CPU/numpy if needed
                if hasattr(result, 'cpu'):
                    result = result.cpu()
                if hasattr(result, 'numpy'):
                    result = result.numpy()
                
                return result
        else:
            # Generic model call
            return self.primary_model(processed_input)
    
    def _run_batch_inference(self, processed_batch: List[Any], timeout: float) -> List[Any]:
        """Run batch model inference with timeout."""
        # Stack inputs if they're tensors
        if TORCH_AVAILABLE and all(hasattr(item, 'shape') for item in processed_batch):
            batch_input = torch.stack(processed_batch)
            with torch.no_grad():
                batch_result = self.primary_model(batch_input)
                # Split results back into list
                return [result for result in batch_result]
        else:
            # Process individually
            return [self._run_inference(item, timeout / len(processed_batch)) for item in processed_batch]
    
    def _check_resource_limits(self):
        """Check if resource usage is within limits."""
        # Check memory usage
        memory_percent = psutil.virtual_memory().percent
        if memory_percent > self.ml_config.max_memory_usage:
            raise PredictionError(f"Memory usage too high: {memory_percent:.1f}%")
        
        # Check CPU usage
        cpu_percent = psutil.cpu_percent(interval=0.1)
        if cpu_percent > self.ml_config.max_cpu_usage:
            raise PredictionError(f"CPU usage too high: {cpu_percent:.1f}%")
        
        # Check GPU memory if available
        if TORCH_AVAILABLE and torch.cuda.is_available():
            gpu_memory_percent = (torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()) * 100
            if gpu_memory_percent > self.ml_config.max_gpu_memory_usage:
                raise PredictionError(f"GPU memory usage too high: {gpu_memory_percent:.1f}%")
    
    def _get_memory_usage(self) -> int:
        """Get current memory usage in MB."""
        return psutil.Process().memory_info().rss // (1024 * 1024)
    
    def _get_gpu_memory_usage(self) -> int:
        """Get current GPU memory usage in MB."""
        if TORCH_AVAILABLE and torch.cuda.is_available():
            return torch.cuda.memory_allocated() // (1024 * 1024)
        return 0
    
    def _get_cached_prediction(self, input_data: Any) -> Any:
        """Get cached prediction result."""
        if not self.ml_config.cache_predictions:
            return None
        
        cache_key = self._generate_cache_key(input_data)
        with self.cache_lock:
            cached_item = self.prediction_cache.get(cache_key)
            if cached_item:
                # Check if cache entry is still valid (simple TTL)
                if time.time() - cached_item['timestamp'] < 3600:  # 1 hour TTL
                    return cached_item['result']
                else:
                    # Remove expired entry
                    del self.prediction_cache[cache_key]
        
        return None
    
    def _cache_prediction(self, input_data: Any, result: Any):
        """Cache prediction result."""
        if not self.ml_config.cache_predictions:
            return
        
        cache_key = self._generate_cache_key(input_data)
        with self.cache_lock:
            # Limit cache size
            if len(self.prediction_cache) > 1000:
                # Remove oldest entries
                oldest_keys = sorted(self.prediction_cache.keys(), 
                                   key=lambda k: self.prediction_cache[k]['timestamp'])[:100]
                for key in oldest_keys:
                    del self.prediction_cache[key]
            
            self.prediction_cache[cache_key] = {
                'result': result,
                'timestamp': time.time()
            }
    
    def _generate_cache_key(self, input_data: Any) -> str:
        """Generate cache key for input data."""
        # Simple hash-based key generation
        if hasattr(input_data, 'tobytes'):
            return str(hash(input_data.tobytes()))
        else:
            return str(hash(str(input_data)))
    
    def _try_prediction_fallback(self, input_data: Any, circuit_error: CircuitBreakerOpenException) -> Any:
        """Try fallback mechanisms when circuit is open."""
        logger.info(f"Primary model circuit open, trying fallbacks for prediction")
        
        # Try cached prediction first
        if self.ml_config.enable_cached_fallback:
            cached_result = self._get_cached_prediction(input_data)
            if cached_result is not None:
                self.ml_metrics.record_prediction(0, 0, success=True, cached=True, fallback=True)
                return cached_result
        
        # Try fallback model
        if self.ml_config.enable_lightweight_fallback and self.fallback_model:
            try:
                result = self._run_fallback_prediction(input_data)
                self.ml_metrics.record_prediction(0, 0, success=True, fallback=True)
                return result
            except Exception as e:
                logger.error(f"Fallback model prediction failed: {e}")
        
        # No fallback available
        raise PredictionError(f"No fallback available for prediction: {circuit_error}")
    
    def _run_fallback_prediction(self, input_data: Any) -> Any:
        """Run prediction using fallback model."""
        if not self.fallback_model:
            raise ModelNotAvailableError("Fallback model not available")
        
        try:
            processed_input = self._preprocess_input(input_data)
            
            if TORCH_AVAILABLE and hasattr(self.fallback_model, '__call__'):
                with torch.no_grad():
                    result = self.fallback_model(processed_input)
                    if hasattr(result, 'cpu'):
                        result = result.cpu()
                    if hasattr(result, 'numpy'):
                        result = result.numpy()
                    return result
            else:
                return self.fallback_model(processed_input)
                
        except Exception as e:
            raise PredictionFailedError(f"Fallback model prediction failed: {str(e)}")
    
    def _warmup_model(self):
        """Warm up model with dummy data."""
        if not self.model_loaded or not self.primary_model:
            return
        
        try:
            logger.info("Warming up model...")
            # Create dummy input based on expected input shape
            if TORCH_AVAILABLE:
                dummy_input = torch.randn(1, 10, device=self.model_device)  # Adjust shape as needed
                with torch.no_grad():
                    _ = self.primary_model(dummy_input)
            
            logger.info("Model warmup completed")
        except Exception as e:
            logger.warning(f"Model warmup failed: {e}")
    
    def _start_resource_monitoring(self):
        """Start background resource monitoring."""
        if self._monitoring_active:
            return
        
        self._monitoring_active = True
        
        def monitor_resources():
            while self._monitoring_active:
                try:
                    # Monitor memory and CPU
                    memory_mb = self._get_memory_usage()
                    cpu_percent = psutil.cpu_percent(interval=1.0)
                    gpu_memory_mb = self._get_gpu_memory_usage()
                    
                    self.ml_metrics.record_resource_usage(memory_mb, gpu_memory_mb, cpu_percent)
                    
                    # Log warnings for high usage
                    if memory_mb > self.ml_config.max_model_memory_mb:
                        logger.warning(f"High memory usage: {memory_mb}MB")
                    
                    if cpu_percent > self.ml_config.max_cpu_usage:
                        logger.warning(f"High CPU usage: {cpu_percent:.1f}%")
                    
                    time.sleep(30)  # Check every 30 seconds
                    
                except Exception as e:
                    logger.error(f"Error in resource monitoring: {e}")
                    time.sleep(60)  # Wait longer on error
        
        self._resource_monitor_thread = threading.Thread(target=monitor_resources, daemon=True)
        self._resource_monitor_thread.start()
        
        logger.info(f"Started resource monitoring for ML circuit '{self.config.name}'")
    
    def _stop_resource_monitoring(self):
        """Stop background resource monitoring."""
        self._monitoring_active = False
        if self._resource_monitor_thread and self._resource_monitor_thread.is_alive():
            self._resource_monitor_thread = None
    
    def get_ml_metrics(self) -> Dict[str, Any]:
        """Get ML-specific metrics."""
        base_metrics = self.get_metrics()
        ml_metrics = self.ml_metrics.get_metrics()
        
        additional_metrics = {
            'model_loaded': self.model_loaded,
            'model_device': self.model_device,
            'primary_model_available': self.primary_model is not None,
            'fallback_model_available': self.fallback_model is not None,
            'queue_size': self.prediction_queue.size(),
            'cache_size': len(self.prediction_cache),
            'current_memory_usage_mb': self._get_memory_usage(),
            'current_gpu_memory_mb': self._get_gpu_memory_usage(),
            'current_cpu_percent': psutil.cpu_percent()
        }
        
        return {**base_metrics, **ml_metrics, **additional_metrics}
    
    def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive ML health check."""
        health_status = {
            'model_loaded': self.model_loaded,
            'circuit_state': self.get_state().value,
            'primary_model_healthy': False,
            'fallback_model_healthy': False,
            'resource_usage_ok': True
        }
        
        # Test primary model
        if self.primary_model:
            try:
                # Simple inference test
                if TORCH_AVAILABLE:
                    test_input = torch.randn(1, 10, device=self.model_device)
                    with torch.no_grad():
                        _ = self.primary_model(test_input)
                health_status['primary_model_healthy'] = True
            except Exception as e:
                health_status['primary_model_error'] = str(e)
        
        # Test fallback model
        if self.fallback_model:
            try:
                if TORCH_AVAILABLE:
                    test_input = torch.randn(1, 10, device=self.model_device)
                    with torch.no_grad():
                        _ = self.fallback_model(test_input)
                health_status['fallback_model_healthy'] = True
            except Exception as e:
                health_status['fallback_model_error'] = str(e)
        
        # Check resource usage
        memory_percent = psutil.virtual_memory().percent
        cpu_percent = psutil.cpu_percent()
        
        if (memory_percent > self.ml_config.max_memory_usage or 
            cpu_percent > self.ml_config.max_cpu_usage):
            health_status['resource_usage_ok'] = False
        
        health_status['memory_usage_percent'] = memory_percent
        health_status['cpu_usage_percent'] = cpu_percent
        
        if TORCH_AVAILABLE and torch.cuda.is_available():
            gpu_memory_used = torch.cuda.memory_allocated()
            gpu_memory_total = torch.cuda.max_memory_allocated()
            health_status['gpu_memory_usage_percent'] = (gpu_memory_used / gpu_memory_total) * 100
        
        return health_status
    
    def cleanup(self):
        """Cleanup ML circuit breaker resources."""
        # Stop monitoring
        self._stop_resource_monitoring()
        self._stop_health_check()
        
        # Clear cache
        with self.cache_lock:
            self.prediction_cache.clear()
        
        # Clear queue
        self.prediction_queue.clear()
        
        # Clear GPU memory if available
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info(f"ML circuit breaker '{self.config.name}' cleaned up")


# Convenience functions for ML circuit breaker setup

def create_ml_circuit_breaker(name: str, **kwargs) -> MLCircuitBreaker:
    """
    Create and register an ML circuit breaker.
    
    Args:
        name: Circuit breaker name
        **kwargs: Additional configuration parameters
        
    Returns:
        Configured MLCircuitBreaker instance
    """
    config = MLCircuitBreakerConfig(name=name, **kwargs)
    circuit_breaker = MLCircuitBreaker(config)
    register_circuit_breaker(circuit_breaker)
    return circuit_breaker


def ml_circuit_breaker(name: str, **kwargs):
    """
    Decorator for ML operations with circuit breaker protection.
    
    Usage:
        audio_model_cb = ml_circuit_breaker(
            name="audio_model",
            model_path="/path/to/model.pth",
            fallback_model_path="/path/to/lightweight_model.pth"
        )
        
        @audio_model_cb
        def predict_audio_species(audio_data):
            return model.predict(audio_data)
    """
    circuit_breaker = create_ml_circuit_breaker(name, **kwargs)
    return circuit_breaker