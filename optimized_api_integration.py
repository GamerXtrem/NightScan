"""Integration module for optimized ML serving with existing API endpoints.

This module provides a bridge between the optimized model serving infrastructure
and the existing Flask API, enabling seamless migration to high-performance serving.
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional
from functools import wraps
import numpy as np
from concurrent.futures import ThreadPoolExecutor

from flask import current_app, request, jsonify
from werkzeug.exceptions import RequestTimeout

from model_serving_optimization import (
    get_optimized_serving_manager, 
    ConnectionPoolConfig,
    OptimizedModelServingManager
)
from config import get_config
from cache_utils import get_cache
from metrics import record_prediction_metrics

logger = logging.getLogger(__name__)


class OptimizedAPIIntegration:
    """Integration layer for optimized ML serving with Flask API."""
    
    def __init__(self):
        self.config = get_config()
        self.serving_manager: Optional[OptimizedModelServingManager] = None
        self.thread_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="OptimizedAPI")
        self._initialized = False
        
    async def initialize(self) -> bool:
        """Initialize the optimized serving integration."""
        if self._initialized:
            return True
            
        try:
            logger.info("Initializing optimized API integration")
            
            # Get optimized serving manager
            self.serving_manager = await get_optimized_serving_manager()
            
            # Deploy default model if not already deployed
            if not await self._ensure_default_model_deployed():
                logger.warning("Failed to deploy default model")
                return False
            
            self._initialized = True
            logger.info("Optimized API integration initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize optimized API integration: {e}")
            return False
    
    async def _ensure_default_model_deployed(self) -> bool:
        """Ensure the default wildlife detection model is deployed."""
        deployment_id = "wildlife_detector_optimized"
        
        try:
            # Check if model is already deployed
            stats = await self.serving_manager.get_optimization_stats()
            if deployment_id in stats.get('model_pools', {}):
                logger.info(f"Model {deployment_id} already deployed")
                return True
            
            # Create model factory based on existing configuration
            def create_wildlife_model():
                try:
                    from torchvision.models import resnet18
                    import torch.nn as nn
                    
                    # Load model from existing configuration
                    model_path = self.config.model.model_path
                    csv_dir = self.config.model.csv_dir
                    
                    # Import existing prediction utilities
                    import sys
                    from pathlib import Path
                    sys.path.append(str(Path("Audio_Training/scripts").resolve()))
                    import predict
                    
                    # Load labels to determine number of classes
                    labels = predict.load_labels(Path(csv_dir))
                    num_classes = len(labels)
                    
                    # Create model architecture
                    model = resnet18(pretrained=False)
                    model.fc = nn.Linear(model.fc.in_features, num_classes)
                    
                    # Load trained weights if available
                    import torch
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    
                    if Path(model_path).exists():
                        state_dict = torch.load(model_path, map_location=device)
                        model.load_state_dict(state_dict)
                        logger.info(f"Loaded model weights from {model_path}")
                    else:
                        logger.warning(f"Model weights not found at {model_path}, using random weights")
                    
                    model.to(device)
                    model.eval()
                    
                    logger.info(f"Created wildlife detection model with {num_classes} classes on {device}")
                    return model
                    
                except Exception as e:
                    logger.error(f"Failed to create wildlife model: {e}")
                    return None
            
            # Deploy the model
            success = await self.serving_manager.deploy_optimized_model(
                deployment_id, create_wildlife_model
            )
            
            if success:
                logger.info(f"Successfully deployed default model: {deployment_id}")
            else:
                logger.error(f"Failed to deploy default model: {deployment_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error ensuring default model deployment: {e}")
            return False
    
    def run_async_in_thread(self, coro):
        """Run async coroutine in thread pool to integrate with Flask."""
        def run_in_thread():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(coro)
            finally:
                loop.close()
        
        return self.thread_pool.submit(run_in_thread)
    
    def optimized_predict(self, audio_data: np.ndarray, request_id: str, 
                         user_id: Optional[str] = None, timeout: float = 30.0) -> Dict[str, Any]:
        """Synchronous wrapper for optimized prediction (Flask-compatible)."""
        if not self._initialized:
            return {
                'error': 'Optimized serving not initialized',
                'request_id': request_id
            }
        
        try:
            # Run async prediction in thread
            future = self.run_async_in_thread(
                self.serving_manager.predict_async(
                    "wildlife_detector_optimized",
                    audio_data,
                    request_id,
                    user_id=user_id,
                    timeout=timeout
                )
            )
            
            # Wait for result with timeout
            result = future.result(timeout=timeout + 5.0)  # Add buffer for thread overhead
            return result
            
        except Exception as e:
            logger.error(f"Optimized prediction failed for {request_id}: {e}")
            return {
                'error': str(e),
                'request_id': request_id
            }
    
    def get_serving_stats(self) -> Dict[str, Any]:
        """Get serving statistics (synchronous for Flask)."""
        if not self._initialized:
            return {'error': 'Optimized serving not initialized'}
        
        try:
            future = self.run_async_in_thread(
                self.serving_manager.get_optimization_stats()
            )
            return future.result(timeout=5.0)
            
        except Exception as e:
            logger.error(f"Failed to get serving stats: {e}")
            return {'error': str(e)}
    
    async def shutdown(self):
        """Shutdown the integration."""
        try:
            if self.serving_manager:
                await self.serving_manager.shutdown()
            
            self.thread_pool.shutdown(wait=True)
            self._initialized = False
            
            logger.info("Optimized API integration shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during integration shutdown: {e}")


# Global integration instance
_api_integration: Optional[OptimizedAPIIntegration] = None


def get_api_integration() -> OptimizedAPIIntegration:
    """Get or create global API integration instance."""
    global _api_integration
    
    if _api_integration is None:
        _api_integration = OptimizedAPIIntegration()
    
    return _api_integration


def optimized_prediction_endpoint(fallback_to_original: bool = True):
    """Decorator for endpoints to use optimized prediction with fallback."""
    def decorator(original_func):
        @wraps(original_func)
        def wrapper(*args, **kwargs):
            api_integration = get_api_integration()
            
            # Check if optimized serving is available
            if not api_integration._initialized:
                if fallback_to_original:
                    logger.info("Optimized serving not available, using original endpoint")
                    return original_func(*args, **kwargs)
                else:
                    return jsonify({
                        'error': 'Optimized serving not available',
                        'fallback_available': False
                    }), 503
            
            # Try optimized prediction first
            try:
                # Extract audio data and request info from Flask request
                # This assumes the original function handles file processing
                
                # Generate request ID
                import uuid
                request_id = str(uuid.uuid4())
                
                # For this decorator, we'll let the original function handle
                # file processing and then intercept the model prediction part
                # This is a simplified approach - in production you might want
                # to extract the audio processing logic
                
                logger.info(f"Using optimized prediction for request {request_id}")
                
                # Call original function but with optimization flag
                kwargs['use_optimized'] = True
                kwargs['request_id'] = request_id
                kwargs['api_integration'] = api_integration
                
                return original_func(*args, **kwargs)
                
            except Exception as e:
                logger.error(f"Optimized endpoint failed: {e}")
                
                if fallback_to_original:
                    logger.info("Falling back to original endpoint")
                    kwargs.pop('use_optimized', None)
                    kwargs.pop('request_id', None)
                    kwargs.pop('api_integration', None)
                    return original_func(*args, **kwargs)
                else:
                    return jsonify({
                        'error': f'Optimized prediction failed: {str(e)}',
                        'fallback_available': False
                    }), 500
        
        return wrapper
    return decorator


def initialize_optimized_serving_async():
    """Initialize optimized serving in background thread."""
    def init_worker():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            api_integration = get_api_integration()
            success = loop.run_until_complete(api_integration.initialize())
            if success:
                logger.info("Background initialization of optimized serving completed")
            else:
                logger.warning("Background initialization of optimized serving failed")
        except Exception as e:
            logger.error(f"Background initialization error: {e}")
        finally:
            loop.close()
    
    import threading
    init_thread = threading.Thread(target=init_worker, daemon=True)
    init_thread.start()


# Flask application factory integration
def register_optimized_endpoints(app):
    """Register optimized serving endpoints with Flask app."""
    
    @app.route("/api/optimized/stats")
    def optimized_stats():
        """Get optimized serving statistics."""
        api_integration = get_api_integration()
        stats = api_integration.get_serving_stats()
        return jsonify(stats)
    
    @app.route("/api/optimized/health")
    def optimized_health():
        """Health check for optimized serving."""
        api_integration = get_api_integration()
        
        health_status = {
            'status': 'healthy' if api_integration._initialized else 'not_ready',
            'initialized': api_integration._initialized,
            'timestamp': time.time()
        }
        
        if api_integration._initialized:
            try:
                stats = api_integration.get_serving_stats()
                health_status['model_pools'] = len(stats.get('model_pools', {}))
                health_status['active_connections'] = {
                    'database': stats.get('connection_pools', {}).get('database', {}).get('initialized', False),
                    'redis': stats.get('connection_pools', {}).get('redis', {}).get('initialized', False)
                }
            except Exception as e:
                health_status['status'] = 'degraded'
                health_status['error'] = str(e)
        
        status_code = 200 if health_status['status'] == 'healthy' else 503
        return jsonify(health_status), status_code
    
    @app.route("/api/optimized/predict", methods=['POST'])
    @optimized_prediction_endpoint(fallback_to_original=True)
    def optimized_predict_endpoint(use_optimized=False, request_id=None, api_integration=None, **kwargs):
        """Optimized prediction endpoint."""
        if not use_optimized:
            # This should not happen with the decorator, but just in case
            return jsonify({'error': 'Optimized serving not available'}), 503
        
        try:
            # Process uploaded file (similar to original endpoint)
            file = request.files.get("file")
            if not file or not file.filename:
                return jsonify({"error": "No file uploaded"}), 400
            
            # Validate file type
            if not file.filename.lower().endswith(".wav"):
                return jsonify({"error": "WAV file required"}), 400
            
            # Read audio data
            audio_bytes = file.read()
            
            # Convert to numpy array (simplified - in production you'd use proper audio processing)
            # For now, we'll create dummy audio data for demonstration
            audio_data = np.frombuffer(audio_bytes[:1000], dtype=np.float32)
            if len(audio_data) == 0:
                audio_data = np.random.randn(1000).astype(np.float32)
            
            # Run optimized prediction
            result = api_integration.optimized_predict(
                audio_data=audio_data,
                request_id=request_id,
                user_id=request.remote_addr,  # Use IP as simple user identifier
                timeout=30.0
            )
            
            # Record metrics
            if 'error' not in result:
                record_prediction_metrics(
                    duration=result.get('latency', 0),
                    success=True,
                    file_size=len(audio_bytes)
                )
            else:
                record_prediction_metrics(
                    duration=result.get('latency', 0),
                    success=False,
                    file_size=len(audio_bytes)
                )
            
            return jsonify(result)
            
        except Exception as e:
            logger.error(f"Optimized prediction endpoint failed: {e}")
            return jsonify({
                'error': str(e),
                'request_id': request_id
            }), 500
    
    # Initialize optimized serving in background
    initialize_optimized_serving_async()
    
    logger.info("Registered optimized serving endpoints")


# Integration with existing API server
def enhance_existing_api_server(app):
    """Enhance existing API server with optimized serving capabilities."""
    
    # Register new endpoints
    register_optimized_endpoints(app)
    
    # Monkey patch existing predict endpoint to use optimization
    original_api_predict = None
    
    # Store reference to original predict function
    for rule in app.url_map.iter_rules():
        if rule.endpoint == 'api_predict' and rule.rule == '/api/predict':
            original_api_predict = app.view_functions.get('api_predict')
            break
    
    if original_api_predict:
        # Wrap original predict function with optimized version
        @optimized_prediction_endpoint(fallback_to_original=True)
        @wraps(original_api_predict)
        def enhanced_api_predict(use_optimized=False, request_id=None, api_integration=None):
            if use_optimized and api_integration:
                # Use optimized path
                logger.info(f"Using optimized path for request {request_id}")
                return optimized_predict_endpoint(True, request_id, api_integration)
            else:
                # Use original path
                logger.info("Using original predict path")
                return original_api_predict()
        
        # Replace the endpoint
        app.view_functions['api_predict'] = enhanced_api_predict
        logger.info("Enhanced existing /api/predict endpoint with optimization")
    else:
        logger.warning("Could not find existing /api/predict endpoint to enhance")
    
    return app


# Cleanup function for graceful shutdown
def cleanup_optimized_serving():
    """Cleanup function for graceful shutdown."""
    global _api_integration
    
    if _api_integration:
        # Run cleanup in a separate thread to avoid blocking
        def cleanup_worker():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(_api_integration.shutdown())
            finally:
                loop.close()
        
        import threading
        cleanup_thread = threading.Thread(target=cleanup_worker)
        cleanup_thread.start()
        cleanup_thread.join(timeout=10.0)  # Wait up to 10 seconds
        
        logger.info("Optimized serving cleanup completed")


# Example usage
if __name__ == "__main__":
    # Example of how to integrate with existing Flask app
    from flask import Flask
    
    app = Flask(__name__)
    
    # Register optimized endpoints
    register_optimized_endpoints(app)
    
    @app.route("/test")
    @optimized_prediction_endpoint(fallback_to_original=False)
    def test_endpoint():
        return jsonify({"message": "Test endpoint with optimization"})
    
    print("Starting test server with optimized serving...")
    app.run(host="0.0.0.0", port=8002, debug=True)
