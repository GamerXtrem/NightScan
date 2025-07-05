"""Model deployment and serving system for NightScan."""

import os
import json
import logging
import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from pathlib import Path
import hashlib
import threading
from contextlib import contextmanager

import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict, deque
from torch.utils.data import DataLoader

from model_versioning import get_model_registry, get_ab_test_manager, ModelStatus
from config import get_config

logger = logging.getLogger(__name__)


@dataclass
class DeploymentConfig:
    """Configuration for model deployment."""
    model_id: str
    version: str
    deployment_id: str
    environment: str  # 'staging', 'production'
    resources: Dict[str, Any]  # CPU, memory, GPU requirements
    scaling: Dict[str, Any]  # Auto-scaling configuration
    health_check: Dict[str, Any]  # Health check configuration
    traffic_percentage: float = 100.0
    rollback_on_error: bool = True
    canary_deployment: bool = False


@dataclass
class ModelPerformanceMetrics:
    """Performance metrics for deployed models."""
    model_id: str
    version: str
    deployment_id: str
    timestamp: datetime
    requests_per_second: float
    average_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    error_rate: float
    throughput: float
    memory_usage_mb: float
    cpu_usage_percent: float
    gpu_usage_percent: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            **asdict(self),
            'timestamp': self.timestamp.isoformat()
        }


class ModelLoader:
    """Handles loading and management of model instances."""
    
    def __init__(self):
        self.loaded_models = {}  # deployment_id -> model instance
        self.model_metadata = {}  # deployment_id -> metadata
        self.model_locks = {}  # deployment_id -> threading.Lock
        self.model_registry = get_model_registry()
    
    def load_model(self, deployment_config: DeploymentConfig) -> bool:
        """Load a model for deployment."""
        deployment_id = deployment_config.deployment_id
        
        try:
            # Load model from registry
            model_data = self.model_registry.load_model(
                deployment_config.model_id, 
                deployment_config.version
            )
            
            if not model_data:
                logger.error(f"Failed to load model {deployment_config.model_id}:{deployment_config.version}")
                return False
            
            # Create model instance based on metadata
            model_instance = self._create_model_instance(model_data)
            
            if not model_instance:
                logger.error(f"Failed to create model instance for {deployment_id}")
                return False
            
            # Store loaded model
            self.loaded_models[deployment_id] = model_instance
            self.model_metadata[deployment_id] = {
                'config': deployment_config,
                'model_data': model_data,
                'loaded_at': datetime.now(),
                'last_used': datetime.now()
            }
            self.model_locks[deployment_id] = threading.Lock()
            
            logger.info(f"Loaded model {deployment_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model {deployment_id}: {e}")
            return False
    
    def unload_model(self, deployment_id: str) -> bool:
        """Unload a model from memory."""
        try:
            if deployment_id in self.loaded_models:
                # Clean up model
                del self.loaded_models[deployment_id]
                del self.model_metadata[deployment_id]
                del self.model_locks[deployment_id]
                
                # Force garbage collection
                import gc
                gc.collect()
                
                # Clear CUDA cache if using GPU
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                logger.info(f"Unloaded model {deployment_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to unload model {deployment_id}: {e}")
            return False
    
    def get_model(self, deployment_id: str) -> Optional[nn.Module]:
        """Get a loaded model instance."""
        if deployment_id in self.loaded_models:
            # Update last used timestamp
            self.model_metadata[deployment_id]['last_used'] = datetime.now()
            return self.loaded_models[deployment_id]
        
        return None
    
    def get_model_lock(self, deployment_id: str) -> Optional[threading.Lock]:
        """Get the lock for a model (for thread-safe inference)."""
        return self.model_locks.get(deployment_id)
    
    def list_loaded_models(self) -> List[Dict[str, Any]]:
        """List all currently loaded models."""
        models = []
        
        for deployment_id, metadata in self.model_metadata.items():
            models.append({
                'deployment_id': deployment_id,
                'model_id': metadata['config'].model_id,
                'version': metadata['config'].version,
                'loaded_at': metadata['loaded_at'].isoformat(),
                'last_used': metadata['last_used'].isoformat()
            })
        
        return models
    
    def cleanup_unused_models(self, max_idle_minutes: int = 30) -> int:
        """Clean up models that haven't been used recently."""
        cleaned_count = 0
        cutoff_time = datetime.now() - timedelta(minutes=max_idle_minutes)
        
        deployment_ids_to_remove = []
        
        for deployment_id, metadata in self.model_metadata.items():
            if metadata['last_used'] < cutoff_time:
                deployment_ids_to_remove.append(deployment_id)
        
        for deployment_id in deployment_ids_to_remove:
            if self.unload_model(deployment_id):
                cleaned_count += 1
        
        if cleaned_count > 0:
            logger.info(f"Cleaned up {cleaned_count} unused models")
        
        return cleaned_count
    
    def _create_model_instance(self, model_data: Dict[str, Any]) -> Optional[nn.Module]:
        """Create model instance from loaded data."""
        try:
            metadata = model_data['metadata']
            state_dict = model_data['state_dict']
            
            # Create model architecture based on metadata
            architecture = metadata.architecture
            
            if architecture == 'resnet18':
                # Create ResNet18 model for wildlife detection
                from torchvision.models import resnet18
                model = resnet18(pretrained=False)
                
                # Modify for wildlife classification
                num_classes = len(metadata.training_config.get('labels', []))
                model.fc = nn.Linear(model.fc.in_features, num_classes)
                
            elif architecture == 'efficientnet_b0':
                # Create EfficientNet model
                from torchvision.models import efficientnet_b0
                model = efficientnet_b0(pretrained=False)
                
                num_classes = len(metadata.training_config.get('labels', []))
                model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
                
            else:
                logger.error(f"Unsupported architecture: {architecture}")
                return None
            
            # Load state dict
            model.load_state_dict(state_dict)
            
            # Set to evaluation mode
            model.eval()
            
            # Move to appropriate device
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = model.to(device)
            
            logger.info(f"Created {architecture} model with {num_classes} classes on {device}")
            return model
            
        except Exception as e:
            logger.error(f"Failed to create model instance: {e}")
            return None


class InferenceService:
    """Service for running model inference."""
    
    def __init__(self, model_loader: ModelLoader):
        self.model_loader = model_loader
        self.ab_test_manager = get_ab_test_manager()
        self.performance_tracker = PerformanceTracker()
        
        # Inference configuration
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = 1  # Single inference by default
        
    async def predict(self, audio_data: np.ndarray, request_id: str, 
                     experiment_id: Optional[str] = None) -> Dict[str, Any]:
        """Run prediction on audio data."""
        start_time = time.time()
        
        try:
            # Get model for request (considering A/B tests)
            model_version = self.ab_test_manager.get_model_for_request(request_id, experiment_id)
            
            if not model_version:
                return {
                    'error': 'No model available for request',
                    'request_id': request_id
                }
            
            # Find deployment for this model version
            deployment_id = self._get_deployment_for_model(model_version)
            
            if not deployment_id:
                return {
                    'error': f'No deployment found for model {model_version}',
                    'request_id': request_id
                }
            
            # Get model instance
            model = self.model_loader.get_model(deployment_id)
            model_lock = self.model_loader.get_model_lock(deployment_id)
            
            if not model or not model_lock:
                return {
                    'error': f'Model not loaded: {deployment_id}',
                    'request_id': request_id
                }
            
            # Run inference with thread safety
            with model_lock:
                predictions = await self._run_inference(model, audio_data)
            
            inference_time = time.time() - start_time
            
            # Record performance metrics
            self.performance_tracker.record_inference(
                deployment_id, inference_time, True
            )
            
            # Record A/B test result if applicable
            if experiment_id:
                # Extract variant from model version mapping
                variant = self._get_variant_for_model(experiment_id, model_version)
                if variant:
                    metrics = {
                        'inference_time': inference_time,
                        'confidence': max(predictions['probabilities']) if predictions.get('probabilities') else 0
                    }
                    self.ab_test_manager.record_result(experiment_id, request_id, variant, metrics)
            
            result = {
                'request_id': request_id,
                'model_version': model_version,
                'deployment_id': deployment_id,
                'predictions': predictions,
                'inference_time': inference_time,
                'timestamp': datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            inference_time = time.time() - start_time
            
            # Record error metrics
            deployment_id = self._get_deployment_for_model(model_version) if 'model_version' in locals() else 'unknown'
            self.performance_tracker.record_inference(deployment_id, inference_time, False)
            
            logger.error(f"Inference failed for request {request_id}: {e}")
            
            return {
                'error': str(e),
                'request_id': request_id,
                'inference_time': inference_time
            }
    
    async def _run_inference(self, model: nn.Module, audio_data: np.ndarray) -> Dict[str, Any]:
        """Run actual model inference."""
        try:
            # Convert audio data to tensor
            if len(audio_data.shape) == 1:
                # Add batch and channel dimensions
                audio_tensor = torch.from_numpy(audio_data).float().unsqueeze(0).unsqueeze(0)
            else:
                audio_tensor = torch.from_numpy(audio_data).float()
            
            # Move to device
            audio_tensor = audio_tensor.to(self.device)
            
            # Run inference
            with torch.no_grad():
                outputs = model(audio_tensor)
                
                # Apply softmax to get probabilities
                probabilities = torch.softmax(outputs, dim=1)
                
                # Get top predictions
                top_probs, top_indices = torch.topk(probabilities, k=min(5, outputs.size(1)))
                
                # Convert to numpy and lists
                top_probs = top_probs.cpu().numpy().flatten().tolist()
                top_indices = top_indices.cpu().numpy().flatten().tolist()
            
            # Format results
            predictions = {
                'top_predictions': [
                    {
                        'class_index': int(idx),
                        'probability': float(prob)
                    }
                    for idx, prob in zip(top_indices, top_probs)
                ],
                'probabilities': top_probs,
                'predicted_class': int(top_indices[0]),
                'confidence': float(top_probs[0])
            }
            
            return predictions
            
        except Exception as e:
            logger.error(f"Model inference failed: {e}")
            raise
    
    def _get_deployment_for_model(self, model_version: str) -> Optional[str]:
        """Get deployment ID for a model version."""
        # In a real implementation, this would query a deployment registry
        # For now, we'll use a simple mapping
        loaded_models = self.model_loader.list_loaded_models()
        
        for model_info in loaded_models:
            model_key = f"{model_info['model_id']}:{model_info['version']}"
            if model_key == model_version:
                return model_info['deployment_id']
        
        return None
    
    def _get_variant_for_model(self, experiment_id: str, model_version: str) -> Optional[str]:
        """Get variant name for a model version in an experiment."""
        experiment_status = self.ab_test_manager.get_experiment_status(experiment_id)
        
        if not experiment_status:
            return None
        
        model_variants = experiment_status['config']['model_variants']
        
        for variant, version in model_variants.items():
            if version == model_version:
                return variant
        
        return None


class PerformanceTracker:
    """Tracks performance metrics for deployed models."""
    
    def __init__(self):
        self.metrics_history = defaultdict(list)  # deployment_id -> list of metrics
        self.current_metrics = {}  # deployment_id -> current aggregated metrics
        self.lock = threading.Lock()
        
        # Metrics windows
        self.window_size = 100  # Number of recent requests to track
        self.aggregation_interval = 60  # Seconds between aggregations
        
        # Start background aggregation
        self._start_background_aggregation()
    
    def record_inference(self, deployment_id: str, inference_time: float, success: bool) -> None:
        """Record an inference result."""
        with self.lock:
            if deployment_id not in self.metrics_history:
                self.metrics_history[deployment_id] = deque(maxlen=self.window_size)
            
            self.metrics_history[deployment_id].append({
                'timestamp': time.time(),
                'inference_time': inference_time,
                'success': success
            })
    
    def get_current_metrics(self, deployment_id: str) -> Optional[ModelPerformanceMetrics]:
        """Get current performance metrics for a deployment."""
        with self.lock:
            return self.current_metrics.get(deployment_id)
    
    def get_all_metrics(self) -> Dict[str, ModelPerformanceMetrics]:
        """Get current metrics for all deployments."""
        with self.lock:
            return self.current_metrics.copy()
    
    def _aggregate_metrics(self) -> None:
        """Aggregate recent metrics for all deployments."""
        current_time = time.time()
        window_start = current_time - 60  # Last minute
        
        with self.lock:
            for deployment_id, history in self.metrics_history.items():
                # Filter to recent metrics
                recent_metrics = [
                    m for m in history 
                    if m['timestamp'] >= window_start
                ]
                
                if not recent_metrics:
                    continue
                
                # Calculate aggregated metrics
                total_requests = len(recent_metrics)
                successful_requests = sum(1 for m in recent_metrics if m['success'])
                failed_requests = total_requests - successful_requests
                
                inference_times = [m['inference_time'] for m in recent_metrics]
                
                # Calculate latency percentiles
                if inference_times:
                    sorted_times = sorted(inference_times)
                    avg_latency = np.mean(sorted_times) * 1000  # Convert to ms
                    p95_latency = np.percentile(sorted_times, 95) * 1000
                    p99_latency = np.percentile(sorted_times, 99) * 1000
                else:
                    avg_latency = p95_latency = p99_latency = 0
                
                # Calculate rates
                time_window = 60  # seconds
                requests_per_second = total_requests / time_window
                error_rate = (failed_requests / total_requests) * 100 if total_requests > 0 else 0
                throughput = successful_requests / time_window
                
                # Get system metrics (simplified)
                memory_usage = self._get_memory_usage()
                cpu_usage = self._get_cpu_usage()
                gpu_usage = self._get_gpu_usage()
                
                # Create metrics object
                metrics = ModelPerformanceMetrics(
                    model_id="wildlife_detector",  # Simplified
                    version="latest",  # Simplified
                    deployment_id=deployment_id,
                    timestamp=datetime.fromtimestamp(current_time),
                    requests_per_second=requests_per_second,
                    average_latency_ms=avg_latency,
                    p95_latency_ms=p95_latency,
                    p99_latency_ms=p99_latency,
                    error_rate=error_rate,
                    throughput=throughput,
                    memory_usage_mb=memory_usage,
                    cpu_usage_percent=cpu_usage,
                    gpu_usage_percent=gpu_usage
                )
                
                self.current_metrics[deployment_id] = metrics
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            return 0.0
    
    def _get_cpu_usage(self) -> float:
        """Get current CPU usage percentage."""
        try:
            import psutil
            return psutil.cpu_percent(interval=1)
        except ImportError:
            return 0.0
    
    def _get_gpu_usage(self) -> Optional[float]:
        """Get current GPU usage percentage."""
        try:
            if torch.cuda.is_available():
                return torch.cuda.utilization()
            return None
        except:
            return None
    
    def _start_background_aggregation(self) -> None:
        """Start background thread for metrics aggregation."""
        def aggregation_loop():
            while True:
                try:
                    self._aggregate_metrics()
                    time.sleep(self.aggregation_interval)
                except Exception as e:
                    logger.error(f"Metrics aggregation failed: {e}")
                    time.sleep(self.aggregation_interval)
        
        aggregation_thread = threading.Thread(target=aggregation_loop, daemon=True)
        aggregation_thread.start()


class DeploymentManager:
    """Manages model deployments and lifecycle."""
    
    def __init__(self):
        self.model_loader = ModelLoader()
        self.inference_service = InferenceService(self.model_loader)
        self.performance_tracker = self.inference_service.performance_tracker
        
        self.deployments = {}  # deployment_id -> DeploymentConfig
        self.deployment_status = {}  # deployment_id -> status info
        
        self.model_registry = get_model_registry()
        self.ab_test_manager = get_ab_test_manager()
    
    def deploy_model(self, config: DeploymentConfig) -> bool:
        """Deploy a model with the given configuration."""
        try:
            logger.info(f"Starting deployment {config.deployment_id}")
            
            # Validate model exists
            model_data = self.model_registry.load_model(config.model_id, config.version)
            if not model_data:
                logger.error(f"Model {config.model_id}:{config.version} not found")
                return False
            
            # Load model into memory
            if not self.model_loader.load_model(config):
                logger.error(f"Failed to load model for deployment {config.deployment_id}")
                return False
            
            # Store deployment configuration
            self.deployments[config.deployment_id] = config
            self.deployment_status[config.deployment_id] = {
                'status': 'deployed',
                'deployed_at': datetime.now(),
                'health': 'healthy',
                'last_health_check': datetime.now()
            }
            
            # Update model status in registry
            self.model_registry.update_status(config.model_id, config.version, ModelStatus.DEPLOYED)
            
            logger.info(f"Successfully deployed {config.deployment_id}")
            return True
            
        except Exception as e:
            logger.error(f"Deployment failed for {config.deployment_id}: {e}")
            
            # Clean up on failure
            self.model_loader.unload_model(config.deployment_id)
            
            if config.deployment_id in self.deployments:
                del self.deployments[config.deployment_id]
            if config.deployment_id in self.deployment_status:
                del self.deployment_status[config.deployment_id]
            
            return False
    
    def undeploy_model(self, deployment_id: str) -> bool:
        """Undeploy a model."""
        try:
            if deployment_id not in self.deployments:
                logger.warning(f"Deployment {deployment_id} not found")
                return False
            
            # Unload model from memory
            self.model_loader.unload_model(deployment_id)
            
            # Remove from tracking
            config = self.deployments[deployment_id]
            del self.deployments[deployment_id]
            del self.deployment_status[deployment_id]
            
            # Update model status in registry
            self.model_registry.update_status(config.model_id, config.version, ModelStatus.TESTING)
            
            logger.info(f"Successfully undeployed {deployment_id}")
            return True
            
        except Exception as e:
            logger.error(f"Undeployment failed for {deployment_id}: {e}")
            return False
    
    def list_deployments(self) -> List[Dict[str, Any]]:
        """List all current deployments."""
        deployments = []
        
        for deployment_id, config in self.deployments.items():
            status = self.deployment_status.get(deployment_id, {})
            metrics = self.performance_tracker.get_current_metrics(deployment_id)
            
            deployment_info = {
                'deployment_id': deployment_id,
                'model_id': config.model_id,
                'version': config.version,
                'environment': config.environment,
                'status': status.get('status', 'unknown'),
                'health': status.get('health', 'unknown'),
                'deployed_at': status.get('deployed_at', datetime.now()).isoformat(),
                'traffic_percentage': config.traffic_percentage
            }
            
            # Add performance metrics if available
            if metrics:
                deployment_info['performance'] = {
                    'requests_per_second': metrics.requests_per_second,
                    'average_latency_ms': metrics.average_latency_ms,
                    'error_rate': metrics.error_rate
                }
            
            deployments.append(deployment_info)
        
        return deployments
    
    def get_deployment_status(self, deployment_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed status of a deployment."""
        if deployment_id not in self.deployments:
            return None
        
        config = self.deployments[deployment_id]
        status = self.deployment_status[deployment_id]
        metrics = self.performance_tracker.get_current_metrics(deployment_id)
        
        return {
            'deployment_id': deployment_id,
            'config': asdict(config),
            'status': status,
            'performance_metrics': metrics.to_dict() if metrics else None,
            'loaded_models': self.model_loader.list_loaded_models()
        }
    
    def health_check(self, deployment_id: str) -> bool:
        """Perform health check on a deployment."""
        if deployment_id not in self.deployments:
            return False
        
        try:
            # Check if model is loaded
            model = self.model_loader.get_model(deployment_id)
            if not model:
                return False
            
            # Run a simple test inference
            test_input = torch.randn(1, 1, 1000)  # Simple test tensor
            
            with torch.no_grad():
                _ = model(test_input)
            
            # Update health status
            self.deployment_status[deployment_id]['health'] = 'healthy'
            self.deployment_status[deployment_id]['last_health_check'] = datetime.now()
            
            return True
            
        except Exception as e:
            logger.error(f"Health check failed for {deployment_id}: {e}")
            self.deployment_status[deployment_id]['health'] = 'unhealthy'
            self.deployment_status[deployment_id]['last_health_check'] = datetime.now()
            return False
    
    async def run_inference(self, audio_data: np.ndarray, request_id: str, 
                          experiment_id: Optional[str] = None) -> Dict[str, Any]:
        """Run inference using the deployed models."""
        return await self.inference_service.predict(audio_data, request_id, experiment_id)


# Global deployment manager instance
_deployment_manager: Optional[DeploymentManager] = None


def get_deployment_manager() -> DeploymentManager:
    """Get or create global deployment manager instance."""
    global _deployment_manager
    
    if _deployment_manager is None:
        _deployment_manager = DeploymentManager()
    
    return _deployment_manager


# CLI interface
if __name__ == "__main__":
    import argparse
    from collections import defaultdict, deque
    
    parser = argparse.ArgumentParser(description="NightScan Model Deployment")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Deploy command
    deploy_parser = subparsers.add_parser('deploy', help='Deploy a model')
    deploy_parser.add_argument('model_id', help='Model ID')
    deploy_parser.add_argument('version', help='Model version')
    deploy_parser.add_argument('--deployment-id', required=True, help='Deployment ID')
    deploy_parser.add_argument('--environment', default='staging', help='Environment')
    
    # Undeploy command
    undeploy_parser = subparsers.add_parser('undeploy', help='Undeploy a model')
    undeploy_parser.add_argument('deployment_id', help='Deployment ID')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List deployments')
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Get deployment status')
    status_parser.add_argument('deployment_id', help='Deployment ID')
    
    # Health check command
    health_parser = subparsers.add_parser('health', help='Health check')
    health_parser.add_argument('deployment_id', help='Deployment ID')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        exit(1)
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    deployment_manager = get_deployment_manager()
    
    if args.command == 'deploy':
        config = DeploymentConfig(
            model_id=args.model_id,
            version=args.version,
            deployment_id=args.deployment_id,
            environment=args.environment,
            resources={'cpu': 2, 'memory': '4Gi'},
            scaling={'min_replicas': 1, 'max_replicas': 3},
            health_check={'interval': 30, 'timeout': 10}
        )
        
        success = deployment_manager.deploy_model(config)
        print(f"Deployment {'successful' if success else 'failed'}")
    
    elif args.command == 'undeploy':
        success = deployment_manager.undeploy_model(args.deployment_id)
        print(f"Undeployment {'successful' if success else 'failed'}")
    
    elif args.command == 'list':
        deployments = deployment_manager.list_deployments()
        for deployment in deployments:
            print(f"{deployment['deployment_id']} - {deployment['model_id']}:{deployment['version']} - {deployment['status']}")
    
    elif args.command == 'status':
        status = deployment_manager.get_deployment_status(args.deployment_id)
        if status:
            print(json.dumps(status, indent=2, default=str))
        else:
            print(f"Deployment {args.deployment_id} not found")
    
    elif args.command == 'health':
        healthy = deployment_manager.health_check(args.deployment_id)
        print(f"Health check: {'healthy' if healthy else 'unhealthy'}")