"""Machine learning model versioning and A/B testing system for NightScan."""

import os
import json
import logging
import hashlib
import shutil
import pickle
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
from enum import Enum
import numpy as np
from collections import defaultdict, deque

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import get_config

logger = logging.getLogger(__name__)


class ModelStatus(Enum):
    """Model status enumeration."""
    TRAINING = "training"
    VALIDATION = "validation"
    TESTING = "testing"
    DEPLOYED = "deployed"
    RETIRED = "retired"
    FAILED = "failed"


class ExperimentStatus(Enum):
    """A/B test experiment status."""
    DRAFT = "draft"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


@dataclass
class ModelMetadata:
    """Metadata for a model version."""
    model_id: str
    version: str
    name: str
    description: str
    created_at: datetime
    status: ModelStatus
    framework: str  # 'pytorch', 'tensorflow', etc.
    architecture: str  # 'resnet18', 'efficientnet', etc.
    dataset_version: str
    training_config: Dict[str, Any]
    metrics: Dict[str, float]
    file_size_bytes: int
    checksum: str
    parent_version: Optional[str] = None
    tags: List[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            **asdict(self),
            'status': self.status.value,
            'created_at': self.created_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelMetadata':
        data['status'] = ModelStatus(data['status'])
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        return cls(**data)


@dataclass
class ExperimentConfig:
    """Configuration for A/B testing experiment."""
    experiment_id: str
    name: str
    description: str
    model_variants: Dict[str, str]  # variant_name -> model_version
    traffic_allocation: Dict[str, float]  # variant_name -> percentage
    success_metrics: List[str]
    minimum_sample_size: int
    maximum_duration_days: int
    confidence_level: float = 0.95
    power: float = 0.8
    
    def validate(self) -> bool:
        """Validate experiment configuration."""
        # Check traffic allocation sums to 100%
        total_traffic = sum(self.traffic_allocation.values())
        if abs(total_traffic - 100.0) > 0.01:
            return False
        
        # Check all variants have models
        for variant in self.traffic_allocation:
            if variant not in self.model_variants:
                return False
        
        return True


@dataclass
class ExperimentResult:
    """Result data for an experiment variant."""
    variant_name: str
    model_version: str
    sample_size: int
    metrics: Dict[str, float]
    confidence_intervals: Dict[str, Tuple[float, float]]
    statistical_significance: Dict[str, bool]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ModelRegistry:
    """Registry for managing model versions."""
    
    def __init__(self, registry_path: str = "models/registry"):
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(parents=True, exist_ok=True)
        
        self.metadata_file = self.registry_path / "metadata.json"
        self.models_dir = self.registry_path / "models"
        self.models_dir.mkdir(exist_ok=True)
        
        # Load existing metadata
        self.metadata = self._load_metadata()
    
    def register_model(self, model: nn.Module, metadata: ModelMetadata, 
                      model_file: Optional[Path] = None) -> bool:
        """Register a new model version."""
        try:
            # Save model to registry
            model_path = self.models_dir / f"{metadata.model_id}_{metadata.version}.pth"
            
            if model_file:
                # Copy existing model file
                shutil.copy2(model_file, model_path)
            else:
                # Save model state dict
                torch.save(model.state_dict(), model_path)
            
            # Calculate checksum
            metadata.checksum = self._calculate_checksum(model_path)
            metadata.file_size_bytes = model_path.stat().st_size
            
            # Store metadata
            model_key = f"{metadata.model_id}:{metadata.version}"
            self.metadata[model_key] = metadata
            self._save_metadata()
            
            logger.info(f"Registered model {model_key}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register model {metadata.model_id}:{metadata.version}: {e}")
            return False
    
    def load_model(self, model_id: str, version: str) -> Optional[Dict[str, Any]]:
        """Load a model version."""
        model_key = f"{model_id}:{version}"
        
        if model_key not in self.metadata:
            logger.error(f"Model {model_key} not found in registry")
            return None
        
        try:
            metadata = self.metadata[model_key]
            model_path = self.models_dir / f"{model_id}_{version}.pth"
            
            if not model_path.exists():
                logger.error(f"Model file not found: {model_path}")
                return None
            
            # Verify checksum
            if self._calculate_checksum(model_path) != metadata.checksum:
                logger.error(f"Model checksum mismatch for {model_key}")
                return None
            
            # Load model state dict
            state_dict = torch.load(model_path, map_location='cpu')
            
            return {
                'state_dict': state_dict,
                'metadata': metadata,
                'model_path': model_path
            }
            
        except Exception as e:
            logger.error(f"Failed to load model {model_key}: {e}")
            return None
    
    def list_models(self, model_id: Optional[str] = None, 
                   status: Optional[ModelStatus] = None) -> List[ModelMetadata]:
        """List models with optional filtering."""
        models = []
        
        for model_key, metadata in self.metadata.items():
            # Filter by model_id
            if model_id and metadata.model_id != model_id:
                continue
            
            # Filter by status
            if status and metadata.status != status:
                continue
            
            models.append(metadata)
        
        # Sort by creation date (newest first)
        models.sort(key=lambda m: m.created_at, reverse=True)
        return models
    
    def get_latest_version(self, model_id: str, 
                          status: Optional[ModelStatus] = None) -> Optional[ModelMetadata]:
        """Get the latest version of a model."""
        models = self.list_models(model_id=model_id, status=status)
        return models[0] if models else None
    
    def update_status(self, model_id: str, version: str, status: ModelStatus) -> bool:
        """Update model status."""
        model_key = f"{model_id}:{version}"
        
        if model_key not in self.metadata:
            return False
        
        self.metadata[model_key].status = status
        self._save_metadata()
        
        logger.info(f"Updated {model_key} status to {status.value}")
        return True
    
    def retire_model(self, model_id: str, version: str) -> bool:
        """Retire a model version."""
        return self.update_status(model_id, version, ModelStatus.RETIRED)
    
    def delete_model(self, model_id: str, version: str) -> bool:
        """Delete a model version (use with caution)."""
        model_key = f"{model_id}:{version}"
        
        if model_key not in self.metadata:
            return False
        
        try:
            # Remove model file
            model_path = self.models_dir / f"{model_id}_{version}.pth"
            if model_path.exists():
                model_path.unlink()
            
            # Remove metadata
            del self.metadata[model_key]
            self._save_metadata()
            
            logger.info(f"Deleted model {model_key}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete model {model_key}: {e}")
            return False
    
    def compare_models(self, model1: Tuple[str, str], model2: Tuple[str, str]) -> Dict[str, Any]:
        """Compare two model versions."""
        model1_key = f"{model1[0]}:{model1[1]}"
        model2_key = f"{model2[0]}:{model2[1]}"
        
        if model1_key not in self.metadata or model2_key not in self.metadata:
            return {'error': 'One or both models not found'}
        
        meta1 = self.metadata[model1_key]
        meta2 = self.metadata[model2_key]
        
        comparison = {
            'model1': {
                'id': model1_key,
                'metrics': meta1.metrics,
                'created_at': meta1.created_at.isoformat(),
                'file_size': meta1.file_size_bytes
            },
            'model2': {
                'id': model2_key,
                'metrics': meta2.metrics,
                'created_at': meta2.created_at.isoformat(),
                'file_size': meta2.file_size_bytes
            },
            'metric_differences': {}
        }
        
        # Calculate metric differences
        for metric in meta1.metrics:
            if metric in meta2.metrics:
                diff = meta2.metrics[metric] - meta1.metrics[metric]
                improvement = ((meta2.metrics[metric] - meta1.metrics[metric]) / meta1.metrics[metric]) * 100
                comparison['metric_differences'][metric] = {
                    'absolute_difference': diff,
                    'percentage_improvement': improvement
                }
        
        return comparison
    
    def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate SHA256 checksum of file."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()
    
    def _load_metadata(self) -> Dict[str, ModelMetadata]:
        """Load model metadata from file."""
        if not self.metadata_file.exists():
            return {}
        
        try:
            with open(self.metadata_file, 'r') as f:
                data = json.load(f)
            
            metadata = {}
            for model_key, model_data in data.items():
                metadata[model_key] = ModelMetadata.from_dict(model_data)
            
            return metadata
            
        except Exception as e:
            logger.error(f"Failed to load model metadata: {e}")
            return {}
    
    def _save_metadata(self) -> None:
        """Save model metadata to file."""
        try:
            data = {}
            for model_key, metadata in self.metadata.items():
                data[model_key] = metadata.to_dict()
            
            with open(self.metadata_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save model metadata: {e}")


class ABTestManager:
    """Manager for A/B testing experiments."""
    
    def __init__(self, model_registry: ModelRegistry, experiments_path: str = "experiments"):
        self.model_registry = model_registry
        self.experiments_path = Path(experiments_path)
        self.experiments_path.mkdir(parents=True, exist_ok=True)
        
        self.experiments_file = self.experiments_path / "experiments.json"
        self.results_dir = self.experiments_path / "results"
        self.results_dir.mkdir(exist_ok=True)
        
        # Load existing experiments
        self.experiments = self._load_experiments()
        
        # Traffic routing state
        self.traffic_router = TrafficRouter()
        
        # Result collectors
        self.result_collectors = {}
    
    def create_experiment(self, config: ExperimentConfig) -> bool:
        """Create a new A/B testing experiment."""
        if not config.validate():
            logger.error(f"Invalid experiment configuration: {config.experiment_id}")
            return False
        
        if config.experiment_id in self.experiments:
            logger.error(f"Experiment {config.experiment_id} already exists")
            return False
        
        # Validate that all model versions exist
        for variant, model_version in config.model_variants.items():
            model_id, version = model_version.split(':')
            if not self.model_registry.load_model(model_id, version):
                logger.error(f"Model {model_version} not found for variant {variant}")
                return False
        
        try:
            # Create experiment record
            experiment = {
                'config': asdict(config),
                'status': ExperimentStatus.DRAFT.value,
                'created_at': datetime.now().isoformat(),
                'started_at': None,
                'ended_at': None,
                'results': {}
            }
            
            self.experiments[config.experiment_id] = experiment
            self._save_experiments()
            
            # Initialize result collector
            self.result_collectors[config.experiment_id] = ResultCollector(
                config.experiment_id, 
                list(config.model_variants.keys()),
                config.success_metrics
            )
            
            logger.info(f"Created experiment {config.experiment_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create experiment {config.experiment_id}: {e}")
            return False
    
    def start_experiment(self, experiment_id: str) -> bool:
        """Start an A/B testing experiment."""
        if experiment_id not in self.experiments:
            logger.error(f"Experiment {experiment_id} not found")
            return False
        
        experiment = self.experiments[experiment_id]
        
        if experiment['status'] != ExperimentStatus.DRAFT.value:
            logger.error(f"Experiment {experiment_id} is not in draft status")
            return False
        
        try:
            # Update experiment status
            experiment['status'] = ExperimentStatus.RUNNING.value
            experiment['started_at'] = datetime.now().isoformat()
            
            # Configure traffic routing
            config = ExperimentConfig(**experiment['config'])
            self.traffic_router.configure_experiment(experiment_id, config)
            
            self._save_experiments()
            
            logger.info(f"Started experiment {experiment_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start experiment {experiment_id}: {e}")
            return False
    
    def stop_experiment(self, experiment_id: str, reason: str = "Manual stop") -> bool:
        """Stop an A/B testing experiment."""
        if experiment_id not in self.experiments:
            return False
        
        experiment = self.experiments[experiment_id]
        
        if experiment['status'] != ExperimentStatus.RUNNING.value:
            return False
        
        try:
            # Update experiment status
            experiment['status'] = ExperimentStatus.COMPLETED.value
            experiment['ended_at'] = datetime.now().isoformat()
            experiment['stop_reason'] = reason
            
            # Remove from traffic routing
            self.traffic_router.remove_experiment(experiment_id)
            
            # Generate final results
            if experiment_id in self.result_collectors:
                final_results = self.result_collectors[experiment_id].get_final_results()
                experiment['results'] = final_results
                
                # Save detailed results
                results_file = self.results_dir / f"{experiment_id}_results.json"
                with open(results_file, 'w') as f:
                    json.dump(final_results, f, indent=2)
            
            self._save_experiments()
            
            logger.info(f"Stopped experiment {experiment_id}: {reason}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to stop experiment {experiment_id}: {e}")
            return False
    
    def get_model_for_request(self, request_id: str, experiment_id: Optional[str] = None) -> Optional[str]:
        """Get model version for a specific request (traffic routing)."""
        if experiment_id and experiment_id in self.experiments:
            experiment = self.experiments[experiment_id]
            if experiment['status'] == ExperimentStatus.RUNNING.value:
                return self.traffic_router.route_request(request_id, experiment_id)
        
        # Default to latest deployed model
        latest_model = self.model_registry.get_latest_version("wildlife_detector", ModelStatus.DEPLOYED)
        return f"{latest_model.model_id}:{latest_model.version}" if latest_model else None
    
    def record_result(self, experiment_id: str, request_id: str, variant: str, 
                     metrics: Dict[str, float]) -> None:
        """Record experiment result."""
        if (experiment_id in self.result_collectors and 
            self.experiments[experiment_id]['status'] == ExperimentStatus.RUNNING.value):
            
            self.result_collectors[experiment_id].record_result(request_id, variant, metrics)
    
    def get_experiment_status(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """Get current status of an experiment."""
        if experiment_id not in self.experiments:
            return None
        
        experiment = self.experiments[experiment_id]
        status = {
            'experiment_id': experiment_id,
            'status': experiment['status'],
            'config': experiment['config'],
            'created_at': experiment['created_at'],
            'started_at': experiment.get('started_at'),
            'ended_at': experiment.get('ended_at')
        }
        
        # Add real-time results if running
        if (experiment['status'] == ExperimentStatus.RUNNING.value and 
            experiment_id in self.result_collectors):
            
            collector = self.result_collectors[experiment_id]
            status['current_results'] = collector.get_current_results()
            status['sample_sizes'] = collector.get_sample_sizes()
            status['duration'] = collector.get_experiment_duration()
        
        return status
    
    def list_experiments(self, status: Optional[ExperimentStatus] = None) -> List[Dict[str, Any]]:
        """List experiments with optional status filtering."""
        experiments = []
        
        for exp_id, experiment in self.experiments.items():
            if status and experiment['status'] != status.value:
                continue
            
            experiments.append({
                'experiment_id': exp_id,
                'name': experiment['config']['name'],
                'status': experiment['status'],
                'created_at': experiment['created_at'],
                'variants': list(experiment['config']['model_variants'].keys())
            })
        
        # Sort by creation date (newest first)
        experiments.sort(key=lambda e: e['created_at'], reverse=True)
        return experiments
    
    def analyze_experiment(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """Perform statistical analysis of experiment results."""
        if experiment_id not in self.experiments:
            return None
        
        experiment = self.experiments[experiment_id]
        
        if experiment_id not in self.result_collectors:
            return None
        
        collector = self.result_collectors[experiment_id]
        analysis = collector.perform_statistical_analysis()
        
        # Add experiment metadata
        analysis['experiment_metadata'] = {
            'experiment_id': experiment_id,
            'name': experiment['config']['name'],
            'status': experiment['status'],
            'variants': experiment['config']['model_variants'],
            'traffic_allocation': experiment['config']['traffic_allocation']
        }
        
        return analysis
    
    def _load_experiments(self) -> Dict[str, Dict[str, Any]]:
        """Load experiments from file."""
        if not self.experiments_file.exists():
            return {}
        
        try:
            with open(self.experiments_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load experiments: {e}")
            return {}
    
    def _save_experiments(self) -> None:
        """Save experiments to file."""
        try:
            with open(self.experiments_file, 'w') as f:
                json.dump(self.experiments, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save experiments: {e}")


class TrafficRouter:
    """Routes traffic between model variants in A/B tests."""
    
    def __init__(self):
        self.experiment_configs = {}
        self.request_assignments = {}  # request_id -> variant
    
    def configure_experiment(self, experiment_id: str, config: ExperimentConfig) -> None:
        """Configure traffic routing for an experiment."""
        self.experiment_configs[experiment_id] = config
    
    def remove_experiment(self, experiment_id: str) -> None:
        """Remove experiment from routing."""
        if experiment_id in self.experiment_configs:
            del self.experiment_configs[experiment_id]
    
    def route_request(self, request_id: str, experiment_id: str) -> Optional[str]:
        """Route a request to a model variant."""
        if experiment_id not in self.experiment_configs:
            return None
        
        config = self.experiment_configs[experiment_id]
        
        # Check if request already has assignment
        assignment_key = f"{experiment_id}:{request_id}"
        if assignment_key in self.request_assignments:
            variant = self.request_assignments[assignment_key]
            return config.model_variants.get(variant)
        
        # Assign to variant based on traffic allocation
        variant = self._assign_variant(request_id, config.traffic_allocation)
        self.request_assignments[assignment_key] = variant
        
        return config.model_variants.get(variant)
    
    def _assign_variant(self, request_id: str, traffic_allocation: Dict[str, float]) -> str:
        """Assign request to variant using consistent hashing."""
        # Use hash of request_id for consistent assignment
        hash_value = int(hashlib.md5(request_id.encode()).hexdigest(), 16)
        percentage = (hash_value % 10000) / 100.0  # 0-99.99%
        
        cumulative = 0.0
        for variant, allocation in traffic_allocation.items():
            cumulative += allocation
            if percentage < cumulative:
                return variant
        
        # Fallback to first variant
        return list(traffic_allocation.keys())[0]


class ResultCollector:
    """Collects and analyzes A/B test results."""
    
    def __init__(self, experiment_id: str, variants: List[str], metrics: List[str]):
        self.experiment_id = experiment_id
        self.variants = variants
        self.metrics = metrics
        self.started_at = datetime.now()
        
        # Storage for results
        self.results = {variant: deque() for variant in variants}
        self.total_samples = {variant: 0 for variant in variants}
        
    def record_result(self, request_id: str, variant: str, metrics: Dict[str, float]) -> None:
        """Record a result for the experiment."""
        if variant not in self.variants:
            return
        
        result = {
            'request_id': request_id,
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics
        }
        
        self.results[variant].append(result)
        self.total_samples[variant] += 1
        
        # Keep only recent results to manage memory
        if len(self.results[variant]) > 10000:
            self.results[variant].popleft()
    
    def get_current_results(self) -> Dict[str, Any]:
        """Get current aggregated results."""
        results = {}
        
        for variant in self.variants:
            variant_results = list(self.results[variant])
            
            if not variant_results:
                results[variant] = {
                    'sample_size': 0,
                    'metrics': {}
                }
                continue
            
            # Calculate average metrics
            avg_metrics = {}
            for metric in self.metrics:
                values = [r['metrics'].get(metric, 0) for r in variant_results if metric in r['metrics']]
                avg_metrics[metric] = np.mean(values) if values else 0
            
            results[variant] = {
                'sample_size': len(variant_results),
                'metrics': avg_metrics
            }
        
        return results
    
    def get_sample_sizes(self) -> Dict[str, int]:
        """Get current sample sizes for each variant."""
        return self.total_samples.copy()
    
    def get_experiment_duration(self) -> float:
        """Get experiment duration in hours."""
        return (datetime.now() - self.started_at).total_seconds() / 3600
    
    def perform_statistical_analysis(self) -> Dict[str, Any]:
        """Perform statistical analysis of results."""
        analysis = {
            'experiment_id': self.experiment_id,
            'analysis_timestamp': datetime.now().isoformat(),
            'duration_hours': self.get_experiment_duration(),
            'variants': {},
            'comparisons': {},
            'recommendations': []
        }
        
        # Analyze each variant
        for variant in self.variants:
            variant_results = list(self.results[variant])
            
            if not variant_results:
                analysis['variants'][variant] = {
                    'sample_size': 0,
                    'metrics': {},
                    'confidence_intervals': {}
                }
                continue
            
            # Calculate statistics for each metric
            variant_analysis = {
                'sample_size': len(variant_results),
                'metrics': {},
                'confidence_intervals': {}
            }
            
            for metric in self.metrics:
                values = [r['metrics'].get(metric, 0) for r in variant_results if metric in r['metrics']]
                
                if values:
                    mean_val = np.mean(values)
                    std_val = np.std(values)
                    
                    # Calculate 95% confidence interval
                    n = len(values)
                    margin_of_error = 1.96 * (std_val / np.sqrt(n))
                    
                    variant_analysis['metrics'][metric] = {
                        'mean': mean_val,
                        'std': std_val,
                        'min': np.min(values),
                        'max': np.max(values)
                    }
                    
                    variant_analysis['confidence_intervals'][metric] = {
                        'lower': mean_val - margin_of_error,
                        'upper': mean_val + margin_of_error
                    }
            
            analysis['variants'][variant] = variant_analysis
        
        # Perform pairwise comparisons
        variants_list = list(self.variants)
        for i in range(len(variants_list)):
            for j in range(i + 1, len(variants_list)):
                variant_a = variants_list[i]
                variant_b = variants_list[j]
                
                comparison = self._compare_variants(variant_a, variant_b)
                comparison_key = f"{variant_a}_vs_{variant_b}"
                analysis['comparisons'][comparison_key] = comparison
        
        # Generate recommendations
        analysis['recommendations'] = self._generate_recommendations(analysis)
        
        return analysis
    
    def _compare_variants(self, variant_a: str, variant_b: str) -> Dict[str, Any]:
        """Compare two variants statistically."""
        results_a = list(self.results[variant_a])
        results_b = list(self.results[variant_b])
        
        comparison = {
            'variant_a': variant_a,
            'variant_b': variant_b,
            'sample_sizes': {
                variant_a: len(results_a),
                variant_b: len(results_b)
            },
            'metric_comparisons': {}
        }
        
        for metric in self.metrics:
            values_a = [r['metrics'].get(metric, 0) for r in results_a if metric in r['metrics']]
            values_b = [r['metrics'].get(metric, 0) for r in results_b if metric in r['metrics']]
            
            if values_a and values_b:
                mean_a = np.mean(values_a)
                mean_b = np.mean(values_b)
                
                # Simple statistical comparison (in production, use proper t-test)
                improvement = ((mean_b - mean_a) / mean_a) * 100 if mean_a != 0 else 0
                
                comparison['metric_comparisons'][metric] = {
                    'mean_a': mean_a,
                    'mean_b': mean_b,
                    'improvement_percent': improvement,
                    'winner': variant_b if mean_b > mean_a else variant_a
                }
        
        return comparison
    
    def _generate_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on analysis."""
        recommendations = []
        
        # Check sample sizes
        total_samples = sum(v['sample_size'] for v in analysis['variants'].values())
        if total_samples < 1000:
            recommendations.append("Consider collecting more data before making decisions (< 1000 total samples)")
        
        # Check for clear winners
        for comparison_key, comparison in analysis['comparisons'].items():
            for metric, metric_comparison in comparison['metric_comparisons'].items():
                improvement = metric_comparison['improvement_percent']
                if abs(improvement) > 5:  # 5% improvement threshold
                    direction = "better" if improvement > 0 else "worse"
                    recommendations.append(
                        f"{comparison['variant_b']} performs {abs(improvement):.1f}% {direction} "
                        f"than {comparison['variant_a']} on {metric}"
                    )
        
        # Duration recommendations
        duration = analysis['duration_hours']
        if duration < 24:
            recommendations.append("Consider running experiment for at least 24 hours to account for daily patterns")
        elif duration > 168:  # 1 week
            recommendations.append("Experiment has been running for over a week. Consider concluding if sufficient data collected")
        
        return recommendations
    
    def get_final_results(self) -> Dict[str, Any]:
        """Get final results for completed experiment."""
        return {
            'experiment_id': self.experiment_id,
            'final_sample_sizes': self.get_sample_sizes(),
            'duration_hours': self.get_experiment_duration(),
            'final_results': self.get_current_results(),
            'statistical_analysis': self.perform_statistical_analysis()
        }


# Global instances
_model_registry: Optional[ModelRegistry] = None
_ab_test_manager: Optional[ABTestManager] = None


def get_model_registry() -> ModelRegistry:
    """Get or create global model registry instance."""
    global _model_registry
    
    if _model_registry is None:
        _model_registry = ModelRegistry()
    
    return _model_registry


def get_ab_test_manager() -> ABTestManager:
    """Get or create global A/B test manager instance."""
    global _ab_test_manager
    
    if _ab_test_manager is None:
        model_registry = get_model_registry()
        _ab_test_manager = ABTestManager(model_registry)
    
    return _ab_test_manager


# CLI interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="NightScan Model Versioning and A/B Testing")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Model registry commands
    registry_parser = subparsers.add_parser('registry', help='Model registry operations')
    registry_subs = registry_parser.add_subparsers(dest='registry_action')
    
    list_parser = registry_subs.add_parser('list', help='List models')
    list_parser.add_argument('--model-id', help='Filter by model ID')
    list_parser.add_argument('--status', help='Filter by status')
    
    compare_parser = registry_subs.add_parser('compare', help='Compare models')
    compare_parser.add_argument('model1', help='First model (format: model_id:version)')
    compare_parser.add_argument('model2', help='Second model (format: model_id:version)')
    
    # A/B testing commands
    experiment_parser = subparsers.add_parser('experiment', help='A/B testing operations')
    experiment_subs = experiment_parser.add_subparsers(dest='experiment_action')
    
    list_exp_parser = experiment_subs.add_parser('list', help='List experiments')
    list_exp_parser.add_argument('--status', help='Filter by status')
    
    status_parser = experiment_subs.add_parser('status', help='Get experiment status')
    status_parser.add_argument('experiment_id', help='Experiment ID')
    
    start_parser = experiment_subs.add_parser('start', help='Start experiment')
    start_parser.add_argument('experiment_id', help='Experiment ID')
    
    stop_parser = experiment_subs.add_parser('stop', help='Stop experiment')
    stop_parser.add_argument('experiment_id', help='Experiment ID')
    stop_parser.add_argument('--reason', default='Manual stop', help='Stop reason')
    
    analyze_parser = experiment_subs.add_parser('analyze', help='Analyze experiment')
    analyze_parser.add_argument('experiment_id', help='Experiment ID')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        exit(1)
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    if args.command == 'registry':
        registry = get_model_registry()
        
        if args.registry_action == 'list':
            status_filter = ModelStatus(args.status) if args.status else None
            models = registry.list_models(model_id=args.model_id, status=status_filter)
            
            for model in models:
                print(f"{model.model_id}:{model.version} - {model.status.value} - {model.created_at}")
        
        elif args.registry_action == 'compare':
            model1_parts = args.model1.split(':')
            model2_parts = args.model2.split(':')
            
            comparison = registry.compare_models(
                (model1_parts[0], model1_parts[1]),
                (model2_parts[0], model2_parts[1])
            )
            
            print(json.dumps(comparison, indent=2))
    
    elif args.command == 'experiment':
        ab_manager = get_ab_test_manager()
        
        if args.experiment_action == 'list':
            status_filter = ExperimentStatus(args.status) if args.status else None
            experiments = ab_manager.list_experiments(status=status_filter)
            
            for exp in experiments:
                print(f"{exp['experiment_id']} - {exp['name']} - {exp['status']}")
        
        elif args.experiment_action == 'status':
            status = ab_manager.get_experiment_status(args.experiment_id)
            if status:
                print(json.dumps(status, indent=2))
            else:
                print(f"Experiment {args.experiment_id} not found")
        
        elif args.experiment_action == 'start':
            success = ab_manager.start_experiment(args.experiment_id)
            print(f"Start {'successful' if success else 'failed'}")
        
        elif args.experiment_action == 'stop':
            success = ab_manager.stop_experiment(args.experiment_id, args.reason)
            print(f"Stop {'successful' if success else 'failed'}")
        
        elif args.experiment_action == 'analyze':
            analysis = ab_manager.analyze_experiment(args.experiment_id)
            if analysis:
                print(json.dumps(analysis, indent=2))
            else:
                print(f"Analysis not available for experiment {args.experiment_id}")