# NightScan ML Model Versioning and A/B Testing

This document covers the machine learning model versioning and A/B testing system for NightScan.

## Overview

NightScan includes a comprehensive ML operations system that provides:

- **Model Versioning**: Track, compare, and manage different versions of ML models
- **A/B Testing**: Run controlled experiments to compare model performance
- **Model Deployment**: Deploy and serve models with performance monitoring
- **Traffic Routing**: Intelligently route requests between model variants
- **Performance Tracking**: Monitor model performance in real-time
- **Automated Rollback**: Automatically rollback problematic deployments

## Model Versioning

### Model Registry

The model registry tracks all model versions with comprehensive metadata:

```python
from model_versioning import get_model_registry, ModelMetadata, ModelStatus

# Get registry instance
registry = get_model_registry()

# Register a new model
metadata = ModelMetadata(
    model_id="wildlife_detector",
    version="v2.1.0",
    name="Wildlife Detector v2.1",
    description="Improved accuracy with new training data",
    created_at=datetime.now(),
    status=ModelStatus.TRAINING,
    framework="pytorch",
    architecture="resnet18",
    dataset_version="v1.5",
    training_config={
        "epochs": 100,
        "learning_rate": 0.001,
        "batch_size": 32,
        "labels": ["owl", "hawk", "sparrow", "robin"]
    },
    metrics={
        "accuracy": 0.95,
        "precision": 0.93,
        "recall": 0.92,
        "f1_score": 0.925
    },
    file_size_bytes=0,  # Will be calculated
    checksum="",  # Will be calculated
    tags=["production", "high-accuracy"]
)

# Register model with PyTorch model instance
success = registry.register_model(model, metadata)
```

### Model Lifecycle

Models progress through different statuses:

1. **TRAINING**: Model is being trained
2. **VALIDATION**: Model is being validated
3. **TESTING**: Model is being tested
4. **DEPLOYED**: Model is deployed in production
5. **RETIRED**: Model is no longer active
6. **FAILED**: Model failed validation or deployment

### Model Comparison

Compare different model versions:

```python
# Compare two model versions
comparison = registry.compare_models(
    ("wildlife_detector", "v2.0.0"),
    ("wildlife_detector", "v2.1.0")
)

print(f"Accuracy improvement: {comparison['metric_differences']['accuracy']['percentage_improvement']:.2f}%")
```

### CLI Usage

```bash
# List all models
python model_versioning.py registry list

# Filter by model ID and status
python model_versioning.py registry list --model-id wildlife_detector --status deployed

# Compare models
python model_versioning.py registry compare wildlife_detector:v2.0.0 wildlife_detector:v2.1.0
```

## A/B Testing

### Experiment Configuration

Set up A/B tests to compare model variants:

```python
from model_versioning import get_ab_test_manager, ExperimentConfig

# Get A/B test manager
ab_manager = get_ab_test_manager()

# Create experiment configuration
config = ExperimentConfig(
    experiment_id="accuracy_improvement_test",
    name="Test v2.1 vs v2.0 Accuracy",
    description="Compare new model version against current production",
    model_variants={
        "control": "wildlife_detector:v2.0.0",
        "treatment": "wildlife_detector:v2.1.0"
    },
    traffic_allocation={
        "control": 70.0,    # 70% traffic
        "treatment": 30.0   # 30% traffic
    },
    success_metrics=["accuracy", "confidence", "inference_time"],
    minimum_sample_size=1000,
    maximum_duration_days=14,
    confidence_level=0.95
)

# Create and start experiment
ab_manager.create_experiment(config)
ab_manager.start_experiment("accuracy_improvement_test")
```

### Traffic Routing

The system automatically routes requests between variants:

```python
# Get model for specific request (A/B testing handled automatically)
model_version = ab_manager.get_model_for_request(
    request_id="user_123_request_456",
    experiment_id="accuracy_improvement_test"
)

# Model version will be either "wildlife_detector:v2.0.0" or "wildlife_detector:v2.1.0"
# based on traffic allocation and consistent hashing
```

### Result Collection

Record experiment results:

```python
# Record result for a request
ab_manager.record_result(
    experiment_id="accuracy_improvement_test",
    request_id="user_123_request_456",
    variant="treatment",
    metrics={
        "accuracy": 0.96,
        "confidence": 0.89,
        "inference_time": 0.234
    }
)
```

### Statistical Analysis

Get comprehensive experiment analysis:

```python
# Get experiment analysis
analysis = ab_manager.analyze_experiment("accuracy_improvement_test")

print(f"Experiment status: {analysis['experiment_metadata']['status']}")
print(f"Duration: {analysis['duration_hours']:.1f} hours")

for comparison_key, comparison in analysis['comparisons'].items():
    for metric, result in comparison['metric_comparisons'].items():
        improvement = result['improvement_percent']
        winner = result['winner']
        print(f"{metric}: {winner} wins by {improvement:.2f}%")
```

### CLI Usage

```bash
# List experiments
python model_versioning.py experiment list

# Get experiment status
python model_versioning.py experiment status accuracy_improvement_test

# Start experiment
python model_versioning.py experiment start accuracy_improvement_test

# Stop experiment
python model_versioning.py experiment stop accuracy_improvement_test --reason "Sufficient data collected"

# Analyze results
python model_versioning.py experiment analyze accuracy_improvement_test
```

## Model Deployment

### Deployment Configuration

Deploy models with comprehensive configuration:

```python
from model_deployment import get_deployment_manager, DeploymentConfig

# Get deployment manager
deployment_manager = get_deployment_manager()

# Create deployment configuration
config = DeploymentConfig(
    model_id="wildlife_detector",
    version="v2.1.0",
    deployment_id="wildlife_detector_v2_1_prod",
    environment="production",
    resources={
        "cpu": 2,
        "memory": "4Gi",
        "gpu": 0
    },
    scaling={
        "min_replicas": 2,
        "max_replicas": 10,
        "target_cpu_utilization": 70
    },
    health_check={
        "interval": 30,
        "timeout": 10,
        "failure_threshold": 3
    },
    traffic_percentage=100.0,
    rollback_on_error=True,
    canary_deployment=False
)

# Deploy model
success = deployment_manager.deploy_model(config)
```

### Model Serving

Run inference with deployed models:

```python
import numpy as np

# Create sample audio data
audio_data = np.random.randn(22050)  # 1 second of audio at 22050 Hz

# Run inference
result = await deployment_manager.run_inference(
    audio_data=audio_data,
    request_id="inference_123",
    experiment_id="accuracy_improvement_test"  # Optional A/B test
)

print(f"Predicted class: {result['predictions']['predicted_class']}")
print(f"Confidence: {result['predictions']['confidence']:.3f}")
print(f"Inference time: {result['inference_time']:.3f}s")
```

### Performance Monitoring

Monitor model performance in real-time:

```python
# Get deployment performance metrics
metrics = deployment_manager.performance_tracker.get_current_metrics("wildlife_detector_v2_1_prod")

if metrics:
    print(f"Requests/sec: {metrics.requests_per_second:.2f}")
    print(f"Average latency: {metrics.average_latency_ms:.1f}ms")
    print(f"P95 latency: {metrics.p95_latency_ms:.1f}ms")
    print(f"Error rate: {metrics.error_rate:.2f}%")
    print(f"Memory usage: {metrics.memory_usage_mb:.1f}MB")
    print(f"CPU usage: {metrics.cpu_usage_percent:.1f}%")
```

### CLI Usage

```bash
# Deploy model
python model_deployment.py deploy wildlife_detector v2.1.0 --deployment-id wildlife_detector_v2_1_prod --environment production

# List deployments
python model_deployment.py list

# Get deployment status
python model_deployment.py status wildlife_detector_v2_1_prod

# Health check
python model_deployment.py health wildlife_detector_v2_1_prod

# Undeploy model
python model_deployment.py undeploy wildlife_detector_v2_1_prod
```

## Integration with NightScan API

### Enhanced Prediction API

The prediction API automatically integrates with the ML versioning system:

```python
# In api_v1.py - enhanced prediction endpoint
@api_v1.route('/predict', methods=['POST'])
def predict_audio():
    # ... file validation ...
    
    # Generate request ID for tracking
    request_id = f"req_{int(time.time())}_{random.randint(1000, 9999)}"
    
    # Check for experiment participation
    experiment_id = request.args.get('experiment_id')
    
    # Get deployment manager
    deployment_manager = get_deployment_manager()
    
    # Run inference with A/B testing
    result = await deployment_manager.run_inference(
        audio_data=audio_data,
        request_id=request_id,
        experiment_id=experiment_id
    )
    
    # Return enhanced result with versioning info
    return jsonify({
        "request_id": request_id,
        "model_version": result["model_version"],
        "predictions": result["predictions"],
        "inference_time": result["inference_time"],
        "experiment_info": {
            "experiment_id": experiment_id,
            "variant": result.get("variant")
        } if experiment_id else None
    })
```

### Automatic Result Collection

Results are automatically collected for analysis:

```python
# Results are automatically recorded in the inference service
# No manual intervention required for A/B test data collection

# Analysis can be accessed via API
@api_v1.route('/experiments/<experiment_id>/analysis')
@login_required
def get_experiment_analysis(experiment_id):
    ab_manager = get_ab_test_manager()
    analysis = ab_manager.analyze_experiment(experiment_id)
    
    if analysis:
        return jsonify(analysis)
    else:
        return jsonify({'error': 'Experiment not found'}), 404
```

## Best Practices

### Model Development Workflow

1. **Training**: Train new model with versioned datasets
2. **Validation**: Validate model performance with test data
3. **Registration**: Register model in the model registry
4. **Testing**: Deploy to staging environment for testing
5. **A/B Testing**: Run controlled experiment in production
6. **Deployment**: Full deployment if A/B test is successful
7. **Monitoring**: Continuous monitoring of production performance

### Experiment Design

1. **Clear Hypothesis**: Define what you're testing and expected outcomes
2. **Sufficient Sample Size**: Ensure statistical significance
3. **Representative Traffic**: Use representative user traffic
4. **Duration**: Run experiments long enough to account for variability
5. **Multiple Metrics**: Track both performance and business metrics
6. **Gradual Rollout**: Start with small traffic percentage

### Performance Optimization

1. **Model Caching**: Cache frequently used models in memory
2. **Batch Inference**: Process multiple requests together when possible
3. **GPU Utilization**: Use GPUs for compute-intensive models
4. **Load Balancing**: Distribute load across multiple model instances
5. **Resource Monitoring**: Monitor and adjust resource allocation

## Monitoring and Alerting

### Key Metrics

Monitor these critical metrics:

- **Model Performance**: Accuracy, precision, recall, F1-score
- **Inference Performance**: Latency, throughput, error rate
- **System Performance**: CPU, memory, GPU utilization
- **Business Metrics**: User satisfaction, conversion rates

### Alert Configuration

Set up alerts for:

```python
# Example Prometheus alerts
alerts:
  - alert: ModelInferenceLatencyHigh
    expr: model_inference_latency_p95 > 1000  # 1 second
    for: 5m
    annotations:
      summary: "Model inference latency is high"
      
  - alert: ModelErrorRateHigh
    expr: model_error_rate > 5  # 5%
    for: 2m
    annotations:
      summary: "Model error rate is high"
      
  - alert: ABTestSignificantResult
    expr: ab_test_confidence > 0.95
    annotations:
      summary: "A/B test has reached statistical significance"
```

### Automated Actions

Configure automated responses:

```python
# Example automated rollback
if error_rate > 10:  # 10% error rate
    deployment_manager.rollback_deployment(deployment_id)
    send_alert("Automatic rollback triggered due to high error rate")

# Example automatic experiment stopping
if experiment_duration > max_duration or sample_size > minimum_sample_size:
    if statistical_significance_achieved():
        ab_manager.stop_experiment(experiment_id, "Sufficient data collected")
```

## Troubleshooting

### Common Issues

1. **Model Loading Failures**
   ```bash
   # Check model registry
   python model_versioning.py registry list --model-id your_model_id
   
   # Verify model file exists
   ls -la models/registry/models/
   
   # Check model metadata
   cat models/registry/metadata.json
   ```

2. **A/B Test Issues**
   ```bash
   # Check experiment status
   python model_versioning.py experiment status your_experiment_id
   
   # Verify traffic allocation
   python model_versioning.py experiment analyze your_experiment_id
   
   # Check request routing
   grep "request_id" /var/log/nightscan/ab_test.log
   ```

3. **Deployment Problems**
   ```bash
   # Check deployment status
   python model_deployment.py status your_deployment_id
   
   # Run health check
   python model_deployment.py health your_deployment_id
   
   # Check resource usage
   python model_deployment.py list
   ```

### Performance Issues

1. **High Latency**
   - Check model size and complexity
   - Verify hardware resources (CPU/GPU)
   - Consider model optimization (quantization, pruning)
   - Use batch inference for multiple requests

2. **High Memory Usage**
   - Unload unused models
   - Optimize model architecture
   - Use model sharding for large models
   - Configure garbage collection

3. **Low Accuracy**
   - Compare with baseline models
   - Check data quality and preprocessing
   - Verify model training configuration
   - Analyze A/B test results

## Security Considerations

### Model Security

1. **Model Integrity**: Verify model checksums before deployment
2. **Access Control**: Restrict model registry access to authorized users
3. **Audit Logging**: Log all model operations and experiments
4. **Data Privacy**: Ensure training data privacy compliance

### Experiment Security

1. **Request Isolation**: Ensure experiments don't leak sensitive data
2. **Result Privacy**: Protect experiment results and analysis
3. **Traffic Analysis**: Prevent traffic analysis attacks
4. **Rollback Security**: Secure rollback procedures

## Advanced Features

### Multi-Armed Bandit Testing

For dynamic traffic allocation:

```python
# Example multi-armed bandit configuration
bandit_config = {
    "algorithm": "thompson_sampling",
    "exploration_rate": 0.1,
    "reward_metric": "user_satisfaction",
    "update_frequency": 3600  # 1 hour
}
```

### Canary Deployments

Gradual rollout of new models:

```python
# Example canary deployment
canary_config = DeploymentConfig(
    # ... other config ...
    canary_deployment=True,
    traffic_percentage=5.0,  # Start with 5% traffic
    canary_success_threshold=0.95,
    canary_duration_hours=24
)
```

### Model Ensembling

Combine multiple models for better performance:

```python
# Example ensemble configuration
ensemble_config = {
    "models": [
        "wildlife_detector:v2.0.0",
        "wildlife_detector:v2.1.0",
        "wildlife_detector_specialized:v1.0.0"
    ],
    "voting_strategy": "weighted_average",
    "weights": [0.4, 0.4, 0.2]
}
```

This comprehensive ML versioning and A/B testing system ensures reliable, data-driven model development and deployment for NightScan.