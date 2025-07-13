# Circuit Breakers Configuration Guide

This document explains how to configure and manage circuit breakers in NightScan.

## Overview

NightScan uses a centralized circuit breaker system to provide fault tolerance and prevent cascade failures across the application. Circuit breakers monitor the health of external services and automatically fail fast when services are unavailable.

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Web App       │────│  Circuit Breaker │────│   Database      │
│   (Flask)       │    │     Manager      │    │   (PostgreSQL)  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   API v1        │────│  Centralized     │────│   Redis Cache   │
│   (REST API)    │    │  Configuration   │    │   (Redis)       │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ Celery Workers  │────│                  │────│   ML Services   │
│   (Tasks)       │    │                  │    │   (PyTorch)     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## Configuration File

Circuit breakers are configured via `config/circuit_breakers.json`:

```json
{
  "settings": {
    "enabled": true,
    "monitoring_enabled": true,
    "default_failure_threshold": 5,
    "default_timeout": 60.0,
    "environment": "production"
  },
  "services": {
    "web_app": {
      "enabled": true,
      "database": {
        "read_timeout": 2.0,
        "write_timeout": 5.0,
        "failure_threshold": 3
      },
      "cache": {
        "redis_host": "localhost",
        "redis_port": 6379,
        "enable_memory_fallback": true
      }
    }
  }
}
```

## Service Types

### 1. Web Application (`web_app`)
- **Database**: User authentication, file operations
- **Cache**: Session storage, temporary data
- **HTTP**: External notification services

### 2. API v1 (`api_v1`) 
- **Database**: Quota management, user data
- **Cache**: Prediction caching
- **ML**: Model inference for predictions
- **HTTP**: External API calls

### 3. Celery Workers (`celery_workers`)
- **Database**: Prediction storage
- **Cache**: Result caching
- **HTTP**: ML API calls, notifications
- **ML**: Local model inference

### 4. ML Prediction (`ml_prediction`)
- **ML**: Core model operations
- **Cache**: Prediction result caching

### 5. Notification Service (`notification_service`)
- **HTTP**: Push notification services
- **Database**: Notification logging

## Circuit Breaker Types

### Database Circuit Breaker
Protects database operations with read/write differentiation:

```python
from circuit_breaker_config import get_database_circuit_breaker

db_circuit = get_database_circuit_breaker("web_app", db_session=db.session)

# Protected query
with db_circuit.transaction():
    user = User(username="test")
    db.session.add(user)
    # Auto-committed by circuit breaker
```

**Features:**
- Read-only fallback mode
- Cache-based fallbacks
- Connection pool monitoring
- Slow query detection

### Cache Circuit Breaker  
Protects Redis operations with memory/disk fallbacks:

```python
from circuit_breaker_config import get_cache_circuit_breaker

cache_circuit = get_cache_circuit_breaker("api_v1")

# Protected cache operation
try:
    result = cache_circuit.get("prediction:abc123")
except CircuitBreakerOpenException:
    # Circuit open - use fallback
    result = fallback_cache.get("prediction:abc123")
```

**Features:**
- Memory cache fallback (LRU)
- Disk cache fallback  
- Session storage fallback
- Task queue fallback

### HTTP Circuit Breaker
Protects external HTTP calls with retry logic:

```python
from circuit_breaker_config import get_http_circuit_breaker

http_circuit = get_http_circuit_breaker("celery_workers", 
                                       service_type="ml_api")

# Protected HTTP request
response = http_circuit.post("/predict", json=data)
```

**Features:**
- Exponential backoff retry
- Connection pooling
- Request/response validation
- Health check monitoring

### ML Circuit Breaker
Protects ML model operations with intelligent fallbacks:

```python
from circuit_breaker_config import get_ml_circuit_breaker

ml_circuit = get_ml_circuit_breaker("api_v1", 
                                   model_path="/path/to/model.pth")

# Protected ML prediction
result = ml_circuit.predict(audio_data)
```

**Features:**
- Lightweight model fallback
- Prediction caching
- Resource monitoring (GPU/CPU/Memory)
- Model warming and preloading

## Fallback Strategies

### 1. Database Fallbacks
- **Circuit Open**: Use cache for reads, queue writes
- **Slow Queries**: Switch to read-only mode
- **Connection Issues**: Retry with exponential backoff

### 2. Cache Fallbacks
- **Redis Down**: Memory cache → Disk cache → Database
- **Memory Full**: LRU eviction with disk persistence
- **Disk Issues**: Memory-only mode

### 3. HTTP Fallbacks  
- **Service Down**: Cached responses, default values
- **Timeout**: Shortened timeouts, async queuing
- **Rate Limited**: Exponential backoff, circuit opening

### 4. ML Fallbacks
- **Model Unavailable**: Lightweight model, cached predictions
- **Resource Exhaustion**: Queue management, batch reduction
- **Inference Timeout**: Fallback to simpler models

## Monitoring & Management

### Health Check Endpoints

```bash
# Overall circuit breaker health
curl http://localhost:8000/circuit-breakers/health

# Detailed metrics
curl http://localhost:8000/circuit-breakers/metrics
```

### Management Script

```bash
# Check status
python scripts/manage_circuit_breakers.py status

# View metrics
python scripts/manage_circuit_breakers.py metrics

# Reset all circuits
python scripts/manage_circuit_breakers.py reset

# Reset specific service
python scripts/manage_circuit_breakers.py reset --service web_app

# Monitor continuously
python scripts/manage_circuit_breakers.py monitor --interval 30
```

### Environment Variables

```bash
# Enable/disable circuit breakers
CIRCUIT_BREAKERS_ENABLED=true

# Configuration file location
CIRCUIT_BREAKER_CONFIG=config/circuit_breakers.json

# Service-specific overrides
REDIS_HOST=redis.example.com
REDIS_PORT=6379
MODEL_PATH=/models/nightscan_v2.pth
```

## Circuit States

### 🟢 CLOSED (Normal Operation)
- All requests pass through
- Failures are monitored
- Circuit opens on threshold breach

### 🔴 OPEN (Failing Fast)
- Requests immediately fail
- Fallback mechanisms activated
- Periodic health checks attempt recovery

### 🟡 HALF_OPEN (Testing Recovery)
- Limited requests allowed through
- Success closes circuit
- Failure reopens circuit immediately

## Configuration Parameters

### Global Settings
- `enabled`: Enable/disable all circuit breakers
- `monitoring_enabled`: Enable metrics collection
- `default_failure_threshold`: Failures before opening (default: 5)
- `default_timeout`: Seconds before trying recovery (default: 60)
- `default_success_threshold`: Successes to close from half-open (default: 3)

### Service-Specific Settings
- `read_timeout`: Database read operation timeout
- `write_timeout`: Database write operation timeout  
- `inference_timeout`: ML model inference timeout
- `connect_timeout`: HTTP connection timeout
- `max_retries`: Maximum retry attempts
- `enable_*_fallback`: Enable specific fallback mechanisms

## Best Practices

### 1. Configuration
- Use environment-specific configs (dev/staging/prod)
- Set appropriate timeouts for each service
- Enable fallbacks for non-critical operations
- Monitor circuit breaker metrics

### 2. Fallback Design
- Ensure fallbacks don't cascade failures
- Use degraded functionality over complete failure
- Cache successful responses for fallback use
- Design for eventual consistency

### 3. Monitoring
- Set up alerts for circuit state changes
- Monitor failure rates and response times
- Track fallback usage patterns
- Regular health check validation

### 4. Testing
- Test circuit breaker behavior under load
- Validate fallback mechanisms work correctly
- Practice manual circuit resets
- Chaos engineering for resilience testing

## Troubleshooting

### Circuit Stuck Open
```bash
# Check health status
python scripts/manage_circuit_breakers.py status

# View detailed metrics
python scripts/manage_circuit_breakers.py metrics

# Reset if necessary
python scripts/manage_circuit_breakers.py reset --service problematic_service
```

### High Failure Rates
1. Check service health
2. Review timeout settings
3. Verify network connectivity  
4. Check resource availability

### Fallback Not Working
1. Verify fallback configuration
2. Check fallback service health
3. Review error logs
4. Test fallback mechanisms

### Performance Issues
1. Check circuit breaker overhead
2. Review timeout configurations
3. Monitor resource usage
4. Optimize fallback strategies

## Integration Examples

### Flask Route Protection
```python
@app.route('/api/data')
def get_data():
    db_circuit = get_database_circuit_breaker("web_app")
    
    try:
        data = db_circuit.execute_query(
            "SELECT * FROM data WHERE active = true",
            read_only=True
        )
        return jsonify(data)
    except CircuitBreakerOpenException:
        # Use cached data as fallback
        return jsonify(get_cached_data())
```

### Celery Task Protection
```python
@celery.task
def process_prediction(file_path):
    ml_circuit = get_ml_circuit_breaker("celery_workers")
    
    try:
        result = ml_circuit.predict(load_audio(file_path))
        return result
    except CircuitBreakerOpenException:
        # Queue for later processing
        queue_for_retry.delay(file_path)
        return {"status": "queued", "reason": "ml_service_unavailable"}
```

### API Client Protection
```python
def call_external_api(data):
    http_circuit = get_http_circuit_breaker("api_v1", 
                                           service_type="external")
    
    try:
        response = http_circuit.post("/external/api", json=data)
        return response.json()
    except CircuitBreakerOpenException:
        # Use default response or cached data
        return {"status": "unavailable", "fallback": True}
```