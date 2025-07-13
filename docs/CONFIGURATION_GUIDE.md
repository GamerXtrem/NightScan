# NightScan Configuration Guide

## Overview

NightScan uses a centralized configuration system based on dataclasses with optional Pydantic validation. The configuration is simple, clear, and appropriate for a project in development.

## Configuration Structure

### Main Configuration (`config.py`)

The configuration is organized into logical sections:

- **DatabaseConfig** - Database connection settings
- **RedisConfig** - Redis cache configuration  
- **SecurityConfig** - Security settings (secret keys, HTTPS, etc.)
- **PortConfig** - Service port assignments
- **RateLimitConfig** - Rate limiting rules
- **ModelConfig** - ML model paths and settings
- **LoggingConfig** - Logging configuration
- **FileUploadConfig** - File upload limits
- **ApiConfig** - API settings
- **OptimizedServingConfig** - Performance optimization
- **RaspberryPiConfig** - Raspberry Pi specific settings

### Environment Configuration (`config/environments.py`)

Pre-configured settings for different environments:
- **development** - Local development with relaxed security
- **testing** - Unit testing with in-memory database
- **staging** - Pre-production environment
- **production** - Production with full security
- **raspberry_pi** - Optimized for Raspberry Pi devices

## Usage

### Basic Usage

```python
from config import get_config

# Get configuration
config = get_config()

# Access settings
database_url = config.database.uri
web_port = config.ports.web_app
model_path = config.model.audio_model_path
```

### Environment Variables

Override any setting using environment variables with `NIGHTSCAN_` prefix:

```bash
export NIGHTSCAN_DATABASE__URI="postgresql://user:pass@host/db"
export NIGHTSCAN_PORTS__WEB_APP=8080
export NIGHTSCAN_MODEL__DEVICE="cuda"
```

### Loading Environment-Specific Config

```python
from config.environments import load_environment_config

# Load production config
config = load_environment_config('production')

# Auto-detect from NIGHTSCAN_ENV
export NIGHTSCAN_ENV=staging
config = load_environment_config()
```

## Port Configuration

All service ports are centralized to avoid conflicts:

```python
config.ports.web_app         # 8000
config.ports.api_v1          # 8001
config.ports.prediction_api  # 8002
config.ports.ml_service      # 8003
# ... etc
```

## ML Model Configuration

Separate paths for audio and photo models:

```python
config.model.audio_model_path  # "models/audio/best_model.pth"
config.model.photo_model_path  # "models/photo/best_model.pth"
config.model.device           # "auto" (auto-detects cuda/cpu)
config.model.batch_size       # 32
```

## Security Configuration

```python
config.security.secret_key         # Auto-generated if not set
config.security.force_https        # True in production
config.security.password_min_length # 10 characters minimum
```

## Production Requirements

When running in production (`NIGHTSCAN_ENV=production`), these environment variables are required:

- `DATABASE_URL` - PostgreSQL connection string
- `REDIS_URL` - Redis connection string  
- `SECRET_KEY` - Flask secret key
- `CSRF_SECRET_KEY` - CSRF protection key

## Example Configuration File

Create `config/production.json`:

```json
{
  "environment": "production",
  "database": {
    "pool_size": 50
  },
  "security": {
    "password_min_length": 16
  },
  "model": {
    "batch_size": 64,
    "device": "cuda"
  }
}
```

## Testing Configuration

```python
# Run with test configuration
export NIGHTSCAN_ENV=testing
python your_script.py

# Or in code
from config.environments import load_environment_config
config = load_environment_config('testing')
```

## Validation

The configuration includes validation to ensure:
- Required settings are present in production
- File paths exist when needed
- Port conflicts are detected
- Upload limits are sensible

Run validation:
```python
from config import get_config, validate_config

config = get_config()
errors = validate_config(config)
if errors:
    print("Configuration errors:", errors)
```

## Simple and Maintainable

This configuration system is intentionally simple:
- No complex encryption or secrets management (can be added later)
- No migration tools (not needed during development)
- Clear structure that's easy to understand
- Gradual complexity - features can be added as needed

For a project in development, this provides the right balance of structure and simplicity.