# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

NightScan is a wildlife detection and monitoring system that uses machine learning to recognize nocturnal animals through audio and image analysis. The system consists of multiple components including a web interface, prediction APIs, mobile app, and edge computing capabilities.

**Key Architecture:** Edge-cloud hybrid system with unified prediction API, modular Flask web application, React Native mobile client, and Raspberry Pi edge devices for field deployment.

## Common Development Commands

### Python Backend

```bash
# Install dependencies
pip install -r requirements.txt

# Run tests
pytest tests/ -v --cov=. --cov-report=xml --cov-report=term

# Run specific test types
pytest -m unit        # Unit tests only
pytest -m integration # Integration tests only
pytest -m performance # Performance tests only

# Code formatting and linting
black .              # Format code
isort .              # Sort imports
ruff check .         # Run linter
mypy . --ignore-missing-imports  # Type checking

# Security scanning
bandit -r .
safety check

# Start web application
export SECRET_KEY=$(python -c 'import secrets; print(secrets.token_hex(32))')
export SQLALCHEMY_DATABASE_URI="postgresql://user:pass@localhost/nightscan"
export PREDICT_API_URL="http://localhost:8001/api/predict"
gunicorn -w 4 -b 0.0.0.0:8000 web.app:application

# Start Celery worker
celery -A web.tasks worker --loglevel=info

# Start unified prediction API (preferred)
export MODEL_PATH="models/best_model.pth"
export CSV_DIR="data/processed/csv"
gunicorn -w 4 -b 0.0.0.0:8002 unified_prediction_system.unified_prediction_api:application

# Start legacy audio-only API
gunicorn -w 4 -b 0.0.0.0:8001 Audio_Training.scripts.api_server:application
```

### iOS App (React Native)

```bash
cd ios-app
npm install
npm test            # Run Jest tests
npm start           # Start Expo development server
npm run ios         # Start iOS simulator
```

### Docker

```bash
# Development environment
docker-compose up -d

# Production deployment
docker-compose -f docker-compose.production.yml up -d

# Testing environment 
docker-compose -f docker-compose.test.yml build
docker-compose -f docker-compose.test.yml run --rm web pytest

# Monitoring stack
docker-compose -f docker-compose.monitoring.yml up -d
```

### Machine Learning

```bash
# Train audio model
python audio_training_efficientnet/train_audio.py --epochs 50 --batch-size 32

# Train photo model  
python picture_training_enhanced/train_photo.py --epochs 100 --batch-size 16

# Create lightweight models for edge deployment
python create_light_models.py --audio-model models/best_model.pth --output mobile_models/

# Test model performance
python scripts/test-performance.py --model-path models/best_model.pth
```

### Raspberry Pi Edge Deployment

```bash
# Setup Pi hardware and services
cd nightscan_pi
sudo ./setup_pi.sh

# Configure camera and audio
sudo ./Hardware/configure_camera_boot.sh
sudo ./Hardware/configure_respeaker_audio.sh

# Run edge detection service
python Program/main.py --config nightscan_config.db
```

## Architecture Overview

### Service Ports
- Web Application: 8000
- API v1: 8001
- Prediction API: 8002
- ML Service: 8003
- Analytics Dashboard: 8008
- WebSocket: 8012
- Redis: 6379
- PostgreSQL: 5432

### Key Components and Their Responsibilities

1. **Web Application (`web/app.py`)**
   - Main Flask application handling user interface, authentication, and file uploads
   - Uses blueprints for modular organization
   - Implements security features including CSP with nonces and rate limiting

2. **API Layer (`api_v1.py`)**
   - RESTful API with OpenAPI documentation
   - Handles programmatic access with authentication
   - Implements quota management and data retention endpoints

3. **Unified Prediction System (`unified_prediction_system/`)**
   - Single entry point for audio and image predictions
   - Automatic file type detection and model routing via `prediction_router.py`
   - Supports batch processing and model pooling through `model_manager.py`
   - File type detection in `file_type_detector.py` handles .wav, .npy, .jpg, .jpeg formats

4. **Configuration (`unified_config.py`)**
   - Modern unified configuration system replacing legacy `config.py`
   - Environment-based configs (development/staging/production) in `config/unified/`
   - Pydantic validation with dataclass components (DatabaseConfig, CacheConfig, etc.)
   - Automatic secret generation and environment variable mapping

5. **Database Schema (`database/create_database.sql`)**
   - PostgreSQL with tables for users, predictions, detections, quotas, and retention
   - Supports tiered plans and usage tracking via `quota_manager.py`
   - Implements archival and data retention strategies
   - Connection pooling and circuit breakers for reliability

6. **Security Module (`security/`)**
   - Modular security components: auth, encryption, validation, rate limiting
   - CSP nonce management in `csp_nonce.py` for dynamic security headers
   - Sensitive data sanitization and secure logging in `secure_logging.py`
   - File upload validation and streaming support in `secure_uploads.py`

7. **Circuit Breaker System**
   - Database, cache, and HTTP circuit breakers in `circuit_breaker_config.py`
   - Protects against cascading failures in microservice architecture
   - Configurable failure thresholds and recovery timeouts

8. **Analytics & Monitoring**
   - Real-time analytics dashboard in `analytics_dashboard.py`
   - Prometheus metrics collection in `metrics.py`
   - Cache monitoring and health checks
   - Performance tracking and quota usage monitoring

### Critical Implementation Details

1. **Security Considerations**
   - All routes require CSRF protection via Flask-WTF
   - Content Security Policy uses dynamic nonces stored in `g.csp_nonce`
   - Session management supports multiple backends (Redis, filesystem, database)
   - Rate limiting is implemented per endpoint with configurable limits

2. **Performance Optimizations**
   - Database queries are optimized to avoid N+1 problems
   - Connection pooling for PostgreSQL and Redis
   - Streaming file uploads to handle large files
   - Model instance pooling for ML predictions
   - Pagination for data exports

3. **Async Processing**
   - Celery workers handle prediction tasks
   - Redis pub/sub for real-time notifications
   - WebSocket support for live updates

4. **Data Management**
   - Tiered retention policies based on subscription plans
   - Automatic archival of old predictions
   - Quota tracking with daily and monthly limits

### Testing Strategy

- Unit tests focus on individual components
- Integration tests verify service interactions
- Performance tests ensure scalability
- Security tests validate authentication and authorization
- Use pytest markers to run specific test suites

### Environment Variables

Critical environment variables that must be set:
- `SECRET_KEY` - Flask secret key for sessions (auto-generated if not set)
- `SQLALCHEMY_DATABASE_URI` - PostgreSQL connection string 
- `REDIS_URL` - Redis connection for caching and sessions
- `PREDICT_API_URL` - URL for prediction service (defaults to unified API on 8002)
- `MODEL_PATH` - Path to trained ML model
- `CSV_DIR` - Directory containing training CSVs
- `NIGHTSCAN_ENV` - Environment name (development/staging/production)
- `DOMAIN_NAME` - Domain for production SSL certificates
- `ADMIN_EMAIL` - Admin email for Let's Encrypt certificates

**Modern Configuration:**
Use `unified_config.py` with environment-specific JSON files in `config/unified/` for better organization and validation.

### Common Debugging Tips

1. Check logs in `logs/` directory for application errors
2. Use `flask shell` to interact with database models
3. Monitor Redis with `redis-cli monitor` for caching issues
4. Use `celery flower` to monitor task queue status
5. Database queries can be profiled with `SQLALCHEMY_ECHO=True`

### Important Conventions

- All API responses follow consistent JSON structure with `success`, `data`, and `error` fields
- Database migrations should be backward compatible
- Security headers are mandatory for all routes
- File uploads are limited to 100MB with configurable user quotas via `quota_manager.py`
- All timestamps are stored in UTC
- Use circuit breakers for external service calls to prevent cascade failures
- Implement proper exception handling using the custom exceptions in `exceptions.py`
- Follow the unified configuration pattern: prefer `unified_config.py` over legacy config files
- CSP nonces are required for inline scripts - use `@csp_nonce_required` decorator
- All sensitive operations should use the security module components
- Model predictions should route through `unified_prediction_system` for consistency

### Code Quality Standards

- Line length: 120 characters (Black/Ruff configured)
- Use type hints where possible (MyPy validation enabled)
- Test markers: `@pytest.mark.unit`, `@pytest.mark.integration`, `@pytest.mark.performance`
- Security scanning with Bandit is enforced
- ML code allows relaxed naming conventions (N803, N806 exceptions)