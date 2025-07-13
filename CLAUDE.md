# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

NightScan is a wildlife detection and monitoring system that uses machine learning to recognize nocturnal animals through audio and image analysis. The system consists of multiple components including a web interface, prediction APIs, mobile app, and edge computing capabilities.

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

# Start prediction API
export MODEL_PATH="models/best_model.pth"
export CSV_DIR="data/processed/csv"
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
# Build and run with docker-compose
docker-compose -f docker-compose.production.yml up -d
docker-compose -f docker-compose.test.yml build
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
   - Automatic file type detection and model routing
   - Supports batch processing and model pooling

4. **Configuration (`config.py`)**
   - Centralized configuration with environment variable support
   - Validates settings and provides defaults
   - Manages database, Redis, security, and ML configurations

5. **Database Schema (`database/create_database.sql`)**
   - PostgreSQL with tables for users, predictions, detections, quotas, and retention
   - Supports tiered plans and usage tracking
   - Implements archival and data retention strategies

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
- `SECRET_KEY` - Flask secret key for sessions
- `SQLALCHEMY_DATABASE_URI` - PostgreSQL connection string
- `REDIS_URL` - Redis connection for caching and sessions
- `PREDICT_API_URL` - URL for prediction service
- `MODEL_PATH` - Path to trained ML model
- `CSV_DIR` - Directory containing training CSVs

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
- File uploads are limited to 100MB with configurable user quotas
- All timestamps are stored in UTC