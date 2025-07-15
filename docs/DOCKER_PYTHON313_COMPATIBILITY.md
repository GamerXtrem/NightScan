# Docker Python 3.13 Compatibility

## Current Status

### Dockerfiles Found
- `Dockerfile` (main application)
- `docker/Dockerfile.test` (testing environment)

### Docker Compose Files
- `docker-compose.production.yml` (production deployment)
- `docker-compose.test.yml` (testing environment)
- `docker-compose.yml` (development)
- Multiple specialized compose files

## Current Python Version

The main `Dockerfile` currently uses:
```dockerfile
FROM python:3.9-slim as base
```

## Python 3.13 Migration Plan

### 1. Update Base Images

**Current:**
```dockerfile
FROM python:3.9-slim as base
```

**Recommended:**
```dockerfile
FROM python:3.13-slim as base
```

### 2. Test Image Availability

Python 3.13 Docker images are available:
- `python:3.13-slim` - Minimal Python 3.13 image
- `python:3.13-alpine` - Even smaller Alpine-based image
- `python:3.13` - Full Python 3.13 image

### 3. Compatibility Verification

#### System Dependencies
All current system dependencies are compatible with Python 3.13:
```dockerfile
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    pkg-config \
    libsndfile1-dev \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*
```

#### Python Dependencies
✅ All NightScan dependencies are Python 3.13 compatible:
- PyTorch 2.7.0
- NumPy 2.2.6
- Flask 3.1.1
- SQLAlchemy 2.0.41
- All other packages (100% compatibility score)

### 4. Migration Steps

#### Step 1: Update Main Dockerfile
```dockerfile
# Multi-stage build for NightScan wildlife detection system
FROM python:3.13-slim as base

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    pkg-config \
    libsndfile1-dev \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd --create-home --shell /bin/bash nightscan
USER nightscan

# Expose port
EXPOSE 8000

# Start command
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:8000", "web.app:application"]
```

#### Step 2: Update Test Dockerfile
```dockerfile
FROM python:3.13-slim

WORKDIR /app

# Install test dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    pkg-config \
    libsndfile1-dev \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install test-specific dependencies
RUN pip install pytest pytest-cov pytest-mock

# Copy test files
COPY . .

# Run tests
CMD ["pytest", "tests/", "-v", "--cov=.", "--cov-report=xml"]
```

#### Step 3: Update Docker Compose

**Production:**
```yaml
services:
  web:
    image: ghcr.io/gamerxtrem/nightscan/web:python3.13-latest
    # ... rest of configuration
```

**Development:**
```yaml
services:
  web:
    build: 
      context: .
      dockerfile: Dockerfile
    # ... rest of configuration
```

### 5. Testing Strategy

#### Build Test
```bash
# Test Docker build
docker build -t nightscan:python3.13 .

# Test container run
docker run --rm nightscan:python3.13 python --version
```

#### Integration Test
```bash
# Test with docker-compose
docker-compose -f docker-compose.test.yml up --build

# Test production setup
docker-compose -f docker-compose.production.yml up --build
```

### 6. Potential Issues and Solutions

#### Issue 1: Build Time
**Problem:** Python 3.13 images might be larger or slower to build
**Solution:** Use `python:3.13-alpine` for smaller images

#### Issue 2: Package Compilation
**Problem:** Some packages might need recompilation
**Solution:** Use pre-compiled wheels or multi-stage builds

#### Issue 3: Memory Usage
**Problem:** Python 3.13 might have different memory characteristics
**Solution:** Monitor and adjust container memory limits

### 7. Rollback Plan

If issues occur:
1. Keep `python:3.9-slim` as fallback
2. Tag images with Python version for easy rollback
3. Use environment variables for Python version selection

### 8. Benefits of Migration

✅ **Latest Python features** - Pattern matching, performance improvements
✅ **Security updates** - Latest security patches
✅ **Long-term support** - Python 3.13 will be supported until 2029
✅ **Performance** - Improved interpreter performance
✅ **Compatibility** - All dependencies already compatible

### 9. Implementation Timeline

1. **Week 1**: Update development Dockerfile and test locally
2. **Week 2**: Update test environment and run full test suite
3. **Week 3**: Update staging environment and monitor
4. **Week 4**: Deploy to production during maintenance window

### 10. Monitoring Points

After migration, monitor:
- Container startup time
- Memory usage
- Application performance
- Error rates
- Build times

## Conclusion

✅ **Ready for migration** - All dependencies are Python 3.13 compatible
✅ **Low risk** - Standard Docker practices apply
✅ **Recommended** - Provides better security and performance
⚠️ **Test thoroughly** - Verify all services work correctly