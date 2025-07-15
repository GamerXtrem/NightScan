# Multi-stage build for NightScan wildlife detection system
# Support for multiple Python versions including 3.13
ARG PYTHON_VERSION=3.13
FROM python:${PYTHON_VERSION}-slim as base

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
# Use Python 3.13 compatible requirements if available
ARG PYTHON_VERSION=3.13
COPY requirements*.txt ./
RUN if [ "${PYTHON_VERSION}" = "3.13" ] && [ -f requirements-python313.txt ]; then \
        pip install --no-cache-dir -r requirements-python313.txt; \
    else \
        pip install --no-cache-dir -r requirements.txt; \
    fi

# Production stage
FROM base as production

# Create non-root user
RUN useradd --create-home --shell /bin/bash nightscan
USER nightscan

# Copy application code
COPY --chown=nightscan:nightscan . .

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV FLASK_APP=web.app
ENV NIGHTSCAN_ENV=production

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Run the application
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "4", "--timeout", "120", "web.app:application"]

# Development stage
FROM base as development

# Install development dependencies
ARG PYTHON_VERSION=3.13
COPY requirements-dev.txt .
RUN if [ "${PYTHON_VERSION}" = "3.13" ] && [ -f requirements-python313.txt ]; then \
        echo "Using Python 3.13 requirements for development"; \
    fi && \
    pip install --no-cache-dir -r requirements-dev.txt

# Copy application code
COPY . .

# Set environment variables for development
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV FLASK_APP=web.app
ENV FLASK_ENV=development
ENV NIGHTSCAN_ENV=development

# Expose port
EXPOSE 8000

# Run in development mode
CMD ["flask", "run", "--host=0.0.0.0", "--port=8000", "--debug"]

# Prediction API stage
FROM base as prediction-api

# Copy model files and prediction server
COPY Audio_Training/ ./Audio_Training/
COPY models/ ./models/

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Health check for prediction API
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8001/api/health || exit 1

# Expose prediction API port
EXPOSE 8001

# Run prediction server
CMD ["python", "Audio_Training/scripts/api_server.py", "--host", "0.0.0.0", "--port", "8001"]