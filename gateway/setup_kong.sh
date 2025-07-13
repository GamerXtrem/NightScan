#!/bin/bash

# Kong Gateway Setup Script for NightScan
# This script configures Kong with JWT authentication and other plugins

set -e

KONG_ADMIN_URL="${KONG_ADMIN_URL:-http://localhost:8081}"
JWT_SECRET="${JWT_SECRET_KEY:-your-secret-key-here}"

echo "Setting up Kong Gateway for NightScan..."

# Wait for Kong to be ready
echo "Waiting for Kong to be ready..."
until curl -s "${KONG_ADMIN_URL}" > /dev/null; do
    sleep 2
done

echo "Kong is ready!"

# Function to create or update a service
create_service() {
    local name=$1
    local url=$2
    
    echo "Creating service: $name"
    curl -i -X PUT "${KONG_ADMIN_URL}/services/${name}" \
        -H "Content-Type: application/json" \
        -d "{
            \"name\": \"${name}\",
            \"url\": \"${url}\",
            \"retries\": 5,
            \"connect_timeout\": 5000,
            \"write_timeout\": 60000,
            \"read_timeout\": 60000
        }"
    echo ""
}

# Function to create or update a route
create_route() {
    local name=$1
    local service=$2
    local paths=$3
    local methods=$4
    
    echo "Creating route: $name"
    curl -i -X PUT "${KONG_ADMIN_URL}/routes/${name}" \
        -H "Content-Type: application/json" \
        -d "{
            \"name\": \"${name}\",
            \"service\": {\"name\": \"${service}\"},
            \"paths\": ${paths},
            \"methods\": ${methods},
            \"strip_path\": false
        }"
    echo ""
}

# Create services
create_service "web-service" "http://web:8000"
create_service "prediction-service" "http://prediction:8002"
create_service "analytics-service" "http://analytics:8008"

# Create routes
create_route "auth-routes" "web-service" '[\"/api/auth\"]' '["GET","POST","OPTIONS"]'
create_route "api-routes" "web-service" '[\"/api/v1\"]' '["GET","POST","PUT","DELETE","OPTIONS"]'
create_route "prediction-routes" "prediction-service" '[\"/api/predict\"]' '["POST","OPTIONS"]'
create_route "analytics-routes" "analytics-service" '[\"/api/analytics\"]' '["GET","OPTIONS"]'

# Enable CORS plugin globally
echo "Enabling CORS plugin..."
curl -i -X POST "${KONG_ADMIN_URL}/plugins" \
    -H "Content-Type: application/json" \
    -d '{
        "name": "cors",
        "config": {
            "origins": ["*"],
            "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
            "headers": ["Accept", "Authorization", "Content-Type", "X-Requested-With"],
            "exposed_headers": ["X-Auth-Token"],
            "credentials": true,
            "max_age": 3600
        }
    }'
echo ""

# Enable rate limiting globally
echo "Enabling rate limiting..."
curl -i -X POST "${KONG_ADMIN_URL}/plugins" \
    -H "Content-Type: application/json" \
    -d '{
        "name": "rate-limiting",
        "config": {
            "minute": 100,
            "policy": "local"
        }
    }'
echo ""

# Configure JWT plugin for protected routes
echo "Configuring JWT authentication..."

# First, create a consumer
curl -i -X PUT "${KONG_ADMIN_URL}/consumers/default-jwt-consumer" \
    -H "Content-Type: application/json" \
    -d '{
        "username": "default-jwt-consumer",
        "custom_id": "jwt-auth"
    }'
echo ""

# Create JWT credential for the consumer
curl -i -X POST "${KONG_ADMIN_URL}/consumers/default-jwt-consumer/jwt" \
    -H "Content-Type: application/json" \
    -d "{
        \"key\": \"${JWT_SECRET}\",
        \"algorithm\": \"HS256\"
    }"
echo ""

# Apply JWT plugin to API routes (not auth routes)
curl -i -X POST "${KONG_ADMIN_URL}/routes/api-routes/plugins" \
    -H "Content-Type: application/json" \
    -d '{
        "name": "jwt",
        "config": {
            "claims_to_verify": ["exp"],
            "header_names": ["Authorization"],
            "cookie_names": ["jwt_token"]
        }
    }'
echo ""

# Apply JWT to prediction routes
curl -i -X POST "${KONG_ADMIN_URL}/routes/prediction-routes/plugins" \
    -H "Content-Type: application/json" \
    -d '{
        "name": "jwt",
        "config": {
            "claims_to_verify": ["exp"],
            "header_names": ["Authorization"]
        }
    }'
echo ""

# Apply JWT to analytics routes
curl -i -X POST "${KONG_ADMIN_URL}/routes/analytics-routes/plugins" \
    -H "Content-Type: application/json" \
    -d '{
        "name": "jwt",
        "config": {
            "claims_to_verify": ["exp"],
            "header_names": ["Authorization"]
        }
    }'
echo ""

# Enable request ID plugin for tracing
echo "Enabling request ID plugin..."
curl -i -X POST "${KONG_ADMIN_URL}/plugins" \
    -H "Content-Type: application/json" \
    -d '{
        "name": "correlation-id",
        "config": {
            "header_name": "X-Request-Id",
            "generator": "uuid",
            "echo_downstream": true
        }
    }'
echo ""

# Enable Prometheus plugin for monitoring
echo "Enabling Prometheus metrics..."
curl -i -X POST "${KONG_ADMIN_URL}/plugins" \
    -H "Content-Type: application/json" \
    -d '{
        "name": "prometheus"
    }'
echo ""

echo "Kong Gateway setup completed!"
echo ""
echo "Gateway endpoints:"
echo "  - HTTP Proxy: http://localhost:8080"
echo "  - HTTPS Proxy: https://localhost:8443"
echo "  - Admin API: http://localhost:8081"
echo "  - Metrics: http://localhost:8080/metrics"
echo ""
echo "Test the gateway:"
echo "  curl http://localhost:8080/api/auth/login -X POST -H 'Content-Type: application/json' -d '{\"username\":\"test\",\"password\":\"test\"}'"