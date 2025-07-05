#!/bin/bash

# NightScan Deployment Script
# This script deploys the NightScan application to Kubernetes

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
NAMESPACE="nightscan"
DOCKER_REGISTRY="${DOCKER_REGISTRY:-localhost:5000}"
VERSION="${VERSION:-latest}"

log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[WARNING] $1${NC}"
}

error() {
    echo -e "${RED}[ERROR] $1${NC}"
    exit 1
}

success() {
    echo -e "${GREEN}[SUCCESS] $1${NC}"
}

# Check dependencies
check_dependencies() {
    log "Checking dependencies..."
    
    command -v kubectl >/dev/null 2>&1 || error "kubectl is required but not installed"
    command -v docker >/dev/null 2>&1 || error "docker is required but not installed"
    
    # Check if kubectl can connect to cluster
    kubectl cluster-info >/dev/null 2>&1 || error "Cannot connect to Kubernetes cluster"
    
    success "Dependencies check passed"
}

# Build Docker images
build_images() {
    log "Building Docker images..."
    
    # Build web application image
    docker build -t ${DOCKER_REGISTRY}/nightscan/web-app:${VERSION} --target production .
    
    # Build prediction API image
    docker build -t ${DOCKER_REGISTRY}/nightscan/prediction-api:${VERSION} --target prediction-api .
    
    success "Docker images built successfully"
}

# Push images to registry
push_images() {
    log "Pushing images to registry..."
    
    docker push ${DOCKER_REGISTRY}/nightscan/web-app:${VERSION}
    docker push ${DOCKER_REGISTRY}/nightscan/prediction-api:${VERSION}
    
    success "Images pushed to registry"
}

# Create namespace
create_namespace() {
    log "Creating namespace..."
    
    if kubectl get namespace ${NAMESPACE} >/dev/null 2>&1; then
        warn "Namespace ${NAMESPACE} already exists"
    else
        kubectl apply -f k8s/namespace.yaml
        success "Namespace created"
    fi
}

# Deploy database and cache
deploy_infrastructure() {
    log "Deploying infrastructure components..."
    
    # Deploy PostgreSQL
    kubectl apply -f k8s/postgres.yaml
    
    # Deploy Redis
    kubectl apply -f k8s/redis.yaml
    
    # Wait for infrastructure to be ready
    log "Waiting for infrastructure to be ready..."
    kubectl wait --for=condition=available --timeout=300s deployment/postgres -n ${NAMESPACE}
    kubectl wait --for=condition=available --timeout=300s deployment/redis -n ${NAMESPACE}
    
    success "Infrastructure deployed successfully"
}

# Deploy application services
deploy_application() {
    log "Deploying application services..."
    
    # Update image tags in manifests
    sed -i.bak "s|image: nightscan/|image: ${DOCKER_REGISTRY}/nightscan/|g" k8s/prediction-api.yaml
    sed -i.bak "s|image: nightscan/|image: ${DOCKER_REGISTRY}/nightscan/|g" k8s/web-app.yaml
    sed -i.bak "s|:latest|:${VERSION}|g" k8s/prediction-api.yaml
    sed -i.bak "s|:latest|:${VERSION}|g" k8s/web-app.yaml
    
    # Deploy prediction API
    kubectl apply -f k8s/prediction-api.yaml
    
    # Deploy web application
    kubectl apply -f k8s/web-app.yaml
    
    # Deploy autoscaling
    kubectl apply -f k8s/hpa.yaml
    
    # Wait for deployments to be ready
    log "Waiting for application deployments to be ready..."
    kubectl wait --for=condition=available --timeout=600s deployment/prediction-api -n ${NAMESPACE}
    kubectl wait --for=condition=available --timeout=300s deployment/web-app -n ${NAMESPACE}
    
    # Restore original manifests
    mv k8s/prediction-api.yaml.bak k8s/prediction-api.yaml
    mv k8s/web-app.yaml.bak k8s/web-app.yaml
    
    success "Application services deployed successfully"
}

# Deploy monitoring
deploy_monitoring() {
    log "Deploying monitoring stack..."
    
    kubectl apply -f k8s/monitoring.yaml
    
    # Wait for monitoring to be ready
    log "Waiting for monitoring to be ready..."
    kubectl wait --for=condition=available --timeout=300s deployment/prometheus -n ${NAMESPACE}
    kubectl wait --for=condition=available --timeout=300s deployment/grafana -n ${NAMESPACE}
    
    success "Monitoring stack deployed successfully"
}

# Initialize database
init_database() {
    log "Initializing database..."
    
    # Wait for web app to be running
    kubectl wait --for=condition=ready pod -l app=web-app -n ${NAMESPACE} --timeout=300s
    
    # Get a web app pod
    WEB_POD=$(kubectl get pods -l app=web-app -n ${NAMESPACE} -o jsonpath='{.items[0].metadata.name}')
    
    # Run database initialization
    kubectl exec -n ${NAMESPACE} ${WEB_POD} -- python -c "
from web.app import create_app
app = create_app()
with app.app_context():
    from web.app import db
    db.create_all()
    print('Database initialized successfully')
"
    
    success "Database initialized"
}

# Check deployment status
check_status() {
    log "Checking deployment status..."
    
    echo ""
    echo "=== PODS ==="
    kubectl get pods -n ${NAMESPACE}
    
    echo ""
    echo "=== SERVICES ==="
    kubectl get services -n ${NAMESPACE}
    
    echo ""
    echo "=== INGRESS ==="
    kubectl get ingress -n ${NAMESPACE}
    
    echo ""
    echo "=== HPA ==="
    kubectl get hpa -n ${NAMESPACE}
    
    # Check if all deployments are ready
    if kubectl wait --for=condition=available --timeout=10s deployment --all -n ${NAMESPACE} >/dev/null 2>&1; then
        success "All deployments are ready!"
    else
        warn "Some deployments are not ready yet"
    fi
}

# Port forward for local access
setup_port_forwarding() {
    log "Setting up port forwarding for local access..."
    
    echo ""
    echo "To access the application locally, run these commands in separate terminals:"
    echo ""
    echo "# Web Application"
    echo "kubectl port-forward -n ${NAMESPACE} service/web-service 8000:8000"
    echo "# Then visit: http://localhost:8000"
    echo ""
    echo "# Grafana Dashboard"
    echo "kubectl port-forward -n ${NAMESPACE} service/grafana-service 3000:3000"
    echo "# Then visit: http://localhost:3000 (admin/admin)"
    echo ""
    echo "# Prometheus"
    echo "kubectl port-forward -n ${NAMESPACE} service/prometheus-service 9090:9090"
    echo "# Then visit: http://localhost:9090"
    echo ""
}

# Main deployment function
main() {
    log "Starting NightScan deployment..."
    
    check_dependencies
    
    if [[ "${SKIP_BUILD:-false}" != "true" ]]; then
        build_images
        push_images
    else
        warn "Skipping image build (SKIP_BUILD=true)"
    fi
    
    create_namespace
    deploy_infrastructure
    deploy_application
    
    if [[ "${SKIP_MONITORING:-false}" != "true" ]]; then
        deploy_monitoring
    else
        warn "Skipping monitoring deployment (SKIP_MONITORING=true)"
    fi
    
    if [[ "${SKIP_DB_INIT:-false}" != "true" ]]; then
        init_database
    else
        warn "Skipping database initialization (SKIP_DB_INIT=true)"
    fi
    
    check_status
    setup_port_forwarding
    
    success "NightScan deployment completed successfully!"
}

# Parse command line arguments
case "${1:-deploy}" in
    "deploy")
        main
        ;;
    "status")
        check_status
        ;;
    "cleanup")
        log "Cleaning up deployment..."
        kubectl delete namespace ${NAMESPACE} --ignore-not-found
        success "Cleanup completed"
        ;;
    "help")
        echo "Usage: $0 [deploy|status|cleanup|help]"
        echo ""
        echo "Commands:"
        echo "  deploy   - Deploy the complete NightScan application (default)"
        echo "  status   - Check deployment status"
        echo "  cleanup  - Delete all resources"
        echo "  help     - Show this help"
        echo ""
        echo "Environment variables:"
        echo "  DOCKER_REGISTRY - Docker registry URL (default: localhost:5000)"
        echo "  VERSION         - Image version tag (default: latest)"
        echo "  SKIP_BUILD      - Skip building images (default: false)"
        echo "  SKIP_MONITORING - Skip monitoring deployment (default: false)"
        echo "  SKIP_DB_INIT    - Skip database initialization (default: false)"
        ;;
    *)
        error "Unknown command: $1. Use 'help' for usage information."
        ;;
esac