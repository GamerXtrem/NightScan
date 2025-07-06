#!/bin/bash

# Enhanced NightScan Deployment Script
# Supports CI/CD pipeline integration with advanced deployment strategies

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
NAMESPACE="${NAMESPACE:-nightscan}"
ENVIRONMENT="${ENVIRONMENT:-staging}"
DOCKER_REGISTRY="${DOCKER_REGISTRY:-ghcr.io}"
IMAGE_NAME="${IMAGE_NAME:-gamerxtrem/nightscan}"
VERSION="${VERSION:-latest}"
DEPLOYMENT_STRATEGY="${DEPLOYMENT_STRATEGY:-rolling}"
HEALTH_CHECK_TIMEOUT="${HEALTH_CHECK_TIMEOUT:-300}"
ROLLBACK_ON_FAILURE="${ROLLBACK_ON_FAILURE:-true}"

# Logging functions
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

info() {
    echo -e "${PURPLE}[INFO] $1${NC}"
}

# Usage function
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

ENVIRONMENT VARIABLES:
  NAMESPACE              Kubernetes namespace (default: nightscan)
  ENVIRONMENT            Deployment environment: staging|production (default: staging)
  DOCKER_REGISTRY        Docker registry URL (default: ghcr.io)
  IMAGE_NAME             Docker image name (default: gamerxtrem/nightscan)
  VERSION                Image version/tag (default: latest)
  DEPLOYMENT_STRATEGY    Deployment strategy: rolling|blue-green|canary (default: rolling)
  HEALTH_CHECK_TIMEOUT   Health check timeout in seconds (default: 300)
  ROLLBACK_ON_FAILURE    Rollback on deployment failure (default: true)

OPTIONS:
  -h, --help             Show this help message
  -v, --version VERSION  Set deployment version
  -e, --env ENVIRONMENT  Set environment (staging|production)
  -s, --strategy STRATEGY Set deployment strategy
  -n, --namespace NS     Set Kubernetes namespace
  --dry-run              Show what would be deployed without actually deploying
  --skip-tests           Skip post-deployment tests
  --force                Force deployment even if health checks fail

EXAMPLES:
  # Deploy to staging
  $0 --env staging --version v1.2.3
  
  # Blue-green production deployment
  $0 --env production --strategy blue-green --version v1.2.3
  
  # Dry run to see what would be deployed
  $0 --dry-run --env production

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            usage
            exit 0
            ;;
        -v|--version)
            VERSION="$2"
            shift 2
            ;;
        -e|--env)
            ENVIRONMENT="$2"
            shift 2
            ;;
        -s|--strategy)
            DEPLOYMENT_STRATEGY="$2"
            shift 2
            ;;
        -n|--namespace)
            NAMESPACE="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --skip-tests)
            SKIP_TESTS=true
            shift
            ;;
        --force)
            FORCE_DEPLOY=true
            shift
            ;;
        *)
            error "Unknown option: $1"
            ;;
    esac
done

# Validate environment
if [[ "$ENVIRONMENT" != "staging" && "$ENVIRONMENT" != "production" ]]; then
    error "Environment must be 'staging' or 'production', got: $ENVIRONMENT"
fi

# Adjust namespace for environment
if [[ "$ENVIRONMENT" == "production" ]]; then
    NAMESPACE="${NAMESPACE}-production"
else
    NAMESPACE="${NAMESPACE}-staging"
fi

# Full image name
FULL_IMAGE_NAME="${DOCKER_REGISTRY}/${IMAGE_NAME}:${VERSION}"

log "🚀 Starting NightScan Enhanced Deployment"
info "Environment: $ENVIRONMENT"
info "Namespace: $NAMESPACE"
info "Image: $FULL_IMAGE_NAME"
info "Strategy: $DEPLOYMENT_STRATEGY"
info "Dry Run: ${DRY_RUN:-false}"

# Check dependencies
check_dependencies() {
    log "🔍 Checking dependencies..."
    
    local required_tools=("kubectl" "docker" "jq")
    for tool in "${required_tools[@]}"; do
        if ! command -v "$tool" >/dev/null 2>&1; then
            error "$tool is required but not installed"
        fi
    done
    
    # Check kubectl connection
    if ! kubectl cluster-info >/dev/null 2>&1; then
        error "Cannot connect to Kubernetes cluster"
    fi
    
    # Check if namespace exists
    if ! kubectl get namespace "$NAMESPACE" >/dev/null 2>&1; then
        warn "Namespace $NAMESPACE does not exist, creating it..."
        if [[ "$DRY_RUN" != "true" ]]; then
            kubectl create namespace "$NAMESPACE"
        fi
    fi
    
    success "Dependencies check passed"
}

# Pre-deployment security check
pre_deployment_security_check() {
    log "🔒 Running pre-deployment security checks..."
    
    # Run security validation script if it exists
    if [[ -f "$PROJECT_ROOT/scripts/security-check.sh" ]]; then
        info "Running NightScan security validation..."
        bash "$PROJECT_ROOT/scripts/security-check.sh" || {
            if [[ "$FORCE_DEPLOY" != "true" ]]; then
                error "Security checks failed. Use --force to override."
            else
                warn "Security checks failed but deployment forced"
            fi
        }
    fi
    
    # Check for External Secrets Operator
    if kubectl get crd externalsecrets.external-secrets.io >/dev/null 2>&1; then
        info "External Secrets Operator is available"
        
        # Verify external secrets are synced
        if kubectl get externalsecrets -n "$NAMESPACE" >/dev/null 2>&1; then
            info "External secrets configured for namespace $NAMESPACE"
        else
            warn "No external secrets found for namespace $NAMESPACE"
        fi
    else
        warn "External Secrets Operator not found"
    fi
    
    success "Security checks completed"
}

# Pull and verify image
verify_image() {
    log "📦 Verifying Docker image..."
    
    if [[ "$DRY_RUN" == "true" ]]; then
        info "Dry run: Would verify image $FULL_IMAGE_NAME"
        return 0
    fi
    
    # Try to pull the image
    if ! docker pull "$FULL_IMAGE_NAME" >/dev/null 2>&1; then
        error "Failed to pull image: $FULL_IMAGE_NAME"
    fi
    
    # Get image info
    local image_info
    image_info=$(docker inspect "$FULL_IMAGE_NAME" --format='{{.Created}}' 2>/dev/null || echo "unknown")
    info "Image created: $image_info"
    
    # Security scan (if trivy is available)
    if command -v trivy >/dev/null 2>&1; then
        info "Running security scan on image..."
        trivy image --exit-code 1 --severity HIGH,CRITICAL "$FULL_IMAGE_NAME" || {
            if [[ "$FORCE_DEPLOY" != "true" ]]; then
                error "Image security scan failed. Use --force to override."
            else
                warn "Image security scan failed but deployment forced"
            fi
        }
    fi
    
    success "Image verification completed"
}

# Deploy using rolling update strategy
deploy_rolling() {
    log "🔄 Deploying with rolling update strategy..."
    
    local manifests_dir="$PROJECT_ROOT/k8s"
    
    # Update image in manifests
    local temp_dir
    temp_dir=$(mktemp -d)
    
    # Copy and update manifests
    cp -r "$manifests_dir"/* "$temp_dir/"
    
    # Update image references
    find "$temp_dir" -name '*.yaml' -o -name '*.yml' | while read -r file; do
        if grep -q "image:" "$file"; then
            sed -i.bak "s|image: .*nightscan.*|image: $FULL_IMAGE_NAME|g" "$file"
            rm -f "${file}.bak"
        fi
    done
    
    if [[ "$DRY_RUN" == "true" ]]; then
        info "Dry run: Would apply the following manifests:"
        find "$temp_dir" -name '*.yaml' -o -name '*.yml' | sort
        rm -rf "$temp_dir"
        return 0
    fi
    
    # Apply External Secrets first
    if [[ -f "$temp_dir/secrets-management.yaml" ]]; then
        info "Applying External Secrets configuration..."
        kubectl apply -f "$temp_dir/secrets-management.yaml" -n "$NAMESPACE"
    fi
    
    # Apply other manifests
    info "Applying Kubernetes manifests..."
    kubectl apply -f "$temp_dir" -n "$NAMESPACE"
    
    # Wait for rollout to complete
    info "Waiting for deployment rollout..."
    kubectl rollout status deployment/web-app -n "$NAMESPACE" --timeout=${HEALTH_CHECK_TIMEOUT}s
    kubectl rollout status deployment/prediction-api -n "$NAMESPACE" --timeout=${HEALTH_CHECK_TIMEOUT}s || true
    
    rm -rf "$temp_dir"
    success "Rolling deployment completed"
}

# Deploy using blue-green strategy
deploy_blue_green() {
    log "🔵🟢 Deploying with blue-green strategy..."
    
    local current_color
    current_color=$(kubectl get service web-app -n "$NAMESPACE" -o jsonpath='{.spec.selector.color}' 2>/dev/null || echo "blue")
    
    local new_color
    if [[ "$current_color" == "blue" ]]; then
        new_color="green"
    else
        new_color="blue"
    fi
    
    info "Current color: $current_color, deploying to: $new_color"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        info "Dry run: Would deploy to $new_color environment"
        return 0
    fi
    
    # Deploy to new color
    local temp_dir
    temp_dir=$(mktemp -d)
    cp -r "$PROJECT_ROOT/k8s"/* "$temp_dir/"
    
    # Update manifests for new color
    find "$temp_dir" -name '*.yaml' -o -name '*.yml' | while read -r file; do
        sed -i.bak "s|image: .*nightscan.*|image: $FULL_IMAGE_NAME|g" "$file"
        sed -i.bak "s|color: .*|color: $new_color|g" "$file"
        sed -i.bak "s|nightscan-[a-z]*|nightscan-$new_color|g" "$file"
        rm -f "${file}.bak"
    done
    
    # Deploy new version
    kubectl apply -f "$temp_dir" -n "$NAMESPACE"
    
    # Wait for new deployment to be ready
    kubectl rollout status deployment/web-app-${new_color} -n "$NAMESPACE" --timeout=${HEALTH_CHECK_TIMEOUT}s
    
    # Run health checks on new deployment
    if run_health_checks "$new_color"; then
        info "Switching traffic to $new_color..."
        
        # Update service to point to new color
        kubectl patch service web-app -n "$NAMESPACE" -p '{"spec":{"selector":{"color":"'"$new_color"'"}}}'
        
        # Wait a bit then cleanup old deployment
        sleep 30
        kubectl delete deployment "web-app-${current_color}" -n "$NAMESPACE" --ignore-not-found=true
        
        success "Blue-green deployment completed successfully"
    else
        error "Health checks failed for $new_color deployment"
    fi
    
    rm -rf "$temp_dir"
}

# Run health checks
run_health_checks() {
    local color="${1:-}"
    log "🏥 Running health checks..."
    
    if [[ "$SKIP_TESTS" == "true" ]]; then
        info "Skipping health checks"
        return 0
    fi
    
    if [[ "$DRY_RUN" == "true" ]]; then
        info "Dry run: Would run health checks"
        return 0
    fi
    
    local service_name="web-app"
    if [[ -n "$color" ]]; then
        service_name="web-app-${color}"
    fi
    
    # Wait for pods to be ready
    local max_attempts=30
    local attempt=0
    
    while [[ $attempt -lt $max_attempts ]]; do
        local ready_pods
        ready_pods=$(kubectl get pods -n "$NAMESPACE" -l app=web-app -o jsonpath='{.items[*].status.containerStatuses[*].ready}' | grep -o true | wc -l || echo 0)
        
        if [[ $ready_pods -gt 0 ]]; then
            info "Found $ready_pods ready pods"
            break
        fi
        
        info "Waiting for pods to be ready... (attempt $((attempt + 1))/$max_attempts)"
        sleep 10
        ((attempt++))
    done
    
    if [[ $attempt -eq $max_attempts ]]; then
        error "Timed out waiting for pods to be ready"
    fi
    
    # Test health endpoints
    info "Testing health endpoints..."
    
    # Port forward to test locally
    local port_forward_pid
    kubectl port-forward -n "$NAMESPACE" service/"$service_name" 8080:80 >/dev/null 2>&1 &
    port_forward_pid=$!
    
    sleep 5
    
    local health_check_passed=true
    
    # Test health endpoint
    if ! curl -f http://localhost:8080/health >/dev/null 2>&1; then
        warn "Health endpoint check failed"
        health_check_passed=false
    fi
    
    # Test ready endpoint
    if ! curl -f http://localhost:8080/ready >/dev/null 2>&1; then
        warn "Ready endpoint check failed"
        health_check_passed=false
    fi
    
    # Cleanup port forward
    kill $port_forward_pid 2>/dev/null || true
    
    if [[ "$health_check_passed" == "true" ]]; then
        success "Health checks passed"
        return 0
    else
        error "Health checks failed"
        return 1
    fi
}

# Rollback deployment
rollback_deployment() {
    log "⏪ Rolling back deployment..."
    
    if [[ "$DRY_RUN" == "true" ]]; then
        info "Dry run: Would rollback deployment"
        return 0
    fi
    
    # Rollback deployments
    kubectl rollout undo deployment/web-app -n "$NAMESPACE"
    kubectl rollout undo deployment/prediction-api -n "$NAMESPACE" || true
    
    # Wait for rollback to complete
    kubectl rollout status deployment/web-app -n "$NAMESPACE" --timeout=300s
    
    success "Rollback completed"
}

# Post-deployment tasks
post_deployment_tasks() {
    log "📋 Running post-deployment tasks..."
    
    if [[ "$DRY_RUN" == "true" ]]; then
        info "Dry run: Would run post-deployment tasks"
        return 0
    fi
    
    # Update deployment annotations
    kubectl annotate deployment web-app -n "$NAMESPACE" \
        deployment.kubernetes.io/revision-history-limit=10 \
        nightscan.io/deployed-by="enhanced-deploy-script" \
        nightscan.io/deployed-at="$(date -Iseconds)" \
        nightscan.io/image="$FULL_IMAGE_NAME" \
        nightscan.io/environment="$ENVIRONMENT" \
        --overwrite
    
    # Cleanup old ReplicaSets (keep last 3)
    kubectl delete replicaset -n "$NAMESPACE" \
        $(kubectl get replicaset -n "$NAMESPACE" -o jsonpath='{.items[?(@.spec.replicas==0)].metadata.name}' | cut -d' ' -f4-) \
        2>/dev/null || true
    
    success "Post-deployment tasks completed"
}

# Generate deployment report
generate_deployment_report() {
    log "📊 Generating deployment report..."
    
    local report_file="deployment-report-$(date +%Y%m%d-%H%M%S).json"
    
    cat > "$report_file" << EOF
{
  "deployment": {
    "timestamp": "$(date -Iseconds)",
    "environment": "$ENVIRONMENT",
    "namespace": "$NAMESPACE",
    "strategy": "$DEPLOYMENT_STRATEGY",
    "image": "$FULL_IMAGE_NAME",
    "version": "$VERSION",
    "dry_run": ${DRY_RUN:-false}
  },
  "kubernetes": {
    "cluster": "$(kubectl config current-context)",
    "server": "$(kubectl config view --minify -o jsonpath='{.clusters[0].cluster.server}')"
  },
  "status": "completed"
}
EOF
    
    info "Deployment report saved: $report_file"
    
    # Also output to stdout for CI/CD
    if [[ -n "$GITHUB_STEP_SUMMARY" ]]; then
        echo "## 🚀 Deployment Summary" >> "$GITHUB_STEP_SUMMARY"
        echo "" >> "$GITHUB_STEP_SUMMARY"
        echo "- **Environment**: $ENVIRONMENT" >> "$GITHUB_STEP_SUMMARY"
        echo "- **Namespace**: $NAMESPACE" >> "$GITHUB_STEP_SUMMARY"
        echo "- **Image**: $FULL_IMAGE_NAME" >> "$GITHUB_STEP_SUMMARY"
        echo "- **Strategy**: $DEPLOYMENT_STRATEGY" >> "$GITHUB_STEP_SUMMARY"
        echo "- **Deployed**: $(date)" >> "$GITHUB_STEP_SUMMARY"
        echo "" >> "$GITHUB_STEP_SUMMARY"
    fi
}

# Main deployment function
main() {
    local start_time
    start_time=$(date +%s)
    
    # Run deployment steps
    check_dependencies
    pre_deployment_security_check
    verify_image
    
    # Choose deployment strategy
    case "$DEPLOYMENT_STRATEGY" in
        "rolling")
            deploy_rolling
            ;;
        "blue-green")
            deploy_blue_green
            ;;
        "canary")
            error "Canary deployment not yet implemented"
            ;;
        *)
            error "Unknown deployment strategy: $DEPLOYMENT_STRATEGY"
            ;;
    esac
    
    # Run health checks
    if ! run_health_checks; then
        if [[ "$ROLLBACK_ON_FAILURE" == "true" ]]; then
            warn "Health checks failed, rolling back..."
            rollback_deployment
        else
            error "Health checks failed and rollback disabled"
        fi
    fi
    
    # Post-deployment tasks
    post_deployment_tasks
    generate_deployment_report
    
    local end_time
    end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    success "✅ NightScan deployment completed successfully in ${duration}s"
    
    if [[ "$DRY_RUN" != "true" ]]; then
        info "🔗 Check your application at:"
        if [[ "$ENVIRONMENT" == "production" ]]; then
            info "   https://nightscan.example.com"
        else
            info "   https://staging.nightscan.example.com"
        fi
    fi
}

# Run main function
main "$@"
