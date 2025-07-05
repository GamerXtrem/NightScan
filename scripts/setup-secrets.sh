#!/bin/bash

# NightScan Secure Secrets Setup Script
# This script securely initializes secrets in HashiCorp Vault or AWS Secrets Manager

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
VAULT_ADDR="${VAULT_ADDR:-http://localhost:8200}"
VAULT_TOKEN="${VAULT_TOKEN:-}"
AWS_REGION="${AWS_REGION:-us-east-1}"
SECRET_BACKEND="${SECRET_BACKEND:-vault}" # vault or aws-secrets-manager

log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

success() {
    echo -e "${GREEN}[SUCCESS] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[WARNING] $1${NC}"
}

error() {
    echo -e "${RED}[ERROR] $1${NC}"
    exit 1
}

# Generate secure random passwords
generate_password() {
    local length="${1:-32}"
    openssl rand -base64 $length | tr -d "=+/" | cut -c1-$length
}

# Generate secret key
generate_secret_key() {
    python3 -c "import secrets; print(secrets.token_urlsafe(32))"
}

# Check dependencies
check_dependencies() {
    log "Checking dependencies..."
    
    if [ "$SECRET_BACKEND" = "vault" ]; then
        command -v vault >/dev/null 2>&1 || error "HashiCorp Vault CLI is required but not installed"
        [ -z "$VAULT_TOKEN" ] && error "VAULT_TOKEN environment variable is required"
    elif [ "$SECRET_BACKEND" = "aws-secrets-manager" ]; then
        command -v aws >/dev/null 2>&1 || error "AWS CLI is required but not installed"
        aws sts get-caller-identity >/dev/null 2>&1 || error "AWS credentials not configured"
    else
        error "Unknown SECRET_BACKEND: $SECRET_BACKEND. Use 'vault' or 'aws-secrets-manager'"
    fi
    
    command -v openssl >/dev/null 2>&1 || error "OpenSSL is required but not installed"
    command -v python3 >/dev/null 2>&1 || error "Python 3 is required but not installed"
    
    success "Dependencies check passed"
}

# Initialize Vault secrets
setup_vault_secrets() {
    log "Setting up secrets in HashiCorp Vault..."
    
    export VAULT_ADDR="$VAULT_ADDR"
    export VAULT_TOKEN="$VAULT_TOKEN"
    
    # Enable KV secrets engine if not already enabled
    vault secrets list | grep -q "secret/" || vault secrets enable -path=secret kv-v2
    
    # Generate secrets
    local db_password=$(generate_password 24)
    local redis_password=$(generate_password 24)
    local secret_key=$(generate_secret_key)
    local csrf_secret=$(generate_secret_key)
    local jwt_secret=$(generate_secret_key)
    local encryption_key=$(generate_secret_key)
    
    # Store database secrets
    vault kv put secret/nightscan/postgres \
        username="nightscan" \
        password="$db_password" \
        database="nightscan"
    
    # Store Redis secrets
    vault kv put secret/nightscan/redis \
        password="$redis_password"
    
    # Store application secrets
    vault kv put secret/nightscan/app \
        secret_key="$secret_key" \
        csrf_secret_key="$csrf_secret" \
        jwt_secret="$jwt_secret" \
        encryption_key="$encryption_key"
    
    # Store API keys (user will need to update these)
    vault kv put secret/nightscan/aws \
        access_key_id="CHANGE_ME" \
        secret_access_key="CHANGE_ME"
    
    vault kv put secret/nightscan/external-apis \
        openai_api_key="CHANGE_ME"
    
    vault kv put secret/nightscan/notifications \
        slack_webhook_url="CHANGE_ME"
    
    vault kv put secret/nightscan/smtp \
        username="noreply@nightscan.com" \
        password="CHANGE_ME" \
        host="smtp.gmail.com" \
        port="587"
    
    success "Vault secrets configured successfully"
    
    # Display summary
    cat << EOF

${GREEN}=== VAULT SECRETS SUMMARY ===${NC}

✅ Database credentials stored at: secret/nightscan/postgres
✅ Redis credentials stored at: secret/nightscan/redis  
✅ Application secrets stored at: secret/nightscan/app
✅ AWS credentials template at: secret/nightscan/aws
✅ External API keys template at: secret/nightscan/external-apis
✅ Notification config template at: secret/nightscan/notifications
✅ SMTP config template at: secret/nightscan/smtp

${YELLOW}⚠️  ACTION REQUIRED:${NC}
Update the following secrets with real values:
- AWS credentials: vault kv put secret/nightscan/aws access_key_id="..." secret_access_key="..."
- OpenAI API key: vault kv put secret/nightscan/external-apis openai_api_key="..."
- Slack webhook: vault kv put secret/nightscan/notifications slack_webhook_url="..."
- SMTP password: vault kv put secret/nightscan/smtp password="..."

EOF
}

# Initialize AWS Secrets Manager secrets
setup_aws_secrets() {
    log "Setting up secrets in AWS Secrets Manager..."
    
    # Generate secrets
    local db_password=$(generate_password 24)
    local redis_password=$(generate_password 24)
    local secret_key=$(generate_secret_key)
    local csrf_secret=$(generate_secret_key)
    local jwt_secret=$(generate_secret_key)
    local encryption_key=$(generate_secret_key)
    
    # Store database secrets
    aws secretsmanager create-secret \
        --region "$AWS_REGION" \
        --name "nightscan/postgres" \
        --description "NightScan PostgreSQL credentials" \
        --secret-string "{\"username\":\"nightscan\",\"password\":\"$db_password\",\"database\":\"nightscan\"}" \
        2>/dev/null || \
    aws secretsmanager update-secret \
        --region "$AWS_REGION" \
        --secret-id "nightscan/postgres" \
        --secret-string "{\"username\":\"nightscan\",\"password\":\"$db_password\",\"database\":\"nightscan\"}"
    
    # Store Redis secrets
    aws secretsmanager create-secret \
        --region "$AWS_REGION" \
        --name "nightscan/redis" \
        --description "NightScan Redis credentials" \
        --secret-string "{\"password\":\"$redis_password\"}" \
        2>/dev/null || \
    aws secretsmanager update-secret \
        --region "$AWS_REGION" \
        --secret-id "nightscan/redis" \
        --secret-string "{\"password\":\"$redis_password\"}"
    
    # Store application secrets
    aws secretsmanager create-secret \
        --region "$AWS_REGION" \
        --name "nightscan/app" \
        --description "NightScan application secrets" \
        --secret-string "{\"secret_key\":\"$secret_key\",\"csrf_secret_key\":\"$csrf_secret\",\"jwt_secret\":\"$jwt_secret\",\"encryption_key\":\"$encryption_key\"}" \
        2>/dev/null || \
    aws secretsmanager update-secret \
        --region "$AWS_REGION" \
        --secret-id "nightscan/app" \
        --secret-string "{\"secret_key\":\"$secret_key\",\"csrf_secret_key\":\"$csrf_secret\",\"jwt_secret\":\"$jwt_secret\",\"encryption_key\":\"$encryption_key\"}"
    
    # Store API keys templates
    aws secretsmanager create-secret \
        --region "$AWS_REGION" \
        --name "nightscan/aws" \
        --description "NightScan AWS credentials" \
        --secret-string "{\"access_key_id\":\"CHANGE_ME\",\"secret_access_key\":\"CHANGE_ME\"}" \
        2>/dev/null || true
    
    aws secretsmanager create-secret \
        --region "$AWS_REGION" \
        --name "nightscan/external-apis" \
        --description "NightScan external API keys" \
        --secret-string "{\"openai_api_key\":\"CHANGE_ME\"}" \
        2>/dev/null || true
    
    success "AWS Secrets Manager secrets configured successfully"
    
    # Display summary
    cat << EOF

${GREEN}=== AWS SECRETS MANAGER SUMMARY ===${NC}

✅ Database credentials stored: nightscan/postgres
✅ Redis credentials stored: nightscan/redis
✅ Application secrets stored: nightscan/app
✅ AWS credentials template: nightscan/aws
✅ External API keys template: nightscan/external-apis

${YELLOW}⚠️  ACTION REQUIRED:${NC}
Update the following secrets with real values:
- AWS credentials: aws secretsmanager update-secret --secret-id nightscan/aws --secret-string '{"access_key_id":"...","secret_access_key":"..."}'
- OpenAI API: aws secretsmanager update-secret --secret-id nightscan/external-apis --secret-string '{"openai_api_key":"..."}'

EOF
}

# Setup Kubernetes RBAC for External Secrets
setup_kubernetes_rbac() {
    log "Setting up Kubernetes RBAC for External Secrets..."
    
    cat << EOF | kubectl apply -f -
apiVersion: v1
kind: ServiceAccount
metadata:
  name: external-secrets
  namespace: external-secrets
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: external-secrets-controller
rules:
- apiGroups: [""]
  resources: ["secrets"]
  verbs: ["create", "update", "delete", "get", "list", "watch"]
- apiGroups: [""]
  resources: ["events"]
  verbs: ["create", "patch"]
- apiGroups: ["external-secrets.io"]
  resources: ["*"]
  verbs: ["*"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: external-secrets-controller
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: ClusterRole
  name: external-secrets-controller
subjects:
- kind: ServiceAccount
  name: external-secrets
  namespace: external-secrets
EOF
    
    success "Kubernetes RBAC configured for External Secrets"
}

# Verify secrets are accessible
verify_secrets() {
    log "Verifying secrets accessibility..."
    
    if [ "$SECRET_BACKEND" = "vault" ]; then
        vault kv get secret/nightscan/postgres >/dev/null && success "✅ Database secrets accessible"
        vault kv get secret/nightscan/redis >/dev/null && success "✅ Redis secrets accessible"
        vault kv get secret/nightscan/app >/dev/null && success "✅ Application secrets accessible"
    elif [ "$SECRET_BACKEND" = "aws-secrets-manager" ]; then
        aws secretsmanager get-secret-value --region "$AWS_REGION" --secret-id "nightscan/postgres" >/dev/null && success "✅ Database secrets accessible"
        aws secretsmanager get-secret-value --region "$AWS_REGION" --secret-id "nightscan/redis" >/dev/null && success "✅ Redis secrets accessible"
        aws secretsmanager get-secret-value --region "$AWS_REGION" --secret-id "nightscan/app" >/dev/null && success "✅ Application secrets accessible"
    fi
}

# Main execution
main() {
    log "=== NightScan Secure Secrets Setup ==="
    log "Backend: $SECRET_BACKEND"
    
    check_dependencies
    
    if [ "$SECRET_BACKEND" = "vault" ]; then
        setup_vault_secrets
    elif [ "$SECRET_BACKEND" = "aws-secrets-manager" ]; then
        setup_aws_secrets
    fi
    
    setup_kubernetes_rbac
    verify_secrets
    
    cat << EOF

${GREEN}=== SETUP COMPLETE ===${NC}

Next steps:
1. Install External Secrets Operator: kubectl apply -f k8s/secrets-management.yaml
2. Update placeholder values for API keys and external services
3. Deploy NightScan with: ./scripts/deploy.sh

${YELLOW}Security Notes:${NC}
- All secrets are now stored securely outside of your codebase
- Rotate secrets regularly using your secret management system
- Monitor secret access through audit logs
- Use principle of least privilege for secret access

EOF
    
    success "Secure secrets setup completed successfully!"
}

# Parse command line arguments
case "${1:-setup}" in
    "setup")
        main
        ;;
    "verify")
        check_dependencies
        verify_secrets
        ;;
    "help")
        echo "Usage: $0 [setup|verify|help]"
        echo ""
        echo "Commands:"
        echo "  setup   - Set up secure secrets (default)"
        echo "  verify  - Verify secrets are accessible"
        echo "  help    - Show this help"
        echo ""
        echo "Environment variables:"
        echo "  SECRET_BACKEND  - 'vault' or 'aws-secrets-manager' (default: vault)"
        echo "  VAULT_ADDR      - Vault server address (default: http://localhost:8200)"
        echo "  VAULT_TOKEN     - Vault authentication token (required for vault)"
        echo "  AWS_REGION      - AWS region (default: us-east-1)"
        ;;
    *)
        error "Unknown command: $1. Use 'help' for usage information."
        ;;
esac