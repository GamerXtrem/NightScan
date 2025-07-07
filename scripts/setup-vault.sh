#!/bin/bash

# Setup HashiCorp Vault for NightScan Secrets Management
# This script configures Vault for secure secret storage and rotation

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
VAULT_VERSION="1.15.4"
VAULT_ADDR="${VAULT_ADDR:-http://127.0.0.1:8200}"
VAULT_CONFIG_DIR="$PROJECT_ROOT/vault"
VAULT_DATA_DIR="$VAULT_CONFIG_DIR/data"
VAULT_LOGS_DIR="$VAULT_CONFIG_DIR/logs"

# Logging functions
log() {
    echo -e "${BLUE}[$(date +'%H:%M:%S')] $1${NC}"
}

success() {
    echo -e "${GREEN}[SUCCESS] $1${NC}"
}

error() {
    echo -e "${RED}[ERROR] $1${NC}"
    exit 1
}

warn() {
    echo -e "${YELLOW}[WARNING] $1${NC}"
}

# Check if Vault is installed
check_vault_installed() {
    if command -v vault >/dev/null 2>&1; then
        log "Vault already installed: $(vault version)"
        return 0
    fi
    return 1
}

# Install Vault
install_vault() {
    log "Installing HashiCorp Vault v${VAULT_VERSION}..."
    
    # Detect OS
    OS=$(uname -s | tr '[:upper:]' '[:lower:]')
    ARCH=$(uname -m)
    
    case "$ARCH" in
        x86_64) ARCH="amd64" ;;
        aarch64) ARCH="arm64" ;;
        *) error "Unsupported architecture: $ARCH" ;;
    esac
    
    # Download Vault
    VAULT_URL="https://releases.hashicorp.com/vault/${VAULT_VERSION}/vault_${VAULT_VERSION}_${OS}_${ARCH}.zip"
    
    log "Downloading Vault from $VAULT_URL..."
    curl -sSL "$VAULT_URL" -o /tmp/vault.zip
    
    # Install
    unzip -o /tmp/vault.zip -d /tmp/
    sudo mv /tmp/vault /usr/local/bin/
    sudo chmod +x /usr/local/bin/vault
    
    # Cleanup
    rm -f /tmp/vault.zip
    
    success "Vault installed successfully"
}

# Create Vault configuration
create_vault_config() {
    log "Creating Vault configuration..."
    
    mkdir -p "$VAULT_CONFIG_DIR" "$VAULT_DATA_DIR" "$VAULT_LOGS_DIR"
    
    # Create Vault config file
    cat > "$VAULT_CONFIG_DIR/config.hcl" <<EOF
# Vault configuration for NightScan

ui = true
disable_mlock = true

storage "file" {
  path = "${VAULT_DATA_DIR}"
}

listener "tcp" {
  address       = "127.0.0.1:8200"
  tls_disable   = "true"  # Use reverse proxy for TLS
}

api_addr = "${VAULT_ADDR}"

log_level = "info"
log_file = "${VAULT_LOGS_DIR}/vault.log"
log_rotate_duration = "24h"
log_rotate_max_files = 7

telemetry {
  prometheus_retention_time = "30s"
  disable_hostname = true
}

# Default lease duration
default_lease_ttl = "768h"  # 32 days
max_lease_ttl = "8760h"     # 365 days
EOF

    # Create systemd service
    sudo tee /etc/systemd/system/vault.service > /dev/null <<EOF
[Unit]
Description=HashiCorp Vault - Secret Management
Documentation=https://www.vaultproject.io/docs/
Requires=network-online.target
After=network-online.target
ConditionFileNotEmpty=${VAULT_CONFIG_DIR}/config.hcl

[Service]
Type=notify
User=vault
Group=vault
ProtectSystem=full
ProtectHome=read-only
PrivateTmp=yes
PrivateDevices=yes
SecureBits=keep-caps
NoNewPrivileges=yes
AmbientCapabilities=CAP_IPC_LOCK
CapabilityBoundingSet=CAP_SYSLOG CAP_IPC_LOCK
ExecStart=/usr/local/bin/vault server -config=${VAULT_CONFIG_DIR}/config.hcl
ExecReload=/bin/kill --signal HUP \$MAINPID
KillMode=process
Restart=on-failure
RestartSec=5
TimeoutStopSec=30
LimitNOFILE=65536
LimitMEMLOCK=infinity

[Install]
WantedBy=multi-user.target
EOF

    # Create vault user
    if ! id -u vault >/dev/null 2>&1; then
        sudo useradd --system --home /var/lib/vault --shell /bin/false vault
    fi
    
    # Set permissions
    sudo chown -R vault:vault "$VAULT_CONFIG_DIR"
    sudo chmod 700 "$VAULT_DATA_DIR"
    
    success "Vault configuration created"
}

# Initialize Vault
initialize_vault() {
    log "Initializing Vault..."
    
    # Start Vault
    sudo systemctl daemon-reload
    sudo systemctl enable vault
    sudo systemctl start vault
    
    sleep 5
    
    # Check if already initialized
    if vault status 2>/dev/null | grep -q "Initialized.*true"; then
        warn "Vault already initialized"
        return 0
    fi
    
    # Initialize with 5 key shares, 3 required to unseal
    INIT_OUTPUT=$(vault operator init -key-shares=5 -key-threshold=3 -format=json)
    
    # Save keys securely
    echo "$INIT_OUTPUT" > "$VAULT_CONFIG_DIR/init-keys.json"
    chmod 600 "$VAULT_CONFIG_DIR/init-keys.json"
    
    # Extract root token
    ROOT_TOKEN=$(echo "$INIT_OUTPUT" | jq -r '.root_token')
    echo "$ROOT_TOKEN" > "$VAULT_CONFIG_DIR/.vault-token"
    chmod 600 "$VAULT_CONFIG_DIR/.vault-token"
    
    # Auto-unseal with first 3 keys (for dev/test only)
    echo "$INIT_OUTPUT" | jq -r '.unseal_keys_b64[:3][]' | while read -r KEY; do
        vault operator unseal "$KEY"
    done
    
    success "Vault initialized and unsealed"
    warn "IMPORTANT: Save the unseal keys and root token from $VAULT_CONFIG_DIR/init-keys.json"
}

# Configure Vault for NightScan
configure_nightscan_vault() {
    log "Configuring Vault for NightScan..."
    
    # Login with root token
    export VAULT_TOKEN=$(cat "$VAULT_CONFIG_DIR/.vault-token")
    
    # Enable audit logging
    vault audit enable file file_path="$VAULT_LOGS_DIR/audit.log" || true
    
    # Enable KV v2 secrets engine
    vault secrets enable -path=nightscan -version=2 kv || true
    
    # Create policies
    cat > "$VAULT_CONFIG_DIR/nightscan-app-policy.hcl" <<EOF
# Policy for NightScan application

# Read secrets
path "nightscan/data/app/*" {
  capabilities = ["read", "list"]
}

# Read database credentials
path "nightscan/data/database/*" {
  capabilities = ["read"]
}

# Read API keys
path "nightscan/data/api-keys/*" {
  capabilities = ["read"]
}

# Token self-renewal
path "auth/token/renew-self" {
  capabilities = ["update"]
}
EOF

    vault policy write nightscan-app "$VAULT_CONFIG_DIR/nightscan-app-policy.hcl"
    
    # Create admin policy
    cat > "$VAULT_CONFIG_DIR/nightscan-admin-policy.hcl" <<EOF
# Policy for NightScan administrators

# Full access to nightscan secrets
path "nightscan/*" {
  capabilities = ["create", "read", "update", "delete", "list"]
}

# Manage policies
path "sys/policies/*" {
  capabilities = ["create", "read", "update", "delete", "list"]
}

# Manage auth methods
path "sys/auth/*" {
  capabilities = ["create", "read", "update", "delete", "list"]
}

# Audit log access
path "sys/audit/*" {
  capabilities = ["read", "list"]
}
EOF

    vault policy write nightscan-admin "$VAULT_CONFIG_DIR/nightscan-admin-policy.hcl"
    
    # Enable AppRole auth for applications
    vault auth enable approle || true
    
    # Create AppRole for NightScan app
    vault write auth/approle/role/nightscan \
        token_policies="nightscan-app" \
        token_ttl=1h \
        token_max_ttl=4h \
        secret_id_ttl=24h \
        secret_id_num_uses=0
    
    # Get role ID
    ROLE_ID=$(vault read -field=role_id auth/approle/role/nightscan/role-id)
    echo "$ROLE_ID" > "$VAULT_CONFIG_DIR/nightscan-role-id"
    
    # Generate secret ID
    SECRET_ID=$(vault write -field=secret_id -f auth/approle/role/nightscan/secret-id)
    echo "$SECRET_ID" > "$VAULT_CONFIG_DIR/nightscan-secret-id"
    
    success "Vault configured for NightScan"
}

# Store initial secrets
store_initial_secrets() {
    log "Storing initial NightScan secrets in Vault..."
    
    # Database credentials
    vault kv put nightscan/database/postgres \
        username="nightscan" \
        password="$(openssl rand -base64 32)" \
        host="postgres" \
        port="5432" \
        database="nightscan" \
        ssl_mode="require"
    
    # Redis credentials
    vault kv put nightscan/database/redis \
        password="$(openssl rand -base64 32)" \
        host="redis" \
        port="6379" \
        db="0"
    
    # Application secrets
    vault kv put nightscan/app/secrets \
        secret_key="$(openssl rand -base64 64)" \
        csrf_secret_key="$(openssl rand -base64 32)" \
        jwt_secret="$(openssl rand -base64 32)"
    
    # API keys (placeholders)
    vault kv put nightscan/api-keys/external \
        smtp_password="CONFIGURE_SMTP_PASSWORD" \
        aws_access_key="CONFIGURE_AWS_ACCESS_KEY" \
        aws_secret_key="CONFIGURE_AWS_SECRET_KEY" \
        grafana_api_key="$(openssl rand -base64 32)"
    
    # Encryption keys
    vault kv put nightscan/encryption/keys \
        data_encryption_key="$(openssl rand -base64 32)" \
        file_encryption_key="$(openssl rand -base64 32)"
    
    success "Initial secrets stored in Vault"
}

# Create secret rotation script
create_rotation_script() {
    log "Creating secret rotation script..."
    
    cat > "$PROJECT_ROOT/scripts/rotate-secrets.sh" <<'EOF'
#!/bin/bash

# Rotate NightScan secrets in Vault

set -euo pipefail

VAULT_ADDR="${VAULT_ADDR:-http://127.0.0.1:8200}"
ROTATION_LOG="/var/log/nightscan/secret-rotation.log"

log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1" | tee -a "$ROTATION_LOG"
}

# Rotate database password
rotate_database_password() {
    log "Rotating database password..."
    
    NEW_PASSWORD=$(openssl rand -base64 32)
    
    # Update in Vault
    vault kv patch nightscan/database/postgres password="$NEW_PASSWORD"
    
    # Update in PostgreSQL (requires psql)
    PGPASSWORD=$(vault kv get -field=password nightscan/database/postgres) \
    psql -h postgres -U nightscan -d nightscan -c \
        "ALTER USER nightscan PASSWORD '$NEW_PASSWORD';"
    
    log "Database password rotated successfully"
}

# Rotate Redis password
rotate_redis_password() {
    log "Rotating Redis password..."
    
    NEW_PASSWORD=$(openssl rand -base64 32)
    
    # Update in Vault
    vault kv patch nightscan/database/redis password="$NEW_PASSWORD"
    
    # Update in Redis
    redis-cli -h redis CONFIG SET requirepass "$NEW_PASSWORD"
    redis-cli -h redis CONFIG REWRITE
    
    log "Redis password rotated successfully"
}

# Rotate application secrets
rotate_app_secrets() {
    log "Rotating application secrets..."
    
    vault kv patch nightscan/app/secrets \
        secret_key="$(openssl rand -base64 64)" \
        jwt_secret="$(openssl rand -base64 32)"
    
    # Signal application to reload
    docker-compose -f docker-compose.production.yml restart web prediction-api
    
    log "Application secrets rotated successfully"
}

# Main rotation
main() {
    log "Starting secret rotation..."
    
    rotate_database_password
    rotate_redis_password
    rotate_app_secrets
    
    log "Secret rotation completed"
}

# Run with lock to prevent concurrent rotations
(
    flock -n 200 || { echo "Rotation already in progress"; exit 1; }
    main
) 200>/var/lock/nightscan-secret-rotation.lock
EOF

    chmod +x "$PROJECT_ROOT/scripts/rotate-secrets.sh"
    
    # Create systemd timer for automatic rotation
    sudo tee /etc/systemd/system/nightscan-secret-rotation.timer > /dev/null <<EOF
[Unit]
Description=NightScan Secret Rotation Timer
Requires=nightscan-secret-rotation.service

[Timer]
OnCalendar=monthly
Persistent=true

[Install]
WantedBy=timers.target
EOF

    sudo tee /etc/systemd/system/nightscan-secret-rotation.service > /dev/null <<EOF
[Unit]
Description=NightScan Secret Rotation
After=vault.service

[Service]
Type=oneshot
ExecStart=$PROJECT_ROOT/scripts/rotate-secrets.sh
User=nightscan
Group=nightscan
StandardOutput=journal
StandardError=journal
EOF

    sudo systemctl daemon-reload
    sudo systemctl enable nightscan-secret-rotation.timer
    
    success "Secret rotation configured"
}

# Create Vault client wrapper
create_vault_client() {
    log "Creating Vault client wrapper..."
    
    cat > "$PROJECT_ROOT/vault/vault-client.py" <<'EOF'
#!/usr/bin/env python3
"""
Vault client wrapper for NightScan
Handles authentication and secret retrieval
"""

import os
import sys
import json
import time
import logging
from typing import Dict, Any, Optional
import hvac
from functools import lru_cache

logger = logging.getLogger(__name__)

class VaultClient:
    def __init__(self, vault_addr: str = None, role_id: str = None, secret_id: str = None):
        self.vault_addr = vault_addr or os.environ.get('VAULT_ADDR', 'http://127.0.0.1:8200')
        self.role_id = role_id or os.environ.get('VAULT_ROLE_ID')
        self.secret_id = secret_id or os.environ.get('VAULT_SECRET_ID')
        self.client = None
        self.token_expiry = 0
        self._authenticate()
    
    def _authenticate(self):
        """Authenticate with Vault using AppRole"""
        try:
            self.client = hvac.Client(url=self.vault_addr)
            
            if self.role_id and self.secret_id:
                # AppRole authentication
                response = self.client.auth.approle.login(
                    role_id=self.role_id,
                    secret_id=self.secret_id
                )
                self.token_expiry = time.time() + response['auth']['lease_duration']
            else:
                # Try token from file
                token_file = os.path.expanduser('~/.vault-token')
                if os.path.exists(token_file):
                    with open(token_file, 'r') as f:
                        self.client.token = f.read().strip()
            
            if not self.client.is_authenticated():
                raise Exception("Failed to authenticate with Vault")
                
            logger.info("Successfully authenticated with Vault")
            
        except Exception as e:
            logger.error(f"Vault authentication failed: {e}")
            raise
    
    def _ensure_authenticated(self):
        """Ensure we have a valid token"""
        if time.time() > self.token_expiry - 300:  # Renew 5 min before expiry
            self._authenticate()
    
    @lru_cache(maxsize=128)
    def get_secret(self, path: str, key: str = None) -> Any:
        """Get secret from Vault with caching"""
        self._ensure_authenticated()
        
        try:
            # Read from KV v2
            response = self.client.secrets.kv.v2.read_secret_version(
                path=path,
                mount_point='nightscan'
            )
            
            data = response['data']['data']
            
            if key:
                return data.get(key)
            return data
            
        except Exception as e:
            logger.error(f"Failed to get secret {path}: {e}")
            raise
    
    def get_database_url(self, db_type: str = 'postgres') -> str:
        """Get database connection URL"""
        if db_type == 'postgres':
            creds = self.get_secret('database/postgres')
            return f"postgresql://{creds['username']}:{creds['password']}@{creds['host']}:{creds['port']}/{creds['database']}?sslmode={creds['ssl_mode']}"
        elif db_type == 'redis':
            creds = self.get_secret('database/redis')
            return f"redis://:{creds['password']}@{creds['host']}:{creds['port']}/{creds['db']}"
        else:
            raise ValueError(f"Unknown database type: {db_type}")
    
    def get_app_config(self) -> Dict[str, Any]:
        """Get all application configuration"""
        return {
            'SECRET_KEY': self.get_secret('app/secrets', 'secret_key'),
            'CSRF_SECRET_KEY': self.get_secret('app/secrets', 'csrf_secret_key'),
            'JWT_SECRET': self.get_secret('app/secrets', 'jwt_secret'),
            'DATABASE_URL': self.get_database_url('postgres'),
            'REDIS_URL': self.get_database_url('redis'),
            'DATA_ENCRYPTION_KEY': self.get_secret('encryption/keys', 'data_encryption_key'),
        }

# Singleton instance
_vault_client = None

def get_vault_client() -> VaultClient:
    """Get or create Vault client singleton"""
    global _vault_client
    if _vault_client is None:
        _vault_client = VaultClient()
    return _vault_client

if __name__ == '__main__':
    # Test connection
    client = get_vault_client()
    print("Vault connection successful")
    print(f"Database URL: {client.get_database_url()}")
EOF

    success "Vault client wrapper created"
}

# Main installation flow
main() {
    log "🔐 Setting up HashiCorp Vault for NightScan..."
    
    # Check if Vault is already installed
    if ! check_vault_installed; then
        install_vault
    fi
    
    # Create configuration
    create_vault_config
    
    # Initialize Vault
    initialize_vault
    
    # Configure for NightScan
    configure_nightscan_vault
    
    # Store initial secrets
    store_initial_secrets
    
    # Create rotation script
    create_rotation_script
    
    # Create client wrapper
    create_vault_client
    
    echo ""
    success "✅ Vault setup completed successfully!"
    echo ""
    echo "📋 Important information:"
    echo "   - Vault UI: http://localhost:8200"
    echo "   - Unseal keys: $VAULT_CONFIG_DIR/init-keys.json"
    echo "   - Root token: $VAULT_CONFIG_DIR/.vault-token"
    echo "   - App Role ID: $VAULT_CONFIG_DIR/nightscan-role-id"
    echo "   - App Secret ID: $VAULT_CONFIG_DIR/nightscan-secret-id"
    echo ""
    echo "🔐 Next steps:"
    echo "   1. Save unseal keys in a secure location"
    echo "   2. Configure real SMTP/AWS credentials in Vault"
    echo "   3. Update docker-compose to use Vault client"
    echo "   4. Test secret rotation with: ./scripts/rotate-secrets.sh"
    echo ""
}

# Run main function
main