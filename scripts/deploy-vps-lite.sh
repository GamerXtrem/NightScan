#!/bin/bash

# Déploiement optimisé NightScan pour VPS Lite Infomaniak
# Usage: ./deploy-vps-lite.sh [version] [environment]

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
VERSION="${1:-latest}"
ENVIRONMENT="${2:-production}"
COMPOSE_FILE="docker-compose.production.yml"
MONITORING_FILE="docker-compose.monitoring.yml"

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

# Check system requirements for VPS Lite
check_vps_requirements() {
    log "🔍 Vérification des prérequis VPS Lite..."
    
    # Check memory (minimum 4GB)
    TOTAL_MEM=$(free -g | awk '/^Mem:/{print $2}')
    if [ "$TOTAL_MEM" -lt 3 ]; then
        error "Mémoire insuffisante: ${TOTAL_MEM}GB (minimum 4GB requis)"
    fi
    
    # Check available disk space (minimum 10GB free)
    AVAILABLE_DISK=$(df -h / | awk 'NR==2{print $4}' | sed 's/G.*//')
    if [ "$AVAILABLE_DISK" -lt 10 ]; then
        error "Espace disque insuffisant: ${AVAILABLE_DISK}GB (minimum 10GB requis)"
    fi
    
    # Check Docker
    if ! command -v docker >/dev/null 2>&1; then
        error "Docker n'est pas installé"
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose >/dev/null 2>&1; then
        error "Docker Compose n'est pas installé"
    fi
    
    # Check environment file
    if [ ! -f "$PROJECT_ROOT/.env.production" ]; then
        error "Fichier .env.production manquant"
    fi
    
    success "Prérequis VPS validés (${TOTAL_MEM}GB RAM, ${AVAILABLE_DISK}GB libre)"
}

# Setup environment for VPS Lite
setup_vps_environment() {
    log "⚙️ Configuration de l'environnement VPS..."
    
    # Create directories
    mkdir -p logs backups monitoring/{loki,promtail,prometheus,grafana}
    
    # Set correct permissions
    chown -R 1000:1000 logs
    chmod 755 backups
    
    # Create Docker network if it doesn't exist
    docker network ls | grep nightscan-net || docker network create nightscan-net
    
    # Pull images first
    log "📦 Téléchargement des images Docker..."
    docker-compose -f "$COMPOSE_FILE" pull
    
    success "Environnement VPS configuré"
}

# Optimize for VPS Lite resources
optimize_for_vps() {
    log "🔧 Optimisation pour VPS Lite..."
    
    # Set Docker daemon options for VPS
    cat > /tmp/docker-daemon.json << EOF
{
    "log-driver": "json-file",
    "log-opts": {
        "max-size": "10m",
        "max-file": "3"
    },
    "max-concurrent-downloads": 3,
    "max-concurrent-uploads": 5,
    "storage-driver": "overlay2",
    "storage-opts": [
        "overlay2.override_kernel_check=true"
    ]
}
EOF
    
    if [ -f /etc/docker/daemon.json ]; then
        cp /etc/docker/daemon.json /etc/docker/daemon.json.backup
    fi
    
    cp /tmp/docker-daemon.json /etc/docker/daemon.json
    systemctl reload docker || warn "Impossible de recharger Docker daemon"
    
    # System optimizations for VPS
    cat > /tmp/vps-sysctl.conf << EOF
# Network optimizations for VPS
net.core.somaxconn = 65535
net.ipv4.tcp_max_syn_backlog = 8192
net.ipv4.ip_local_port_range = 1024 65535
net.ipv4.tcp_tw_reuse = 1
net.ipv4.tcp_fin_timeout = 15

# Memory optimizations for 4GB VPS
vm.swappiness = 10
vm.dirty_ratio = 15
vm.dirty_background_ratio = 5
vm.overcommit_memory = 1

# File system optimizations
fs.file-max = 2097152
fs.inotify.max_user_watches = 524288
EOF
    
    cp /tmp/vps-sysctl.conf /etc/sysctl.d/99-nightscan-vps.conf
    sysctl -p /etc/sysctl.d/99-nightscan-vps.conf || warn "Impossible d'appliquer les optimisations système"
    
    success "Optimisations VPS appliquées"
}

# Pre-deployment checks
pre_deployment_checks() {
    log "🔍 Vérifications pré-déploiement..."
    
    # Check Docker Compose file validity
    docker-compose -f "$COMPOSE_FILE" config >/dev/null || error "Fichier Docker Compose invalide"
    
    # Check environment variables
    if ! grep -q "DB_PASSWORD" .env.production; then
        error "Variable DB_PASSWORD manquante dans .env.production"
    fi
    
    # Check available memory before deployment
    AVAILABLE_MEM=$(free -m | awk 'NR==2{printf "%.1f", $7/1000}')
    if (( $(echo "$AVAILABLE_MEM < 2.0" | bc -l) )); then
        warn "Mémoire disponible faible: ${AVAILABLE_MEM}GB"
    fi
    
    # Check for port conflicts
    if netstat -tuln | grep -q ":80\|:443\|:3000\|:9090"; then
        warn "Certains ports sont déjà utilisés"
    fi
    
    success "Vérifications pré-déploiement terminées"
}

# Deploy application stack
deploy_application() {
    log "🚀 Déploiement de l'application NightScan v${VERSION}..."
    
    # Stop existing containers if any
    if docker-compose -f "$COMPOSE_FILE" ps -q >/dev/null 2>&1; then
        log "Arrêt des conteneurs existants..."
        docker-compose -f "$COMPOSE_FILE" down --timeout 30
    fi
    
    # Start database and cache first
    log "Démarrage de la base de données et du cache..."
    docker-compose -f "$COMPOSE_FILE" up -d postgres redis
    
    # Wait for database to be ready
    log "Attente de la base de données..."
    timeout=60
    while [ $timeout -gt 0 ]; do
        if docker-compose -f "$COMPOSE_FILE" exec -T postgres pg_isready -U nightscan >/dev/null 2>&1; then
            break
        fi
        sleep 2
        ((timeout--))
    done
    
    if [ $timeout -eq 0 ]; then
        error "Timeout: Base de données non prête"
    fi
    
    # Start application services
    log "Démarrage des services applicatifs..."
    docker-compose -f "$COMPOSE_FILE" up -d prediction-api web
    
    # Wait for application to be ready
    log "Attente de l'application..."
    timeout=120
    while [ $timeout -gt 0 ]; do
        if curl -f http://localhost:8000/health >/dev/null 2>&1; then
            break
        fi
        sleep 3
        ((timeout--))
    done
    
    if [ $timeout -eq 0 ]; then
        error "Timeout: Application non prête"
    fi
    
    # Start reverse proxy
    log "Démarrage du reverse proxy..."
    docker-compose -f "$COMPOSE_FILE" up -d nginx-proxy letsencrypt
    
    success "Application déployée avec succès"
}

# Deploy monitoring stack
deploy_monitoring() {
    log "📊 Déploiement du monitoring..."
    
    # Create monitoring configs if they don't exist
    if [ ! -f monitoring/loki/config.yml ]; then
        mkdir -p monitoring/loki
        cat > monitoring/loki/config.yml << 'EOF'
auth_enabled: false

server:
  http_listen_port: 3100
  grpc_listen_port: 9096

common:
  path_prefix: /loki
  storage:
    filesystem:
      chunks_directory: /loki/chunks
      rules_directory: /loki/rules
  replication_factor: 1
  ring:
    kvstore:
      store: inmemory

schema_config:
  configs:
    - from: 2020-10-24
      store: boltdb-shipper
      object_store: filesystem
      schema: v11
      index:
        prefix: index_
        period: 24h

limits_config:
  retention_period: 168h
  enforce_metric_name: false
  reject_old_samples: true
  reject_old_samples_max_age: 168h
  max_entries_limit_per_query: 5000
  max_streams_per_user: 10000
  max_line_size: 256KB
  max_label_value_length: 4096
  max_label_name_length: 1024
  max_label_names_per_series: 30
EOF
    fi
    
    # Start monitoring stack
    docker-compose -f "$MONITORING_FILE" up -d
    
    success "Monitoring déployé"
}

# Post-deployment validation
post_deployment_validation() {
    log "✅ Validation post-déploiement..."
    
    # Check all services are running
    FAILED_SERVICES=$(docker-compose -f "$COMPOSE_FILE" ps | grep -v "Up" | grep -c "Exit\|Restarting" || true)
    if [ "$FAILED_SERVICES" -gt 0 ]; then
        error "$FAILED_SERVICES service(s) en échec"
    fi
    
    # Check health endpoints
    if ! curl -f http://localhost:8000/health >/dev/null 2>&1; then
        error "Health check de l'application échoué"
    fi
    
    if ! curl -f http://localhost:8001/api/health >/dev/null 2>&1; then
        error "Health check de l'API de prédiction échoué"
    fi
    
    # Check resource usage
    MEMORY_USAGE=$(docker stats --no-stream --format "table {{.Container}}\t{{.MemUsage}}" | grep nightscan | awk '{sum += $2} END {print sum}')
    log "Utilisation mémoire totale: ${MEMORY_USAGE}MB"
    
    # Check disk usage
    DISK_USAGE=$(df -h / | awk 'NR==2{print $5}' | sed 's/%//')
    if [ "$DISK_USAGE" -gt 80 ]; then
        warn "Utilisation disque élevée: ${DISK_USAGE}%"
    fi
    
    success "Validation terminée - Application opérationnelle"
}

# Setup automated backups
setup_backups() {
    log "💾 Configuration des sauvegardes..."
    
    cat > "$PROJECT_ROOT/scripts/backup-vps.sh" << 'EOF'
#!/bin/bash
BACKUP_DIR="/opt/backups/nightscan"
DATE=$(date +%Y%m%d_%H%M%S)
RETENTION_DAYS=7

mkdir -p $BACKUP_DIR

# Database backup
docker exec nightscan-db pg_dump -U nightscan nightscan | gzip > $BACKUP_DIR/db_$DATE.sql.gz

# Volumes backup
docker run --rm -v nightscan_upload_data:/data -v $BACKUP_DIR:/backup alpine tar czf /backup/uploads_$DATE.tar.gz -C /data .
docker run --rm -v nightscan_model_cache:/data -v $BACKUP_DIR:/backup alpine tar czf /backup/models_$DATE.tar.gz -C /data .

# Configuration backup
tar czf $BACKUP_DIR/config_$DATE.tar.gz docker-compose.production.yml .env.production nginx/ monitoring/

# Cleanup old backups
find $BACKUP_DIR -name "*.gz" -mtime +$RETENTION_DAYS -delete

echo "Backup completed: $DATE"
EOF
    
    chmod +x "$PROJECT_ROOT/scripts/backup-vps.sh"
    
    # Add to crontab if not already present
    if ! crontab -l 2>/dev/null | grep -q "backup-vps.sh"; then
        (crontab -l 2>/dev/null; echo "0 3 * * * $PROJECT_ROOT/scripts/backup-vps.sh >> /var/log/nightscan-backup.log 2>&1") | crontab -
    fi
    
    success "Sauvegardes automatiques configurées"
}

# Main deployment function
main() {
    log "🚀 Début du déploiement NightScan VPS Lite v${VERSION}"
    
    cd "$PROJECT_ROOT"
    
    # Load environment
    if [ -f .env.production ]; then
        set -a
        source .env.production
        set +a
    fi
    
    # Run deployment steps
    check_vps_requirements
    setup_vps_environment
    optimize_for_vps
    pre_deployment_checks
    deploy_application
    deploy_monitoring
    post_deployment_validation
    setup_backups
    
    success "🎉 Déploiement NightScan VPS Lite terminé avec succès!"
    
    log "📱 Application disponible sur:"
    log "   - Web: https://${DOMAIN_NAME:-localhost}"
    log "   - API: https://api.${DOMAIN_NAME:-localhost}/api/health"
    log "   - Monitoring: https://monitoring.${DOMAIN_NAME:-localhost}"
    
    log "📊 Utilisation des ressources:"
    docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}"
}

# Run main function
main "$@"