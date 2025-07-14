#!/bin/bash
# Script déploiement production NightScan avec Blue-Green deployment
set -e

echo "🚀 DÉPLOIEMENT PRODUCTION NIGHTSCAN"
echo "==================================="

# Configuration
PRODUCTION_HOST="${PRODUCTION_SERVER:-production.nightscan.com}"
PRODUCTION_USER="${PRODUCTION_USER:-deploy}"
DEPLOYMENT_DIR="/opt/nightscan"
BLUE_DIR="$DEPLOYMENT_DIR/blue"
GREEN_DIR="$DEPLOYMENT_DIR/green"
CURRENT_LINK="$DEPLOYMENT_DIR/current"

# Couleurs
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_blue() {
    echo -e "${BLUE}[BLUE]${NC} $1"
}

# Vérification prérequis production
check_production_prerequisites() {
    log_info "Vérification prérequis production..."
    
    if [[ -z "$PRODUCTION_SSH_KEY" ]]; then
        log_error "PRODUCTION_SSH_KEY non défini"
        exit 1
    fi
    
    if [[ -z "$DATABASE_URL" ]]; then
        log_error "DATABASE_URL non défini"
        exit 1
    fi
    
    if [[ -z "$GITHUB_SHA" ]]; then
        log_error "GITHUB_SHA non défini"
        exit 1
    fi
    
    # Vérifier que c'est bien main branch
    if [[ "$GITHUB_REF" != "refs/heads/main" ]]; then
        log_error "Déploiement production autorisé seulement depuis main"
        exit 1
    fi
    
    # Tests pré-déploiement passés
    if [[ ! -f "pre_deployment_check.passed" ]]; then
        log_error "Tests pré-déploiement non passés"
        exit 1
    fi
    
    log_info "✅ Prérequis production validés"
}

# Configuration SSH sécurisée
setup_production_ssh() {
    log_info "Configuration SSH production..."
    
    mkdir -p ~/.ssh
    echo "$PRODUCTION_SSH_KEY" > ~/.ssh/production_key
    chmod 600 ~/.ssh/production_key
    
    # Configuration SSH stricte pour production
    cat > ~/.ssh/config << EOF
Host production
    HostName $PRODUCTION_HOST
    User $PRODUCTION_USER
    IdentityFile ~/.ssh/production_key
    StrictHostKeyChecking yes
    UserKnownHostsFile ~/.ssh/known_hosts_production
    ServerAliveInterval 60
    ServerAliveCountMax 3
EOF
    
    # Host key verification
    ssh-keyscan -H "$PRODUCTION_HOST" > ~/.ssh/known_hosts_production
    
    log_info "✅ SSH production configuré"
}

# Déterminer environnement actuel (blue/green)
detect_current_environment() {
    log_info "Détection environnement actuel..."
    
    CURRENT_ENV=$(ssh production << 'EOF'
        if [[ -L "$CURRENT_LINK" ]]; then
            CURRENT_TARGET=$(readlink "$CURRENT_LINK")
            if [[ "$CURRENT_TARGET" == *"blue"* ]]; then
                echo "blue"
            elif [[ "$CURRENT_TARGET" == *"green"* ]]; then
                echo "green"
            else
                echo "unknown"
            fi
        else
            echo "none"
        fi
EOF
    )
    
    if [[ "$CURRENT_ENV" == "blue" ]]; then
        DEPLOY_ENV="green"
        STANDBY_ENV="blue"
    else
        DEPLOY_ENV="blue"
        STANDBY_ENV="green"
    fi
    
    log_info "Environnement actuel: $CURRENT_ENV"
    log_info "Déploiement vers: $DEPLOY_ENV"
    
    export CURRENT_ENV DEPLOY_ENV STANDBY_ENV
}

# Backup production critique
backup_production() {
    log_info "Backup production critique..."
    
    ssh production << EOF
        set -e
        
        BACKUP_TIMESTAMP=\$(date +%Y%m%d_%H%M%S)
        BACKUP_PATH="/opt/nightscan/backups/production_\$BACKUP_TIMESTAMP"
        mkdir -p "\$BACKUP_PATH"
        
        # Backup base données avec compression
        pg_dump "$DATABASE_URL" | gzip > "\$BACKUP_PATH/database.sql.gz"
        
        # Backup Redis
        redis-cli --rdb "\$BACKUP_PATH/redis.rdb"
        
        # Backup configuration
        cp -r "$DEPLOYMENT_DIR/config" "\$BACKUP_PATH/"
        
        # Backup uploads/models
        tar -czf "\$BACKUP_PATH/user_data.tar.gz" \
            "$DEPLOYMENT_DIR/uploads" \
            "$DEPLOYMENT_DIR/models" 2>/dev/null || true
        
        # Upload vers S3 si configuré
        if command -v aws &> /dev/null && [[ -n "\$AWS_BACKUP_BUCKET" ]]; then
            aws s3 sync "\$BACKUP_PATH" "s3://\$AWS_BACKUP_BUCKET/nightscan/\$BACKUP_TIMESTAMP/"
        fi
        
        echo "✅ Backup créé: \$BACKUP_PATH"
        echo "\$BACKUP_PATH" > /tmp/latest_backup_path
EOF
    
    log_info "✅ Backup production terminé"
}

# Déploiement Blue-Green
deploy_to_environment() {
    log_info "Déploiement vers environnement $DEPLOY_ENV..."
    
    # Préparation archive optimisée
    tar -czf nightscan-production.tar.gz \
        --exclude='.git' \
        --exclude='node_modules' \
        --exclude='__pycache__' \
        --exclude='.pytest_cache' \
        --exclude='htmlcov' \
        --exclude='logs/*' \
        --exclude='*.pyc' \
        .
    
    # Upload sécurisé
    scp nightscan-production.tar.gz production:/tmp/
    
    ssh production << EOF
        set -e
        
        DEPLOY_DIR="$DEPLOYMENT_DIR/$DEPLOY_ENV"
        
        # Nettoyage environnement cible
        rm -rf "\$DEPLOY_DIR"
        mkdir -p "\$DEPLOY_DIR"
        
        # Extraction
        cd "\$DEPLOY_DIR"
        tar -xzf /tmp/nightscan-production.tar.gz
        rm /tmp/nightscan-production.tar.gz
        
        # Installation dépendances dans environnement isolé
        python3 -m venv venv
        source venv/bin/activate
        pip install --upgrade pip
        pip install --no-cache-dir -r requirements.txt
        
        # Configuration production
        cp config/unified/production.json config/current.json
        
        # Permissions sécurisées
        chown -R nightscan:nightscan "\$DEPLOY_DIR"
        chmod -R 750 "\$DEPLOY_DIR"
        chmod 640 "\$DEPLOY_DIR/config/current.json"
        
        echo "✅ Déploiement $DEPLOY_ENV terminé"
EOF
    
    log_info "✅ Déploiement vers $DEPLOY_ENV terminé"
}

# Tests environnement déployé
test_deployed_environment() {
    log_info "Tests environnement $DEPLOY_ENV..."
    
    ssh production << EOF
        set -e
        
        cd "$DEPLOYMENT_DIR/$DEPLOY_ENV"
        source venv/bin/activate
        
        # Variables environnement production
        export NIGHTSCAN_ENV=production
        export NIGHTSCAN_CONFIG_FILE="$DEPLOYMENT_DIR/$DEPLOY_ENV/config/current.json"
        export DATABASE_URL="$DATABASE_URL"
        
        # Tests rapides
        python -c "import web.app; print('✅ Import web.app OK')"
        python -c "import api_v1; print('✅ Import api_v1 OK')"
        python -c "import unified_config; print('✅ Import unified_config OK')"
        
        # Test connexions critiques
        python -c "
import psycopg2
import redis
import os

# Test PostgreSQL
conn = psycopg2.connect(os.environ['DATABASE_URL'])
cursor = conn.cursor()
cursor.execute('SELECT 1')
print('✅ PostgreSQL connexion OK')
cursor.close()
conn.close()

# Test Redis
r = redis.from_url(os.environ.get('REDIS_URL', 'redis://localhost:6379'))
r.ping()
print('✅ Redis connexion OK')
"
        
        echo "✅ Tests environnement $DEPLOY_ENV réussis"
EOF
    
    log_info "✅ Tests environnement $DEPLOY_ENV réussis"
}

# Démarrage services dans nouvel environnement
start_services_in_environment() {
    log_info "Démarrage services dans $DEPLOY_ENV..."
    
    ssh production << EOF
        set -e
        
        cd "$DEPLOYMENT_DIR/$DEPLOY_ENV"
        
        # Ports pour nouvel environnement
        if [[ "$DEPLOY_ENV" == "blue" ]]; then
            WEB_PORT=8010
            API_PORT=8011
            PREDICTION_PORT=8012
        else
            WEB_PORT=8020
            API_PORT=8021
            PREDICTION_PORT=8022
        fi
        
        # Démarrage avec docker-compose spécifique
        if [[ -f "docker-compose.production-$DEPLOY_ENV.yml" ]]; then
            docker-compose -f "docker-compose.production-$DEPLOY_ENV.yml" up -d
        else
            # Démarrage manuel avec ports dédiés
            source venv/bin/activate
            export NIGHTSCAN_ENV=production
            export NIGHTSCAN_CONFIG_FILE="$DEPLOYMENT_DIR/$DEPLOY_ENV/config/current.json"
            
            # Web application
            nohup gunicorn -w 4 -b "0.0.0.0:\$WEB_PORT" web.app:application \
                --access-logfile logs/web-access.log \
                --error-logfile logs/web-error.log \
                --pid logs/web.pid \
                --daemon
            
            # API v1
            nohup gunicorn -w 4 -b "0.0.0.0:\$API_PORT" api_v1:app \
                --access-logfile logs/api-access.log \
                --error-logfile logs/api-error.log \
                --pid logs/api.pid \
                --daemon
            
            # Prediction API
            nohup python unified_prediction_system/unified_prediction_api.py \
                --port=\$PREDICTION_PORT \
                > logs/prediction.log 2>&1 &
            echo \$! > logs/prediction.pid
            
            # Celery worker (partagé entre environnements)
            if ! pgrep -f "celery.*worker" > /dev/null; then
                nohup celery -A web.tasks worker --loglevel=info \
                    > logs/celery.log 2>&1 &
                echo \$! > logs/celery.pid
            fi
        fi
        
        echo "✅ Services $DEPLOY_ENV démarrés"
        echo "Web: \$WEB_PORT, API: \$API_PORT, Prediction: \$PREDICTION_PORT"
EOF
    
    log_info "✅ Services $DEPLOY_ENV démarrés"
}

# Tests santé nouvel environnement
health_check_new_environment() {
    log_info "Tests santé nouvel environnement..."
    
    # Récupérer ports depuis serveur
    PORTS=$(ssh production << EOF
        if [[ "$DEPLOY_ENV" == "blue" ]]; then
            echo "8010 8011 8012"
        else
            echo "8020 8021 8022"
        fi
EOF
    )
    
    WEB_PORT=$(echo $PORTS | cut -d' ' -f1)
    API_PORT=$(echo $PORTS | cut -d' ' -f2)
    PREDICTION_PORT=$(echo $PORTS | cut -d' ' -f3)
    
    # Attendre démarrage
    sleep 45
    
    # Tests santé via SSH tunnel (plus sécurisé)
    ssh -L "9080:localhost:$WEB_PORT" \
        -L "9081:localhost:$API_PORT" \
        -L "9082:localhost:$PREDICTION_PORT" \
        production sleep 60 &
    SSH_PID=$!
    
    sleep 10
    
    # Tests endpoints
    if curl -f "http://localhost:9080/health" > /dev/null 2>&1; then
        log_info "✅ Web app $DEPLOY_ENV OK"
    else
        log_error "❌ Web app $DEPLOY_ENV échec"
        kill $SSH_PID
        return 1
    fi
    
    if curl -f "http://localhost:9081/api/v1/health" > /dev/null 2>&1; then
        log_info "✅ API $DEPLOY_ENV OK"
    else
        log_error "❌ API $DEPLOY_ENV échec"
        kill $SSH_PID
        return 1
    fi
    
    if curl -f "http://localhost:9082/health" > /dev/null 2>&1; then
        log_info "✅ Prediction API $DEPLOY_ENV OK"
    else
        log_warn "⚠️ Prediction API $DEPLOY_ENV non accessible"
    fi
    
    kill $SSH_PID
    
    log_info "✅ Tests santé $DEPLOY_ENV réussis"
}

# Basculement traffic (Blue-Green switch)
switch_traffic() {
    log_info "Basculement traffic vers $DEPLOY_ENV..."
    
    ssh production << EOF
        set -e
        
        # Mise à jour load balancer/nginx
        if [[ "$DEPLOY_ENV" == "blue" ]]; then
            UPSTREAM_PORTS="8010 8011"
        else
            UPSTREAM_PORTS="8020 8021"
        fi
        
        # Mise à jour configuration nginx
        cat > /etc/nginx/sites-available/nightscan << 'NGINX_CONF'
upstream nightscan_web {
    server localhost:$(echo $UPSTREAM_PORTS | cut -d' ' -f1);
}

upstream nightscan_api {
    server localhost:$(echo $UPSTREAM_PORTS | cut -d' ' -f2);
}

server {
    listen 80;
    listen 443 ssl http2;
    server_name production.nightscan.com;
    
    # SSL configuration
    ssl_certificate /etc/ssl/certs/nightscan.pem;
    ssl_certificate_key /etc/ssl/private/nightscan.key;
    
    location / {
        proxy_pass http://nightscan_web;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }
    
    location /api/ {
        proxy_pass http://nightscan_api;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }
}
NGINX_CONF
        
        # Test configuration nginx
        nginx -t
        
        # Rechargement graceful nginx
        nginx -s reload
        
        # Mise à jour symlink current
        ln -sfn "$DEPLOYMENT_DIR/$DEPLOY_ENV" "$CURRENT_LINK"
        
        echo "✅ Traffic basculé vers $DEPLOY_ENV"
EOF
    
    log_info "✅ Traffic basculé vers $DEPLOY_ENV"
}

# Tests post-basculement
post_switch_tests() {
    log_info "Tests post-basculement..."
    
    sleep 30
    
    # Tests depuis l'extérieur
    PRODUCTION_URL="https://$PRODUCTION_HOST"
    
    if curl -f "$PRODUCTION_URL/health" > /dev/null 2>&1; then
        log_info "✅ Production web accessible"
    else
        log_error "❌ Production web non accessible"
        return 1
    fi
    
    if curl -f "$PRODUCTION_URL/api/v1/health" > /dev/null 2>&1; then
        log_info "✅ Production API accessible"
    else
        log_error "❌ Production API non accessible"
        return 1
    fi
    
    # Tests fonctionnels
    python scripts/production_smoke_tests.py
    
    log_info "✅ Tests post-basculement réussis"
}

# Arrêt ancien environnement
stop_old_environment() {
    log_info "Arrêt ancien environnement $STANDBY_ENV..."
    
    ssh production << EOF
        set -e
        
        cd "$DEPLOYMENT_DIR/$STANDBY_ENV"
        
        # Arrêt services
        if [[ -f "docker-compose.production-$STANDBY_ENV.yml" ]]; then
            docker-compose -f "docker-compose.production-$STANDBY_ENV.yml" down
        else
            # Arrêt manuel via PID files
            for pid_file in logs/*.pid; do
                if [[ -f "\$pid_file" ]]; then
                    kill \$(cat "\$pid_file") 2>/dev/null || true
                    rm "\$pid_file"
                fi
            done
        fi
        
        echo "✅ Ancien environnement $STANDBY_ENV arrêté"
EOF
    
    log_info "✅ Ancien environnement $STANDBY_ENV arrêté"
}

# Rollback en cas d'échec
emergency_rollback() {
    log_error "ÉCHEC DÉPLOIEMENT - ROLLBACK D'URGENCE!"
    
    ssh production << EOF
        set -e
        
        # Restaurer ancien environnement
        if [[ "$CURRENT_ENV" != "none" ]]; then
            ln -sfn "$DEPLOYMENT_DIR/$CURRENT_ENV" "$CURRENT_LINK"
            
            # Redémarrer services si nécessaire
            cd "$DEPLOYMENT_DIR/$CURRENT_ENV"
            
            # Vérifier si services tournent
            if ! curl -f "http://localhost:8000/health" > /dev/null 2>&1; then
                # Redémarrage services
                if [[ -f "docker-compose.production-$CURRENT_ENV.yml" ]]; then
                    docker-compose -f "docker-compose.production-$CURRENT_ENV.yml" up -d
                fi
            fi
            
            # Recharger nginx
            nginx -s reload
            
            echo "✅ Rollback vers $CURRENT_ENV terminé"
        fi
        
        # Arrêter environnement défaillant
        cd "$DEPLOYMENT_DIR/$DEPLOY_ENV"
        if [[ -f "docker-compose.production-$DEPLOY_ENV.yml" ]]; then
            docker-compose -f "docker-compose.production-$DEPLOY_ENV.yml" down
        fi
EOF
    
    log_error "Rollback terminé - Production restaurée"
    exit 1
}

# Nettoyage post-déploiement
cleanup_deployment() {
    log_info "Nettoyage post-déploiement..."
    
    ssh production << EOF
        set -e
        
        # Nettoyage anciens backups (>30 jours)
        find "/opt/nightscan/backups" -name "production_*" -mtime +30 -exec rm -rf {} \;
        
        # Nettoyage logs anciens
        find "/opt/nightscan/*/logs" -name "*.log" -mtime +7 -exec gzip {} \;
        find "/opt/nightscan/*/logs" -name "*.log.gz" -mtime +30 -delete
        
        # Nettoyage Docker si utilisé
        docker system prune -f
        
        echo "✅ Nettoyage terminé"
EOF
    
    log_info "✅ Nettoyage terminé"
}

# Fonction principale
main() {
    # Trap pour rollback automatique
    trap emergency_rollback ERR
    
    check_production_prerequisites
    setup_production_ssh
    detect_current_environment
    backup_production
    deploy_to_environment
    test_deployed_environment
    start_services_in_environment
    health_check_new_environment
    switch_traffic
    post_switch_tests
    stop_old_environment
    cleanup_deployment
    
    log_info "🎉 DÉPLOIEMENT PRODUCTION RÉUSSI!"
    echo "===================================="
    echo "Environment: $DEPLOY_ENV"
    echo "SHA: $GITHUB_SHA"
    echo "Timestamp: $(date)"
    echo "Production URL: https://$PRODUCTION_HOST"
}

# Exécution
main "$@"