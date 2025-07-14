#!/bin/bash
# Script déploiement staging NightScan
set -e

echo "🚀 DÉPLOIEMENT STAGING NIGHTSCAN"
echo "================================="

# Configuration
STAGING_HOST="${STAGING_SERVER:-staging.nightscan.com}"
STAGING_USER="${STAGING_USER:-deploy}"
DEPLOYMENT_DIR="/opt/nightscan"
BACKUP_DIR="/opt/nightscan/backups"

# Couleurs pour output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Vérification prérequis
check_prerequisites() {
    log_info "Vérification prérequis déploiement..."
    
    if [[ -z "$STAGING_SSH_KEY" ]]; then
        log_error "STAGING_SSH_KEY non défini"
        exit 1
    fi
    
    if [[ -z "$GITHUB_SHA" ]]; then
        log_error "GITHUB_SHA non défini"
        exit 1
    fi
    
    # Vérifier connectivité
    if ! ping -c 1 "$STAGING_HOST" > /dev/null 2>&1; then
        log_error "Impossible de joindre $STAGING_HOST"
        exit 1
    fi
    
    log_info "✅ Prérequis validés"
}

# Préparation clés SSH
setup_ssh() {
    log_info "Configuration SSH..."
    
    mkdir -p ~/.ssh
    echo "$STAGING_SSH_KEY" > ~/.ssh/staging_key
    chmod 600 ~/.ssh/staging_key
    
    # Ajout host connu
    ssh-keyscan -H "$STAGING_HOST" >> ~/.ssh/known_hosts
    
    log_info "✅ SSH configuré"
}

# Backup avant déploiement
backup_current_deployment() {
    log_info "Sauvegarde déploiement actuel..."
    
    ssh -i ~/.ssh/staging_key "$STAGING_USER@$STAGING_HOST" << EOF
        set -e
        
        # Créer répertoire backup avec timestamp
        BACKUP_TIMESTAMP=\$(date +%Y%m%d_%H%M%S)
        BACKUP_PATH="$BACKUP_DIR/pre_deploy_\$BACKUP_TIMESTAMP"
        mkdir -p "\$BACKUP_PATH"
        
        # Backup application
        if [[ -d "$DEPLOYMENT_DIR/current" ]]; then
            cp -r "$DEPLOYMENT_DIR/current" "\$BACKUP_PATH/application"
        fi
        
        # Backup base données
        pg_dump \$NIGHTSCAN_DATABASE_URI | gzip > "\$BACKUP_PATH/database.sql.gz"
        
        # Backup Redis
        redis-cli --rdb "\$BACKUP_PATH/redis.rdb"
        
        echo "✅ Backup créé: \$BACKUP_PATH"
EOF
    
    log_info "✅ Backup terminé"
}

# Déploiement code
deploy_application() {
    log_info "Déploiement application..."
    
    # Créer archive déploiement
    tar -czf nightscan-staging.tar.gz \
        --exclude='.git' \
        --exclude='node_modules' \
        --exclude='__pycache__' \
        --exclude='.pytest_cache' \
        --exclude='htmlcov' \
        .
    
    # Upload et extraction
    scp -i ~/.ssh/staging_key nightscan-staging.tar.gz "$STAGING_USER@$STAGING_HOST:/tmp/"
    
    ssh -i ~/.ssh/staging_key "$STAGING_USER@$STAGING_HOST" << EOF
        set -e
        
        # Créer répertoire release
        RELEASE_DIR="$DEPLOYMENT_DIR/releases/\$(date +%Y%m%d_%H%M%S)_${GITHUB_SHA:0:8}"
        mkdir -p "\$RELEASE_DIR"
        
        # Extraire application
        cd "\$RELEASE_DIR"
        tar -xzf /tmp/nightscan-staging.tar.gz
        rm /tmp/nightscan-staging.tar.gz
        
        # Installer dépendances
        python3 -m venv venv
        source venv/bin/activate
        pip install --upgrade pip
        pip install -r requirements.txt
        
        # Configuration staging
        cp config/unified/staging.json config/current.json
        
        # Symlink vers current
        ln -sfn "\$RELEASE_DIR" "$DEPLOYMENT_DIR/current"
        
        echo "✅ Application déployée: \$RELEASE_DIR"
EOF
    
    log_info "✅ Application déployée"
}

# Migration base données
run_migrations() {
    log_info "Exécution migrations base données..."
    
    ssh -i ~/.ssh/staging_key "$STAGING_USER@$STAGING_HOST" << EOF
        set -e
        cd "$DEPLOYMENT_DIR/current"
        source venv/bin/activate
        
        # Variables environnement staging
        export NIGHTSCAN_ENV=staging
        export NIGHTSCAN_CONFIG_FILE="$DEPLOYMENT_DIR/current/config/current.json"
        
        # Migrations (si Alembic utilisé)
        if [[ -f "alembic.ini" ]]; then
            alembic upgrade head
        fi
        
        # Mise à jour schéma si scripts SQL
        if [[ -f "database/migrations/staging.sql" ]]; then
            psql \$NIGHTSCAN_DATABASE_URI -f database/migrations/staging.sql
        fi
        
        echo "✅ Migrations terminées"
EOF
    
    log_info "✅ Migrations terminées"
}

# Redémarrage services
restart_services() {
    log_info "Redémarrage services..."
    
    ssh -i ~/.ssh/staging_key "$STAGING_USER@$STAGING_HOST" << EOF
        set -e
        cd "$DEPLOYMENT_DIR/current"
        
        # Arrêt services existants
        pkill -f "nightscan" || true
        
        # Redémarrage avec docker-compose
        if [[ -f "docker-compose.staging.yml" ]]; then
            docker-compose -f docker-compose.staging.yml down
            docker-compose -f docker-compose.staging.yml up -d
        else
            # Démarrage manuel
            source venv/bin/activate
            export NIGHTSCAN_ENV=staging
            
            # Web application
            nohup gunicorn -w 2 -b 0.0.0.0:8000 web.app:application > logs/web.log 2>&1 &
            
            # API v1
            nohup gunicorn -w 2 -b 0.0.0.0:8001 api_v1:app > logs/api.log 2>&1 &
            
            # Prediction API
            nohup python unified_prediction_system/unified_prediction_api.py > logs/prediction.log 2>&1 &
            
            # Celery worker
            nohup celery -A web.tasks worker --loglevel=info > logs/celery.log 2>&1 &
        fi
        
        echo "✅ Services redémarrés"
EOF
    
    log_info "✅ Services redémarrés"
}

# Tests post-déploiement
run_post_deployment_tests() {
    log_info "Tests post-déploiement..."
    
    # Attendre démarrage services
    sleep 30
    
    # Tests santé endpoints
    STAGING_URL="https://$STAGING_HOST"
    
    # Test web app
    if curl -f "$STAGING_URL/health" > /dev/null 2>&1; then
        log_info "✅ Web app accessible"
    else
        log_error "❌ Web app non accessible"
        return 1
    fi
    
    # Test API
    if curl -f "$STAGING_URL/api/v1/health" > /dev/null 2>&1; then
        log_info "✅ API accessible"
    else
        log_error "❌ API non accessible"
        return 1
    fi
    
    # Test prediction API
    if curl -f "$STAGING_URL:8002/health" > /dev/null 2>&1; then
        log_info "✅ Prediction API accessible"
    else
        log_warn "⚠️ Prediction API non accessible"
    fi
    
    # Tests fonctionnels
    python scripts/staging_validation.py
    
    log_info "✅ Tests post-déploiement réussis"
}

# Nettoyage anciennes releases
cleanup_old_releases() {
    log_info "Nettoyage anciennes releases..."
    
    ssh -i ~/.ssh/staging_key "$STAGING_USER@$STAGING_HOST" << EOF
        set -e
        
        # Garder seulement les 5 dernières releases
        cd "$DEPLOYMENT_DIR/releases"
        ls -t | tail -n +6 | xargs -r rm -rf
        
        # Nettoyage anciens backups (>7 jours)
        find "$BACKUP_DIR" -name "pre_deploy_*" -mtime +7 -exec rm -rf {} \;
        
        echo "✅ Nettoyage terminé"
EOF
    
    log_info "✅ Nettoyage terminé"
}

# Rollback en cas d'échec
rollback_deployment() {
    log_error "Échec déploiement - Rollback..."
    
    ssh -i ~/.ssh/staging_key "$STAGING_USER@$STAGING_HOST" << EOF
        set -e
        
        # Trouver dernière release valide
        LAST_RELEASE=\$(ls -t "$DEPLOYMENT_DIR/releases" | sed -n '2p')
        
        if [[ -n "\$LAST_RELEASE" ]]; then
            ln -sfn "$DEPLOYMENT_DIR/releases/\$LAST_RELEASE" "$DEPLOYMENT_DIR/current"
            
            # Redémarrer services
            cd "$DEPLOYMENT_DIR/current"
            docker-compose -f docker-compose.staging.yml restart || {
                pkill -f "nightscan"
                # Redémarrage manuel des services
            }
            
            echo "✅ Rollback vers \$LAST_RELEASE"
        else
            echo "❌ Aucune release précédente trouvée"
        fi
EOF
    
    log_error "Rollback terminé"
    exit 1
}

# Fonction principale
main() {
    # Trap pour rollback automatique en cas d'erreur
    trap rollback_deployment ERR
    
    check_prerequisites
    setup_ssh
    backup_current_deployment
    deploy_application
    run_migrations
    restart_services
    run_post_deployment_tests
    cleanup_old_releases
    
    log_info "🎉 DÉPLOIEMENT STAGING RÉUSSI!"
    echo "================================="
    echo "SHA: $GITHUB_SHA"
    echo "Host: $STAGING_HOST"
    echo "Timestamp: $(date)"
}

# Exécution
main "$@"