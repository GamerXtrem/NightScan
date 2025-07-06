#!/bin/bash

# ============================================================================
# 🚨 Script de Rollback d'Urgence NightScan VPS Lite
# ============================================================================

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
LOG_FILE="/tmp/nightscan_rollback.log"
BACKUP_DIR="/home/$(whoami)/backups"

# Couleurs
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Variables
ROLLBACK_TYPE=""
TARGET_VERSION=""
BACKUP_FILE=""
FORCE_ROLLBACK=false

# Fonctions utilitaires
log() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')] $1${NC}" | tee -a "$LOG_FILE"
}

success() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')] ✅ $1${NC}" | tee -a "$LOG_FILE"
}

error() {
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] ❌ $1${NC}" | tee -a "$LOG_FILE"
    exit 1
}

warn() {
    echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')] ⚠️  $1${NC}" | tee -a "$LOG_FILE"
}

# Confirmation sécurisée
confirm_rollback() {
    echo ""
    echo -e "${RED}⚠️  ATTENTION: PROCÉDURE DE ROLLBACK D'URGENCE${NC}"
    echo -e "${RED}Cette opération va:${NC}"
    echo -e "${RED}  1. Arrêter tous les services actuels${NC}"
    echo -e "${RED}  2. Restaurer une version précédente${NC}"
    echo -e "${RED}  3. Redémarrer les services${NC}"
    echo ""
    
    if [ "$FORCE_ROLLBACK" = false ]; then
        read -p "Êtes-vous sûr de vouloir continuer? (tapez 'ROLLBACK' pour confirmer): " confirmation
        if [ "$confirmation" != "ROLLBACK" ]; then
            log "Rollback annulé par l'utilisateur"
            exit 0
        fi
    fi
    
    log "🚨 Rollback confirmé - Démarrage de la procédure d'urgence"
}

# Vérifications préalables
pre_rollback_checks() {
    log "🔍 Vérifications préalables..."
    
    # Vérifier que nous sommes dans le bon répertoire
    if [ ! -f "$PROJECT_ROOT/docker-compose.production.yml" ]; then
        error "Fichier docker-compose.production.yml introuvable - Mauvais répertoire?"
    fi
    
    # Vérifier Docker
    if ! docker info >/dev/null 2>&1; then
        error "Docker non accessible - Permissions insuffisantes?"
    fi
    
    # Créer backup pré-rollback
    log "💾 Création backup pré-rollback..."
    create_pre_rollback_backup
    
    success "Vérifications préalables OK"
}

# Créer backup avant rollback
create_pre_rollback_backup() {
    local backup_name="pre_rollback_$(date +%Y%m%d_%H%M%S).tar.gz"
    local backup_path="$BACKUP_DIR/$backup_name"
    
    mkdir -p "$BACKUP_DIR"
    
    cd "$PROJECT_ROOT"
    
    # Backup configuration actuelle
    tar -czf "$backup_path" \
        docker-compose.production.yml \
        docker-compose.monitoring.yml \
        secrets/ \
        nginx/ \
        monitoring/ \
        ssl/ \
        2>/dev/null || true
        
    success "Backup pré-rollback créé: $backup_name"
}

# Arrêt d'urgence des services
emergency_stop() {
    log "🛑 Arrêt d'urgence des services..."
    
    cd "$PROJECT_ROOT"
    
    # Arrêter services de monitoring en premier
    log "📊 Arrêt monitoring..."
    docker-compose -f docker-compose.monitoring.yml down --remove-orphans || true
    
    # Arrêter services principaux
    log "🐳 Arrêt services principaux..."
    docker-compose -f docker-compose.production.yml down --remove-orphans || true
    
    # Arrêter SSL si présent
    if [ -f "docker-compose.ssl.yml" ]; then
        log "🔒 Arrêt services SSL..."
        docker-compose -f docker-compose.ssl.yml down --remove-orphans || true
    fi
    
    # Nettoyer containers orphelins
    log "🧹 Nettoyage containers orphelins..."
    docker container prune -f || true
    
    success "Services arrêtés"
}

# Rollback version Git
rollback_git_version() {
    log "🔄 Rollback version Git vers $TARGET_VERSION..."
    
    cd "$PROJECT_ROOT"
    
    # Sauvegarder état actuel
    local current_commit=$(git rev-parse HEAD)
    echo "$current_commit" > .rollback_previous_commit
    
    # Checkout version cible
    if ! git checkout "$TARGET_VERSION" 2>/dev/null; then
        error "Impossible de checkout la version $TARGET_VERSION"
    fi
    
    success "Version Git restaurée: $TARGET_VERSION"
}

# Restauration depuis backup
rollback_from_backup() {
    log "📦 Restoration depuis backup: $BACKUP_FILE..."
    
    if [ ! -f "$BACKUP_DIR/$BACKUP_FILE" ]; then
        error "Fichier backup introuvable: $BACKUP_DIR/$BACKUP_FILE"
    fi
    
    cd "$PROJECT_ROOT"
    
    # Extraire backup
    if ! tar -xzf "$BACKUP_DIR/$BACKUP_FILE" 2>/dev/null; then
        error "Impossible d'extraire le backup $BACKUP_FILE"
    fi
    
    success "Backup restauré: $BACKUP_FILE"
}

# Rollback configuration seulement
rollback_config_only() {
    log "⚙️ Rollback configuration uniquement..."
    
    cd "$PROJECT_ROOT"
    
    # Restaurer configuration Docker Compose
    if [ -f "docker-compose.production.yml.backup" ]; then
        cp docker-compose.production.yml.backup docker-compose.production.yml
        success "Configuration production restaurée"
    fi
    
    if [ -f "docker-compose.monitoring.yml.backup" ]; then
        cp docker-compose.monitoring.yml.backup docker-compose.monitoring.yml
        success "Configuration monitoring restaurée"
    fi
    
    # Restaurer secrets si backup disponible
    if [ -d "secrets.backup" ]; then
        rm -rf secrets/
        cp -r secrets.backup secrets/
        success "Secrets restaurés"
    fi
    
    success "Configuration restaurée"
}

# Rollback containers seulement
rollback_containers_only() {
    log "🐳 Rollback containers vers version précédente..."
    
    cd "$PROJECT_ROOT"
    
    # Utiliser tags précédents pour les images
    if grep -q "VERSION" docker-compose.production.yml; then
        # Modifier VERSION dans .env ou docker-compose
        if [ -f "secrets/production/.env" ]; then
            sed -i.backup 's/VERSION=.*/VERSION=previous/' secrets/production/.env || true
        fi
    fi
    
    # Pull images précédentes
    log "📥 Téléchargement images précédentes..."
    docker-compose -f docker-compose.production.yml pull || warn "Impossible de pull certaines images"
    
    success "Images containers restaurées"
}

# Redémarrage des services
restart_services() {
    log "🚀 Redémarrage des services..."
    
    cd "$PROJECT_ROOT"
    
    # Vérifier configuration avant redémarrage
    if ! docker-compose -f docker-compose.production.yml config >/dev/null 2>&1; then
        error "Configuration Docker Compose invalide"
    fi
    
    # Redémarrer services principaux
    log "🐳 Démarrage services principaux..."
    if ! docker-compose -f docker-compose.production.yml up -d; then
        error "Échec redémarrage services principaux"
    fi
    
    # Attendre que les services soient prêts
    log "⏳ Attente démarrage services (60s)..."
    sleep 60
    
    # Redémarrer monitoring
    if [ -f "docker-compose.monitoring.yml" ]; then
        log "📊 Démarrage monitoring..."
        docker-compose -f docker-compose.monitoring.yml up -d || warn "Échec démarrage monitoring"
    fi
    
    success "Services redémarrés"
}

# Tests post-rollback
post_rollback_tests() {
    log "🧪 Tests post-rollback..."
    
    # Test basique des containers
    local unhealthy=$(docker ps --filter "health=unhealthy" --format "{{.Names}}" | wc -l)
    if [ "$unhealthy" -eq 0 ]; then
        success "Tous les containers sont healthy"
    else
        warn "$unhealthy containers unhealthy détectés"
    fi
    
    # Test connectivité si domaine configuré
    if [ -n "${DOMAIN_NAME:-}" ]; then
        if curl -f -s -k --connect-timeout 10 "https://$DOMAIN_NAME/health" >/dev/null 2>&1; then
            success "Application accessible via HTTPS"
        else
            warn "Application non accessible via HTTPS"
        fi
    fi
    
    # Test logs pour erreurs critiques
    local errors=$(docker-compose -f docker-compose.production.yml logs --since="5m" 2>&1 | grep -i "error\|critical\|fatal" | wc -l)
    if [ "$errors" -eq 0 ]; then
        success "Aucune erreur critique dans les logs"
    else
        warn "$errors erreurs détectées dans les logs"
    fi
}

# Créer point de restauration
create_restore_point() {
    log "📋 Création point de restauration..."
    
    local restore_info="$PROJECT_ROOT/.rollback_info"
    
    cat > "$restore_info" << EOF
# Informations de rollback - $(date)
ROLLBACK_TIMESTAMP=$(date +%s)
ROLLBACK_TYPE=$ROLLBACK_TYPE
PREVIOUS_VERSION=$TARGET_VERSION
BACKUP_USED=$BACKUP_FILE
ROLLBACK_REASON="Emergency rollback"
EOF
    
    success "Point de restauration créé"
}

# Afficher résumé
show_rollback_summary() {
    echo ""
    echo -e "${BLUE}=====================================${NC}"
    echo -e "${BLUE}🚨 ROLLBACK D'URGENCE TERMINÉ${NC}"
    echo -e "${BLUE}=====================================${NC}"
    echo ""
    
    echo -e "${GREEN}✅ Type de rollback: $ROLLBACK_TYPE${NC}"
    echo -e "${GREEN}✅ Services redémarrés${NC}"
    echo -e "${GREEN}✅ Tests post-rollback effectués${NC}"
    echo ""
    
    echo -e "${YELLOW}📋 Actions recommandées:${NC}"
    echo -e "   1. Vérifier fonctionnement application"
    echo -e "   2. Surveiller logs système"
    echo -e "   3. Alerter équipe du rollback"
    echo -e "   4. Analyser cause du problème"
    echo -e "   5. Planifier correctif"
    echo ""
    
    echo -e "${BLUE}📁 Logs rollback: $LOG_FILE${NC}"
    echo -e "${BLUE}📦 Backup pré-rollback: $BACKUP_DIR/pre_rollback_*${NC}"
    echo ""
    
    success "Rollback d'urgence terminé avec succès!"
}

# Fonction principale
main() {
    case "$ROLLBACK_TYPE" in
        "git")
            confirm_rollback
            pre_rollback_checks
            emergency_stop
            rollback_git_version
            restart_services
            ;;
        "backup")
            confirm_rollback
            pre_rollback_checks
            emergency_stop
            rollback_from_backup
            restart_services
            ;;
        "config")
            confirm_rollback
            pre_rollback_checks
            emergency_stop
            rollback_config_only
            restart_services
            ;;
        "containers")
            confirm_rollback
            pre_rollback_checks
            emergency_stop
            rollback_containers_only
            restart_services
            ;;
        "quick")
            # Rollback rapide sans confirmation
            FORCE_ROLLBACK=true
            log "🚨 ROLLBACK RAPIDE - Redémarrage des services"
            emergency_stop
            sleep 5
            restart_services
            ;;
        *)
            error "Type de rollback non spécifié. Utilisez --help pour voir les options."
            ;;
    esac
    
    # Actions communes
    post_rollback_tests
    create_restore_point
    show_rollback_summary
}

# Gestion des options
while [[ $# -gt 0 ]]; do
    case $1 in
        --git)
            ROLLBACK_TYPE="git"
            TARGET_VERSION="$2"
            shift 2
            ;;
        --backup)
            ROLLBACK_TYPE="backup"
            BACKUP_FILE="$2"
            shift 2
            ;;
        --config)
            ROLLBACK_TYPE="config"
            shift
            ;;
        --containers)
            ROLLBACK_TYPE="containers"
            shift
            ;;
        --quick)
            ROLLBACK_TYPE="quick"
            shift
            ;;
        --force)
            FORCE_ROLLBACK=true
            shift
            ;;
        --help)
            echo "🚨 Script de Rollback d'Urgence NightScan VPS Lite"
            echo ""
            echo "Usage: $0 [TYPE] [OPTIONS]"
            echo ""
            echo "Types de rollback:"
            echo "  --git VERSION        Rollback vers commit/tag Git"
            echo "  --backup FILE        Rollback depuis fichier backup"
            echo "  --config             Rollback configuration uniquement"
            echo "  --containers         Rollback containers uniquement"
            echo "  --quick              Rollback rapide (redémarrage services)"
            echo ""
            echo "Options:"
            echo "  --force              Pas de confirmation"
            echo "  --help               Afficher cette aide"
            echo ""
            echo "Exemples:"
            echo "  $0 --git main                    # Rollback vers branche main"
            echo "  $0 --git v1.2.0                 # Rollback vers tag v1.2.0"
            echo "  $0 --backup backup_20240101.tar.gz # Rollback depuis backup"
            echo "  $0 --quick --force              # Rollback rapide sans confirmation"
            echo ""
            echo "⚠️  ATTENTION: Utilisez uniquement en cas d'urgence!"
            exit 0
            ;;
        *)
            error "Option inconnue: $1. Utilisez --help pour voir les options."
            ;;
    esac
done

# Vérifications de base
if [ -z "$ROLLBACK_TYPE" ]; then
    error "Type de rollback requis. Utilisez --help pour voir les options."
fi

if [ "$ROLLBACK_TYPE" = "git" ] && [ -z "$TARGET_VERSION" ]; then
    error "Version Git requise pour rollback git"
fi

if [ "$ROLLBACK_TYPE" = "backup" ] && [ -z "$BACKUP_FILE" ]; then
    error "Fichier backup requis pour rollback backup"
fi

# Vérifier répertoire backup
mkdir -p "$BACKUP_DIR"

# Exécution
log "🚨 Démarrage rollback d'urgence NightScan VPS Lite..."
main "$@"