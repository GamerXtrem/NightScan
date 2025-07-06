#!/bin/bash

# Configuration sécurisée des secrets pour NightScan VPS Lite
# Usage: ./setup-secrets.sh --env production

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
ENVIRONMENT="${1:-production}"

log() {
    echo -e "${BLUE}[$(date +'%H:%M:%S')] $1${NC}"
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

# Vérifier les prérequis
check_prerequisites() {
    log "🔍 Vérification des prérequis..."
    
    # Vérifier openssl
    if ! command -v openssl >/dev/null 2>&1; then
        error "openssl n'est pas installé"
    fi
    
    # Vérifier git
    if ! command -v git >/dev/null 2>&1; then
        error "git n'est pas installé"
    fi
    
    # Vérifier si on est dans un repo git
    if ! git rev-parse --git-dir >/dev/null 2>&1; then
        error "Pas dans un repository git"
    fi
    
    success "Prérequis validés"
}

# Générer des secrets forts
generate_secrets() {
    log "🔐 Génération des secrets sécurisés..."
    
    # Créer le répertoire secrets
    mkdir -p "$PROJECT_ROOT/secrets/$ENVIRONMENT"
    
    # Générer les secrets
    DB_PASSWORD=$(openssl rand -base64 32 | tr -d "=+/" | cut -c1-25)
    REDIS_PASSWORD=$(openssl rand -base64 32 | tr -d "=+/" | cut -c1-25)
    SECRET_KEY=$(openssl rand -base64 64 | tr -d "=+/")
    CSRF_SECRET_KEY=$(openssl rand -base64 32 | tr -d "=+/")
    JWT_SECRET=$(openssl rand -base64 32 | tr -d "=+/")
    GRAFANA_PASSWORD=$(openssl rand -base64 16 | tr -d "=+/" | cut -c1-12)
    
    success "Secrets générés"
}

# Créer le fichier .env sécurisé
create_env_file() {
    log "📝 Création du fichier .env.$ENVIRONMENT..."
    
    ENV_FILE="$PROJECT_ROOT/secrets/$ENVIRONMENT/.env"
    
    cat > "$ENV_FILE" << EOF
# NightScan VPS Lite - Variables d'environnement $ENVIRONMENT
# Généré le $(date)
# ATTENTION: Ce fichier contient des secrets sensibles

# === DOMAIN CONFIGURATION ===
DOMAIN_NAME=nightscan.yourdomain.com
ADMIN_EMAIL=admin@yourdomain.com

# === DATABASE CONFIGURATION ===
DB_PASSWORD=$DB_PASSWORD

# === REDIS CONFIGURATION ===
REDIS_PASSWORD=$REDIS_PASSWORD

# === APPLICATION SECRETS ===
SECRET_KEY=$SECRET_KEY
CSRF_SECRET_KEY=$CSRF_SECRET_KEY
JWT_SECRET=$JWT_SECRET

# === MONITORING ===
GRAFANA_PASSWORD=$GRAFANA_PASSWORD

# === EMAIL NOTIFICATIONS (À configurer) ===
SMTP_HOST=smtp.yourdomain.com
SMTP_PORT=587
SMTP_USER=alerts@yourdomain.com
SMTP_PASSWORD=CHANGE_ME_SMTP_PASSWORD

# === DOCKER IMAGE VERSIONS ===
VERSION=latest

# === RESOURCE LIMITS (VPS Lite optimized) ===
WEB_MEMORY_LIMIT=1g
WEB_CPU_LIMIT=1.0
PREDICTION_MEMORY_LIMIT=2g
PREDICTION_CPU_LIMIT=1.5
DB_MEMORY_LIMIT=500m
DB_CPU_LIMIT=0.5
REDIS_MEMORY_LIMIT=200m
REDIS_CPU_LIMIT=0.3

# === PERFORMANCE TUNING ===
GUNICORN_WORKERS=2
GUNICORN_THREADS=2
API_BATCH_SIZE=4
MAX_WORKERS=2

# === BACKUP CONFIGURATION ===
BACKUP_RETENTION_DAYS=7
BACKUP_SCHEDULE="0 3 * * *"

# === SSL CONFIGURATION ===
LETSENCRYPT_EMAIL=\${ADMIN_EMAIL}
LETSENCRYPT_HOST=\${DOMAIN_NAME},www.\${DOMAIN_NAME},api.\${DOMAIN_NAME},monitoring.\${DOMAIN_NAME}

# === SECURITY ===
FORCE_HTTPS=true
SECURE_COOKIES=true
HSTS_ENABLED=true
EOF
    
    # Permissions restrictives
    chmod 600 "$ENV_FILE"
    
    success "Fichier .env.$ENVIRONMENT créé avec permissions 600"
}

# Initialiser git-crypt si pas déjà fait
init_git_crypt() {
    log "🔒 Configuration de git-crypt..."
    
    # Vérifier si git-crypt est installé
    if ! command -v git-crypt >/dev/null 2>&1; then
        warn "git-crypt n'est pas installé. Installation recommandée:"
        echo "  macOS: brew install git-crypt"
        echo "  Ubuntu: apt-get install git-crypt"
        echo "  Manual: https://github.com/AGWA/git-crypt"
        
        # Utiliser alternative simple avec permissions
        setup_simple_encryption
        return
    fi
    
    # Vérifier si git-crypt est déjà initialisé
    if [ ! -f ".git-crypt/.git-crypt" ]; then
        log "Initialisation de git-crypt..."
        git-crypt init
        
        # Créer .gitattributes pour chiffrer les secrets
        if [ ! -f ".gitattributes" ]; then
            echo "secrets/**/.env filter=git-crypt diff=git-crypt" > .gitattributes
            echo "*.key filter=git-crypt diff=git-crypt" >> .gitattributes
            echo "*.pem filter=git-crypt diff=git-crypt" >> .gitattributes
        else
            if ! grep -q "secrets/\*\*/\.env" .gitattributes; then
                echo "secrets/**/.env filter=git-crypt diff=git-crypt" >> .gitattributes
                echo "*.key filter=git-crypt diff=git-crypt" >> .gitattributes
                echo "*.pem filter=git-crypt diff=git-crypt" >> .gitattributes
            fi
        fi
        
        success "git-crypt initialisé"
    else
        success "git-crypt déjà configuré"
    fi
}

# Alternative simple sans git-crypt
setup_simple_encryption() {
    log "🔐 Configuration encryption simple (sans git-crypt)..."
    
    # Ajouter secrets/ au .gitignore
    if [ ! -f ".gitignore" ]; then
        touch .gitignore
    fi
    
    if ! grep -q "secrets/" .gitignore; then
        echo "" >> .gitignore
        echo "# Secrets (sensibles)" >> .gitignore
        echo "secrets/" >> .gitignore
        echo ".env.production" >> .gitignore
        echo ".env.staging" >> .gitignore
    fi
    
    warn "Secrets non versionnés (ajoutés à .gitignore)"
    warn "⚠️  Important: Sauvegarder les secrets séparément!"
}

# Supprimer les secrets hardcodés
remove_hardcoded_secrets() {
    log "🧹 Suppression des secrets hardcodés..."
    
    # Lister les fichiers avec secrets par défaut
    SECRET_FILES=()
    
    # Rechercher dans tous les fichiers
    while IFS= read -r -d '' file; do
        if grep -l "nightscan_secret\|redis_secret\|your-secret-key" "$file" 2>/dev/null; then
            SECRET_FILES+=("$file")
        fi
    done < <(find "$PROJECT_ROOT" -type f \( -name "*.py" -o -name "*.yml" -o -name "*.yaml" \) -print0)
    
    if [ ${#SECRET_FILES[@]} -eq 0 ]; then
        success "Aucun secret hardcodé trouvé"
        return
    fi
    
    echo "📋 Fichiers avec secrets hardcodés détectés:"
    for file in "${SECRET_FILES[@]}"; do
        echo "  - $file"
    done
    
    warn "⚠️  Veuillez remplacer manuellement les secrets hardcodés par des variables d'environnement"
    warn "   Exemple: DB_PASSWORD=\${DB_PASSWORD} au lieu de DB_PASSWORD=nightscan_secret"
}

# Fonction principale
main() {
    log "🛡️  Configuration des secrets sécurisés NightScan VPS Lite"
    log "Environment: $ENVIRONMENT"
    
    cd "$PROJECT_ROOT"
    
    check_prerequisites
    generate_secrets
    create_env_file
    init_git_crypt
    setup_simple_encryption
    remove_hardcoded_secrets
    
    success "🎉 Configuration des secrets terminée!"
    
    echo ""
    echo "📋 PROCHAINES ÉTAPES:"
    echo "1. 🔍 Vérifier et modifier secrets/$ENVIRONMENT/.env selon vos besoins"
    echo "2. 🔄 Remplacer les secrets hardcodés dans le code"
    echo "3. 🧪 Tester avec: ./scripts/deploy-vps-lite.sh"
    echo "4. 🔒 Sauvegarder les secrets en lieu sûr"
    echo ""
    echo "⚠️  ATTENTION:"
    echo "- Ne jamais commiter le répertoire secrets/ sans chiffrement"
    echo "- Utiliser des mots de passe uniques pour chaque environnement"
    echo "- Effectuer une rotation régulière des secrets"
}

# Parse arguments
case "${1:-}" in
    --env)
        ENVIRONMENT="$2"
        ;;
    --help|-h)
        echo "Usage: $0 [--env ENVIRONMENT]"
        echo "  ENVIRONMENT: production, staging (default: production)"
        exit 0
        ;;
esac

main "$@"