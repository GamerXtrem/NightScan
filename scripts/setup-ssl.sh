#!/bin/bash

# Script de configuration SSL/TLS automatique pour NightScan VPS Lite
# Usage: ./setup-ssl.sh [--staging]

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

log() {
    echo -e "${BLUE}[$(date +'%H:%M:%S')] $1${NC}"
}

success() {
    echo -e "${GREEN}✅ $1${NC}"
}

error() {
    echo -e "${RED}❌ $1${NC}"
}

warn() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

# Variables
STAGING=false
ENV_FILE="$PROJECT_ROOT/secrets/production/.env"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --staging)
            STAGING=true
            shift
            ;;
        *)
            echo "Usage: $0 [--staging]"
            exit 1
            ;;
    esac
done

# Charger les variables d'environnement
if [ ! -f "$ENV_FILE" ]; then
    error "Fichier .env non trouvé: $ENV_FILE"
    error "Exécutez d'abord: ./scripts/setup-secrets.sh --env production"
    exit 1
fi

source "$ENV_FILE"

if [ -z "$DOMAIN_NAME" ] || [ -z "$ADMIN_EMAIL" ]; then
    error "Variables DOMAIN_NAME et ADMIN_EMAIL requises dans .env"
    exit 1
fi

log "🔒 Configuration SSL/TLS pour NightScan VPS Lite"
log "=============================================="
log "Domaine: $DOMAIN_NAME"
log "Email: $ADMIN_EMAIL"
log "Mode: $([ "$STAGING" = true ] && echo "STAGING" || echo "PRODUCTION")"

# Vérifier que Docker et docker-compose sont installés
check_dependencies() {
    log "🔍 Vérification des dépendances..."
    
    if ! command -v docker &> /dev/null; then
        error "Docker n'est pas installé"
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        error "Docker Compose n'est pas installé"
        exit 1
    fi
    
    success "Dépendances vérifiées"
}

# Créer la structure de répertoires SSL
setup_ssl_directories() {
    log "📁 Création de la structure SSL..."
    
    cd "$PROJECT_ROOT"
    
    # Créer répertoires avec permissions appropriées
    sudo mkdir -p ssl/letsencrypt ssl/challenges
    sudo mkdir -p logs/nginx logs/certbot logs/ssl-monitor
    sudo mkdir -p nginx/conf.d
    
    # Permissions sécurisées
    sudo chown -R $USER:$USER ssl/ logs/ nginx/
    sudo chmod 755 ssl/ logs/ nginx/
    sudo chmod 700 ssl/letsencrypt
    
    success "Structure SSL créée"
}

# Créer le fichier d'authentification pour monitoring
setup_monitoring_auth() {
    log "🔐 Configuration authentification monitoring..."
    
    if [ ! -f "$PROJECT_ROOT/nginx/.htpasswd" ]; then
        # Générer mot de passe pour monitoring
        MONITORING_PASSWORD=$(openssl rand -base64 16 | tr -d "=+/")
        
        # Créer fichier htpasswd avec htpasswd ou openssl
        if command -v htpasswd &> /dev/null; then
            echo "$MONITORING_PASSWORD" | htpasswd -ci "$PROJECT_ROOT/nginx/.htpasswd" admin
        else
            # Fallback avec openssl
            HASH=$(openssl passwd -apr1 "$MONITORING_PASSWORD")
            echo "admin:$HASH" > "$PROJECT_ROOT/nginx/.htpasswd"
        fi
        
        chmod 600 "$PROJECT_ROOT/nginx/.htpasswd"
        
        # Sauvegarder le mot de passe
        echo "MONITORING_PASSWORD=$MONITORING_PASSWORD" >> "$ENV_FILE"
        
        success "Authentification monitoring configurée"
        warn "Mot de passe monitoring: $MONITORING_PASSWORD"
        warn "Sauvegardez ce mot de passe en lieu sûr!"
    else
        success "Authentification monitoring déjà configurée"
    fi
}

# Configurer le certificat SSL
setup_ssl_certificate() {
    log "🛡️  Configuration certificat SSL..."
    
    cd "$PROJECT_ROOT"
    
    # Créer réseau Docker si nécessaire
    docker network create nightscan_network 2>/dev/null || true
    
    # Démarrer Nginx seul d'abord (pour Let's Encrypt challenge)
    log "Démarrage Nginx temporaire..."
    
    # Configuration temporaire pour obtenir le certificat
    cat > nginx/nginx.temp.conf << EOF
events {
    worker_connections 1024;
}

http {
    server {
        listen 80;
        server_name $DOMAIN_NAME www.$DOMAIN_NAME api.$DOMAIN_NAME monitoring.$DOMAIN_NAME;
        
        location /.well-known/acme-challenge/ {
            root /var/www/certbot;
            allow all;
        }
        
        location / {
            return 200 'NightScan SSL Setup';
            add_header Content-Type text/plain;
        }
    }
}
EOF
    
    # Démarrer Nginx temporaire
    docker run -d --name nginx_temp \
        --network nightscan_network \
        -p 80:80 \
        -v "$PROJECT_ROOT/nginx/nginx.temp.conf:/etc/nginx/nginx.conf:ro" \
        -v "$PROJECT_ROOT/ssl/challenges:/var/www/certbot" \
        nginx:1.25-alpine
    
    sleep 5
    
    # Obtenir le certificat SSL
    log "Demande de certificat SSL..."
    
    if [ "$STAGING" = true ]; then
        STAGING_FLAG="--staging"
        warn "Mode STAGING activé - certificat de test uniquement"
    else
        STAGING_FLAG=""
    fi
    
    # Exécuter Certbot
    docker run --rm \
        --network nightscan_network \
        -v "$PROJECT_ROOT/ssl/letsencrypt:/etc/letsencrypt" \
        -v "$PROJECT_ROOT/ssl/challenges:/var/www/certbot" \
        -v "$PROJECT_ROOT/logs/certbot:/var/log/letsencrypt" \
        certbot/certbot:latest \
        certonly \
        --webroot \
        --webroot-path=/var/www/certbot \
        --email "$ADMIN_EMAIL" \
        --agree-tos \
        --no-eff-email \
        $STAGING_FLAG \
        --domains "$DOMAIN_NAME,www.$DOMAIN_NAME,api.$DOMAIN_NAME,monitoring.$DOMAIN_NAME" \
        --verbose
    
    # Arrêter Nginx temporaire
    docker stop nginx_temp && docker rm nginx_temp
    
    success "Certificat SSL obtenu"
}

# Démarrer les services SSL
start_ssl_services() {
    log "🚀 Démarrage des services SSL..."
    
    cd "$PROJECT_ROOT"
    
    # Substituer les variables dans nginx.conf
    envsubst '${DOMAIN_NAME}' < nginx/nginx.production.conf > nginx/nginx.production.processed.conf
    
    # Démarrer les services SSL
    docker-compose -f docker-compose.ssl.yml up -d
    
    # Attendre que les services soient prêts
    log "Attente démarrage des services..."
    sleep 30
    
    # Vérifier que Nginx fonctionne
    if curl -f -s "http://$DOMAIN_NAME" > /dev/null; then
        success "Nginx opérationnel"
    else
        warn "Nginx peut mettre quelques minutes à être complètement opérationnel"
    fi
    
    # Vérifier HTTPS
    if curl -f -s "https://$DOMAIN_NAME" > /dev/null 2>&1; then
        success "HTTPS opérationnel"
    else
        warn "HTTPS peut mettre quelques minutes à être disponible"
    fi
}

# Test final SSL
test_ssl_configuration() {
    log "🔍 Test de la configuration SSL..."
    
    sleep 10
    
    # Test basique HTTPS
    if curl -f -s -I "https://$DOMAIN_NAME" | grep -q "200 OK"; then
        success "HTTPS fonctionne"
    else
        warn "Test HTTPS échoué - vérifiez les logs"
    fi
    
    # Test redirection HTTP vers HTTPS
    if curl -s -I "http://$DOMAIN_NAME" | grep -q "301"; then
        success "Redirection HTTP → HTTPS active"
    else
        warn "Redirection HTTP → HTTPS non détectée"
    fi
    
    # Test sous-domaines
    for subdomain in www api monitoring; do
        if curl -f -s -I "https://$subdomain.$DOMAIN_NAME" > /dev/null 2>&1; then
            success "Sous-domaine $subdomain HTTPS OK"
        else
            warn "Sous-domaine $subdomain peut nécessiter une configuration supplémentaire"
        fi
    done
}

# Afficher les informations finales
show_ssl_info() {
    log "📋 Configuration SSL terminée"
    echo ""
    echo "🔒 Certificat SSL installé pour:"
    echo "  - https://$DOMAIN_NAME"
    echo "  - https://www.$DOMAIN_NAME" 
    echo "  - https://api.$DOMAIN_NAME"
    echo "  - https://monitoring.$DOMAIN_NAME"
    echo ""
    echo "🔧 Services disponibles:"
    echo "  - Application principale: https://$DOMAIN_NAME"
    echo "  - API de prédiction: https://api.$DOMAIN_NAME"
    echo "  - Monitoring Grafana: https://monitoring.$DOMAIN_NAME"
    echo ""
    echo "📊 Monitoring:"
    echo "  - Username: admin"
    echo "  - Password: $(grep MONITORING_PASSWORD "$ENV_FILE" | cut -d= -f2 2>/dev/null || echo "Voir .env")"
    echo ""
    echo "⚙️  Gestion SSL:"
    echo "  - Renouvellement automatique: ✅ Configuré"
    echo "  - Monitoring expiration: ✅ Actif"
    echo "  - Logs SSL: ./logs/certbot/"
    echo ""
    echo "🔄 Commandes utiles:"
    echo "  - Renouveler SSL: docker-compose -f docker-compose.ssl.yml restart ssl-renewer"
    echo "  - Logs Nginx: docker-compose -f docker-compose.ssl.yml logs nginx"
    echo "  - Status SSL: docker-compose -f docker-compose.ssl.yml ps"
}

# Fonction principale
main() {
    check_dependencies
    setup_ssl_directories
    setup_monitoring_auth
    setup_ssl_certificate
    start_ssl_services
    test_ssl_configuration
    show_ssl_info
    
    success "🎉 Configuration SSL/TLS terminée avec succès!"
    
    if [ "$STAGING" = true ]; then
        warn "Mode STAGING utilisé - pour production, relancez sans --staging"
    fi
}

main "$@"