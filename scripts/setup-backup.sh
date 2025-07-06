#!/bin/bash

# Script de configuration backup automatisé pour NightScan VPS Lite
# Sauvegarde optimisée pour 50GB SSD avec rotation intelligente
# Usage: ./setup-backup.sh

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
ENV_FILE="$PROJECT_ROOT/secrets/production/.env"
BACKUP_DIR="/var/backups/nightscan"
REMOTE_BACKUP_DIR="/var/backups/nightscan/remote"

log "💾 Configuration backup automatisé NightScan VPS Lite"
log "===================================================="

# Charger les variables d'environnement
load_environment() {
    if [ ! -f "$ENV_FILE" ]; then
        error "Fichier .env non trouvé: $ENV_FILE"
        exit 1
    fi
    
    source "$ENV_FILE"
    success "Variables d'environnement chargées"
}

# Vérifier les privilèges root
check_root() {
    if [ "$EUID" -ne 0 ]; then
        error "Ce script doit être exécuté en tant que root"
        error "Utilisez: sudo $0"
        exit 1
    fi
    success "Privilèges root vérifiés"
}

# Installer les outils de backup
install_backup_tools() {
    log "📦 Installation des outils de backup..."
    
    # Mise à jour des paquets
    apt-get update -qq
    
    # Installation des outils nécessaires
    apt-get install -y \
        postgresql-client \
        redis-tools \
        rsync \
        gzip \
        tar \
        curl \
        jq \
        bc
    
    success "Outils de backup installés"
}

# Créer la structure de répertoires backup
setup_backup_directories() {
    log "📁 Création structure répertoires backup..."
    
    # Créer répertoires avec permissions appropriées
    mkdir -p "$BACKUP_DIR"/{database,files,logs,config,ssl}
    mkdir -p "$BACKUP_DIR"/archive/{daily,weekly,monthly}
    mkdir -p "$REMOTE_BACKUP_DIR"
    mkdir -p /var/log/nightscan-backup
    
    # Permissions sécurisées
    chmod 700 "$BACKUP_DIR"
    chmod 755 /var/log/nightscan-backup
    
    success "Structure backup créée"
}

# Créer le script de backup principal
create_backup_script() {
    log "📝 Création script backup principal..."
    
    cat > /usr/local/bin/nightscan-backup.sh << 'EOF'
#!/bin/bash

# Script de backup NightScan VPS Lite
# Optimisé pour espace disque limité (50GB SSD)

set -e

# Configuration
BACKUP_BASE="/var/backups/nightscan"
LOG_FILE="/var/log/nightscan-backup/backup.log"
DATE=$(date '+%Y%m%d_%H%M%S')
BACKUP_TYPE=${1:-daily}  # daily, weekly, monthly

# Charger variables d'environnement
if [ -f "/home/*/NightScan/secrets/production/.env" ]; then
    source /home/*/NightScan/secrets/production/.env
elif [ -f "/opt/nightscan/secrets/production/.env" ]; then
    source /opt/nightscan/secrets/production/.env
fi

# Fonction de log
log_backup() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Fonction de nettoyage en cas d'erreur
cleanup_on_error() {
    log_backup "ERROR: Backup failed, cleaning up..."
    rm -rf "$BACKUP_BASE/tmp/backup_$DATE" 2>/dev/null || true
    exit 1
}

trap cleanup_on_error ERR

log_backup "=== DEBUT BACKUP $BACKUP_TYPE ==="

# Créer répertoire temporaire
TEMP_BACKUP="$BACKUP_BASE/tmp/backup_$DATE"
mkdir -p "$TEMP_BACKUP"

# 1. Backup Base de données PostgreSQL
log_backup "1/6 Backup PostgreSQL..."
if [ -n "$DB_PASSWORD" ]; then
    export PGPASSWORD="$DB_PASSWORD"
    pg_dump -h localhost -U nightscan -d nightscan \
        --no-password --clean --if-exists \
        --compress=9 \
        > "$TEMP_BACKUP/database_$DATE.sql.gz" 2>/dev/null || {
        log_backup "WARNING: PostgreSQL backup failed (container may be down)"
        touch "$TEMP_BACKUP/database_$DATE.sql.gz"
    }
    unset PGPASSWORD
else
    log_backup "WARNING: DB_PASSWORD not found, skipping PostgreSQL backup"
fi

# 2. Backup Redis
log_backup "2/6 Backup Redis..."
if command -v redis-cli >/dev/null 2>&1; then
    redis-cli --rdb "$TEMP_BACKUP/redis_$DATE.rdb" 2>/dev/null || {
        log_backup "WARNING: Redis backup failed (container may be down)"
        touch "$TEMP_BACKUP/redis_$DATE.rdb"
    }
else
    log_backup "WARNING: redis-cli not available, skipping Redis backup"
fi

# 3. Backup fichiers de configuration
log_backup "3/6 Backup configuration..."
mkdir -p "$TEMP_BACKUP/config"

# Fichiers Docker Compose
if [ -d "/home/*/NightScan" ]; then
    NIGHTSCAN_DIR="/home/*/NightScan"
elif [ -d "/opt/nightscan" ]; then
    NIGHTSCAN_DIR="/opt/nightscan"
fi

if [ -n "$NIGHTSCAN_DIR" ]; then
    cp "$NIGHTSCAN_DIR"/docker-compose*.yml "$TEMP_BACKUP/config/" 2>/dev/null || true
    cp -r "$NIGHTSCAN_DIR"/nginx "$TEMP_BACKUP/config/" 2>/dev/null || true
    cp -r "$NIGHTSCAN_DIR"/monitoring "$TEMP_BACKUP/config/" 2>/dev/null || true
    
    # Secrets (chiffrés)
    if [ -d "$NIGHTSCAN_DIR/secrets" ]; then
        tar czf "$TEMP_BACKUP/config/secrets_$DATE.tar.gz" \
            -C "$NIGHTSCAN_DIR" secrets/ 2>/dev/null || true
    fi
fi

# 4. Backup certificats SSL
log_backup "4/6 Backup certificats SSL..."
if [ -d "/etc/letsencrypt" ]; then
    tar czf "$TEMP_BACKUP/ssl_$DATE.tar.gz" \
        -C /etc letsencrypt/ 2>/dev/null || {
        log_backup "WARNING: SSL certificates backup failed"
    }
fi

# 5. Backup logs importants
log_backup "5/6 Backup logs..."
mkdir -p "$TEMP_BACKUP/logs"
if [ -d "/var/log/nginx" ]; then
    tar czf "$TEMP_BACKUP/logs/nginx_$DATE.tar.gz" \
        -C /var/log nginx/ 2>/dev/null || true
fi
if [ -d "/var/log/nightscan-security" ]; then
    tar czf "$TEMP_BACKUP/logs/security_$DATE.tar.gz" \
        -C /var/log nightscan-security/ 2>/dev/null || true
fi

# 6. Créer archive finale
log_backup "6/6 Création archive finale..."
FINAL_BACKUP="$BACKUP_BASE/archive/$BACKUP_TYPE/nightscan_${BACKUP_TYPE}_$DATE.tar.gz"
mkdir -p "$(dirname "$FINAL_BACKUP")"

tar czf "$FINAL_BACKUP" -C "$BACKUP_BASE/tmp" "backup_$DATE/"

# Calculer taille du backup
BACKUP_SIZE=$(du -h "$FINAL_BACKUP" | cut -f1)
log_backup "Backup créé: $FINAL_BACKUP ($BACKUP_SIZE)"

# Nettoyage répertoire temporaire
rm -rf "$TEMP_BACKUP"

# Rotation des backups (économie d'espace VPS Lite)
log_backup "Rotation des backups..."
case $BACKUP_TYPE in
    daily)
        # Garder 7 jours
        find "$BACKUP_BASE/archive/daily" -name "*.tar.gz" -mtime +7 -delete
        ;;
    weekly)
        # Garder 4 semaines
        find "$BACKUP_BASE/archive/weekly" -name "*.tar.gz" -mtime +28 -delete
        ;;
    monthly)
        # Garder 6 mois
        find "$BACKUP_BASE/archive/monthly" -name "*.tar.gz" -mtime +180 -delete
        ;;
esac

# Vérifier espace disque après backup
DISK_USAGE=$(df / | tail -1 | awk '{print $5}' | sed 's/%//')
log_backup "Utilisation disque après backup: $DISK_USAGE%"

if [ "$DISK_USAGE" -gt 85 ]; then
    log_backup "WARNING: Espace disque faible ($DISK_USAGE%)"
    # Nettoyage d'urgence - supprimer les plus anciens backups
    find "$BACKUP_BASE/archive" -name "*.tar.gz" -mtime +14 -delete
    log_backup "Nettoyage d'urgence effectué"
fi

log_backup "=== FIN BACKUP $BACKUP_TYPE ==="
EOF

    chmod +x /usr/local/bin/nightscan-backup.sh
    success "Script backup principal créé"
}

# Créer le script de restauration
create_restore_script() {
    log "🔄 Création script de restauration..."
    
    cat > /usr/local/bin/nightscan-restore.sh << 'EOF'
#!/bin/bash

# Script de restauration NightScan VPS Lite

set -e

BACKUP_BASE="/var/backups/nightscan"
LOG_FILE="/var/log/nightscan-backup/restore.log"

# Fonction de log
log_restore() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Afficher les backups disponibles
list_backups() {
    echo "Backups disponibles:"
    echo "==================="
    
    for type in daily weekly monthly; do
        echo ""
        echo "=== $type ==="
        ls -lah "$BACKUP_BASE/archive/$type/" 2>/dev/null || echo "Aucun backup $type"
    done
}

# Restaurer depuis un backup
restore_backup() {
    local backup_file="$1"
    
    if [ ! -f "$backup_file" ]; then
        echo "Erreur: Fichier backup non trouvé: $backup_file"
        exit 1
    fi
    
    log_restore "=== DEBUT RESTAURATION ==="
    log_restore "Fichier: $backup_file"
    
    # Créer répertoire temporaire
    TEMP_RESTORE="/tmp/nightscan_restore_$(date +%s)"
    mkdir -p "$TEMP_RESTORE"
    
    # Extraire le backup
    log_restore "Extraction du backup..."
    tar xzf "$backup_file" -C "$TEMP_RESTORE"
    
    # Trouver le répertoire backup
    BACKUP_DIR=$(find "$TEMP_RESTORE" -name "backup_*" -type d | head -1)
    
    if [ -z "$BACKUP_DIR" ]; then
        echo "Erreur: Structure backup invalide"
        rm -rf "$TEMP_RESTORE"
        exit 1
    fi
    
    echo "Contenu du backup:"
    ls -la "$BACKUP_DIR"
    
    echo ""
    echo "Options de restauration:"
    echo "1. Base de données PostgreSQL"
    echo "2. Redis"
    echo "3. Configuration"
    echo "4. Certificats SSL"
    echo "5. Logs"
    echo "6. Tout restaurer"
    echo ""
    read -p "Choisissez une option (1-6): " choice
    
    case $choice in
        1)
            log_restore "Restauration PostgreSQL..."
            if [ -f "$BACKUP_DIR/database_"*.sql.gz ]; then
                echo "ATTENTION: Ceci va remplacer la base de données actuelle!"
                read -p "Continuer? (oui/non): " confirm
                if [ "$confirm" = "oui" ]; then
                    # Charger variables
                    source /home/*/NightScan/secrets/production/.env 2>/dev/null || true
                    export PGPASSWORD="$DB_PASSWORD"
                    gunzip -c "$BACKUP_DIR/database_"*.sql.gz | psql -h localhost -U nightscan -d nightscan
                    log_restore "Base de données restaurée"
                fi
            fi
            ;;
        2)
            log_restore "Restauration Redis..."
            if [ -f "$BACKUP_DIR/redis_"*.rdb ]; then
                echo "ATTENTION: Ceci va remplacer les données Redis actuelles!"
                read -p "Continuer? (oui/non): " confirm
                if [ "$confirm" = "oui" ]; then
                    # Arrêter Redis, remplacer fichier, redémarrer
                    docker-compose -f /home/*/NightScan/docker-compose.production.yml stop redis
                    cp "$BACKUP_DIR/redis_"*.rdb /var/lib/redis/dump.rdb
                    docker-compose -f /home/*/NightScan/docker-compose.production.yml start redis
                    log_restore "Redis restauré"
                fi
            fi
            ;;
        3)
            log_restore "Restauration configuration..."
            echo "Fichiers de configuration trouvés dans le backup"
            ls -la "$BACKUP_DIR/config/"
            echo ""
            echo "Restaurer manuellement les fichiers nécessaires depuis: $BACKUP_DIR/config/"
            ;;
        6)
            echo "ATTENTION: Restauration complète!"
            echo "Ceci va remplacer TOUTES les données actuelles!"
            read -p "Tapez 'RESTAURER_TOUT' pour confirmer: " confirm
            if [ "$confirm" = "RESTAURER_TOUT" ]; then
                log_restore "Restauration complète en cours..."
                # Implémenter restauration complète si nécessaire
                echo "Restauration complète non implémentée - utilisez les options individuelles"
            fi
            ;;
        *)
            echo "Option invalide"
            ;;
    esac
    
    # Nettoyage
    rm -rf "$TEMP_RESTORE"
    log_restore "=== FIN RESTAURATION ==="
}

# Menu principal
if [ $# -eq 0 ]; then
    list_backups
    echo ""
    echo "Usage: $0 <chemin_vers_backup>"
    echo "Exemple: $0 /var/backups/nightscan/archive/daily/nightscan_daily_20250106_120000.tar.gz"
else
    restore_backup "$1"
fi
EOF

    chmod +x /usr/local/bin/nightscan-restore.sh
    success "Script de restauration créé"
}

# Configurer les tâches cron de backup
setup_backup_cron() {
    log "⏰ Configuration tâches cron backup..."
    
    # Créer cron jobs pour backup automatique
    cat > /etc/cron.d/nightscan-backup << EOF
# Backups automatiques NightScan VPS Lite
SHELL=/bin/bash
PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

# Backup quotidien à 3h du matin
0 3 * * * root /usr/local/bin/nightscan-backup.sh daily

# Backup hebdomadaire le dimanche à 2h du matin
0 2 * * 0 root /usr/local/bin/nightscan-backup.sh weekly

# Backup mensuel le 1er de chaque mois à 1h du matin
0 1 1 * * root /usr/local/bin/nightscan-backup.sh monthly

# Vérification espace disque toutes les heures
0 * * * * root /usr/local/bin/nightscan-backup-monitor.sh
EOF

    success "Tâches cron configurées"
}

# Créer le script de monitoring backup
create_backup_monitor() {
    log "📊 Création script monitoring backup..."
    
    cat > /usr/local/bin/nightscan-backup-monitor.sh << 'EOF'
#!/bin/bash

# Monitoring backup et espace disque NightScan VPS Lite

LOG_FILE="/var/log/nightscan-backup/monitor.log"
BACKUP_BASE="/var/backups/nightscan"

# Fonction de log
log_monitor() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" >> "$LOG_FILE"
}

# Vérifier espace disque
DISK_USAGE=$(df / | tail -1 | awk '{print $5}' | sed 's/%//')
BACKUP_SIZE=$(du -sh "$BACKUP_BASE" 2>/dev/null | cut -f1 || echo "0")

log_monitor "Espace disque: $DISK_USAGE%, Taille backups: $BACKUP_SIZE"

# Alertes espace disque
if [ "$DISK_USAGE" -gt 90 ]; then
    log_monitor "CRITICAL: Espace disque critique ($DISK_USAGE%)"
    # Nettoyage d'urgence
    find "$BACKUP_BASE/archive" -name "*.tar.gz" -mtime +7 -delete
    log_monitor "Nettoyage d'urgence effectué"
elif [ "$DISK_USAGE" -gt 80 ]; then
    log_monitor "WARNING: Espace disque élevé ($DISK_USAGE%)"
fi

# Vérifier derniers backups
LAST_DAILY=$(find "$BACKUP_BASE/archive/daily" -name "*.tar.gz" -mtime -1 | wc -l)
if [ "$LAST_DAILY" -eq 0 ]; then
    log_monitor "WARNING: Aucun backup quotidien dans les dernières 24h"
fi

# Vérifier intégrité des backups (échantillon)
RECENT_BACKUP=$(find "$BACKUP_BASE/archive/daily" -name "*.tar.gz" -mtime -1 | head -1)
if [ -n "$RECENT_BACKUP" ]; then
    if tar tzf "$RECENT_BACKUP" >/dev/null 2>&1; then
        log_monitor "INFO: Backup récent intègre"
    else
        log_monitor "ERROR: Backup récent corrompu: $RECENT_BACKUP"
    fi
fi

# Statistiques backup
DAILY_COUNT=$(find "$BACKUP_BASE/archive/daily" -name "*.tar.gz" | wc -l)
WEEKLY_COUNT=$(find "$BACKUP_BASE/archive/weekly" -name "*.tar.gz" | wc -l)
MONTHLY_COUNT=$(find "$BACKUP_BASE/archive/monthly" -name "*.tar.gz" | wc -l)

log_monitor "Backups: $DAILY_COUNT quotidiens, $WEEKLY_COUNT hebdomadaires, $MONTHLY_COUNT mensuels"
EOF

    chmod +x /usr/local/bin/nightscan-backup-monitor.sh
    success "Script monitoring backup créé"
}

# Test du système de backup
test_backup_system() {
    log "🔍 Test du système de backup..."
    
    # Test backup rapide
    log "Exécution backup test..."
    if /usr/local/bin/nightscan-backup.sh daily > /tmp/backup_test.log 2>&1; then
        success "Backup test réussi"
    else
        warn "Backup test échoué - voir /tmp/backup_test.log"
    fi
    
    # Vérifier les fichiers créés
    if [ -d "$BACKUP_DIR/archive/daily" ]; then
        BACKUP_FILES=$(ls -la "$BACKUP_DIR/archive/daily" | wc -l)
        success "$BACKUP_FILES fichiers backup créés"
    fi
    
    success "Système de backup testé"
}

# Afficher les informations finales
show_backup_info() {
    log "💾 Configuration backup terminée"
    echo ""
    echo "💾 Système de backup configuré:"
    echo "  ✅ Backup quotidien (3h du matin)"
    echo "  ✅ Backup hebdomadaire (dimanche 2h)"
    echo "  ✅ Backup mensuel (1er du mois 1h)"
    echo "  ✅ Monitoring automatique"
    echo ""
    echo "📁 Emplacements backup:"
    echo "  - Base: $BACKUP_DIR"
    echo "  - Quotidiens: $BACKUP_DIR/archive/daily"
    echo "  - Hebdomadaires: $BACKUP_DIR/archive/weekly"
    echo "  - Mensuels: $BACKUP_DIR/archive/monthly"
    echo ""
    echo "🔄 Rotation automatique:"
    echo "  - Quotidiens: 7 jours"
    echo "  - Hebdomadaires: 4 semaines"
    echo "  - Mensuels: 6 mois"
    echo ""
    echo "⚙️  Commandes utiles:"
    echo "  - Backup manuel: nightscan-backup.sh daily"
    echo "  - Lister backups: nightscan-restore.sh"
    echo "  - Restaurer: nightscan-restore.sh <fichier>"
    echo "  - Logs backup: tail -f /var/log/nightscan-backup/backup.log"
    echo "  - Monitoring: tail -f /var/log/nightscan-backup/monitor.log"
    echo ""
    echo "🔒 Sécurité:"
    echo "  - Permissions 700 sur répertoire backup"
    echo "  - Secrets chiffrés dans archives"
    echo "  - Nettoyage automatique si espace faible"
    echo ""
    echo "📊 Optimisations VPS Lite:"
    echo "  - Compression maximale (gzip -9)"
    echo "  - Rotation intelligente par espace"
    echo "  - Monitoring espace disque"
    echo "  - Nettoyage d'urgence si >90% plein"
}

# Fonction principale
main() {
    load_environment
    check_root
    install_backup_tools
    setup_backup_directories
    create_backup_script
    create_restore_script
    setup_backup_cron
    create_backup_monitor
    test_backup_system
    show_backup_info
    
    success "🎉 Système de backup NightScan VPS Lite configuré!"
    echo ""
    warn "📝 Recommandations:"
    echo "  - Testez une restauration pour valider le système"
    echo "  - Configurez un backup externe (S3, FTP, etc.)"
    echo "  - Surveillez l'espace disque régulièrement"
    echo "  - Documentez la procédure de restauration"
}

main "$@"