#!/bin/bash
# Script backup automatisé quotidien pour NightScan
# À configurer dans crontab: 0 2 * * * /opt/nightscan/scripts/backup_automation.sh

set -e

# Configuration
BACKUP_DIR="/opt/nightscan/backups"
RETENTION_DAYS=30
DAILY_RETENTION=7
WEEKLY_RETENTION=4
MONTHLY_RETENTION=12

# Timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
DATE_DAY=$(date +%Y%m%d)
DATE_WEEK=$(date +%Y_W%U)
DATE_MONTH=$(date +%Y%m)

# Couleurs pour logs
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Logging
LOGFILE="$BACKUP_DIR/backup.log"
mkdir -p "$BACKUP_DIR"

log() {
    echo -e "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOGFILE"
}

log_info() {
    log "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    log "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    log "${RED}[ERROR]${NC} $1"
}

# Fonction vérification espace disque
check_disk_space() {
    log_info "Vérification espace disque..."
    
    # Vérifier espace libre (minimum 10GB)
    AVAILABLE_GB=$(df "$BACKUP_DIR" | tail -1 | awk '{print int($4/1024/1024)}')
    
    if [[ $AVAILABLE_GB -lt 10 ]]; then
        log_error "Espace disque insuffisant: ${AVAILABLE_GB}GB disponible"
        # Nettoyage d'urgence
        cleanup_old_backups
        
        # Revérifier
        AVAILABLE_GB=$(df "$BACKUP_DIR" | tail -1 | awk '{print int($4/1024/1024)}')
        if [[ $AVAILABLE_GB -lt 5 ]]; then
            log_error "Espace critique - backup annulé"
            exit 1
        fi
    fi
    
    log_info "Espace disponible: ${AVAILABLE_GB}GB"
}

# Backup base de données PostgreSQL
backup_database() {
    log_info "Backup base de données PostgreSQL..."
    
    local backup_file="$BACKUP_DIR/daily/database_${TIMESTAMP}.sql.gz"
    mkdir -p "$BACKUP_DIR/daily"
    
    # Variables environnement
    export PGPASSWORD="$NIGHTSCAN_DB_PASSWORD"
    
    # Backup avec compression
    pg_dump \
        -h "${NIGHTSCAN_DB_HOST:-localhost}" \
        -p "${NIGHTSCAN_DB_PORT:-5432}" \
        -U "${NIGHTSCAN_DB_USER:-nightscan}" \
        -d "${NIGHTSCAN_DB_NAME:-nightscan}" \
        --no-owner \
        --no-privileges \
        --verbose \
        | gzip > "$backup_file"
    
    if [[ ${PIPESTATUS[0]} -eq 0 ]]; then
        local size=$(du -h "$backup_file" | cut -f1)
        log_info "✅ Database backup: $backup_file ($size)"
        
        # Symlink vers latest
        ln -sf "$backup_file" "$BACKUP_DIR/latest_database.sql.gz"
    else
        log_error "❌ Échec backup database"
        rm -f "$backup_file"
        return 1
    fi
}

# Backup Redis
backup_redis() {
    log_info "Backup Redis..."
    
    local backup_file="$BACKUP_DIR/daily/redis_${TIMESTAMP}.rdb"
    mkdir -p "$BACKUP_DIR/daily"
    
    # Backup RDB
    if command -v redis-cli &> /dev/null; then
        redis-cli \
            -h "${REDIS_HOST:-localhost}" \
            -p "${REDIS_PORT:-6379}" \
            --rdb "$backup_file"
        
        if [[ $? -eq 0 ]]; then
            local size=$(du -h "$backup_file" | cut -f1)
            log_info "✅ Redis backup: $backup_file ($size)"
            
            # Compression
            gzip "$backup_file"
            ln -sf "${backup_file}.gz" "$BACKUP_DIR/latest_redis.rdb.gz"
        else
            log_error "❌ Échec backup Redis"
            return 1
        fi
    else
        log_warn "⚠️ redis-cli non disponible"
    fi
}

# Backup fichiers utilisateur
backup_user_data() {
    log_info "Backup fichiers utilisateur..."
    
    local backup_file="$BACKUP_DIR/daily/user_data_${TIMESTAMP}.tar.gz"
    mkdir -p "$BACKUP_DIR/daily"
    
    # Répertoires à sauvegarder
    local data_dirs=(
        "/opt/nightscan/uploads"
        "/opt/nightscan/models"
        "/opt/nightscan/config"
        "/opt/nightscan/logs"
    )
    
    # Créer archive avec exclusions
    tar -czf "$backup_file" \
        --exclude="*.tmp" \
        --exclude="*.log.gz" \
        --exclude="cache/*" \
        "${data_dirs[@]}" 2>/dev/null || true
    
    if [[ -f "$backup_file" ]]; then
        local size=$(du -h "$backup_file" | cut -f1)
        log_info "✅ User data backup: $backup_file ($size)"
        
        ln -sf "$backup_file" "$BACKUP_DIR/latest_user_data.tar.gz"
    else
        log_warn "⚠️ Aucune donnée utilisateur trouvée"
    fi
}

# Backup configuration système
backup_system_config() {
    log_info "Backup configuration système..."
    
    local backup_file="$BACKUP_DIR/daily/system_config_${TIMESTAMP}.tar.gz"
    mkdir -p "$BACKUP_DIR/daily"
    
    # Configurations système importantes
    local config_dirs=(
        "/etc/nginx/sites-available/nightscan"
        "/etc/systemd/system/nightscan*"
        "/etc/ssl/certs/nightscan*"
        "/etc/crontab"
        "/etc/logrotate.d/nightscan"
    )
    
    tar -czf "$backup_file" "${config_dirs[@]}" 2>/dev/null || true
    
    if [[ -f "$backup_file" ]]; then
        local size=$(du -h "$backup_file" | cut -f1)
        log_info "✅ System config backup: $backup_file ($size)"
    fi
}

# Backup code source (si repository local)
backup_source_code() {
    log_info "Backup code source..."
    
    if [[ -d "/opt/nightscan/.git" ]]; then
        local backup_file="$BACKUP_DIR/daily/source_code_${TIMESTAMP}.tar.gz"
        
        cd /opt/nightscan
        
        # Archive avec métadonnées git
        tar -czf "$backup_file" \
            --exclude="node_modules" \
            --exclude="__pycache__" \
            --exclude=".pytest_cache" \
            --exclude="htmlcov" \
            --exclude="logs/*" \
            .
        
        if [[ -f "$backup_file" ]]; then
            local size=$(du -h "$backup_file" | cut -f1)
            log_info "✅ Source code backup: $backup_file ($size)"
            
            # Capture état git
            git log -1 --oneline > "$BACKUP_DIR/daily/git_state_${TIMESTAMP}.txt"
        fi
    else
        log_warn "⚠️ Repository git non trouvé"
    fi
}

# Upload vers stockage distant
upload_to_remote() {
    log_info "Upload vers stockage distant..."
    
    # S3 backup (si configuré)
    if [[ -n "$AWS_BACKUP_BUCKET" ]] && command -v aws &> /dev/null; then
        log_info "Upload vers S3: $AWS_BACKUP_BUCKET"
        
        aws s3 sync "$BACKUP_DIR/daily/" \
            "s3://$AWS_BACKUP_BUCKET/nightscan/daily/$DATE_DAY/" \
            --storage-class STANDARD_IA \
            --only-show-errors
        
        if [[ $? -eq 0 ]]; then
            log_info "✅ Upload S3 réussi"
        else
            log_error "❌ Échec upload S3"
        fi
    fi
    
    # Backup FTP/SFTP (si configuré)
    if [[ -n "$BACKUP_FTP_SERVER" ]]; then
        log_info "Upload vers FTP: $BACKUP_FTP_SERVER"
        
        # Upload via lftp
        lftp -u "$BACKUP_FTP_USER,$BACKUP_FTP_PASSWORD" "$BACKUP_FTP_SERVER" << EOF
mirror -R "$BACKUP_DIR/daily" /nightscan/daily/$DATE_DAY
bye
EOF
        
        if [[ $? -eq 0 ]]; then
            log_info "✅ Upload FTP réussi"
        else
            log_error "❌ Échec upload FTP"
        fi
    fi
    
    # Rsync vers serveur distant (si configuré)
    if [[ -n "$BACKUP_RSYNC_HOST" ]]; then
        log_info "Upload via rsync: $BACKUP_RSYNC_HOST"
        
        rsync -avz --delete \
            "$BACKUP_DIR/daily/" \
            "$BACKUP_RSYNC_USER@$BACKUP_RSYNC_HOST:$BACKUP_RSYNC_PATH/daily/$DATE_DAY/"
        
        if [[ $? -eq 0 ]]; then
            log_info "✅ Upload rsync réussi"
        else
            log_error "❌ Échec upload rsync"
        fi
    fi
}

# Création backups hebdomadaires et mensuels
create_periodic_backups() {
    log_info "Création backups périodiques..."
    
    # Backup hebdomadaire (dimanche)
    if [[ $(date +%u) -eq 7 ]]; then
        log_info "Création backup hebdomadaire..."
        
        mkdir -p "$BACKUP_DIR/weekly"
        
        # Copier backup quotidien vers hebdomadaire
        cp -r "$BACKUP_DIR/daily" "$BACKUP_DIR/weekly/$DATE_WEEK"
        
        log_info "✅ Backup hebdomadaire créé: $DATE_WEEK"
    fi
    
    # Backup mensuel (1er du mois)
    if [[ $(date +%d) -eq 01 ]]; then
        log_info "Création backup mensuel..."
        
        mkdir -p "$BACKUP_DIR/monthly"
        
        # Copier backup quotidien vers mensuel
        cp -r "$BACKUP_DIR/daily" "$BACKUP_DIR/monthly/$DATE_MONTH"
        
        log_info "✅ Backup mensuel créé: $DATE_MONTH"
    fi
}

# Vérification intégrité backups
verify_backup_integrity() {
    log_info "Vérification intégrité backups..."
    
    # Test restoration database (superficiel)
    if [[ -f "$BACKUP_DIR/latest_database.sql.gz" ]]; then
        if zcat "$BACKUP_DIR/latest_database.sql.gz" | head -10 | grep -q "PostgreSQL"; then
            log_info "✅ Backup database semble valide"
        else
            log_error "❌ Backup database corrompu"
        fi
    fi
    
    # Test archives tar
    for archive in "$BACKUP_DIR"/daily/*.tar.gz; do
        if [[ -f "$archive" ]]; then
            if tar -tzf "$archive" > /dev/null 2>&1; then
                log_info "✅ Archive valide: $(basename "$archive")"
            else
                log_error "❌ Archive corrompue: $(basename "$archive")"
                rm -f "$archive"
            fi
        fi
    done
}

# Nettoyage anciens backups
cleanup_old_backups() {
    log_info "Nettoyage anciens backups..."
    
    # Nettoyage backups quotidiens
    find "$BACKUP_DIR/daily" -type f -mtime +$DAILY_RETENTION -delete 2>/dev/null || true
    
    # Nettoyage backups hebdomadaires
    if [[ -d "$BACKUP_DIR/weekly" ]]; then
        ls -t "$BACKUP_DIR/weekly" | tail -n +$((WEEKLY_RETENTION + 1)) | \
            xargs -r -I {} rm -rf "$BACKUP_DIR/weekly/{}"
    fi
    
    # Nettoyage backups mensuels
    if [[ -d "$BACKUP_DIR/monthly" ]]; then
        ls -t "$BACKUP_DIR/monthly" | tail -n +$((MONTHLY_RETENTION + 1)) | \
            xargs -r -I {} rm -rf "$BACKUP_DIR/monthly/{}"
    fi
    
    # Nettoyage logs anciens
    find "$BACKUP_DIR" -name "*.log" -mtime +30 -delete 2>/dev/null || true
    
    log_info "✅ Nettoyage terminé"
}

# Envoi rapport backup
send_backup_report() {
    log_info "Envoi rapport backup..."
    
    local report_file="/tmp/backup_report_${TIMESTAMP}.txt"
    
    cat > "$report_file" << EOF
NightScan - Rapport Backup Quotidien
====================================
Date: $(date)
Serveur: $(hostname)

RÉSUMÉ:
- Database: $(test -f "$BACKUP_DIR/latest_database.sql.gz" && echo "✅ OK" || echo "❌ ÉCHEC")
- Redis: $(test -f "$BACKUP_DIR/latest_redis.rdb.gz" && echo "✅ OK" || echo "❌ ÉCHEC")
- User Data: $(test -f "$BACKUP_DIR/latest_user_data.tar.gz" && echo "✅ OK" || echo "❌ ÉCHEC")

TAILLES:
$(du -h "$BACKUP_DIR"/latest_* 2>/dev/null || echo "Aucun backup récent")

ESPACE DISQUE:
$(df -h "$BACKUP_DIR")

LOGS (dernières 20 lignes):
$(tail -20 "$LOGFILE")
EOF
    
    # Envoi email si configuré
    if [[ -n "$BACKUP_EMAIL" ]] && command -v mail &> /dev/null; then
        mail -s "NightScan Backup Report - $(date +%Y-%m-%d)" "$BACKUP_EMAIL" < "$report_file"
        log_info "✅ Rapport envoyé à $BACKUP_EMAIL"
    fi
    
    # Notification Slack si configuré
    if [[ -n "$SLACK_WEBHOOK_URL" ]]; then
        local status_emoji="✅"
        local status_color="good"
        
        # Vérifier si tous les backups sont OK
        if ! test -f "$BACKUP_DIR/latest_database.sql.gz"; then
            status_emoji="❌"
            status_color="danger"
        fi
        
        curl -X POST -H 'Content-type: application/json' \
            --data "{
                \"text\":\"$status_emoji NightScan Backup - $(date +%Y-%m-%d)\",
                \"color\":\"$status_color\",
                \"attachments\":[{
                    \"text\":\"$(tail -10 "$LOGFILE" | sed 's/"/\\"/g')\"
                }]
            }" \
            "$SLACK_WEBHOOK_URL"
        
        log_info "✅ Notification Slack envoyée"
    fi
    
    rm -f "$report_file"
}

# Test restoration (optionnel, lourd)
test_restoration() {
    if [[ "$1" == "--test-restore" ]]; then
        log_info "Test restoration (mode test)..."
        
        # Créer base test temporaire
        local test_db="nightscan_restore_test_$$"
        
        createdb "$test_db" 2>/dev/null || true
        
        # Test restoration database
        if [[ -f "$BACKUP_DIR/latest_database.sql.gz" ]]; then
            zcat "$BACKUP_DIR/latest_database.sql.gz" | \
                psql "$test_db" > /dev/null 2>&1
            
            if [[ $? -eq 0 ]]; then
                log_info "✅ Test restoration database OK"
            else
                log_error "❌ Test restoration database échec"
            fi
            
            # Nettoyage
            dropdb "$test_db" 2>/dev/null || true
        fi
    fi
}

# Fonction principale
main() {
    log_info "🚀 DÉBUT BACKUP AUTOMATISÉ NIGHTSCAN"
    log_info "===================================="
    
    # Trap pour nettoyage en cas d'interruption
    trap 'log_error "Backup interrompu"; exit 1' INT TERM
    
    # Vérifications préliminaires
    check_disk_space
    
    # Backups principaux
    backup_database
    backup_redis
    backup_user_data
    backup_system_config
    backup_source_code
    
    # Upload distant
    upload_to_remote
    
    # Backups périodiques
    create_periodic_backups
    
    # Vérifications
    verify_backup_integrity
    
    # Test restoration si demandé
    test_restoration "$@"
    
    # Nettoyage
    cleanup_old_backups
    
    # Rapport
    send_backup_report
    
    log_info "✅ BACKUP AUTOMATISÉ TERMINÉ"
    log_info "=============================="
}

# Gestion options
case "${1:-}" in
    --test-restore)
        main --test-restore
        ;;
    --cleanup-only)
        cleanup_old_backups
        ;;
    --verify-only)
        verify_backup_integrity
        ;;
    --help)
        echo "Usage: $0 [--test-restore|--cleanup-only|--verify-only|--help]"
        echo ""
        echo "Options:"
        echo "  --test-restore  : Effectue un test de restoration"
        echo "  --cleanup-only  : Nettoyage uniquement"
        echo "  --verify-only   : Vérification intégrité uniquement"
        echo "  --help          : Affiche cette aide"
        echo ""
        echo "Variables environnement:"
        echo "  AWS_BACKUP_BUCKET    : Bucket S3 pour backup distant"
        echo "  BACKUP_EMAIL         : Email pour rapport backup"
        echo "  SLACK_WEBHOOK_URL    : Webhook Slack pour notifications"
        echo "  BACKUP_FTP_SERVER    : Serveur FTP backup"
        echo "  BACKUP_RSYNC_HOST    : Serveur rsync backup"
        ;;
    *)
        main "$@"
        ;;
esac