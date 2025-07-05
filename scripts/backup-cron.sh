#!/bin/bash

# NightScan Automated Backup Script
# This script is designed to be run via cron for regular backups

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LOG_DIR="/var/log/nightscan"
LOG_FILE="$LOG_DIR/backup.log"
LOCK_FILE="/tmp/nightscan_backup.lock"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Create log directory if it doesn't exist
mkdir -p "$LOG_DIR"

# Logging function
log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$LOG_FILE"
}

error() {
    echo -e "${RED}$(date '+%Y-%m-%d %H:%M:%S') - ERROR: $1${NC}" | tee -a "$LOG_FILE"
}

success() {
    echo -e "${GREEN}$(date '+%Y-%m-%d %H:%M:%S') - SUCCESS: $1${NC}" | tee -a "$LOG_FILE"
}

warn() {
    echo -e "${YELLOW}$(date '+%Y-%m-%d %H:%M:%S') - WARNING: $1${NC}" | tee -a "$LOG_FILE"
}

# Check if another backup is already running
check_lock() {
    if [ -f "$LOCK_FILE" ]; then
        local pid=$(cat "$LOCK_FILE")
        if ps -p "$pid" > /dev/null 2>&1; then
            error "Another backup process is already running (PID: $pid)"
            exit 1
        else
            warn "Stale lock file found, removing it"
            rm -f "$LOCK_FILE"
        fi
    fi
    
    # Create lock file
    echo $$ > "$LOCK_FILE"
}

# Remove lock file on exit
cleanup() {
    rm -f "$LOCK_FILE"
}

trap cleanup EXIT

# Check disk space before backup
check_disk_space() {
    local backup_dir="${BACKUP_DIR:-/var/backups/nightscan}"
    local min_space_gb=5
    
    if ! command -v df > /dev/null; then
        warn "df command not available, skipping disk space check"
        return 0
    fi
    
    local available_space=$(df "$backup_dir" | awk 'NR==2 {print int($4/1024/1024)}')
    
    if [ "$available_space" -lt "$min_space_gb" ]; then
        error "Insufficient disk space: ${available_space}GB available, ${min_space_gb}GB required"
        return 1
    fi
    
    log "Disk space check passed: ${available_space}GB available"
    return 0
}

# Send notification (implement based on your notification system)
send_notification() {
    local type="$1"
    local message="$2"
    
    # Example: send email notification
    if command -v mail > /dev/null; then
        echo "$message" | mail -s "NightScan Backup $type" admin@yourdomain.com
    fi
    
    # Example: send to Slack webhook
    if [ ! -z "$SLACK_WEBHOOK_URL" ]; then
        curl -X POST -H 'Content-type: application/json' \
            --data "{\"text\":\"NightScan Backup $type: $message\"}" \
            "$SLACK_WEBHOOK_URL" > /dev/null 2>&1
    fi
    
    # Example: write to syslog
    logger -t nightscan-backup "$type: $message"
}

# Run backup based on schedule type
run_backup() {
    local backup_type="${1:-full}"
    local start_time=$(date +%s)
    
    log "Starting $backup_type backup..."
    
    # Change to project directory
    cd "$PROJECT_DIR"
    
    # Set environment variables
    export PYTHONPATH="$PROJECT_DIR"
    
    # Run the backup
    local backup_cmd="python backup_system.py backup --type=$backup_type"
    
    # Add flags based on backup type
    case "$backup_type" in
        "quick")
            backup_cmd="$backup_cmd --no-models --no-cloud"
            ;;
        "database-only")
            backup_cmd="$backup_cmd --no-uploads --no-models --no-cloud"
            ;;
        "full")
            # Full backup includes everything
            ;;
    esac
    
    # Execute backup command and capture output
    local backup_output
    local backup_exit_code
    
    if backup_output=$(eval "$backup_cmd" 2>&1); then
        backup_exit_code=0
    else
        backup_exit_code=$?
    fi
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    # Parse backup output for key information
    local backup_id=$(echo "$backup_output" | grep -o '"backup_id": "[^"]*"' | cut -d'"' -f4)
    local backup_size=$(echo "$backup_output" | grep -o '"size_bytes": [0-9]*' | cut -d' ' -f2)
    
    if [ $backup_exit_code -eq 0 ]; then
        success "$backup_type backup completed in ${duration}s (ID: $backup_id, Size: $backup_size bytes)"
        send_notification "SUCCESS" "$backup_type backup completed successfully in ${duration}s"
        
        # Log backup details
        echo "$backup_output" >> "$LOG_FILE"
        
        return 0
    else
        error "$backup_type backup failed after ${duration}s"
        error "Backup output: $backup_output"
        send_notification "FAILED" "$backup_type backup failed: $backup_output"
        
        return 1
    fi
}

# Clean up old backups
cleanup_old_backups() {
    log "Starting backup cleanup..."
    
    cd "$PROJECT_DIR"
    export PYTHONPATH="$PROJECT_DIR"
    
    local cleanup_output
    if cleanup_output=$(python backup_system.py cleanup 2>&1); then
        success "Backup cleanup completed"
        log "Cleanup output: $cleanup_output"
        
        # Extract cleanup statistics
        local cleaned_local=$(echo "$cleanup_output" | grep -o '"cleaned_local": [0-9]*' | cut -d' ' -f2)
        local cleaned_cloud=$(echo "$cleanup_output" | grep -o '"cleaned_cloud": [0-9]*' | cut -d' ' -f2)
        
        if [ "$cleaned_local" -gt 0 ] || [ "$cleaned_cloud" -gt 0 ]; then
            log "Cleaned up $cleaned_local local and $cleaned_cloud cloud backups"
        fi
    else
        error "Backup cleanup failed: $cleanup_output"
    fi
}

# Verify recent backups
verify_backups() {
    log "Verifying recent backups..."
    
    cd "$PROJECT_DIR"
    export PYTHONPATH="$PROJECT_DIR"
    
    # Get list of recent backups
    local backups_output
    if backups_output=$(python backup_system.py list 2>&1); then
        # Verify the most recent backup
        local recent_backup_id=$(echo "$backups_output" | jq -r '.[0].backup_id' 2>/dev/null)
        
        if [ ! -z "$recent_backup_id" ] && [ "$recent_backup_id" != "null" ]; then
            log "Verifying backup: $recent_backup_id"
            
            local verify_output
            if verify_output=$(python backup_system.py verify "$recent_backup_id" 2>&1); then
                local verification_success=$(echo "$verify_output" | grep -o '"overall_success": [a-z]*' | cut -d' ' -f2)
                
                if [ "$verification_success" = "true" ]; then
                    success "Backup verification passed for $recent_backup_id"
                else
                    error "Backup verification failed for $recent_backup_id"
                    error "Verification output: $verify_output"
                fi
            else
                error "Backup verification command failed: $verify_output"
            fi
        else
            warn "No recent backups found to verify"
        fi
    else
        error "Failed to list backups for verification: $backups_output"
    fi
}

# Health check before backup
health_check() {
    log "Performing pre-backup health check..."
    
    # Check if application is running
    if ! curl -f -s http://localhost:8000/health > /dev/null 2>&1; then
        warn "Web application health check failed"
    fi
    
    if ! curl -f -s http://localhost:8001/api/health > /dev/null 2>&1; then
        warn "Prediction API health check failed"
    fi
    
    # Check database connectivity
    if command -v pg_isready > /dev/null; then
        if ! pg_isready -h localhost -p 5432 > /dev/null 2>&1; then
            warn "Database connectivity check failed"
        fi
    fi
    
    # Check Redis connectivity
    if command -v redis-cli > /dev/null; then
        if ! redis-cli ping > /dev/null 2>&1; then
            warn "Redis connectivity check failed"
        fi
    fi
    
    log "Health check completed"
}

# Main execution
main() {
    local backup_type="${1:-full}"
    local skip_cleanup="${2:-false}"
    local skip_verification="${3:-false}"
    
    log "=== NightScan Backup Script Started ==="
    log "Backup type: $backup_type"
    log "PID: $$"
    
    # Pre-flight checks
    check_lock
    
    if ! check_disk_space; then
        error "Pre-flight checks failed"
        exit 1
    fi
    
    # Perform health check
    health_check
    
    # Run the backup
    if run_backup "$backup_type"; then
        success "Backup process completed successfully"
        
        # Verify backup if requested
        if [ "$skip_verification" != "true" ]; then
            verify_backups
        fi
        
        # Clean up old backups if requested
        if [ "$skip_cleanup" != "true" ]; then
            cleanup_old_backups
        fi
        
        log "=== NightScan Backup Script Completed Successfully ==="
        exit 0
    else
        error "Backup process failed"
        log "=== NightScan Backup Script Failed ==="
        exit 1
    fi
}

# Parse command line arguments
case "${1:-full}" in
    "full"|"quick"|"database-only")
        main "$@"
        ;;
    "cleanup")
        check_lock
        cleanup_old_backups
        ;;
    "verify")
        check_lock
        verify_backups
        ;;
    "health")
        health_check
        ;;
    "help")
        echo "Usage: $0 [backup_type] [skip_cleanup] [skip_verification]"
        echo ""
        echo "Backup types:"
        echo "  full          - Complete backup (default)"
        echo "  quick         - Backup without models or cloud upload"
        echo "  database-only - Database backup only"
        echo ""
        echo "Other commands:"
        echo "  cleanup       - Clean up old backups only"
        echo "  verify        - Verify recent backups only"
        echo "  health        - Run health check only"
        echo "  help          - Show this help"
        echo ""
        echo "Examples:"
        echo "  $0 full                    # Full backup with cleanup and verification"
        echo "  $0 quick true              # Quick backup, skip cleanup"
        echo "  $0 database-only true true # Database only, skip cleanup and verification"
        ;;
    *)
        error "Unknown command: $1. Use 'help' for usage information."
        exit 1
        ;;
esac