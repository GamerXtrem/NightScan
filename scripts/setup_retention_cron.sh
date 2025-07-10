#!/bin/bash
"""
Script de Configuration des Tâches Cron pour la Rétention des Données
Configure automatiquement les tâches cron pour le nettoyage régulier.
"""

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CLEANUP_SCRIPT="$SCRIPT_DIR/data_retention_cleanup.py"
LOG_DIR="$PROJECT_DIR/logs/retention"
VENV_PATH="${VENV_PATH:-$PROJECT_DIR/venv}"

# Couleurs pour l'affichage
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🕒 Configuration des tâches cron pour la rétention des données${NC}"
echo "================================================="

# Vérification des prérequis
check_prerequisites() {
    echo -e "${YELLOW}Vérification des prérequis...${NC}"
    
    # Vérifier que le script de nettoyage existe
    if [[ ! -f "$CLEANUP_SCRIPT" ]]; then
        echo -e "${RED}❌ Script de nettoyage non trouvé: $CLEANUP_SCRIPT${NC}"
        exit 1
    fi
    
    # Vérifier Python
    if ! command -v python3 &> /dev/null; then
        echo -e "${RED}❌ Python 3 non trouvé${NC}"
        exit 1
    fi
    
    # Créer le dossier de logs
    mkdir -p "$LOG_DIR"
    
    echo -e "${GREEN}✅ Prérequis vérifiés${NC}"
}

# Configuration des tâches cron
setup_cron_jobs() {
    echo -e "${YELLOW}Configuration des tâches cron...${NC}"
    
    # Définir le chemin Python
    if [[ -f "$VENV_PATH/bin/python" ]]; then
        PYTHON_CMD="$VENV_PATH/bin/python"
        echo "Utilisation de l'environnement virtuel: $VENV_PATH"
    else
        PYTHON_CMD="python3"
        echo "Utilisation de Python système"
    fi
    
    # Backup de la crontab actuelle
    BACKUP_FILE="/tmp/crontab_backup_$(date +%Y%m%d_%H%M%S)"
    crontab -l > "$BACKUP_FILE" 2>/dev/null || true
    echo "Sauvegarde de la crontab actuelle: $BACKUP_FILE"
    
    # Créer les nouvelles entrées cron
    cat << EOF > /tmp/nightscan_retention_cron
# NightScan Data Retention Cron Jobs
# Ajouté automatiquement le $(date)

# Nettoyage quotidien des données expirées (3h du matin)
0 3 * * * cd $PROJECT_DIR && $PYTHON_CMD $CLEANUP_SCRIPT --execute --all-users --log-file $LOG_DIR/daily_cleanup.log >> $LOG_DIR/daily_cleanup.log 2>&1

# Vérification des notifications d'expiration (9h du matin)
0 9 * * * cd $PROJECT_DIR && $PYTHON_CMD $CLEANUP_SCRIPT --notify --days-before 7 --log-file $LOG_DIR/notifications.log >> $LOG_DIR/notifications.log 2>&1

# Rapport hebdomadaire de rétention (dimanche à 6h)
0 6 * * 0 cd $PROJECT_DIR && $PYTHON_CMD $CLEANUP_SCRIPT --report --output $LOG_DIR/weekly_report_\$(date +\%Y\%m\%d).json --log-file $LOG_DIR/weekly_report.log >> $LOG_DIR/weekly_report.log 2>&1

# Nettoyage des anciens logs (premier du mois à 2h)
0 2 1 * * find $LOG_DIR -name "*.log" -mtime +30 -delete && find $LOG_DIR -name "weekly_report_*.json" -mtime +90 -delete

EOF

    # Ajouter à la crontab existante
    (crontab -l 2>/dev/null || true; cat /tmp/nightscan_retention_cron) | crontab -
    
    # Nettoyer le fichier temporaire
    rm /tmp/nightscan_retention_cron
    
    echo -e "${GREEN}✅ Tâches cron configurées${NC}"
}

# Test des scripts
test_scripts() {
    echo -e "${YELLOW}Test des scripts de rétention...${NC}"
    
    cd "$PROJECT_DIR"
    
    # Test en mode dry-run
    echo "Test du nettoyage en mode dry-run..."
    if $PYTHON_CMD "$CLEANUP_SCRIPT" --dry-run --all-users --verbose; then
        echo -e "${GREEN}✅ Test de nettoyage réussi${NC}"
    else
        echo -e "${RED}❌ Test de nettoyage échoué${NC}"
        exit 1
    fi
    
    # Test du rapport
    echo "Test de génération de rapport..."
    if $PYTHON_CMD "$CLEANUP_SCRIPT" --report --output "/tmp/test_report.json"; then
        echo -e "${GREEN}✅ Test de rapport réussi${NC}"
        rm -f "/tmp/test_report.json"
    else
        echo -e "${RED}❌ Test de rapport échoué${NC}"
        exit 1
    fi
    
    # Test des notifications
    echo "Test de vérification des notifications..."
    if $PYTHON_CMD "$CLEANUP_SCRIPT" --notify --days-before 7; then
        echo -e "${GREEN}✅ Test de notifications réussi${NC}"
    else
        echo -e "${RED}❌ Test de notifications échoué${NC}"
        exit 1
    fi
}

# Affichage des tâches configurées
show_cron_status() {
    echo -e "${BLUE}📋 Tâches cron configurées:${NC}"
    echo "================================================="
    crontab -l | grep -A 10 -B 2 "NightScan Data Retention" || echo "Aucune tâche trouvée"
    echo ""
    echo -e "${BLUE}📁 Logs disponibles dans: ${NC}$LOG_DIR"
    echo -e "${BLUE}🔧 Pour modifier: ${NC}crontab -e"
    echo -e "${BLUE}📊 Pour tester manuellement: ${NC}$PYTHON_CMD $CLEANUP_SCRIPT --help"
}

# Fonction de désinstallation
uninstall_cron() {
    echo -e "${YELLOW}Suppression des tâches cron NightScan...${NC}"
    
    # Sauvegarder la crontab actuelle
    BACKUP_FILE="/tmp/crontab_backup_uninstall_$(date +%Y%m%d_%H%M%S)"
    crontab -l > "$BACKUP_FILE" 2>/dev/null || true
    
    # Supprimer les lignes NightScan
    crontab -l 2>/dev/null | grep -v "NightScan Data Retention" | grep -v "$CLEANUP_SCRIPT" | crontab -
    
    echo -e "${GREEN}✅ Tâches cron supprimées${NC}"
    echo "Sauvegarde disponible: $BACKUP_FILE"
}

# Menu principal
show_menu() {
    echo ""
    echo -e "${BLUE}Options disponibles:${NC}"
    echo "1. Installer les tâches cron"
    echo "2. Tester les scripts"
    echo "3. Afficher le statut"
    echo "4. Désinstaller les tâches cron"
    echo "5. Quitter"
    echo ""
}

# Main
main() {
    case "${1:-}" in
        "install")
            check_prerequisites
            setup_cron_jobs
            show_cron_status
            ;;
        "test")
            check_prerequisites
            test_scripts
            ;;
        "status")
            show_cron_status
            ;;
        "uninstall")
            uninstall_cron
            ;;
        "help"|"--help"|"-h")
            echo "Usage: $0 [install|test|status|uninstall|help]"
            echo ""
            echo "  install    - Configure les tâches cron"
            echo "  test       - Teste les scripts de rétention"
            echo "  status     - Affiche les tâches configurées"
            echo "  uninstall  - Supprime les tâches cron"
            echo "  help       - Affiche cette aide"
            ;;
        "")
            # Mode interactif
            while true; do
                show_menu
                read -p "Choisissez une option (1-5): " choice
                case $choice in
                    1)
                        check_prerequisites
                        setup_cron_jobs
                        show_cron_status
                        ;;
                    2)
                        test_scripts
                        ;;
                    3)
                        show_cron_status
                        ;;
                    4)
                        uninstall_cron
                        ;;
                    5)
                        echo "Au revoir!"
                        break
                        ;;
                    *)
                        echo -e "${RED}Option invalide${NC}"
                        ;;
                esac
                echo ""
                read -p "Appuyez sur Entrée pour continuer..."
            done
            ;;
        *)
            echo -e "${RED}Option inconnue: $1${NC}"
            echo "Utilisez '$0 help' pour voir les options disponibles"
            exit 1
            ;;
    esac
}

main "$@"