#!/bin/bash

# ============================================================================
# 🧪 Script de Tests Post-Déploiement NightScan VPS Lite
# ============================================================================

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
LOG_FILE="/tmp/nightscan_post_deploy_tests.log"

# Variables d'environnement
DOMAIN_NAME="${DOMAIN_NAME:-}"
TIMEOUT=30

# Couleurs
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Compteurs
TESTS_TOTAL=0
TESTS_PASSED=0
TESTS_FAILED=0

# Fonctions utilitaires
log() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')] $1${NC}" | tee -a "$LOG_FILE"
}

success() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')] ✅ $1${NC}" | tee -a "$LOG_FILE"
    ((TESTS_PASSED++))
}

fail() {
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] ❌ $1${NC}" | tee -a "$LOG_FILE"
    ((TESTS_FAILED++))
}

warn() {
    echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')] ⚠️  $1${NC}" | tee -a "$LOG_FILE"
}

test_start() {
    ((TESTS_TOTAL++))
    log "🧪 Test $TESTS_TOTAL: $1"
}

# Test de connectivité HTTP/HTTPS
test_http_connectivity() {
    test_start "Connectivité HTTP/HTTPS"
    
    if [ -z "$DOMAIN_NAME" ]; then
        fail "Variable DOMAIN_NAME non définie"
        return 1
    fi
    
    # Test HTTPS principal
    if curl -f -s -k --connect-timeout $TIMEOUT "https://$DOMAIN_NAME" > /dev/null; then
        success "HTTPS principal accessible (https://$DOMAIN_NAME)"
    else
        fail "HTTPS principal inaccessible (https://$DOMAIN_NAME)"
    fi
    
    # Test API
    if curl -f -s -k --connect-timeout $TIMEOUT "https://api.$DOMAIN_NAME" > /dev/null; then
        success "API HTTPS accessible (https://api.$DOMAIN_NAME)"
    else
        fail "API HTTPS inaccessible (https://api.$DOMAIN_NAME)"
    fi
    
    # Test monitoring
    if curl -f -s -k --connect-timeout $TIMEOUT "https://monitoring.$DOMAIN_NAME" > /dev/null; then
        success "Monitoring HTTPS accessible (https://monitoring.$DOMAIN_NAME)"
    else
        fail "Monitoring HTTPS inaccessible (https://monitoring.$DOMAIN_NAME)"
    fi
}

# Test des endpoints de santé
test_health_endpoints() {
    test_start "Endpoints de santé"
    
    # Test health application web
    if curl -f -s -k --connect-timeout $TIMEOUT "https://$DOMAIN_NAME/health" | grep -q "ok"; then
        success "Endpoint /health application web OK"
    else
        fail "Endpoint /health application web KO"
    fi
    
    # Test health API
    if curl -f -s -k --connect-timeout $TIMEOUT "https://api.$DOMAIN_NAME/health" | grep -q "ok"; then
        success "Endpoint /health API OK"
    else
        fail "Endpoint /health API KO"
    fi
    
    # Test health Grafana
    if curl -f -s -k --connect-timeout $TIMEOUT "https://monitoring.$DOMAIN_NAME/api/health" > /dev/null; then
        success "Endpoint /api/health Grafana OK"
    else
        fail "Endpoint /api/health Grafana KO"
    fi
}

# Test des services Docker
test_docker_services() {
    test_start "Services Docker"
    
    cd "$PROJECT_ROOT"
    
    # Vérifier services principaux
    if docker-compose -f docker-compose.production.yml ps | grep -q "Up"; then
        success "Services principaux Docker actifs"
    else
        fail "Services principaux Docker inactifs"
    fi
    
    # Vérifier services monitoring
    if docker-compose -f docker-compose.monitoring.yml ps | grep -q "Up"; then
        success "Services monitoring Docker actifs"
    else
        fail "Services monitoring Docker inactifs"
    fi
    
    # Vérifier healthchecks
    local unhealthy_services=$(docker ps --filter "health=unhealthy" --format "table {{.Names}}" | tail -n +2)
    if [ -z "$unhealthy_services" ]; then
        success "Tous les services Docker sont healthy"
    else
        fail "Services Docker unhealthy: $unhealthy_services"
    fi
}

# Test de performance système
test_system_performance() {
    test_start "Performance système"
    
    # Test utilisation mémoire
    local memory_usage=$(free | grep Mem | awk '{printf("%.1f"), $3/$2*100}')
    if (( $(echo "$memory_usage < 85.0" | bc -l) )); then
        success "Utilisation mémoire OK: ${memory_usage}%"
    else
        fail "Utilisation mémoire élevée: ${memory_usage}%"
    fi
    
    # Test utilisation CPU
    local cpu_usage=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | awk -F'%' '{print $1}')
    if (( $(echo "$cpu_usage < 80.0" | bc -l) )); then
        success "Utilisation CPU OK: ${cpu_usage}%"
    else
        warn "Utilisation CPU élevée: ${cpu_usage}%"
    fi
    
    # Test espace disque
    local disk_usage=$(df / | tail -1 | awk '{print $5}' | sed 's/%//')
    if [ "$disk_usage" -lt 80 ]; then
        success "Utilisation disque OK: ${disk_usage}%"
    else
        warn "Utilisation disque élevée: ${disk_usage}%"
    fi
}

# Test SSL/TLS
test_ssl_certificates() {
    test_start "Certificats SSL/TLS"
    
    # Test certificat principal
    if echo | openssl s_client -servername "$DOMAIN_NAME" -connect "$DOMAIN_NAME:443" 2>/dev/null | openssl x509 -noout -dates 2>/dev/null; then
        success "Certificat SSL principal valide"
    else
        fail "Certificat SSL principal invalide"
    fi
    
    # Test certificat API
    if echo | openssl s_client -servername "api.$DOMAIN_NAME" -connect "api.$DOMAIN_NAME:443" 2>/dev/null | openssl x509 -noout -dates 2>/dev/null; then
        success "Certificat SSL API valide"
    else
        fail "Certificat SSL API invalide"
    fi
    
    # Test protocoles SSL
    if nmap --script ssl-enum-ciphers -p 443 "$DOMAIN_NAME" 2>/dev/null | grep -q "TLSv1.2\|TLSv1.3"; then
        success "Protocoles SSL modernes supportés"
    else
        warn "Protocoles SSL modernes non détectés"
    fi
}

# Test de sécurité réseau
test_network_security() {
    test_start "Sécurité réseau"
    
    # Test firewall UFW
    if sudo ufw status | grep -q "Status: active"; then
        success "Firewall UFW actif"
    else
        fail "Firewall UFW inactif"
    fi
    
    # Test fail2ban
    if sudo systemctl is-active fail2ban >/dev/null 2>&1; then
        success "Service fail2ban actif"
    else
        fail "Service fail2ban inactif"
    fi
    
    # Test ports ouverts
    local open_ports=$(ss -tuln | grep :80 | wc -l)
    if [ "$open_ports" -gt 0 ]; then
        success "Port HTTP/HTTPS ouvert"
    else
        fail "Aucun port HTTP/HTTPS ouvert"
    fi
}

# Test de monitoring
test_monitoring() {
    test_start "Monitoring et logs"
    
    # Test Grafana login
    if curl -f -s -k --connect-timeout $TIMEOUT "https://monitoring.$DOMAIN_NAME/login" | grep -q "Grafana"; then
        success "Interface Grafana accessible"
    else
        fail "Interface Grafana inaccessible"
    fi
    
    # Test collecte logs
    if docker logs nightscan-promtail 2>&1 | grep -q "level=info"; then
        success "Promtail collecte des logs"
    else
        warn "Promtail ne collecte pas de logs"
    fi
    
    # Test métriques Prometheus
    if curl -f -s -k --connect-timeout $TIMEOUT "http://localhost:9090/metrics" | grep -q "prometheus_"; then
        success "Prometheus collecte des métriques"
    else
        warn "Prometheus ne collecte pas de métriques"
    fi
}

# Test backup
test_backup_system() {
    test_start "Système de backup"
    
    # Test script backup
    if [ -f "$PROJECT_ROOT/scripts/backup-production.sh" ] && [ -x "$PROJECT_ROOT/scripts/backup-production.sh" ]; then
        success "Script backup présent et exécutable"
    else
        fail "Script backup manquant ou non exécutable"
    fi
    
    # Test répertoire backup
    if [ -d "/home/$(whoami)/backups" ]; then
        success "Répertoire backup configuré"
    else
        warn "Répertoire backup non configuré"
    fi
    
    # Test crontab backup
    if crontab -l 2>/dev/null | grep -q "backup"; then
        success "Backup automatique configuré (crontab)"
    else
        warn "Backup automatique non configuré"
    fi
}

# Test fonctionnel application
test_application_features() {
    test_start "Fonctionnalités application"
    
    # Test page d'accueil
    if curl -f -s -k "https://$DOMAIN_NAME" | grep -q "NightScan"; then
        success "Page d'accueil charge correctement"
    else
        fail "Page d'accueil ne charge pas"
    fi
    
    # Test assets statiques
    if curl -f -s -k "https://$DOMAIN_NAME/static/css/main.css" > /dev/null 2>&1; then
        success "Assets CSS accessibles"
    else
        warn "Assets CSS non accessibles"
    fi
    
    # Test endpoint API version
    if curl -f -s -k "https://api.$DOMAIN_NAME/version" | grep -q "version\|api"; then
        success "Endpoint API version accessible"
    else
        warn "Endpoint API version non accessible"
    fi
}

# Test temps de réponse
test_response_times() {
    test_start "Temps de réponse"
    
    # Test temps réponse page d'accueil
    local response_time=$(curl -o /dev/null -s -k -w "%{time_total}" "https://$DOMAIN_NAME")
    if (( $(echo "$response_time < 3.0" | bc -l) )); then
        success "Temps de réponse page d'accueil OK: ${response_time}s"
    else
        warn "Temps de réponse page d'accueil lent: ${response_time}s"
    fi
    
    # Test temps réponse API
    local api_response_time=$(curl -o /dev/null -s -k -w "%{time_total}" "https://api.$DOMAIN_NAME/health")
    if (( $(echo "$api_response_time < 2.0" | bc -l) )); then
        success "Temps de réponse API OK: ${api_response_time}s"
    else
        warn "Temps de réponse API lent: ${api_response_time}s"
    fi
}

# Vérification logs d'erreurs
test_error_logs() {
    test_start "Logs d'erreurs"
    
    cd "$PROJECT_ROOT"
    
    # Vérifier logs Docker pour erreurs critiques
    local critical_errors=$(docker-compose -f docker-compose.production.yml logs --since="1h" 2>&1 | grep -i "error\|critical\|fatal" | wc -l)
    if [ "$critical_errors" -eq 0 ]; then
        success "Aucune erreur critique dans les logs (dernière heure)"
    else
        warn "$critical_errors erreurs trouvées dans les logs"
    fi
    
    # Vérifier logs système
    local system_errors=$(sudo journalctl --since="1 hour ago" -p err | wc -l)
    if [ "$system_errors" -eq 0 ]; then
        success "Aucune erreur système (dernière heure)"
    else
        warn "$system_errors erreurs système trouvées"
    fi
}

# Générer rapport
generate_report() {
    log "📊 Génération du rapport de tests..."
    
    local success_rate=$((TESTS_PASSED * 100 / TESTS_TOTAL))
    
    echo ""
    echo -e "${BLUE}=====================================${NC}"
    echo -e "${BLUE}📊 RAPPORT TESTS POST-DÉPLOIEMENT${NC}"
    echo -e "${BLUE}=====================================${NC}"
    echo ""
    
    echo -e "${GREEN}✅ Tests réussis: $TESTS_PASSED${NC}"
    echo -e "${RED}❌ Tests échoués: $TESTS_FAILED${NC}"
    echo -e "${BLUE}📊 Total tests: $TESTS_TOTAL${NC}"
    echo -e "${BLUE}🎯 Taux de réussite: $success_rate%${NC}"
    echo ""
    
    if [ "$success_rate" -ge 90 ]; then
        echo -e "${GREEN}🎉 DÉPLOIEMENT VALIDÉ - Excellent score!${NC}"
    elif [ "$success_rate" -ge 75 ]; then
        echo -e "${YELLOW}⚠️ DÉPLOIEMENT ACCEPTABLE - Quelques améliorations nécessaires${NC}"
    else
        echo -e "${RED}❌ DÉPLOIEMENT À CORRIGER - Problèmes critiques détectés${NC}"
    fi
    
    echo ""
    echo -e "${BLUE}📁 Log détaillé: $LOG_FILE${NC}"
    echo ""
}

# Fonction principale
main() {
    log "🧪 Démarrage des tests post-déploiement NightScan VPS Lite..."
    
    # Vérifier si DOMAIN_NAME est défini
    if [ -z "$DOMAIN_NAME" ]; then
        read -p "Nom de domaine (ex: nightscan.com): " DOMAIN_NAME
        export DOMAIN_NAME
    fi
    
    log "🎯 Tests pour le domaine: $DOMAIN_NAME"
    echo ""
    
    # Exécuter tous les tests
    test_docker_services
    test_http_connectivity
    test_health_endpoints
    test_ssl_certificates
    test_system_performance
    test_network_security
    test_monitoring
    test_backup_system
    test_application_features
    test_response_times
    test_error_logs
    
    # Générer rapport final
    generate_report
    
    # Exit code basé sur le taux de réussite
    local success_rate=$((TESTS_PASSED * 100 / TESTS_TOTAL))
    if [ "$success_rate" -ge 90 ]; then
        exit 0
    else
        exit 1
    fi
}

# Gestion des options
while [[ $# -gt 0 ]]; do
    case $1 in
        --domain)
            DOMAIN_NAME="$2"
            shift 2
            ;;
        --timeout)
            TIMEOUT="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --domain DOMAIN     Nom de domaine à tester"
            echo "  --timeout SECONDS   Timeout pour les requêtes (défaut: 30s)"
            echo "  --help              Afficher cette aide"
            exit 0
            ;;
        *)
            echo "Option inconnue: $1"
            exit 1
            ;;
    esac
done

# Vérification des dépendances
if ! command -v curl &> /dev/null; then
    echo "❌ curl n'est pas installé"
    exit 1
fi

if ! command -v docker &> /dev/null; then
    echo "❌ docker n'est pas installé"
    exit 1
fi

if ! command -v bc &> /dev/null; then
    echo "❌ bc n'est pas installé (apt install bc)"
    exit 1
fi

# Exécution
main "$@"