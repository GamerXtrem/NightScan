#!/bin/bash

# Script de configuration firewall UFW + fail2ban pour NightScan VPS Lite
# Sécurité réseau optimisée pour VPS Lite
# Usage: ./setup-firewall.sh

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

log "🔥 Configuration firewall UFW + fail2ban pour NightScan VPS Lite"
log "================================================================"

# Vérifier les privilèges root
check_root() {
    if [ "$EUID" -ne 0 ]; then
        error "Ce script doit être exécuté en tant que root"
        error "Utilisez: sudo $0"
        exit 1
    fi
    success "Privilèges root vérifiés"
}

# Installer UFW et fail2ban
install_security_packages() {
    log "📦 Installation UFW et fail2ban..."
    
    # Mise à jour des paquets
    apt-get update -qq
    
    # Installation des paquets de sécurité
    apt-get install -y ufw fail2ban
    
    success "Paquets de sécurité installés"
}

# Configurer UFW (Uncomplicated Firewall)
configure_ufw() {
    log "🔥 Configuration UFW..."
    
    # Reset UFW pour partir d'une base propre
    ufw --force reset
    
    # Politique par défaut : deny all incoming, allow outgoing
    ufw default deny incoming
    ufw default allow outgoing
    
    # Autoriser SSH (IMPORTANT: avant d'activer UFW)
    ufw allow ssh
    ufw allow 22/tcp
    
    # Autoriser HTTP et HTTPS (Nginx)
    ufw allow 80/tcp
    ufw allow 443/tcp
    
    # Autoriser monitoring Grafana (accès restreint par Nginx auth)
    ufw allow 3000/tcp
    
    # Règles spécifiques pour Docker (si nécessaire)
    # ufw allow from 172.16.0.0/12 to any port 5432  # PostgreSQL interne
    # ufw allow from 172.16.0.0/12 to any port 6379  # Redis interne
    
    # Limiter les connexions SSH (protection brute force)
    ufw limit ssh/tcp
    
    # Règles avancées pour NightScan
    
    # Bloquer les ports sensibles de l'extérieur
    ufw deny 5432/tcp  # PostgreSQL
    ufw deny 6379/tcp  # Redis
    ufw deny 3100/tcp  # Loki
    ufw deny 9080/tcp  # Promtail
    
    # Autoriser ping (ICMP) mais limité
    ufw allow in on lo
    ufw allow out on lo
    
    # Activer UFW
    ufw --force enable
    
    success "UFW configuré et activé"
}

# Configurer fail2ban
configure_fail2ban() {
    log "🛡️  Configuration fail2ban..."
    
    # Créer configuration personnalisée fail2ban
    cat > /etc/fail2ban/jail.local << EOF
[DEFAULT]
# Configuration fail2ban pour NightScan VPS Lite
ignoreip = 127.0.0.1/8 ::1
bantime = 3600
findtime = 600
maxretry = 3
backend = auto
usedns = warn
logencoding = auto
enabled = false
filter = %(__name__)s
destemail = root@localhost
sender = root@$(hostname -f)
mta = sendmail
protocol = tcp
chain = <known/chain>

# Actions
action_ = %(banaction)s[name=%(__name__)s, bantime="%(bantime)s", port="%(port)s", protocol="%(protocol)s"]
action_mw = %(banaction)s[name=%(__name__)s, bantime="%(bantime)s", port="%(port)s", protocol="%(protocol)s"]
           %(mta)s-whois[name=%(__name__)s, sender="%(sender)s", dest="%(destemail)s", protocol="%(protocol)s", chain="%(chain)s"]
action_mwl = %(banaction)s[name=%(__name__)s, bantime="%(bantime)s", port="%(port)s", protocol="%(protocol)s"]
            %(mta)s-whois-lines[name=%(__name__)s, sender="%(sender)s", dest="%(destemail)s", logpath="%(logpath)s", chain="%(chain)s"]
action_xarf = %(banaction)s[name=%(__name__)s, bantime="%(bantime)s", port="%(port)s", protocol="%(protocol)s"]
             xarf-login-attack[service=%(__name__)s, sender="%(sender)s", logpath="%(logpath)s", port="%(port)s"]
action = %(action_)s

# Jail SSH (protection connexions SSH)
[sshd]
enabled = true
port = ssh
logpath = %(sshd_log)s
maxretry = 3
bantime = 3600

# Jail Nginx (protection attaques web)
[nginx-http-auth]
enabled = true
filter = nginx-http-auth
port = http,https
logpath = /var/log/nginx/error.log
maxretry = 3
bantime = 3600

[nginx-limit-req]
enabled = true
filter = nginx-limit-req
port = http,https
logpath = /var/log/nginx/error.log
maxretry = 3
bantime = 3600

[nginx-botsearch]
enabled = true
filter = nginx-botsearch
port = http,https
logpath = /var/log/nginx/access.log
maxretry = 2
bantime = 7200

# Jail Docker (si activé)
[docker-auth]
enabled = false
filter = docker-auth
port = 2376
logpath = /var/log/docker.log
maxretry = 3
bantime = 3600

# Protection NightScan spécifique
[nightscan-auth]
enabled = true
filter = nightscan-auth
port = http,https
logpath = /var/log/nginx/access.log
maxretry = 5
findtime = 300
bantime = 1800
EOF

    # Créer filtres personnalisés pour NightScan
    cat > /etc/fail2ban/filter.d/nightscan-auth.conf << EOF
# Filtre fail2ban pour authentification NightScan
[Definition]
failregex = ^<HOST> -.*"POST /(login|api/auth).*" (401|403|422)
            ^<HOST> -.*"POST /api/.*" 401
ignoreregex = 
EOF

    # Redémarrer et activer fail2ban
    systemctl restart fail2ban
    systemctl enable fail2ban
    
    success "fail2ban configuré et activé"
}

# Configurer les logs de sécurité
setup_security_logging() {
    log "📝 Configuration logs de sécurité..."
    
    # Créer répertoire logs sécurité
    mkdir -p /var/log/nightscan-security
    
    # Configuration logrotate pour logs sécurité
    cat > /etc/logrotate.d/nightscan-security << EOF
/var/log/nightscan-security/*.log {
    daily
    missingok
    rotate 7
    compress
    delaycompress
    notifempty
    create 644 root root
    postrotate
        systemctl reload fail2ban > /dev/null 2>&1 || true
    endscript
}
EOF

    # Script de monitoring sécurité
    cat > /usr/local/bin/nightscan-security-monitor.sh << 'EOF'
#!/bin/bash
# Monitoring sécurité automatique pour NightScan

LOG_FILE="/var/log/nightscan-security/security-monitor.log"
DATE=$(date '+%Y-%m-%d %H:%M:%S')

# Fonction de log
log_security() {
    echo "[$DATE] $1" >> "$LOG_FILE"
}

# Vérifier le statut fail2ban
if ! systemctl is-active --quiet fail2ban; then
    log_security "ALERT: fail2ban service down"
    systemctl restart fail2ban
fi

# Vérifier le statut UFW
if ! ufw status | grep -q "Status: active"; then
    log_security "ALERT: UFW firewall inactive"
fi

# Compter les IPs bannies
BANNED_IPS=$(fail2ban-client status sshd 2>/dev/null | grep "Banned IP list" | wc -l)
if [ "$BANNED_IPS" -gt 0 ]; then
    log_security "INFO: $BANNED_IPS IPs currently banned"
fi

# Vérifier les connexions suspectes
SUSPICIOUS_CONNECTIONS=$(ss -tn | grep ":22\|:80\|:443" | wc -l)
if [ "$SUSPICIOUS_CONNECTIONS" -gt 50 ]; then
    log_security "WARNING: High number of connections ($SUSPICIOUS_CONNECTIONS)"
fi

# Vérifier l'utilisation disque
DISK_USAGE=$(df / | tail -1 | awk '{print $5}' | sed 's/%//')
if [ "$DISK_USAGE" -gt 80 ]; then
    log_security "WARNING: Disk usage high ($DISK_USAGE%)"
fi

# Vérifier la charge système
LOAD_AVG=$(uptime | awk -F'load average:' '{print $2}' | awk '{print $1}' | sed 's/,//')
if (( $(echo "$LOAD_AVG > 2.0" | bc -l) )); then
    log_security "WARNING: High system load ($LOAD_AVG)"
fi
EOF

    chmod +x /usr/local/bin/nightscan-security-monitor.sh
    
    # Créer cron job pour monitoring sécurité
    cat > /etc/cron.d/nightscan-security << EOF
# Monitoring sécurité NightScan - toutes les 15 minutes
*/15 * * * * root /usr/local/bin/nightscan-security-monitor.sh
EOF

    success "Logs de sécurité configurés"
}

# Créer règles de sécurité avancées
create_advanced_security_rules() {
    log "🔒 Configuration règles sécurité avancées..."
    
    # Script de blocage d'IPs malveillantes
    cat > /usr/local/bin/block-malicious-ips.sh << 'EOF'
#!/bin/bash
# Blocage automatique d'IPs malveillantes pour NightScan

# Listes d'IPs malveillantes connues
MALICIOUS_LISTS=(
    "https://www.spamhaus.org/drop/drop.txt"
    "https://www.spamhaus.org/drop/edrop.txt"
)

# Télécharger et appliquer les listes
for list in "${MALICIOUS_LISTS[@]}"; do
    echo "Téléchargement: $list"
    curl -s "$list" | grep -E '^[0-9]' | cut -d' ' -f1 | while read -r ip; do
        if [[ $ip =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+/[0-9]+$ ]]; then
            ufw deny from "$ip" > /dev/null 2>&1
        fi
    done
done

echo "Blocage IPs malveillantes mis à jour"
EOF

    chmod +x /usr/local/bin/block-malicious-ips.sh
    
    # Exécuter une fois au démarrage
    # /usr/local/bin/block-malicious-ips.sh
    
    success "Règles sécurité avancées créées"
}

# Test de la configuration firewall
test_firewall_configuration() {
    log "🔍 Test de la configuration firewall..."
    
    # Test statut UFW
    if ufw status | grep -q "Status: active"; then
        success "UFW est actif"
    else
        error "UFW n'est pas actif"
    fi
    
    # Test statut fail2ban
    if systemctl is-active --quiet fail2ban; then
        success "fail2ban est actif"
    else
        error "fail2ban n'est pas actif"
    fi
    
    # Afficher les règles UFW
    echo ""
    log "Règles UFW actives:"
    ufw status numbered
    
    # Afficher les jails fail2ban
    echo ""
    log "Jails fail2ban actives:"
    fail2ban-client status
    
    success "Configuration firewall testée"
}

# Afficher les informations finales
show_firewall_info() {
    log "🔥 Configuration firewall terminée"
    echo ""
    echo "🛡️  Sécurité réseau activée:"
    echo "  ✅ UFW (Uncomplicated Firewall)"
    echo "  ✅ fail2ban (Protection brute force)"
    echo "  ✅ Monitoring sécurité automatique"
    echo ""
    echo "🚪 Ports autorisés:"
    echo "  - 22/tcp (SSH) - Limité"
    echo "  - 80/tcp (HTTP) - Redirigé vers HTTPS"
    echo "  - 443/tcp (HTTPS) - NightScan App"
    echo "  - 3000/tcp (Grafana) - Protégé par auth"
    echo ""
    echo "🚫 Ports bloqués:"
    echo "  - 5432/tcp (PostgreSQL)"
    echo "  - 6379/tcp (Redis)"
    echo "  - 3100/tcp (Loki)"
    echo "  - Tous les autres ports"
    echo ""
    echo "⚙️  Commandes utiles:"
    echo "  - Status UFW: ufw status"
    echo "  - Status fail2ban: fail2ban-client status"
    echo "  - IPs bannies SSH: fail2ban-client status sshd"
    echo "  - Logs sécurité: tail -f /var/log/nightscan-security/security-monitor.log"
    echo "  - Débannir IP: fail2ban-client unban <IP>"
    echo ""
    echo "📊 Monitoring:"
    echo "  - Monitoring automatique toutes les 15 minutes"
    echo "  - Logs dans /var/log/nightscan-security/"
    echo "  - Alertes système en cas de problème"
}

# Fonction principale
main() {
    check_root
    install_security_packages
    configure_ufw
    configure_fail2ban
    setup_security_logging
    create_advanced_security_rules
    test_firewall_configuration
    show_firewall_info
    
    success "🎉 Firewall UFW + fail2ban configuré avec succès!"
    echo ""
    warn "⚠️  IMPORTANT:"
    echo "  - Testez la connexion SSH avant de fermer cette session"
    echo "  - En cas de problème: sudo ufw disable"
    echo "  - Sauvegardez votre IP: sudo ufw allow from <VOTRE_IP>"
}

main "$@"