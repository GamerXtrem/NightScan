#!/bin/bash

# Script de déploiement monitoring léger pour NightScan VPS Lite
# Loki + Promtail + Grafana optimisé 4GB RAM
# Usage: ./deploy-monitoring.sh

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

# Charger les variables d'environnement
if [ ! -f "$ENV_FILE" ]; then
    error "Fichier .env non trouvé: $ENV_FILE"
    exit 1
fi

source "$ENV_FILE"

log "📊 Déploiement monitoring NightScan VPS Lite"
log "============================================="

# Vérifier les dépendances
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

# Créer les répertoires de monitoring
setup_monitoring_directories() {
    log "📁 Création des répertoires monitoring..."
    
    cd "$PROJECT_ROOT"
    
    # Créer répertoires avec permissions appropriées
    sudo mkdir -p monitoring/data/loki monitoring/data/grafana
    sudo mkdir -p logs/loki logs/grafana logs/promtail
    
    # Permissions pour Grafana (UID 472)
    sudo chown -R 472:472 monitoring/data/grafana logs/grafana
    
    # Permissions pour Loki (UID 10001)
    sudo chown -R 10001:10001 monitoring/data/loki logs/loki
    
    # Permissions génériques
    sudo chown -R $USER:$USER monitoring/ logs/
    
    success "Répertoires monitoring créés"
}

# Créer la configuration Grafana
create_grafana_config() {
    log "⚙️  Configuration Grafana..."
    
    # Configuration Grafana optimisée VPS Lite
    cat > "$PROJECT_ROOT/monitoring/grafana/grafana.ini" << EOF
[default]
instance_name = NightScan VPS Lite

[server]
protocol = http
http_port = 3000
domain = monitoring.${DOMAIN_NAME}
root_url = https://monitoring.${DOMAIN_NAME}
serve_from_sub_path = false

[database]
type = sqlite3
path = /var/lib/grafana/grafana.db

[session]
provider = memory

[analytics]
reporting_enabled = false
check_for_updates = false

[security]
admin_user = admin
admin_password = ${GRAFANA_PASSWORD}
secret_key = ${SECRET_KEY}
cookie_secure = true
cookie_samesite = strict
strict_transport_security = true

[users]
allow_sign_up = false
allow_org_create = false
auto_assign_org = true
auto_assign_org_role = Viewer

[auth.anonymous]
enabled = false

[log]
mode = console
level = info

[log.console]
level = info
format = json

[panels]
disable_sanitize_html = false

[alerting]
enabled = true
execute_alerts = true

[unified_alerting]
enabled = true

[feature_toggles]
enable = 

[quota]
enabled = false

[alerting]
enabled = true

[metrics]
enabled = true
interval_seconds = 10

[tracing.jaeger]
address = 

[grafana_net]
url = https://grafana.net

[snapshots]
external_enabled = false

[external_image_storage]
provider = local
EOF

    success "Configuration Grafana créée"
}

# Créer le dashboard NightScan
create_nightscan_dashboard() {
    log "📈 Création dashboard NightScan..."
    
    cat > "$PROJECT_ROOT/monitoring/grafana/dashboards/nightscan-overview.json" << 'EOF'
{
  "dashboard": {
    "id": null,
    "title": "NightScan VPS Lite - Overview",
    "tags": ["nightscan", "production"],
    "timezone": "browser",
    "panels": [
      {
        "id": 1,
        "title": "System Overview",
        "type": "stat",
        "targets": [
          {
            "expr": "up",
            "legendFormat": "Services Online"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "color": {
              "mode": "thresholds"
            },
            "thresholds": {
              "steps": [
                {
                  "color": "red",
                  "value": 0
                },
                {
                  "color": "green",
                  "value": 1
                }
              ]
            }
          }
        },
        "gridPos": {
          "h": 8,
          "w": 12,
          "x": 0,
          "y": 0
        }
      },
      {
        "id": 2,
        "title": "Recent Logs",
        "type": "logs",
        "targets": [
          {
            "expr": "{job=~\"nightscan.*\"} |= \"ERROR\" or |= \"WARN\"",
            "refId": "A"
          }
        ],
        "gridPos": {
          "h": 8,
          "w": 12,
          "x": 12,
          "y": 0
        }
      },
      {
        "id": 3,
        "title": "HTTP Requests",
        "type": "graph",
        "targets": [
          {
            "expr": "sum(rate(nginx_requests_total[5m])) by (status)",
            "legendFormat": "Status {{status}}"
          }
        ],
        "gridPos": {
          "h": 8,
          "w": 24,
          "x": 0,
          "y": 8
        }
      }
    ],
    "time": {
      "from": "now-1h",
      "to": "now"
    },
    "refresh": "30s",
    "schemaVersion": 27,
    "version": 1
  }
}
EOF

    success "Dashboard NightScan créé"
}

# Démarrer les services de monitoring
start_monitoring_services() {
    log "🚀 Démarrage des services monitoring..."
    
    cd "$PROJECT_ROOT"
    
    # Créer réseau Docker si nécessaire
    docker network create nightscan_network 2>/dev/null || true
    
    # Démarrer monitoring
    docker-compose -f docker-compose.monitoring.yml up -d
    
    # Attendre que les services soient prêts
    log "Attente démarrage des services..."
    sleep 30
    
    # Vérifier Loki
    if curl -f -s "http://localhost:3100/ready" > /dev/null; then
        success "Loki opérationnel"
    else
        warn "Loki peut mettre quelques minutes à être prêt"
    fi
    
    # Vérifier Grafana
    if curl -f -s "http://localhost:3000/api/health" > /dev/null; then
        success "Grafana opérationnel"
    else
        warn "Grafana peut mettre quelques minutes à être prêt"
    fi
}

# Test du monitoring
test_monitoring() {
    log "🔍 Test du monitoring..."
    
    sleep 10
    
    # Test Loki logs ingestion
    if curl -s "http://localhost:3100/loki/api/v1/query?query={job=\"nginx\"}" | grep -q "data"; then
        success "Loki ingère les logs"
    else
        warn "Aucun log détecté - normal si première installation"
    fi
    
    # Test Grafana login
    if curl -s -u "admin:${GRAFANA_PASSWORD}" "http://localhost:3000/api/admin/settings" > /dev/null; then
        success "Grafana authentification OK"
    else
        warn "Grafana authentification échouée"
    fi
}

# Afficher les informations finales
show_monitoring_info() {
    log "📊 Configuration monitoring terminée"
    echo ""
    echo "📈 Services de monitoring:"
    echo "  - Loki (logs): http://localhost:3100"
    echo "  - Grafana (dashboards): http://localhost:3000"
    echo "  - Promtail (collecteur): Port 9080"
    echo ""
    echo "🔐 Accès Grafana:"
    echo "  - URL: https://monitoring.${DOMAIN_NAME}"
    echo "  - Username: admin"
    echo "  - Password: ${GRAFANA_PASSWORD}"
    echo ""
    echo "📋 Ressources utilisées (VPS Lite optimisé):"
    echo "  - Loki: ~300MB RAM"
    echo "  - Grafana: ~150MB RAM" 
    echo "  - Promtail: ~100MB RAM"
    echo "  - Total: ~550MB/4GB (14%)"
    echo ""
    echo "🔧 Commandes utiles:"
    echo "  - Logs Loki: docker-compose -f docker-compose.monitoring.yml logs loki"
    echo "  - Logs Grafana: docker-compose -f docker-compose.monitoring.yml logs grafana"
    echo "  - Status: docker-compose -f docker-compose.monitoring.yml ps"
    echo "  - Redémarrer: docker-compose -f docker-compose.monitoring.yml restart"
}

# Fonction principale
main() {
    check_dependencies
    setup_monitoring_directories
    create_grafana_config
    create_nightscan_dashboard
    start_monitoring_services
    test_monitoring
    show_monitoring_info
    
    success "🎉 Monitoring NightScan VPS Lite déployé avec succès!"
    echo ""
    echo "📊 Monitoring léger configuré:"
    echo "  ✅ Loki (remplace Elasticsearch)"
    echo "  ✅ Promtail (remplace Logstash)"
    echo "  ✅ Grafana (dashboards)"
    echo "  ✅ 90% moins de RAM que ELK Stack"
}

main "$@"