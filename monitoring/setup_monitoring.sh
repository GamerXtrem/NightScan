#!/bin/bash
# Script d'installation monitoring Prometheus + Grafana pour NightScan

set -e

echo "🚀 Installation monitoring avancé NightScan"
echo "============================================="

# Vérifier prérequis
if ! command -v docker &> /dev/null; then
    echo "❌ Docker n'est pas installé"
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose n'est pas installé"
    exit 1
fi

# Créer répertoires nécessaires
echo "📁 Création structure répertoires..."
mkdir -p monitoring/grafana/{dashboards,datasources}
mkdir -p monitoring/alertmanager/templates

# Vérifier fichiers de configuration
echo "🔧 Vérification configuration..."
CONFIG_FILES=(
    "prometheus.yml"
    "alerting_rules.yml"
    "alertmanager.yml"
    "docker-compose.monitoring.yml"
    "grafana/datasources/prometheus.yml"
    "grafana/dashboards/nightscan-overview.json"
)

for file in "${CONFIG_FILES[@]}"; do
    if [[ ! -f "monitoring/$file" ]]; then
        echo "❌ Fichier manquant: monitoring/$file"
        exit 1
    fi
done

echo "✅ Tous les fichiers de configuration présents"

# Configurer permissions Grafana
echo "🔐 Configuration permissions Grafana..."
sudo mkdir -p /var/lib/grafana
sudo chown -R 472:472 /var/lib/grafana

# Démarrer services monitoring
echo "🚀 Démarrage services monitoring..."
cd monitoring
docker-compose -f docker-compose.monitoring.yml up -d

# Attendre que les services démarrent
echo "⏳ Attente démarrage services..."
sleep 30

# Vérifier statut services
echo "🔍 Vérification statut services..."
SERVICES=("prometheus" "grafana" "alertmanager" "node_exporter")

for service in "${SERVICES[@]}"; do
    if docker ps | grep -q "nightscan_$service"; then
        echo "✅ $service démarré"
    else
        echo "❌ $service n'a pas démarré"
        docker logs "nightscan_$service" | tail -10
    fi
done

# Test connectivity
echo "🌐 Test connectivité services..."

# Test Prometheus
if curl -s http://localhost:9090/-/healthy > /dev/null; then
    echo "✅ Prometheus accessible sur http://localhost:9090"
else
    echo "❌ Prometheus non accessible"
fi

# Test Grafana
if curl -s http://localhost:3000/api/health > /dev/null; then
    echo "✅ Grafana accessible sur http://localhost:3000"
    echo "   Login: admin / nightscan_admin_2025"
else
    echo "❌ Grafana non accessible"
fi

# Test Alertmanager
if curl -s http://localhost:9093/-/healthy > /dev/null; then
    echo "✅ Alertmanager accessible sur http://localhost:9093"
else
    echo "❌ Alertmanager non accessible"
fi

# Configuration initiale Grafana
echo "📊 Configuration initiale Grafana..."
sleep 10

# Import dashboard automatique
GRAFANA_URL="http://admin:nightscan_admin_2025@localhost:3000"

echo "📊 Import dashboard NightScan..."
curl -X POST \
  "$GRAFANA_URL/api/dashboards/db" \
  -H "Content-Type: application/json" \
  -d @grafana/dashboards/nightscan-overview.json

# Créer utilisateur lecture seule
echo "👤 Création utilisateur monitoring..."
curl -X POST \
  "$GRAFANA_URL/api/admin/users" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Monitoring User",
    "email": "monitoring@nightscan.com",
    "login": "monitoring",
    "password": "monitoring_readonly_2025",
    "role": "Viewer"
  }'

# Configuration alertes email (si SMTP configuré)
if [[ -n "$SMTP_SERVER" ]]; then
    echo "📧 Configuration alertes email..."
    curl -X POST \
      "$GRAFANA_URL/api/alert-notifications" \
      -H "Content-Type: application/json" \
      -d '{
        "name": "email-notifications",
        "type": "email",
        "settings": {
          "addresses": "devops@nightscan.com"
        }
      }'
fi

echo ""
echo "🎉 Installation monitoring terminée!"
echo "======================================"
echo ""
echo "📊 Accès interfaces:"
echo "   Prometheus: http://localhost:9090"
echo "   Grafana:    http://localhost:3000 (admin/nightscan_admin_2025)"
echo "   Alertmanager: http://localhost:9093"
echo ""
echo "📈 Métriques disponibles:"
echo "   - Services NightScan (web, api, prediction, ml)"
echo "   - Base de données PostgreSQL"
echo "   - Cache Redis"
echo "   - Système (CPU, mémoire, disque)"
echo ""
echo "🚨 Alertes configurées:"
echo "   - Services down (1min)"
echo "   - Erreurs élevées (5min)"
echo "   - Performance dégradée"
echo "   - Ressources système"
echo ""
echo "📋 Prochaines étapes:"
echo "   1. Configurer SMTP pour alertes email"
echo "   2. Personnaliser dashboards Grafana"
echo "   3. Ajuster seuils alertes selon charge"
echo "   4. Ajouter métriques business spécifiques"
echo ""

# Création script de maintenance
cat > monitoring_maintenance.sh << 'EOF'
#!/bin/bash
# Script maintenance monitoring NightScan

echo "🔧 Maintenance monitoring NightScan"

# Nettoyage données anciennes Prometheus (>30 jours)
docker exec nightscan_prometheus \
  promtool query range 'up' \
  --start=$(date -d '30 days ago' --iso-8601) \
  --end=$(date --iso-8601) \
  --step=1h > /dev/null

# Backup configuration Grafana
docker exec nightscan_grafana \
  tar -czf /var/lib/grafana/backup-$(date +%Y%m%d).tar.gz \
  /etc/grafana /var/lib/grafana/grafana.db

# Vérification santé services
services=("prometheus" "grafana" "alertmanager" "node_exporter")
for service in "${services[@]}"; do
    if ! docker ps | grep -q "nightscan_$service"; then
        echo "⚠️ Redémarrage $service"
        docker restart "nightscan_$service"
    fi
done

echo "✅ Maintenance terminée"
EOF

chmod +x monitoring_maintenance.sh

echo "📝 Script maintenance créé: monitoring_maintenance.sh"
echo "   Recommandé: ajouter au cron quotidien"

cd ..
echo "✅ Installation monitoring complète!"