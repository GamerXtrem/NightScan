#!/bin/bash

# NightScan - Script de démarrage unifié
# Utilise .env pour la configuration

set -e  # Exit on error

# Couleurs pour les messages
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}🌙 NightScan - Démarrage de l'application${NC}"

# Charger le fichier .env s'il existe
if [ -f .env ]; then
    echo "📋 Chargement de la configuration depuis .env"
    set -a  # Export automatiquement toutes les variables
    source .env
    set +a
else
    echo -e "${YELLOW}⚠️  Fichier .env non trouvé. Copier .env.example vers .env et le configurer.${NC}"
    exit 1
fi

# Valeurs par défaut si non définies dans .env
export NIGHTSCAN_ENV=${NIGHTSCAN_ENV:-development}
export HOST=${HOST:-0.0.0.0}
export PORT=${PORT:-8000}
export WORKERS=${WORKERS:-2}
export TIMEOUT=${TIMEOUT:-120}

# Vérifier PostgreSQL
echo "🔍 Vérification de PostgreSQL..."
if ! pg_isready -h localhost -U ${DB_USER:-nightscan} >/dev/null 2>&1; then
    echo -e "${RED}❌ PostgreSQL n'est pas accessible${NC}"
    echo "Assurez-vous que PostgreSQL est démarré et configuré correctement"
    exit 1
fi
echo -e "${GREEN}✅ PostgreSQL est accessible${NC}"

# Vérifier Redis (optionnel)
if command -v redis-cli &> /dev/null; then
    echo "🔍 Vérification de Redis..."
    if redis-cli ping >/dev/null 2>&1; then
        echo -e "${GREEN}✅ Redis est accessible${NC}"
    else
        echo -e "${YELLOW}⚠️  Redis n'est pas accessible (optionnel)${NC}"
    fi
fi

# Arrêter les instances existantes
echo "🛑 Arrêt des instances existantes..."
pkill -f gunicorn 2>/dev/null || true
pkill -f "python.*web/app.py" 2>/dev/null || true
sleep 2

# Démarrer selon l'environnement
if [ "$NIGHTSCAN_ENV" = "production" ]; then
    echo -e "${GREEN}🚀 Démarrage en mode PRODUCTION${NC}"
    echo "URL: http://${HOST}:${PORT}"
    
    # Démarrer avec Gunicorn
    exec gunicorn \
        --workers $WORKERS \
        --bind ${HOST}:${PORT} \
        --timeout $TIMEOUT \
        --access-logfile - \
        --error-logfile - \
        --log-level info \
        web.app:application
else
    echo -e "${YELLOW}🔧 Démarrage en mode DÉVELOPPEMENT${NC}"
    echo "URL: http://localhost:${PORT}"
    echo ""
    echo "Identifiants de test:"
    echo "  Username: testuser"
    echo "  Password: testpass123"
    
    # Démarrer avec Flask
    export FLASK_ENV=development
    export FLASK_DEBUG=1
    exec python -m web.app
fi