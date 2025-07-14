# 🌙 NightScan

**NightScan** est un système intelligent de détection et surveillance de la faune nocturne utilisant l'intelligence artificielle pour reconnaître les animaux par l'analyse audio et visuelle.

## 🚀 Installation et Démarrage

**👉 [Guide d'Installation Complet](INSTALLATION_GUIDE.md) 👈**

Le guide complet vous accompagne étape par étape selon votre type d'installation :
- 🖥️ **Installation Locale** (développement/tests)
- 🌐 **VPS Production** (déploiement internet)
- 🏗️ **Docker** (containers)
- 🔧 **Raspberry Pi** (capteurs terrain)

### ⚡ Démarrage Rapide

```bash
# Installation locale simple
git clone https://github.com/votre-org/nightscan.git
cd nightscan
chmod +x setup_local.sh && ./setup_local.sh
source env/bin/activate && python web/app.py
```

**→ Accès :** http://localhost:8000

## 🏗️ Architecture Système

Le projet est organisé en plusieurs composants :

- **🎵 Audio Training** – Outils de préparation audio, spectrogrammes et entraînement de modèles
- **📸 Picture Training** – Scripts pour datasets d'images et reconnaissance visuelle  
- **🌐 Web Interface** – Application Flask avec authentification et upload
- **📱 Mobile App** – Client React Native pour iOS/Android
- **🤖 Unified Prediction** – API unifiée de prédiction audio/photo
- **🔧 Raspberry Pi** – Code pour capteurs terrain embarqués

## 📊 Fonctionnalités Principales

### 🎯 Détection Intelligente
- **Reconnaissance Audio** : Classification automatique des cris d'animaux nocturnes
- **Analyse Visuelle** : Détection et identification par caméra infrarouge
- **Prédiction Edge** : Traitement local sur Raspberry Pi pour temps réel
- **API Unifiée** : Traitement automatique audio/photo avec routage intelligent

### 🌐 Interface Web Moderne
- **Dashboard Temps Réel** : Visualisation détections et statistiques
- **Gestion Utilisateurs** : Authentification sécurisée, quotas, plans
- **Upload Sécurisé** : Traitement fichiers jusqu'à 100MB avec validation
- **Export Données** : CSV, KML pour analyse géospatiale

### 📱 Application Mobile
- **React Native** : Compatible iOS et Android
- **Cartes Interactives** : Affichage géolocalisé des détections
- **Notifications Push** : Alertes temps réel nouvelles détections
- **Mode Hors-ligne** : Synchronisation automatique

### 🔧 Monitoring & Production
- **Métriques Avancées** : Prometheus + Grafana
- **Haute Disponibilité** : Circuit breakers, connection pooling
- **Sécurité Renforcée** : CSP, rate limiting, audit trails
- **Docker Ready** : Déploiement containerisé complet

## 📚 Documentation Technique

### Guides d'Installation et Configuration
- **[Guide d'Installation Complet](INSTALLATION_GUIDE.md)** - Installation pas à pas tous environnements
- **[Guide Docker](README-Docker.md)** - Déploiement containerisé
- **[Guide VPS Production](DEPLOYMENT_GUIDE.md)** - Déploiement production VPS Lite
- **[Configuration Développeur](CLAUDE.md)** - Environnement et commandes développement

### Documentation Spécialisée  
- **[Optimisation ML](docs/ML_SERVING_OPTIMIZATION.md)** - Performance, caching, pooling
- **[Sécurité](docs/SECURITY_AUDIT_EXTERNAL_GUIDE.md)** - Audit, hardening, compliance
- **[Monitoring](docs/OPERATIONS_PROCEDURES.md)** - Métriques, alertes, maintenance
- **[Backup/Recovery](docs/BACKUP_DISASTER_RECOVERY.md)** - Sauvegarde, restauration

### Composants Spécifiques
- **[API Server](docs/en/api_server.md)** - Configuration API de prédiction
- **[Application Mobile](docs/en/mobile_app.md)** - React Native, Expo
- **[Setup Raspberry Pi](docs/en/pi_setup.md)** - Configuration capteurs terrain
- **[Plugin WordPress](docs/en/wordpress_plugin.md)** - Intégration WP

## 🚀 Démarrage Rapide Développeur

### Installation Locale
```bash
git clone https://github.com/votre-org/nightscan.git
cd nightscan
python -m venv env && source env/bin/activate
pip install -r requirements.txt
export SECRET_KEY=$(python -c 'import secrets; print(secrets.token_hex(32))')
export SQLALCHEMY_DATABASE_URI="sqlite:///nightscan.db"
python -c "from web.app import app, db; app.app_context().push(); db.create_all()"
gunicorn -w 4 -b 0.0.0.0:8000 web.app:application
```

### Docker Compose
```bash
git clone https://github.com/votre-org/nightscan.git
cd nightscan  
docker-compose up -d
```

**→ Accès Application :** http://localhost:8000  
**→ Accès API :** http://localhost:8001  
**→ Monitoring :** http://localhost:3000 (Grafana)

## 🧪 Tests et Validation

```bash
# Tests automatiques complets
pytest tests/ -v --cov=. --cov-report=term

# Tests par type
pytest -m unit        # Tests unitaires
pytest -m integration # Tests d'intégration  
pytest -m performance # Tests de performance

# Validation sécurité
bandit -r . && safety check

# Tests application mobile
cd ios-app && npm test
```

## 🌍 Écosystème NightScan

### Applications et Services
- **🌐 Interface Web** : Dashboard principal avec authentification
- **📱 App Mobile** : Client iOS/Android React Native
- **🔧 NightScan Pi** : Système embarqué Raspberry Pi
- **🔌 Plugin WordPress** : Intégration CMS

### Objectifs Utilisateurs
- **🔬 Chercheurs** : Analyse scientifique de la faune nocturne
- **📸 Photographes Animaliers** : Localisation et tracking espèces
- **🌿 Naturalistes** : Observation et documentation biodiversité
- **🏞️ Gestionnaires Espaces** : Monitoring écosystèmes protégés

## 🤝 Contribution et Support

### Contribuer au Projet
- **🐛 Issues** : [Signaler bugs et demandes](https://github.com/votre-org/nightscan/issues)
- **💬 Discussions** : [Communauté](https://github.com/votre-org/nightscan/discussions)
- **🔧 Pull Requests** : Contributions code bienvenues
- **📖 Documentation** : Améliorer guides existants

### Obtenir de l'Aide
- **📚 Documentation Complète** : Toutes les informations dans `/docs/`
- **⚡ Guide Installation** : [INSTALLATION_GUIDE.md](INSTALLATION_GUIDE.md)
- **🔧 Guide Développeur** : [CLAUDE.md](CLAUDE.md)
- **🚨 Dépannage** : Section FAQ dans guide installation

## 📄 Licence et Copyright

**Licence** : GNU General Public License v3.0  
**Copyright** : NightScan Project Contributors

Voir [LICENSE](LICENSE) pour les détails complets.

---

**🌟 NightScan - Révéler les secrets de la nuit grâce à l'IA**

*Projet open-source pour la surveillance intelligente de la faune nocturne*