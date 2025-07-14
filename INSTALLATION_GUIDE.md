# 🚀 Guide d'Installation Complet NightScan

**NightScan** est un système de détection et surveillance de la faune nocturne utilisant l'intelligence artificielle pour reconnaître les animaux par l'analyse audio et visuelle.

## 🎯 Choisir Votre Type d'Installation

Sélectionnez le type d'installation qui correspond à vos besoins :

| Type | Usage | Complexité | Temps |
|------|-------|------------|-------|
| [🖥️ **Installation Locale**](#-installation-locale-développement) | Développement, tests | ⭐⭐ | 30 min |
| [🌐 **VPS Production**](#-déploiement-vps-production) | Production internet | ⭐⭐⭐⭐ | 2h |
| [🏗️ **Docker**](#-déploiement-docker) | Containers, scaling | ⭐⭐⭐ | 45 min |
| [🔧 **Raspberry Pi**](#-configuration-raspberry-pi) | Capteurs terrain | ⭐⭐⭐⭐⭐ | 3h |

---

## 📋 Pré-requis Système

### Minimum Requis
- **OS** : Linux (Ubuntu 20.04+), macOS 10.15+, Windows 10
- **RAM** : 4 GB minimum, 8 GB recommandé  
- **Storage** : 10 GB libre minimum
- **Python** : 3.10+ 
- **Git** : Pour clonage repository

### Pré-requis par Type d'Installation

#### 🖥️ Installation Locale
- Python 3.10+
- Git
- FFmpeg
- PostgreSQL (optionnel, SQLite par défaut)

#### 🌐 VPS Production  
- VPS avec 4GB RAM minimum
- Domaine pointant vers le VPS
- Accès SSH root/sudo
- Docker Engine 20.10+

#### 🏗️ Docker
- Docker Engine 20.10+
- Docker Compose 2.0+
- 4GB RAM disponible

#### 🔧 Raspberry Pi
- Raspberry Pi 4B+ (4GB+ RAM)
- Carte SD 32GB+ (Classe 10)
- Caméra Pi compatible
- Micro ReSpeaker (audio)

---

## ⚡ Installation Rapide (Quick Start)

### Option 1: Installation Locale Simple
```bash
# Cloner le projet
git clone https://github.com/votre-org/nightscan.git
cd nightscan

# Setup automatique
chmod +x setup_local.sh
./setup_local.sh

# Démarrer l'application
source env/bin/activate
python web/app.py
```
**→ Accès :** http://localhost:8000

### Option 2: Docker Compose
```bash
# Cloner et démarrer
git clone https://github.com/votre-org/nightscan.git
cd nightscan
docker-compose up -d

# Vérifier statut
docker-compose ps
```
**→ Accès :** http://localhost:8000

---

## 🔧 Installation Détaillée Étape par Étape

### 🖥️ Installation Locale (Développement)

#### Étape 1: Préparation Environnement
```bash
# Installer dépendances système (Ubuntu/Debian)
sudo apt update
sudo apt install -y python3.10 python3.10-venv python3-pip git ffmpeg portaudio19-dev

# macOS avec Homebrew
brew install python@3.10 git ffmpeg portaudio

# Windows - Installer depuis python.org et ffmpeg.org
```

#### Étape 2: Clonage et Setup
```bash
# Cloner le repository
git clone https://github.com/votre-org/nightscan.git
cd nightscan

# Créer environnement virtuel
python3.10 -m venv env
source env/bin/activate  # Windows: env\Scripts\activate

# Installer dépendances Python
pip install --upgrade pip
pip install -r requirements.txt
```

#### Étape 3: Configuration Base de Données
```bash
# Option A: SQLite (simple, développement)
export SQLALCHEMY_DATABASE_URI="sqlite:///nightscan.db"

# Option B: PostgreSQL (recommandé)
sudo apt install postgresql postgresql-contrib
sudo -u postgres createdb nightscan
export SQLALCHEMY_DATABASE_URI="postgresql://postgres:password@localhost/nightscan"
```

#### Étape 4: Configuration Variables
```bash
# Générer clé secrète
export SECRET_KEY=$(python -c 'import secrets; print(secrets.token_hex(32))')

# URLs des services
export PREDICT_API_URL="http://localhost:8001/api/predict"

# Optionnel: Redis pour cache
export REDIS_URL="redis://localhost:6379/0"
```

#### Étape 5: Initialisation Base de Données
```bash
# Créer tables
python -c "from web.app import app, db; app.app_context().push(); db.create_all()"

# Optionnel: Données de test
python database/scripts/load_test_data.py
```

#### Étape 6: Démarrage Services
```bash
# Terminal 1: Application Web
gunicorn -w 4 -b 0.0.0.0:8000 web.app:application

# Terminal 2: API de Prédiction
export MODEL_PATH="models/best_model.pth"
export CSV_DIR="data/processed/csv"
gunicorn -w 4 -b 0.0.0.0:8001 unified_prediction_system.unified_prediction_api:application

# Terminal 3: Worker Celery (optionnel)
celery -A web.tasks worker --loglevel=info
```

**✅ Validation :** http://localhost:8000 - Page d'accueil accessible

---

### 🌐 Déploiement VPS Production

> **📖 Guide Complet :** [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)

#### Étape 1: Préparation VPS
```bash
# Connexion SSH
ssh root@votre-vps-ip

# Mise à jour système
apt update && apt upgrade -y
apt install -y docker.io docker-compose git ufw fail2ban

# Utilisateur dédié
useradd -m -s /bin/bash nightscan
usermod -aG docker nightscan
su - nightscan
```

#### Étape 2: Clonage et Configuration
```bash
# Cloner projet
git clone https://github.com/votre-org/nightscan.git
cd nightscan

# Configuration domaine
export DOMAIN_NAME="votre-domaine.com"
export ADMIN_EMAIL="admin@votre-domaine.com"

# Générer secrets
./scripts/setup-secrets.sh
```

#### Étape 3: SSL/TLS et Sécurité
```bash
# Configuration SSL automatique
./scripts/setup-ssl.sh

# Firewall
sudo ./scripts/setup-firewall.sh

# Protection fail2ban
sudo ./scripts/setup-fail2ban.sh
```

#### Étape 4: Déploiement Docker
```bash
# Créer réseau
docker network create nightscan-net

# Démarrer services
docker-compose -f docker-compose.production.yml up -d

# Monitoring
docker-compose -f docker-compose.monitoring.yml up -d
```

#### Étape 5: Validation Production
```bash
# Tests automatiques
python scripts/validate-production.py

# Tests manuels
curl -k https://votre-domaine.com/health
curl -k https://api.votre-domaine.com/health
```

**✅ Validation :** https://votre-domaine.com accessible avec SSL

---

### 🏗️ Déploiement Docker

#### Étape 1: Préparation Docker
```bash
# Installer Docker (Ubuntu)
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh

# Installer Docker Compose
curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
chmod +x /usr/local/bin/docker-compose

# Vérification
docker --version
docker-compose --version
```

#### Étape 2: Configuration Environnement
```bash
# Cloner projet
git clone https://github.com/votre-org/nightscan.git
cd nightscan

# Copier configuration
cp .env.example .env

# Éditer variables nécessaires
nano .env
```

#### Étape 3: Build et Démarrage
```bash
# Build images
docker-compose build

# Démarrer stack complète
docker-compose up -d

# Vérifier statut
docker-compose ps
docker-compose logs -f
```

#### Services Déployés
- **Web App** : http://localhost:8000
- **API** : http://localhost:8001  
- **PostgreSQL** : localhost:5432
- **Redis** : localhost:6379
- **Monitoring** : http://localhost:3000 (Grafana)

**✅ Validation :** `docker-compose ps` - Tous services "Up"

---

### 🔧 Configuration Raspberry Pi

> **📖 Guide Détaillé :** [docs/en/pi_setup.md](docs/en/pi_setup.md)

#### Étape 1: Préparation Hardware
```bash
# Image Raspberry Pi OS Lite
# Flash sur carte SD avec Raspberry Pi Imager
# Activer SSH dans raspi-config

# Première connexion
ssh pi@raspberry-pi-ip
sudo apt update && sudo apt upgrade -y
```

#### Étape 2: Installation Capteurs
```bash
# Caméra Pi
sudo raspi-config  # Interface Options > Camera > Enable

# ReSpeaker Audio
cd nightscan_pi/Hardware
chmod +x configure_respeaker_audio.sh
sudo ./configure_respeaker_audio.sh

# Test caméra
libcamera-still -o test.jpg

# Test audio
arecord -D plughw:CARD=seeed2micvoicec,DEV=0 -f cd test.wav
```

#### Étape 3: Installation NightScan Pi
```bash
# Cloner projet
git clone https://github.com/votre-org/nightscan.git
cd nightscan/nightscan_pi

# Installation dépendances
sudo ./setup_pi.sh

# Configuration
python Program/wifi_config.py --setup
```

#### Étape 4: Configuration Services
```bash
# Service principal
sudo cp nightscan.service /etc/systemd/system/
sudo systemctl enable nightscan
sudo systemctl start nightscan

# WiFi manager
sudo cp wifi-manager.service /etc/systemd/system/
sudo systemctl enable wifi-manager
```

**✅ Validation :** LED verte clignote, détections dans logs

---

## 🧪 Validation et Tests

### Tests Fonctionnels de Base
```bash
# Test santé application
curl http://localhost:8000/health

# Test API prédiction
curl -X POST http://localhost:8001/api/predict \
  -F "file=@test_audio.wav"

# Test base de données
python -c "from web.app import db; print('DB OK' if db.engine.execute('SELECT 1').scalar() == 1 else 'DB Error')"
```

### Tests Complets
```bash
# Suite de tests automatiques
pytest tests/ -v --cov=. --cov-report=term

# Tests performance
python scripts/test-performance.py

# Tests sécurité
python scripts/security-audit.py
```

### Validation Mobile App
```bash
cd ios-app
npm install
npm test
npm start  # Démarrer Expo
```

---

## 🎯 Configuration Post-Installation

### 1. Création Premier Utilisateur
```bash
# Via interface web
# → http://localhost:8000/register

# Via CLI
python scripts/create-admin-user.py --email admin@domain.com
```

### 2. Upload Premier Modèle
```bash
# Modèles pré-entraînés dans models/
# Ou entraîner nouveau modèle:
python Audio_Training/scripts/train.py --csv_dir data/processed/csv --model_dir models
```

### 3. Configuration Mobile
```bash
# Dans ios-app/services/api.js
export const API_BASE_URL = 'http://votre-serveur:8000';

# Build app
cd ios-app
npm run build
```

### 4. Monitoring (Production)
- **Grafana** : http://votre-domaine.com:3000
- **Login** : admin / (voir secrets/production/.env)
- **Dashboards** : Système, Docker, Application

---

## 🚨 Dépannage et FAQ

### Problèmes Fréquents

#### ❌ Port déjà utilisé
```bash
# Identifier processus
sudo lsof -i :8000
sudo kill -9 PID

# Changer port
export PORT=8080
gunicorn -w 4 -b 0.0.0.0:$PORT web.app:application
```

#### ❌ Erreur base de données
```bash
# Vérifier connexion
python -c "import psycopg2; print('OK')"

# Réinitialiser DB
dropdb nightscan && createdb nightscan
python -c "from web.app import db; db.create_all()"
```

#### ❌ FFmpeg non trouvé
```bash
# Ubuntu/Debian
sudo apt install ffmpeg

# macOS
brew install ffmpeg

# Vérifier
ffmpeg -version
```

#### ❌ Erreur modèle PyTorch
```bash
# Réinstaller PyTorch
pip uninstall torch torchaudio
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu

# Télécharger modèle
wget https://github.com/votre-org/nightscan-models/releases/download/v1.0/best_model.pth -O models/best_model.pth
```

### FAQ Installation

**Q: Quelle version de Python utiliser ?**  
A: Python 3.10+ recommandé. 3.8+ minimum supporté.

**Q: Peut-on utiliser SQLite en production ?**  
A: Non recommandé. PostgreSQL conseillé pour performances et concurrent access.

**Q: Comment changer le port par défaut ?**  
A: Variable d'environnement `PORT=8080` ou paramètre `-b 0.0.0.0:8080` avec gunicorn.

**Q: L'app mobile ne se connecte pas ?**  
A: Vérifier `API_BASE_URL` dans `ios-app/services/api.js` et firewall serveur.

**Q: Comment activer HTTPS en local ?**  
A: Utiliser `mkcert` pour certificats auto-signés ou reverse proxy nginx.

**Q: Erreur "Model not found" ?**  
A: Vérifier `MODEL_PATH` et télécharger modèle depuis releases GitHub.

---

## 📚 Documentation Complémentaire

### Guides Techniques Détaillés
- [**Configuration Avancée**](docs/CONFIGURATION_GUIDE.md) - Variables, secrets, sécurité
- [**Optimisation ML**](docs/ML_SERVING_OPTIMIZATION.md) - Performance, caching, pooling
- [**Sécurité**](docs/SECURITY_AUDIT_EXTERNAL_GUIDE.md) - Audit, hardening, compliance
- [**Monitoring**](docs/OPERATIONS_PROCEDURES.md) - Métriques, alertes, maintenance
- [**Backup/Recovery**](docs/BACKUP_DISASTER_RECOVERY.md) - Sauvegarde, restauration

### Composants Spécifiques
- [**API Server**](docs/en/api_server.md) - Configuration API de prédiction
- [**Flask App**](docs/en/flask_app.md) - Interface web, routes, auth
- [**Mobile App**](docs/en/mobile_app.md) - React Native, Expo
- [**Nginx Setup**](docs/en/nginx_setup.md) - Reverse proxy, SSL/TLS
- [**WordPress Plugin**](docs/en/wordpress_plugin.md) - Intégration WP

### Architecture et Développement
- [**Architecture Système**](CLAUDE.md) - Composants, ports, responsabilités
- [**CI/CD Pipeline**](docs/CI_CD_GUIDE.md) - Tests, déploiement automatisé
- [**Docker Guide**](README-Docker.md) - Containers, orchestration

---

## 🆘 Support et Communauté

### Obtenir de l'Aide
- **🐛 Issues GitHub** : [Créer une issue](https://github.com/votre-org/nightscan/issues)
- **📖 Documentation** : Guides dans `/docs/`
- **💬 Discussions** : [GitHub Discussions](https://github.com/votre-org/nightscan/discussions)

### Contribuer
- **🔧 Développement** : Fork, branch, pull request
- **📝 Documentation** : Améliorer guides existants
- **🧪 Tests** : Ajouter tests, rapporter bugs
- **🌍 Traductions** : Traduire documentation

---

## ✅ Checklist Installation Réussie

### Installation Locale
- [ ] Python 3.10+ installé et vérifié
- [ ] Repository cloné et dépendances installées
- [ ] Base de données créée et accessible
- [ ] Variables d'environnement configurées
- [ ] Application web accessible sur http://localhost:8000
- [ ] API de prédiction répond sur http://localhost:8001
- [ ] Tests automatiques passent avec `pytest`

### VPS Production
- [ ] VPS configuré avec Docker et domaine
- [ ] SSL/TLS configuré et fonctionnel (A+ rating)
- [ ] Services Docker démarrés et stables
- [ ] Monitoring Grafana accessible
- [ ] Firewall UFW et fail2ban actifs
- [ ] Backups automatiques configurés
- [ ] Tests post-déploiement réussis

### Raspberry Pi
- [ ] Hardware assemblé et fonctionnel
- [ ] Capteurs caméra et audio détectés
- [ ] Service NightScan Pi actif
- [ ] WiFi manager opérationnel
- [ ] Premières détections dans les logs

---

**🎉 Installation Terminée !**

Votre système NightScan est maintenant opérationnel. Consultez la documentation technique pour la configuration avancée et les optimisations.

**Prochaines étapes :**
1. 📱 Configurer l'application mobile
2. 🤖 Entraîner vos propres modèles  
3. 📊 Explorer les dashboards de monitoring
4. 🔧 Optimiser les performances selon vos besoins

---

*Guide créé pour NightScan v2.0 - Mis à jour le 14 juillet 2025*