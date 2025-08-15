# NightScanPi - Guide de Déploiement en Production

## Vue d'Ensemble

Ce guide détaille la configuration et le déploiement d'un dispositif NightScanPi pour un environnement de production. Le système capture automatiquement des sons et images de faune nocturne avec transmission des données vers l'API cloud.

## Prérequis

### Matériel Requis
- Raspberry Pi Zero 2 W (ou Pi 4 pour de meilleures performances)
- Carte microSD 64 Go minimum (format ext4 recommandé)
- Caméra IR-CUT (module CSI)
- Microphone USB (ReSpeaker Mic Array Lite recommandé)
- LEDs infrarouges pour vision nocturne
- Détecteur PIR pour détection de mouvement
- Batterie 18650 + TPL5110 pour gestion d'alimentation
- Panneau solaire 5V 1A pour recharge
- (Optionnel) Module SIM pour connectivité sans Wi-Fi

### Logiciels Requis
- Raspberry Pi OS Lite 64-bit
- Python 3.9+ avec pip
- Accès SSH configuré

## Installation Rapide

### 1. Clone et Configuration Initiale

```bash
# Sur le Raspberry Pi
git clone https://github.com/votre-repo/NightScan.git
cd NightScan/nightscan_pi

# Exécuter l'installation automatisée
./setup_pi.sh

# Configuration pour la production
./configure_production.sh
```

### 2. Configuration Interactive

Le script `configure_production.sh` vous guidera à travers :
- Configuration de l'URL API
- Coordonnées GPS et timezone
- Paramètres audio et caméra
- Module SIM (si applicable)
- Création du service systemd

## Variables d'Environnement

### Variables Critiques (OBLIGATOIRES pour la production)

| Variable | Description | Exemple |
|----------|-------------|---------|
| `NIGHTSCAN_API_URL` | URL de l'API pour upload des fichiers | `https://api.yourdomain.com/upload` |
| `NIGHTSCAN_GPS_COORDS` | Coordonnées GPS (lat,lon) | `46.9,7.4` |

### Variables de Configuration API

| Variable | Description | Défaut |
|----------|-------------|---------|
| `NIGHTSCAN_API_TOKEN` | Token d'authentification API | - |
| `NIGHTSCAN_UPLOAD_RETRIES` | Tentatives de retry upload | `3` |
| `NIGHTSCAN_UPLOAD_TIMEOUT` | Timeout upload (secondes) | `60` |
| `NIGHTSCAN_OFFLINE` | Mode hors-ligne (1=activé) | `0` |

### Variables de Localisation et Temps

| Variable | Description | Défaut |
|----------|-------------|---------|
| `NIGHTSCAN_TIMEZONE` | Fuseau horaire | Auto-détecté |
| `NIGHTSCAN_START_HOUR` | Heure de début (24h) | `18` |
| `NIGHTSCAN_STOP_HOUR` | Heure de fin (24h) | `10` |
| `NIGHTSCAN_SUN_OFFSET` | Offset coucher/lever (min) | `30` |

### Variables Audio

| Variable | Description | Défaut |
|----------|-------------|---------|
| `NIGHTSCAN_AUDIO_THRESHOLD` | Seuil détection audio (0.0-1.0) | `0.5` |
| `NIGHTSCAN_AUDIO_DURATION` | Durée enregistrement (sec) | `8` |
| `NIGHTSCAN_AUDIO_SAMPLE_RATE` | Fréquence échantillonnage | `22050` |

### Variables Caméra

| Variable | Description | Défaut |
|----------|-------------|---------|
| `NIGHTSCAN_CAMERA_RESOLUTION` | Résolution (largeur,hauteur) | `1920,1080` |
| `NIGHTSCAN_CAMERA_ISO` | Sensibilité ISO | `800` |
| `NIGHTSCAN_IR_ENABLED` | LEDs IR activées | `true` |

### Variables Performance

| Variable | Description | Défaut |
|----------|-------------|---------|
| `NIGHTSCAN_FORCE_PI_ZERO` | Force optimisations Pi Zero | Auto-détecté |
| `NIGHTSCAN_MAX_MEMORY_PERCENT` | % mémoire max avant nettoyage | `85` |
| `NIGHTSCAN_THREADS` | Nombre de threads | Auto-détecté |
| `NIGHTSCAN_MAX_STORAGE_PERCENT` | % stockage max SD | `70` |

### Variables Stockage

| Variable | Description | Défaut |
|----------|-------------|---------|
| `NIGHTSCAN_DATA_DIR` | Répertoire données | `/home/pi/nightscan_data` |
| `NIGHTSCAN_LOG` | Fichier log | `/home/pi/nightscan.log` |
| `NIGHTSCAN_KEEP_WAV_FILES` | Garder fichiers .wav | `false` |
| `NIGHTSCAN_SPECTROGRAM_FORMAT` | Format spectrogrammes | `npy` |

### Variables Matériel

| Variable | Description | Défaut |
|----------|-------------|---------|
| `NIGHTSCAN_SIM_DEVICE` | Périphérique module SIM | - |
| `NIGHTSCAN_SIM_BAUDRATE` | Débit SIM | `115200` |
| `NIGHTSCAN_WIFI_SCAN_TIMEOUT` | Timeout scan Wi-Fi (sec) | `30` |

### Variables Debug

| Variable | Description | Défaut |
|----------|-------------|---------|
| `NIGHTSCAN_DEBUG` | Logs de debug | `false` |
| `NIGHTSCAN_CAMERA_DEBUG` | Debug caméra | `false` |
| `NIGHTSCAN_AUDIO_DEBUG` | Debug audio | `false` |
| `NIGHTSCAN_SKIP_HARDWARE` | Skip init hardware (tests) | `false` |

## Déploiement en Production

### 1. Service Systemd

```bash
# Copier le service généré
sudo cp nightscan.service /etc/systemd/system/

# Activer et démarrer
sudo systemctl enable nightscan
sudo systemctl start nightscan

# Vérifier le statut
sudo systemctl status nightscan
```

### 2. Monitoring

```bash
# Logs en temps réel
sudo journalctl -u nightscan -f

# Logs du fichier principal
tail -f ~/nightscan.log

# Vérifier l'espace disque
df -h

# Surveiller la mémoire (Pi Zero)
free -h
```

### 3. Maintenance

#### Sauvegarde Configuration

```bash
# Utiliser le script de sauvegarde
./backup_config.sh

# Sauvegarde manuelle
tar -czf nightscan_backup_$(date +%Y%m%d).tar.gz .env Program/nightscan_config.db ~/sun_times.json
```

#### Rotation des Logs

```bash
# Configurer logrotate
sudo tee /etc/logrotate.d/nightscan << EOF
/home/pi/nightscan.log {
    daily
    rotate 7
    compress
    delaycompress
    missingok
    notifempty
    create 644 pi pi
    postrotate
        systemctl reload nightscan
    endscript
}
EOF
```

### 4. Tests de Validation

#### Test Complet du Système

```bash
# Test de tous les composants
python Program/camera_test.py --all

# Test de détection du capteur
python Program/camera_test.py --detect-sensor

# Test d'upload
python Program/sync.py ~/nightscan_data/spectrograms --url $NIGHTSCAN_API_URL
```

#### Test de Performance

```bash
# Vérifier la mémoire Pi Zero
python -c "
import psutil
mem = psutil.virtual_memory()
print(f'Mémoire: {mem.percent}% utilisée')
print(f'Disponible: {mem.available/1024/1024:.0f}MB')
"
```

## Résolution de Problèmes

### Problèmes Courants

#### 1. Erreur d'Import Modules

**Symptôme :** `ModuleNotFoundError` pour les imports locaux

**Solution :**
```bash
# Vérifier la structure des imports
cd nightscan_pi
python -c "from Program.main import main; print('Imports OK')"
```

#### 2. URL API Non Configurée

**Symptôme :** Uploads échouent avec `example.com`

**Solution :**
```bash
# Vérifier la configuration
grep NIGHTSCAN_API_URL .env

# Reconfigurer si nécessaire
./configure_production.sh
```

#### 3. Caméra Non Détectée

**Symptôme :** Erreurs caméra au démarrage

**Solution :**
```bash
# Vérifier la détection
python Program/camera_sensor_detector.py

# Reconfigurer la caméra
./Hardware/configure_camera_boot.sh
sudo reboot
```

#### 4. Problèmes de Mémoire (Pi Zero)

**Symptôme :** Processus tué par OOM

**Solution :**
```bash
# Vérifier les optimisations Pi Zero
grep NIGHTSCAN_FORCE_PI_ZERO .env

# Réduire la résolution
sed -i 's/NIGHTSCAN_CAMERA_RESOLUTION=.*/NIGHTSCAN_CAMERA_RESOLUTION=1280,720/' .env
```

### Logs de Diagnostic

```bash
# Logs système complets
sudo dmesg | grep -E "(camera|usb|audio)"

# État des services
sudo systemctl status nightscan

# Utilisation des ressources
htop

# Test connectivité réseau
ping -c 4 8.8.8.8
curl -I $NIGHTSCAN_API_URL
```

## Sécurité

### Permissions Fichiers

```bash
# Sécuriser le fichier .env
chmod 600 .env

# Vérifier les permissions du répertoire
ls -la ~/nightscan_data/
```

### Authentification API

Si votre API nécessite une authentification, configurez le token :

```bash
echo "NIGHTSCAN_API_TOKEN=your-secure-token" >> .env
```

### Accès SSH

Pour la production, sécurisez l'accès SSH :

```bash
# Changer le port SSH (optionnel)
sudo sed -i 's/#Port 22/Port 2222/' /etc/ssh/sshd_config

# Désactiver l'authentification par mot de passe
sudo sed -i 's/#PasswordAuthentication yes/PasswordAuthentication no/' /etc/ssh/sshd_config

# Redémarrer SSH
sudo systemctl restart ssh
```

## Support et Maintenance

### Contacts
- Documentation technique : voir `README.md` principal
- Issues : [Repository GitHub Issues]
- Configuration matérielle : voir `Hardware/README.md`

### Mises à Jour

```bash
# Mise à jour du système
sudo apt update && sudo apt upgrade -y

# Mise à jour du code
git pull origin main

# Redémarrer le service
sudo systemctl restart nightscan
```

---

**Note :** Ce système est optimisé pour fonctionner de manière autonome. En cas de problème critique, consultez les logs et n'hésitez pas à redémarrer le dispositif.