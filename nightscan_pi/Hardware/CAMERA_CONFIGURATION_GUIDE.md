# 📸 Guide de Configuration Caméra IR-CUT pour NightScanPi

## 🎯 **Vue d'Ensemble**

Ce guide explique comment configurer correctement une caméra IR-CUT sur Raspberry Pi Zero 2W pour le projet NightScanPi.

## 🔧 **Configuration Automatique (Recommandée)**

### **1. Installation Complète**
```bash
# Cloner le projet
git clone https://github.com/GamerXtrem/NightScan.git
cd NightScan

# Lancer l'installation automatique
./NightScanPi/setup_pi.sh
```

L'installateur détectera automatiquement le Raspberry Pi et proposera de configurer la caméra.

### **2. Configuration Caméra Manuelle**
```bash
# Si vous avez sauté la configuration lors de l'installation
./NightScanPi/Hardware/configure_camera_boot.sh

# Ou avec un capteur spécifique
./NightScanPi/Hardware/configure_camera_boot.sh --sensor imx219
```

## 🛠️ **Configuration Manuelle**

### **1. Prérequis Système**
```bash
# Installer les dépendances
sudo apt update
sudo apt install -y python3-picamera2 libcamera-apps unzip wget

# Ajouter l'utilisateur au groupe video
sudo usermod -a -G video $USER
```

### **2. Configuration `/boot/firmware/config.txt`**

**⚠️ CRITIQUE : Créer une sauvegarde d'abord**
```bash
sudo cp /boot/firmware/config.txt /boot/firmware/config.txt.backup
```

**Ajouter à `/boot/firmware/config.txt` :**
```bash
# NightScanPi Camera Configuration
camera_auto_detect=0
start_x=1
gpu_mem=64

# Pour capteur IMX219 (le plus commun)
dtoverlay=imx219

# OU pour d'autres capteurs :
# dtoverlay=ov5647        # OV5647
# dtoverlay=imx477        # IMX477 (HQ Camera)
# dtoverlay=imx290,clock-frequency=37125000  # IMX290/IMX327
```

## 📋 **Types de Capteurs Supportés**

| Capteur | Configuration | Usage | Résolution |
|---------|---------------|-------|------------|
| **IMX219** | `dtoverlay=imx219` | ✅ **Recommandé** - Polyvalent | 8MP (3280×2464) |
| **OV5647** | `dtoverlay=ov5647` | Caméra Pi originale | 5MP (2592×1944) |
| **IMX477** | `dtoverlay=imx477` | Haute qualité | 12MP (4056×3040) |
| **IMX290** | `dtoverlay=imx290,clock-frequency=37125000` | Vision nocturne | 2MP (1920×1080) |
| **IMX327** | `dtoverlay=imx290,clock-frequency=37125000` | Vision nocturne | 2MP (1920×1080) |
| **OV9281** | `dtoverlay=ov9281` | Obturateur global | 1MP (1280×800) |

## 🔍 **Détection du Capteur**

### **Méthode 1 : Test Libcamera**
```bash
# Raspberry Pi OS Bookworm
rpicam-hello -t 5000 --info-text "%frame"

# Raspberry Pi OS Bullseye
libcamera-hello -t 5000 --info-text "%frame"
```

### **Méthode 2 : Python**
```bash
python3 NightScanPi/Program/camera_test.py --status
```

### **Méthode 3 : Logs Système**
```bash
dmesg | grep -i camera
journalctl | grep -i camera
```

## ⚙️ **Optimisations Pi Zero 2W**

### **Configuration Mémoire Optimisée**
```bash
# Dans /boot/firmware/config.txt
gpu_mem=64                # GPU mémoire réduite pour Pi Zero
gpu_mem_512=64           # Spécifique aux 512MB du Pi Zero 2W
disable_overscan=1       # Utilisation complète de l'écran
disable_splash=1         # Boot plus rapide
```

### **Optimisations Energétiques**
```bash
# Désactiver interfaces inutilisées
dtparam=audio=off        # Si pas besoin d'audio
dtparam=spi=off          # Si pas de SPI
dtparam=i2c_arm=off      # Si pas d'I2C
```

## 🧪 **Tests et Validation**

### **1. Test Système**
```bash
# Test basic libcamera
rpicam-hello -t 5000

# Test capture image
rpicam-still -o test.jpg --width 1920 --height 1080
```

### **2. Test Python**
```bash
# Test diagnostic complet
python3 NightScanPi/Program/camera_test.py --all

# Test capture Python
python3 NightScanPi/Program/camera_test.py --capture
```

### **3. Test Intégration NightScanPi**
```bash
# Test cycle complet
python3 NightScanPi/Program/main.py
```

## 🚨 **Dépannage**

### **Problème : "No cameras available"**
```bash
# Vérifier configuration
grep -A 10 "NightScanPi Camera" /boot/firmware/config.txt

# Vérifier dtoverlay
grep "dtoverlay.*=" /boot/firmware/config.txt

# Redémarrer après configuration
sudo reboot
```

### **Problème : "Camera not detected"**
```bash
# Vérifier auto-detect désactivé
grep "camera_auto_detect" /boot/firmware/config.txt
# Doit montrer: camera_auto_detect=0

# Vérifier connexion physique
vcgencmd get_camera
# Doit montrer: supported=1 detected=1
```

### **Problème : Image de mauvaise qualité**
```bash
# Test avec paramètres manuels
rpicam-still -o test.jpg --shutter 10000 --gain 1.0 --awb auto

# Vérifier fichiers de tuning
ls /usr/share/libcamera/ipa/rpi/*/
```

### **Problème : Erreurs de mémoire**
```bash
# Augmenter gpu_mem dans config.txt
gpu_mem=128

# Ou réduire résolution
python3 NightScanPi/Program/camera_test.py --capture --resolution 640x480
```

## 📁 **Fichiers de Configuration**

### **Structure des Fichiers**
```
NightScanPi/Hardware/
├── configure_camera_boot.sh       # Script configuration automatique
├── config.txt.template             # Template complet config.txt
├── CAMERA_CONFIGURATION_GUIDE.md   # Ce guide
└── RPI IR-CUT Camera               # Documentation capteur
```

### **Fichiers Générés**
```
/opt/nightscan/camera_info.json     # Info capteur détecté
/boot/firmware/config.txt.backup    # Sauvegarde config original
~/test_nightscan_camera.sh          # Script de test
```

## 🔐 **Permissions Requises**

```bash
# Utilisateur dans groupe video
groups $USER | grep video

# Si absent, ajouter :
sudo usermod -a -G video $USER
# Puis redémarrer session
```

## 🎯 **Validation Finale**

### **Checklist de Configuration**
- [ ] **Dépendances installées** (`python3-picamera2`, `libcamera-apps`)
- [ ] **Config.txt modifié** avec `camera_auto_detect=0` et bon `dtoverlay`
- [ ] **Utilisateur dans groupe video**
- [ ] **Système redémarré** après configuration
- [ ] **Camera détectée** par `rpicam-hello`
- [ ] **Python fonctionne** avec `camera_test.py`
- [ ] **Capture d'image réussie**

### **Test de Performance**
```bash
# Test résolution complète
python3 -c "
from NightScanPi.Program.camera_trigger import capture_image
from pathlib import Path
import time

start = time.time()
path = capture_image(Path('test_images'), (1920, 1080))
duration = time.time() - start
print(f'Image {path} captured in {duration:.2f}s')
"
```

## 📞 **Support**

En cas de problème :

1. **Vérifier logs** : `dmesg | grep -i camera`
2. **Tester manuellement** : `rpicam-hello -t 5000`
3. **Vérifier config** : `cat /boot/firmware/config.txt | grep -A 10 NightScanPi`
4. **Restaurer sauvegarde** : `sudo cp /boot/firmware/config.txt.backup /boot/firmware/config.txt`

**🎉 Une fois configurée, votre caméra IR-CUT sera entièrement fonctionnelle avec NightScanPi !**