# ReSpeaker Lite Configuration Guide for NightScanPi

## 🎤 ReSpeaker Lite Overview

Le ReSpeaker Lite est un microphone USB avancé avec double array de microphones MEMS et traitement audio IA embarqué, spécialement conçu pour la capture audio longue distance et la suppression de bruit.

### Spécifications Techniques
- **Processeur**: XMOS XU316 AI Sound and Audio chipset
- **Microphones**: 2 microphones MEMS PDM haute performance
- **Distance de capture**: Jusqu'à 3 mètres
- **Fréquence d'échantillonnage**: **16 kHz maximum**
- **Sensibilité**: -26 dBFS
- **Point de surcharge acoustique**: 120 dBSPL
- **SNR**: 64 dBA
- **Interface**: USB Audio Class 2.0 (UAC2)
- **Canaux**: 2 (stéréo)
- **USB ID**: Vendor `2886`, Product `0019` (mode USB)

### Algorithmes Audio Embarqués
- **Cancellation d'interférences (IC)**
- **Cancellation d'écho acoustique (AEC)**
- **Suppression de bruit (NS)**
- **Contrôle automatique de gain (AGC)**
- **Ratio voix/bruit (VNR)**

## 🚨 Problèmes de Configuration Identifiés

### Avant Correction
```python
# ❌ Configuration incorrecte
RATE = 22050      # Dépasse la limite du ReSpeaker Lite (16kHz max)
CHANNELS = 1      # Sous-utilise les 2 microphones
# Pas de détection USB spécifique
```

### Après Correction
```python
# ✅ Configuration optimisée
RESPEAKER_RATE = 16000     # Respecte la limite hardware
RESPEAKER_CHANNELS = 2     # Utilise les 2 microphones
# Détection automatique du device
# Conversion stereo->mono intelligente
```

## 🔧 Configuration Automatique

### 1. Scripts de Configuration

#### Installation Automatique
```bash
# Exécuter le script de configuration ReSpeaker
sudo chmod +x /path/to/NightScanPi/Hardware/configure_respeaker_audio.sh
sudo /path/to/NightScanPi/Hardware/configure_respeaker_audio.sh
```

#### Test de Détection
```python
# Test de détection ReSpeaker
python3 -c "
from NightScanPi.Program.respeaker_detector import detect_respeaker
device = detect_respeaker()
if device:
    print(f'ReSpeaker détecté: {device.device_name}')
    print(f'Canaux: {device.channels}, Taux: {device.sample_rates}')
else:
    print('ReSpeaker Lite non détecté')
"
```

### 2. Configuration ALSA Automatique

Le script crée automatiquement `/etc/asound.conf` :

```alsa
# Configuration ALSA optimisée pour ReSpeaker Lite
pcm.!default {
    type asym
    playback.pcm "playback"
    capture.pcm "capture"
}

pcm.capture {
    type plug
    slave {
        pcm "hw:ReSpeakerLite,0"
        format S16_LE
        rate 16000
        channels 2
    }
}

pcm.respeaker_mono {
    type route
    slave {
        pcm "respeaker"
        channels 2
    }
    ttable.0.0 0.5
    ttable.0.1 0.5
}
```

### 3. Règles udev

Configuration automatique pour l'ID USB du ReSpeaker :

```udev
# /etc/udev/rules.d/99-respeaker-lite.rules
SUBSYSTEM=="usb", ATTR{idVendor}=="2886", ATTR{idProduct}=="0019", MODE="0666", GROUP="audio"
KERNEL=="controlC[0-9]*", ATTR{id}=="ReSpeakerLite", GROUP="audio", MODE="0664"
```

## 🎯 Intégration NightScanPi

### Détection Automatique

Le système NightScanPi détecte automatiquement le ReSpeaker Lite :

```python
from NightScanPi.Program.audio_capture import record_segment
from pathlib import Path

# Enregistre automatiquement avec ReSpeaker si disponible
record_segment(duration=8, out_path=Path("test.wav"))
# ✅ Utilise 16kHz, 2 canaux, converti en mono
```

### Configuration Dynamique

```python
from NightScanPi.Program.audio_capture import get_audio_config

config = get_audio_config()
print(f"Configuration audio: {config}")
# Output pour ReSpeaker:
# {
#   'device_id': 2,
#   'sample_rate': 16000,
#   'channels': 2,
#   'is_respeaker': True
# }
```

### Avantages de l'Intégration

1. **Détection automatique** du ReSpeaker Lite via USB ID
2. **Configuration optimale** : 16kHz, stéréo -> mono
3. **Fallback intelligent** vers audio par défaut si non détecté
4. **Algorithmes IA embarqués** pour qualité audio supérieure
5. **Capture longue distance** jusqu'à 3 mètres
6. **Suppression de bruit** automatique

## 🧪 Tests et Validation

### Test de Fonctionnement
```bash
# Test complet ReSpeaker
python3 /tmp/test_respeaker.py

# Test enregistrement manuel
arecord -D respeaker -f S16_LE -r 16000 -c 2 -d 5 test_respeaker.wav

# Vérification USB
lsusb | grep 2886:0019
```

### Validation Audio
```bash
# Informations device
aplay -l | grep -i respeaker
arecord -l | grep -i respeaker

# Test capture
arecord -D hw:ReSpeakerLite,0 -f S16_LE -r 16000 -c 2 -d 3 validation.wav
```

## 🔍 Dépannage

### ReSpeaker Non Détecté

1. **Vérifier connexion USB**
   ```bash
   lsusb | grep 2886
   dmesg | grep -i respeaker
   ```

2. **Vérifier firmware USB**
   ```bash
   dfu-util -l | grep 2886:0019
   ```

3. **Vérifier pilotes audio**
   ```bash
   sudo modprobe snd-usb-audio
   sudo systemctl restart pulseaudio
   ```

### Problèmes de Permission
```bash
# Ajouter utilisateur au groupe audio
sudo usermod -a -G audio $USER
sudo udevadm control --reload-rules
sudo udevadm trigger
```

### Firmware I2S vs USB
Le ReSpeaker Lite a deux firmwares :
- **USB mode** (default) : `2886:0019` - Pour NightScanPi
- **I2S mode** : `2886:001a` - Pour XIAO ESP32S3

Vérifier le bon firmware :
```bash
dfu-util -l
# Doit montrer "ver=0205" ou supérieur pour mode USB
```

## 📊 Comparaison Performance

| Aspect | Micro Standard | ReSpeaker Lite |
|--------|---------------|----------------|
| Distance capture | 1m | 3m |
| Suppression bruit | Basique | IA avancée |
| Qualité SNR | ~50dB | 64dB |
| Canaux | 1 | 2 (array) |
| Fréquence max | Variable | 16kHz optimisé |
| Algorithms embarqués | Aucun | AEC, NS, AGC, IC |

## 🎉 Résultats Attendus

Avec la configuration ReSpeaker Lite optimisée :

1. **Qualité audio supérieure** grâce aux algorithmes IA
2. **Capture longue distance** efficace jusqu'à 3m
3. **Suppression de bruit** automatique 
4. **Détection automatique** plug-and-play
5. **Compatibilité parfaite** avec l'écosystème NightScanPi
6. **Conversion intelligente** stéréo -> mono

Le système s'adapte automatiquement et utilise le ReSpeaker Lite si disponible, sinon bascule vers l'audio standard sans intervention utilisateur.