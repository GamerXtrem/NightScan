# Liste de tâches NightScanPi

Cette liste regroupe les actions à mettre en œuvre pour les scripts de `NightScanPi` d'après la documentation du dossier.

## 1. Installation et configuration
- [ ] Flasher **Raspberry Pi OS Lite** sur la carte SD.
- [ ] Activer **SSH** et préparer `wifi_config.py` pour recevoir SSID et mot de passe depuis l'application mobile.
- [ ] Installer les paquets système requis : `python3-pip`, `ffmpeg`, `sox`, `libatlas-base-dev`.
- [ ] Installer les modules Python : `numpy`, `opencv-python`, `soundfile`, `flask`.

## 2. Scripts principaux
- [ ] **main.py** : orchestrer le fonctionnement global (capture sur détection, horaires d'activité, appel des autres scripts).
- [ ] **audio_capture.py** : enregistrer 8 s de son à chaque détection (PIR ou seuil audio) et sauvegarder en `.wav`.
- [ ] **camera_trigger.py** : prendre une photo infrarouge lors de la détection (PIR ou audio).
- [ ] **spectrogram_gen.py** : après 12 h, convertir les `.wav` en spectrogrammes `.npy` et supprimer les `.wav` si la carte SD dépasse 70 % de remplissage.
- [ ] **wifi_config.py** : récupérer les paramètres Wi-Fi envoyés par l'application mobile et les appliquer.
- [ ] **sync.py** : envoyer automatiquement spectrogrammes et photos via Wi-Fi ou module SIM ; prévoir un mode déconnexion permettant la copie manuelle via la carte SD.
- [ ] **utils/energy_manager.py** : contrôler l'alimentation à l'aide du TPL5110 pour que le Pi fonctionne uniquement de 18 h à 10 h.

## 3. Gestion énergétique
- [ ] Implémenter la planification d'arrêt/démarrage dans `energy_manager.py` pour limiter la consommation.
- [ ] Veiller à ce que la génération des spectrogrammes s'effectue après midi pour ne pas gêner les captures nocturnes.

Cette liste pourra être complétée au fur et à mesure de l'avancement du projet.
