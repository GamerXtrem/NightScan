# Liste de tâches NightScanPi

Cette liste regroupe les actions à mettre en œuvre pour les scripts de `NightScanPi` d'après la documentation du dossier.

## 1. Installation et configuration
- [x] Flasher **Raspberry Pi OS Lite** sur la carte SD.
- [x] Activer **SSH** et préparer `wifi_config.py` pour recevoir SSID et mot de passe depuis l'application mobile.
- [x] Installer les paquets système requis : `python3-pip`, `ffmpeg`, `sox`, `libatlas-base-dev`.
- [x] Installer les modules Python : `numpy`, `opencv-python`, `soundfile`, `flask`.

## 2. Scripts principaux
- [x] **main.py** : orchestrer le fonctionnement global (capture sur détection, horaires d'activité, appel des autres scripts).
- [x] **audio_capture.py** : enregistrer 8 s de son à chaque détection (PIR ou seuil audio) et sauvegarder en `.wav`.
- [x] **camera_trigger.py** : prendre une photo infrarouge lors de la détection (PIR ou audio).
- [x] **spectrogram_gen.py** : après 12 h, convertir les `.wav` en spectrogrammes `.npy` et supprimer les `.wav` si la carte SD dépasse 70 % de remplissage.
- [x] **wifi_config.py** : récupérer les paramètres Wi-Fi envoyés par l'application mobile et les appliquer.
- [x] **sync.py** : envoyer automatiquement spectrogrammes et photos via Wi-Fi ou module SIM ; prévoir un mode déconnexion permettant la copie manuelle via la carte SD.
- [x] **utils/energy_manager.py** : contrôler l'alimentation à l'aide du TPL5110 pour que le Pi fonctionne uniquement de 18 h à 10 h.
- [x] Ajouter des tests unitaires pour `camera_trigger.py`.

## 3. Gestion énergétique
- [x] Implémenter la planification d'arrêt/démarrage dans `energy_manager.py` pour limiter la consommation.
- [x] Veiller à ce que la génération des spectrogrammes s'effectue après midi pour ne pas gêner les captures nocturnes.

Cette liste pourra être complétée au fur et à mesure de l'avancement du projet.

## 4. Tâches complémentaires
- [x] Documenter le câblage et les caractéristiques dans le dossier `Hardware/`.
- [x] Intégrer la détection par capteur PIR et seuil audio dans `main.py`.
- [x] Créer un service pour recevoir les identifiants Wi-Fi depuis l'application mobile et appliquer `wifi_config.py`.
- [x] Ajouter la prise en charge du module SIM pour le transfert des données lorsque le Wi-Fi est indisponible.
- [x] Écrire un script d'installation automatisée pour le Raspberry Pi (packages et configuration).
- [x] Ajouter des tests unitaires pour `audio_capture.py` et `main.py`.
- [x] Fournir un exemple de fichier de configuration et activer un journal des erreurs.

## 5. Synchronisation horaire
- [x] Écrire un script `time_config.py` pour saisir l'heure actuelle et la position GPS lors de la première configuration.
- [x] Si aucune position n'est fournie, utiliser par défaut les coordonnées de Berne (46.9480 N, 7.4474 E).
- [x] Déterminer le fuseau horaire à partir de la position avec `timezonefinder` et l'appliquer via `timedatectl`.
- [x] Installer et configurer `chrony` pour maintenir l'heure synchronisée via Wi‑Fi ou module SIM.
- [x] Documenter la procédure dans `NightScanPi/README.md`.

## 6. Cycle jour/nuit automatique
- [x] Ajouter un module `sun_times.py` calculant les heures de lever et coucher du soleil en fonction de la date et des coordonnées GPS (par exemple via `suntime`).
- [x] Conserver ces horaires dans un fichier de référence mis à jour quotidiennement.
- [x] Adapter `energy_manager.py` et `main.py` pour n'activer l'enregistrement que 30 min avant le coucher du soleil jusqu'à 30 min après le lever.
- [x] Écrire des tests unitaires pour vérifier le calcul des horaires et le respect de la fenêtre d'activité.
- [x] Mettre à jour la documentation pour décrire la configuration du système basé sur le cycle solaire.

## 7. Interface de transfert manuel
- [x] Générer une mini page web accessible en local via Flask ou FastAPI.
- [x] Ajouter un bouton **Transférer les données** déclenchant l'envoi des fichiers.
- [x] Vérifier la connexion réseau (ping ou DNS) avant le transfert.
- [x] Envoyer les spectrogrammes `.npy` et les photos du dossier `/data/exports/` vers le VPS.
- [x] Supprimer les fichiers locaux une fois l'envoi confirmé.
- [x] Envoyer une notification à l'app mobile NightScan pour signaler la fin du transfert.

## 8. Réveil Wi-Fi par signal sonore
- [x] Installer `sounddevice`, `numpy`, `scipy` ou `aubio` sur le Pi.
- [x] Écrire un script `wifi_wakeup.py` analysant le flux micro en temps réel (FFT) et activant le Wi-Fi avec `sudo ifconfig wlan0 up` à la détection de 2100 Hz.
- [x] Journaliser chaque détection pour faciliter le debug.
- [x] Générer un son déclencheur `.wav` d'une seconde à 2100 Hz (ou DTMF) et l'intégrer dans l'application iOS.
- [x] Ajouter dans l'app iOS un bouton **Réveiller NightScanPi** jouant ce son à plein volume et affichant « Envoi du signal sonore… ».
- [x] Conserver un état `wifi_awake` dans un fichier `.status` et couper automatiquement le Wi-Fi après 10 min sans connexion.
- [x] Journaliser la durée d'activation du Wi-Fi.
- [ ] Tester différentes fréquences, la distance et le volume nécessaires pour éviter les faux positifs et valider le fonctionnement.
- [x] Vérifier que la détection sonore ne consomme pas trop d’énergie (benchmark).
