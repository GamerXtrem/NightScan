NightScanPi – Piège nocturne autonome audio & photo (Raspberry Pi Zero 2 W)
🎯 Objectif
NightScanPi est un système embarqué dédié à la capture automatisée de sons et d’images de la faune nocturne, fonctionnant sur batterie et panneau solaire, avec envoi des données via Wi-Fi ou module SIM. Il est actif entre 18h et 10h, et transforme les sons détectés en spectrogrammes .npy plus légers pour le transfert.

🧭 Première utilisation (Onboarding)
À réception de l’appareil, l’utilisateur doit :

Insérer une carte microSD (min. 64 Go, format ext4 conseillé)

Alimenter l'appareil (aucun bouton requis, démarrage automatique à la mise sous tension)

Lancer l’application mobile NightScan (iOS / Android)

Depuis l'application :

Configurer le Wi-Fi en envoyant SSID et mot de passe

Saisir la position GPS de l’installation

(Facultatif) Activer l’envoi via module SIM si installé et si un abonnement a été souscrit

Configurer la date et l'heure avec `time_config.py` (coordonnées GPS nécessaires pour le fuseau horaire)

🧩 Composants
Composant	Fonction
Raspberry Pi Zero 2 W	Unité centrale
Caméra IR-Cut (CSI)	Capture photo nocturne
Micro USB	Capture audio
LED infrarouges	Vision de nuit
Détecteur PIR	Détection de mouvement
Carte microSD 64 Go min.	Stockage des données
Batterie 18650 + TPL5110	Alimentation et timer
Panneau solaire 5V 1A	Recharge quotidienne
(Optionnel) Module SIM	Transfert hors Wi-Fi
### Informations matérielles complémentaires

Des fiches détaillées se trouvent dans le répertoire `Hardware/` :

- **Raspberry Pi Zero 2 W** : processeur quad‑cœur 1 GHz, 512 Mo de RAM, Wi‑Fi 2,4 GHz et Bluetooth 4.2. Sa consommation varie entre 0,6 W et 3 W.
- **RPI IR‑CUT Camera** : module caméra CSI avec filtre infrarouge motorisé et LED IR, prévu pour la vision diurne et nocturne. Le courant maximal avoisine 150 mA.
- **ReSpeaker Mic Array Lite** : carte microphonique double basée sur un chipset XMOS XU316 intégrant l’annulation d’écho et la suppression de bruit, avec une LED RGB.

Ces documents décrivent les schémas de raccordement et les réglages avancés (modes HDR de la caméra, mise à jour du micro, etc.).

⏱ Fonctionnement
🕕 De 18h à 10h :

Le système est actif

À chaque détection par le capteur PIR, il capture :

1 photo infrarouge

1 enregistrement audio de 8 secondes (.wav)
a chaque détection audio quand ça dépasse un seuil, il capture:
1 photo 
1 enregistrement audio de 8 secondes

🕛 À partir de 12h :

Les fichiers audio sont transformés en spectrogrammes .npy
Les enregistrements sont rééchantillonnés à 22 050 Hz et convertis en
mel-spectrogrammes exprimés en dB afin de correspondre au traitement de
`predict.py`

Les fichiers .wav sont automatiquement supprimés si la carte SD dépasse 70% de remplissage

📤 Transfert des données
Via Wi-Fi configuré avec l'app mobile NightScan :

Transfert automatique des spectrogrammes et photos

Via module SIM :

Transfert automatique si réseau disponible et abonnement actif

Sinon : l’utilisateur peut retirer la carte SD pour consulter les fichiers localement

📁 Structure des fichiers
swift
Copier
Modifier
/home/pi/nightscanpi/
├── main.py
├── audio_capture.py
├── camera_trigger.py
├── spectrogram_gen.py
├── wifi_config.py
├── sync.py
└── utils/
    └── energy_manager.py
🛠 Installation système
Flasher Raspberry Pi OS Lite 64 bits sur carte SD

Activer SSH et préparer les scripts wifi_config.py pour connexion via application mobile

Installer les dépendances :

bash
Copier
Modifier
sudo apt update
sudo apt install python3-pip ffmpeg sox libatlas-base-dev
pip3 install numpy opencv-python soundfile flask
🔌 Gestion énergétique
TPL5110 coupe automatiquement le courant en dehors de la plage horaire utile

Le Pi est alimenté uniquement de 18h à 10h

Les horaires peuvent être adaptés en définissant les variables
`NIGHTSCAN_START_HOUR` et `NIGHTSCAN_STOP_HOUR` avant l'exécution des scripts
(`energy_manager.py`, `main.py`, etc.).

Le cycle jour/nuit est activé par défaut. Les heures de lever et de coucher
sont enregistrées dans `~/sun_times.json` dès l'installation. Ce fichier est
mis à jour automatiquement si la position ou la date changent. Pour stocker ces
informations ailleurs, définissez `NIGHTSCAN_SUN_FILE`. La marge par rapport au
lever/coucher peut être ajustée via `NIGHTSCAN_SUN_OFFSET` (en minutes).

Le traitement des fichiers audio (.wav → .npy) se fait après 12h, pour éviter les pics de charge pendant la collecte

## Aperçu du dépôt NightScan

Ce dossier `NightScanPi/` représente la partie embarquée du projet. À la racine du dépôt, on trouve notamment :
- `Audio_Training/` et `Picture_Training/` pour la préparation des données et l'entraînement des modèles de reconnaissance.
- `web/` contenant l'application Flask servant d'interface de téléversement et de consultation des prédictions.
- `ios-app/` pour un exemple d'application mobile.
- `wp-plugin/` avec des modules WordPress dédiés aux envois depuis un site et à l'affichage des statistiques.
- `setup_vps_infomaniak.sh` qui automatise le déploiement d'un VPS configuré pour héberger l'API.
- `docs/` où se trouvent des guides complémentaires.

Le `README.md` situé à la racine détaille ces répertoires et explique comment installer l'environnement de test.

## Dossier `Program`
Ce répertoire contient les scripts Python exécutés sur le Raspberry Pi :

- `main.py` orchestre les captures nocturnes.
- `audio_capture.py` enregistre 8 s d'audio.
- `camera_trigger.py` prend une photo infrarouge.
- `spectrogram_gen.py` convertit les fichiers `.wav` en spectrogrammes `.npy`.
- `wifi_config.py` écrit la configuration Wi-Fi reçue via l'application mobile.
- `sync.py` envoie automatiquement les fichiers générés.
- `utils/energy_manager.py` gère la plage horaire d'activité.
- `time_config.py` règle l'heure et le fuseau en début d'installation.

### Synchronisation horaire
Pour définir l'heure et le fuseau, exécutez :
```bash
python time_config.py "2024-01-01 12:00:00" --lat 46.9 --lon 7.4
```
Si aucune coordonnée n'est fournie, celles de Berne sont utilisées.

Pour conserver une heure précise durant l'utilisation, installez et activez
`chrony` qui synchronisera automatiquement l'horloge dès qu'une connexion
(Wi‑Fi ou module SIM) est disponible :

```bash
sudo apt install -y chrony
sudo systemctl enable --now chrony
sudo timedatectl set-ntp true
```
