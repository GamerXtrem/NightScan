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
Flasher Raspberry Pi OS Lite sur carte SD

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

Le traitement des fichiers audio (.wav → .npy) se fait après 12h, pour éviter les pics de charge pendant la collecte

