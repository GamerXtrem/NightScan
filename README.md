# NightScan
🎧 NightScan: Classification de sons animaux à partir de spectrogrammes
📦 Présentation
NightScan est un projet de classification de sons d'animaux basé sur des spectrogrammes générés à partir d'enregistrements audio. Le pipeline comprend la conversion des fichiers audio bruts en spectrogrammes de 8 secondes, l'entraînement de modèles de deep learning, et l'évaluation des performances.

🚀 Fonctionnalités principales
Prétraitement des fichiers audio pour générer des spectrogrammes de durée uniforme.

Entraînement de modèles de classification avec PyTorch.

Évaluation des performances sur des ensembles de validation.

Sauvegarde et chargement des modèles entraînés.

🗂️ Structure du projet
bash
Copier
Modifier
NightScan/
├── data/
│   ├── raw/                # Fichiers audio bruts
│   └── processed/          # Spectrogrammes générés
├── models/                 # Modèles entraînés
├── scripts/
│   ├── preprocess.py       # Script de prétraitement
│   ├── train.py            # Script d'entraînement
│   └── evaluate.py         # Script d'évaluation
├── utils/
│   └── helpers.py          # Fonctions utilitaires
├── requirements.txt        # Dépendances Python
└── README.md               # Ce fichier
🛠️ Installation
Cloner le dépôt :

bash
Copier
Modifier
git clone https://github.com/votre-utilisateur/NightScan.git
cd NightScan
Créer un environnement virtuel et installer les dépendances :

bash
Copier
Modifier
python -m venv env
source env/bin/activate  # Sur Windows : env\Scripts\activate
pip install -r requirements.txt
🧪 Utilisation
Prétraitement des données :

bash
Copier
Modifier
  python scripts/preprocess.py --input_dir data/raw --output_dir data/processed
Entraînement du modèle :

bash
Copier
Modifier
  python scripts/train.py --data_dir data/processed --model_dir models/
Évaluation du modèle :
analyticsvidhya.com

bash
Copier
Modifier
  python scripts/evaluate.py --model_path models/best_model.pth --data_dir data/processed🎧 NightScan: Classification de sons animaux à partir de spectrogrammes
📦 Présentation
NightScan est un projet de classification de sons d'animaux basé sur des spectrogrammes générés à partir d'enregistrements audio. Le pipeline comprend la conversion des fichiers audio bruts en spectrogrammes de 8 secondes, l'entraînement de modèles de deep learning, et l'évaluation des performances.

🚀 Fonctionnalités principales
Prétraitement des fichiers audio pour générer des spectrogrammes de durée uniforme.

Entraînement de modèles de classification avec PyTorch.

Évaluation des performances sur des ensembles de validation.

Sauvegarde et chargement des modèles entraînés.

🗂️ Structure du projet
bash
Copier
Modifier
NightScan/
├── data/
│   ├── raw/                # Fichiers audio bruts
│   └── processed/          # Spectrogrammes générés
├── models/                 # Modèles entraînés
├── scripts/
│   ├── preprocess.py       # Script de prétraitement
│   ├── train.py            # Script d'entraînement
│   └── evaluate.py         # Script d'évaluation
├── utils/
│   └── helpers.py          # Fonctions utilitaires
├── requirements.txt        # Dépendances Python
└── README.md               # Ce fichier
🛠️ Installation
Cloner le dépôt :

bash
Copier
Modifier
git clone https://github.com/votre-utilisateur/NightScan.git
cd NightScan
Créer un environnement virtuel et installer les dépendances :

bash
Copier
Modifier
python -m venv env
source env/bin/activate  # Sur Windows : env\Scripts\activate
pip install -r requirements.txt
🧪 Utilisation
Prétraitement des données :

bash
Copier
Modifier
  python scripts/preprocess.py --input_dir data/raw --output_dir data/processed
Entraînement du modèle :

bash
Copier
Modifier
  python scripts/train.py --data_dir data/processed --model_dir models/
Évaluation du modèle :
analyticsvidhya.com

bash
Copier
Modifier
  python scripts/evaluate.py --model_path models/best_model.pth --data_dir data/processed

  
