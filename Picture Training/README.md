NightScan - Image Classification
Présentation
NightScan est un projet de classification d’animaux à partir d’images fixes (photos). L’objectif est de créer un modèle capable de reconnaître automatiquement une espèce à partir d’une image grâce à un pipeline complet de traitement, d’entraînement et d’évaluation utilisant PyTorch.

Fonctionnalités principales
Chargement structuré des images classées par dossier.

Prétraitement unifié et normalisation des images.

Entraînement d’un modèle CNN (ResNet18) avec option pretrained.

Évaluation sur jeu de validation et sauvegarde automatique du meilleur modèle.

Structure du projet
README.md — ce fichier

data/ — contient les images brutes classées (raw/) et les CSV associés (csv/)

models/ — répertoire de sauvegarde des modèles entraînés

scripts/ — scripts prepare_csv.py, train.py et predict.py pour préparer les données, entraîner un modèle et faire des prédictions

utils/ — fonctions utilitaires

setup.sh — script qui crée automatiquement les dossiers nécessaires

requirements.txt — dépendances Python

Installation
Clonez le dépôt et installez les dépendances dans un environnement virtuel :

bash
Copier
Modifier
git clone https://github.com/votre-utilisateur/NightScan.git
cd NightScan
python -m venv env
source env/bin/activate  # Sur Windows : env\Scripts\activate
pip install -r requirements.txt
./setup.sh
Utilisation
Préparez les fichiers CSV à partir des images puis lancez l'entraînement :

bash
Copier
Modifier
python scripts/prepare_csv.py --input_dir data/raw --output_dir data/csv
python scripts/train.py --csv_dir data/csv --model_dir models/ --pretrained
Le dossier data/raw doit contenir un dossier par espèce (ex. data/raw/bubo_bubo/, data/raw/capreolus_capreolus/, etc.). Les chemins d'accès aux images sont stockés dans train.csv, val.csv, et test.csv en respectant une répartition 70/15/15.

Fonctionnement des scripts
prepare_csv.py
Scanne tous les dossiers de --input_dir.

Crée trois fichiers CSV (train.csv, val.csv, test.csv) avec les colonnes path et label.

Répartition des données configurable (défaut : 70/15/15).

Exemple d'utilisation :

bash
Copier
Modifier
python scripts/prepare_csv.py --input_dir data/raw --output_dir data/csv
train.py
Lit train.csv et val.csv à partir du dossier --csv_dir.

Charge les images, les redimensionne en 224×224 et les normalise selon ImageNet.

Utilise torchvision.transforms pour appliquer les transformations.

Entraîne un modèle ResNet18 (avec ou sans --pretrained).

Sauvegarde automatique du meilleur modèle dans --model_dir/best_model.pth.

Paramètres : --epochs, --batch_size, --lr, --num_workers.

predict.py
Charge un modèle entraîné et prédit l’espèce d’une ou plusieurs images.

Applique les mêmes transformations que lors de l'entraînement.

Affiche le top 3 des classes prédites avec leur score de confiance.

Exemple d’utilisation :

bash
Copier
Modifier
python scripts/predict.py --model_path models/best_model.pth image.jpg
