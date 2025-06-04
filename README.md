# NightScan

## Présentation
NightScan est un projet de classification de sons d'animaux basé sur des spectrogrammes générés à partir d'enregistrements audio. Le pipeline comprend la conversion des fichiers audio bruts en spectrogrammes de 8 secondes, l'entraînement de modèles de deep learning et l'évaluation des performances.

## Fonctionnalités principales
- Prétraitement des fichiers audio pour obtenir des spectrogrammes de durée uniforme.
- Entraînement de modèles de classification avec PyTorch.
- Évaluation des modèles sur des ensembles de validation.
- Sauvegarde et chargement des modèles entraînés.

## Structure du projet
- `README.md` — ce fichier
- `data/` (à créer) — contient les enregistrements (`raw/`) et les spectrogrammes (`processed/`)
- `models/` (à créer) — répertoire pour stocker les modèles entraînés
- `scripts/` (à créer) — scripts `preprocess.py`, `train.py` et `evaluate.py` (optionnels et pas encore fournis)
- `utils/` (à créer) — fonctions utilitaires
- `requirements.txt` (à créer) — dépendances Python

## Installation
Clonez le dépôt puis installez les dépendances dans un environnement virtuel :

```bash
git clone https://github.com/votre-utilisateur/NightScan.git
cd NightScan
python -m venv env
source env/bin/activate  # Sur Windows : env\Scripts\activate
pip install -r requirements.txt
```

## Utilisation
Prétraitez les données, entraînez un modèle et évaluez-le :

**Note :** ces scripts peuvent être absents ou inutiles pour le moment.

```bash
python scripts/preprocess.py --input_dir data/raw --output_dir data/processed
python scripts/train.py --csv_dir data/processed/csv --model_dir models/
python scripts/evaluate.py --model_path models/best_model.pth --data_dir data/processed
```

Le script `train.py` utilise désormais un ResNet‑18 préentraîné d'ImageNet et
remplace sa dernière couche pour s'adapter au nombre de classes présent dans
vos données.

Le dossier `data/raw` doit contenir un sous-répertoire par classe (par exemple `data/raw/chat/`, `data/raw/chien/`, ...). Le prétraitement conserve cette structure et génère des spectrogrammes classés dans `data/processed/spectrograms`. Les fichiers CSV produits dans `data/processed/csv` possèdent désormais deux colonnes : `path` et `label`.

Lorsque le script `preprocess.py` isole un cri mais obtient un segment silencieux (volume inférieur à -60 dBFS), le fichier résultant est supprimé et ignoré.
