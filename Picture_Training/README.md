# NightScan - Image Classification

## Présentation
NightScan est un projet de classification d'animaux à partir de photos. Les images sont listées dans des fichiers CSV puis utilisées pour entraîner un modèle ResNet18 avec PyTorch.

## Fonctionnalités principales
- Chargement structuré des images classées par dossier.
- Normalisation et redimensionnement unifié des images.
- Entraînement d'un modèle CNN (ResNet18) avec option `--pretrained`.
- Évaluation sur jeu de validation et sauvegarde du meilleur modèle.

## Structure du projet
- `README.md` — ce fichier
- `data/` — contient les images brutes (`raw/`) et les CSV (`csv/`)
- `models/` — répertoire pour stocker les modèles entraînés
- `scripts/` — `prepare_csv.py`, `train.py` et `predict.py`
- `utils/` — fonctions utilitaires
- `Picture_Training/setup.sh` — script pour créer les dossiers nécessaires
- `../requirements.txt` — dépendances Python communes (à la racine)

## Installation
Clonez le dépôt puis installez les dépendances dans un environnement virtuel :

```bash
git clone https://github.com/votre-utilisateur/NightScan.git
cd NightScan
python -m venv env
source env/bin/activate  # Sur Windows : env\Scripts\activate
pip install -r requirements.txt
chmod +x Picture_Training/setup.sh
./Picture_Training/setup.sh
```

## Utilisation
Préparez les fichiers CSV à partir des images puis lancez l'entraînement :

```bash
python scripts/prepare_csv.py --input_dir data/raw --output_dir data/csv
python scripts/train.py --csv_dir data/csv --model_dir models/ --pretrained
```

Le dossier `data/raw` doit contenir un sous-dossier par espèce (ex. `data/raw/bubo_bubo/`, `data/raw/capreolus_capreolus/`, ...). Les chemins d'accès aux images sont stockés dans `train.csv`, `val.csv` et `test.csv` selon une répartition 70/15/15 par défaut. Si vous changez ces fractions, chaque valeur doit rester comprise entre 0 et 1 et `train + val` doit être strictement inférieur à 1, sinon le programme refusera de générer les CSV.

### Fonctionnement des scripts

- **`prepare_csv.py`**
  - Scanne tous les dossiers de `--input_dir`.
  - Crée trois fichiers CSV (`train.csv`, `val.csv`, `test.csv`) avec les colonnes `path` et `label`.
  - Répartition des données configurable (par défaut : 70/15/15). Les valeurs fournies doivent être comprises entre 0 et 1 et la somme `train + val` doit rester inférieure à 1.

- **`train.py`**
  - Lit `train.csv` et `val.csv` depuis le dossier indiqué par `--csv_dir`.
  - Charge les images, les redimensionne en 224×224 et les normalise selon ImageNet.
  - Entraîne un ResNet18 (optionnellement pré-entraîné).
  - Sauvegarde automatique du meilleur modèle dans `--model_dir/best_model.pth`.
  - Paramètres : `--epochs`, `--batch_size`, `--lr`, `--num_workers`.

- **`predict.py`**
  - Charge un modèle entraîné et prédit l'espèce d'une ou plusieurs images.
  - Applique les mêmes transformations que lors de l'entraînement.
  - Affiche le top 3 des classes prédites avec leur score de confiance.
  - Nécessite l'argument `--csv_dir` pour charger les noms des classes à partir de `train.csv`.

```bash
python scripts/predict.py --model_path models/best_model.pth --csv_dir data/csv image.jpg
```
