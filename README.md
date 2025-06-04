# NightScan

## Présentation
NightScan est un projet de classification de sons d'animaux basé sur des spectrogrammes générés à partir d'enregistrements audio. Le pipeline comprend la conversion des fichiers audio bruts en spectrogrammes de 8 secondes, l'entraînement de modèles de deep learning et l'évaluation des performances.

## Fonctionnalités principales
- Prétraitement des fichiers audio pour obtenir des spectrogrammes de durée uniforme.
- Entraînement de modèles de classification avec PyTorch.
- Évaluation des modèles sur des ensembles de validation.
- Sauvegarde du meilleur modèle entraîné.

## Structure du projet
- `README.md` — ce fichier
- `data/` (à créer) — contient les enregistrements (`raw/`) et les spectrogrammes (`processed/`)
- `models/` (à créer) — répertoire pour stocker les modèles entraînés
- `scripts/` — scripts `preprocess.py` et `train.py` pour prétraiter les données et entraîner un modèle. Un script d'évaluation pourra être ajouté ultérieurement.
- `utils/` (à créer) — fonctions utilitaires
- `requirements.txt` — dépendances Python

## Installation
Clonez le dépôt puis installez les dépendances dans un environnement virtuel :

```bash
git clone https://github.com/votre-utilisateur/NightScan.git
cd NightScan
python -m venv env
source env/bin/activate  # Sur Windows : env\Scripts\activate
pip install -r requirements.txt
```

Le prétraitement avec `pydub` nécessite également l'outil système **ffmpeg** disponible sur la plupart des distributions Linux et sur Windows.

## Utilisation
Prétraitez les données puis entraînez un modèle :

```bash
python scripts/preprocess.py --input_dir data/raw --output_dir data/processed
python scripts/train.py --csv_dir data/processed/csv --model_dir models/
```

Le dossier `data/raw` doit contenir un sous-répertoire par classe (par exemple `data/raw/chat/`, `data/raw/chien/`, ...). Le prétraitement conserve cette structure et génère des spectrogrammes classés dans `data/processed/spectrograms`. Les fichiers CSV produits dans `data/processed/csv` possèdent désormais deux colonnes : `path` et `label`.

Lorsque le script `preprocess.py` isole un cri mais obtient un segment silencieux (volume inférieur à -60 dBFS), le fichier résultant est supprimé et ignoré.

## Fonctionnement des scripts

### `preprocess.py`

- Convertit récursivement tous les MP3 présents dans `--input_dir` en fichiers WAV placés dans `output_dir/wav`.
- Découpe chaque WAV en segments de 8 secondes grâce à la détection de silence (seuil −40 dBFS). Les segments trop courts sont complétés par du silence et ceux trop longs sont tronqués.
- Les segments dont le volume reste inférieur à −60 dBFS sont ignorés.
- Les segments valides sont enregistrés dans `output_dir/segments` en conservant la structure de dossiers des classes.
- Un mél‑spectrogramme est généré pour chaque segment dans `output_dir/spectrograms`.
- Les chemins de ces spectrogrammes et leurs étiquettes sont sauvegardés dans `train.csv`, `val.csv` et `test.csv` sous `output_dir/csv` selon un partage 70 % / 15 % / 15 %.

### `train.py`

- Lit `train.csv` et `val.csv` depuis le répertoire indiqué par `--csv_dir`.
- Charge chaque spectrogramme `.npy`, le normalise entre 0 et 1 puis le redimensionne en 224×224 et le duplique sur trois canaux pour l'utiliser avec ResNet18.
- Entraîne un réseau ResNet18 (poids initiaux aléatoires) sur GPU si disponible.
- Après chaque époque, affiche la perte et la précision de validation et sauvegarde dans `--model_dir/best_model.pth` le modèle obtenant la meilleure précision.
- Paramètres optionnels : `--epochs` (10 par défaut), `--batch_size` (32) et `--lr` (1e-3).
