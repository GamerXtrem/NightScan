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
- `scripts/` — scripts `preprocess.py`, `train.py` et `predict.py` pour prétraiter les données, entraîner un modèle et effectuer des prédictions.
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

Avant d'exécuter `preprocess.py`, installez également :

```bash
brew install portaudio   # macOS uniquement
pip install pyaudio audioop-lts
```

Le prétraitement avec `pydub` nécessite également l'outil système **ffmpeg** disponible sur la plupart des distributions Linux et sur Windows.
Les spectrogrammes sont désormais calculés avec **torchaudio**.

## Utilisation
Prétraitez les données puis entraînez un modèle :

```bash
python scripts/preprocess.py --input_dir data/raw --output_dir data/processed --workers 4
python scripts/train.py --csv_dir data/processed/csv --model_dir models/ --pretrained
```

Le dossier `data/raw` doit contenir un sous-répertoire par classe (par exemple `data/raw/chat/`, `data/raw/chien/`, ...). Le prétraitement conserve cette structure et génère des spectrogrammes classés dans `data/processed/spectrograms`. Les fichiers CSV produits dans `data/processed/csv` possèdent désormais deux colonnes : `path` et `label`.

Lorsque le script `preprocess.py` isole un cri mais obtient un segment silencieux (volume inférieur à `CHUNK_SILENCE_THRESH`), le fichier résultant est supprimé et ignoré.

## Fonctionnement des scripts

-### `preprocess.py`

- Convertit les MP3 en WAV et recopie directement les fichiers déjà au format WAV dans `output_dir/wav`.
- Découpe chaque WAV en segments de 8 secondes grâce à la détection de silence (`SPLIT_SILENCE_THRESH` à −40 dBFS par défaut). Les segments trop courts sont complétés par du silence et ceux trop longs sont tronqués.
- Les segments dont le volume moyen est inférieur à `CHUNK_SILENCE_THRESH` (−20 dBFS par défaut) sont ignorés.
- Les segments valides sont enregistrés dans `output_dir/segments` en conservant la structure de dossiers des classes.
- Un mél‑spectrogramme est généré pour chaque segment dans `output_dir/spectrograms` à l'aide de **torchaudio**.
- Les chemins de ces spectrogrammes et leurs étiquettes sont sauvegardés dans `train.csv`, `val.csv` et `test.csv` sous `output_dir/csv` selon un partage 70 % / 15 % / 15 %.
- Optionnel : `--workers` permet de paralléliser les conversions et traitements.

### `train.py`

- Lit `train.csv` et `val.csv` depuis le répertoire indiqué par `--csv_dir`.
- Charge chaque spectrogramme `.npy`, le normalise entre 0 et 1 puis le redimensionne en 224×224 et le duplique sur trois canaux pour l'utiliser avec ResNet18.
- Entraîne un réseau ResNet18 sur GPU ou MPS (Apple Silicon) si disponible. Les poids ImageNet peuvent être chargés avec `--pretrained`.
- Après chaque époque, affiche la perte et la précision de validation et sauvegarde dans `--model_dir/best_model.pth` le modèle obtenant la meilleure précision.
- Paramètres optionnels : `--epochs` (10 par défaut), `--batch_size` (32), `--lr` (1e-3) et `--num_workers` (0).

### `predict.py`

- Charge un modèle entraîné et renvoie pour chaque fichier audio les trois classes les plus probables.
  Le fichier est d'abord découpé en segments de 8 s grâce à la même détection de
  silence que dans `preprocess.py`. Les segments quasiment silencieux sont
  ignorés.
- Exemple d'utilisation :

```bash
python scripts/predict.py --model_path models/best_model.pth --csv_dir data/processed/csv sample.wav
```
