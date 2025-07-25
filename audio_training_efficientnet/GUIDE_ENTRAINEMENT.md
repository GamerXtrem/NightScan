# Guide d'entraînement du modèle audio NightScan

Ce guide explique comment entraîner un modèle de classification audio avec des classes dynamiques basées sur la structure des dossiers.

## Structure des données requise

Organisez vos fichiers audio dans des dossiers par classe :

```
audio_data/
├── chant_oiseau/
│   ├── audio1.wav
│   ├── audio2.wav
│   └── ...
├── appel_mammifere/
│   ├── audio1.wav
│   └── ...
├── cri_chouette/
│   └── audio1.wav
└── nouvelle_espece/
    └── audio1.wav
```

**Important** : Le nom du dossier devient le nom de la classe.

## Étapes d'entraînement

### 1. Préparer les données

```bash
cd audio_training_efficientnet

# Scanner les dossiers et créer les fichiers CSV
python prepare_audio_data.py /chemin/vers/audio_data \
    --output-dir data/processed/csv \
    --train-ratio 0.7 \
    --val-ratio 0.15 \
    --test-ratio 0.15 \
    --min-samples 10
```

Options importantes :
- `--min-samples` : Nombre minimum d'échantillons par classe (défaut: 10)
- `--relative-paths` : Utiliser des chemins relatifs dans les CSV

### 2. Entraîner le modèle

```bash
# Entraînement avec les vraies données
python train_audio.py \
    --data-dir /chemin/vers/audio_data \
    --csv-dir data/processed/csv \
    --epochs 50 \
    --batch-size 32 \
    --learning-rate 0.001
```

Options utiles :
- `--spectrogram-dir` : Répertoire pour sauvegarder les spectrogrammes (accélère les entraînements suivants)
- `--model-name` : Modèle à utiliser (efficientnet-b0, efficientnet-b1, etc.)
- `--use-mock-data` : Pour tester avec des données simulées

### 3. Fichiers de sortie

Après l'entraînement, vous trouverez dans `audio_training_efficientnet/models/` :
- `best_model.pth` : Le modèle entraîné avec les classes détectées
- `metadata.json` : Informations sur le modèle et les classes
- `training_history.json` : Historique de l'entraînement

## Utilisation du modèle

Le modèle entraîné est automatiquement compatible avec le système de prédiction unifié. Les classes sont sauvegardées dans le checkpoint et seront chargées automatiquement lors de l'utilisation.

## Ajouter de nouvelles espèces

1. Créez un nouveau dossier avec le nom de l'espèce
2. Ajoutez au moins 10 fichiers WAV
3. Relancez `prepare_audio_data.py`
4. Relancez l'entraînement

## Conseils

- Utilisez au moins 50-100 échantillons par classe pour de bons résultats
- Les fichiers WAV doivent idéalement avoir la même durée (8 secondes)
- Les noms de dossiers doivent être descriptifs et sans espaces (utilisez _ à la place)
- Surveillez la validation accuracy pendant l'entraînement

## Dépannage

**Erreur "FileNotFoundError: classes.json"**
→ Exécutez d'abord `prepare_audio_data.py`

**Mémoire insuffisante**
→ Réduisez le batch size avec `--batch-size 16` ou moins

**Entraînement trop lent**
→ Utilisez `--spectrogram-dir` pour mettre en cache les spectrogrammes