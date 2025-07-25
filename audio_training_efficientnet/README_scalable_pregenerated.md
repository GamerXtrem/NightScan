# Système Scalable avec Spectrogrammes Pré-générés

## Vue d'ensemble

Ce système permet de gérer efficacement l'entraînement sur de très grandes bases de données audio (1500+ classes) en pré-générant tous les spectrogrammes. Cela évite les problèmes de mémoire liés aux augmentations audio en temps réel.

## Workflow

### 1. Créer l'index SQLite

```bash
python train_audio_large_scale.py \
    --create-index \
    --audio-root ~/NightScan/data/audio_data \
    --index-db audio_index_large.db
```

### 2. Pré-générer les spectrogrammes

```bash
python pregenerate_spectrograms_scalable.py \
    --index-db audio_index_large.db \
    --audio-root ~/NightScan/data/audio_data \
    --output-dir ~/NightScan/data/spectrograms_cache \
    --num-workers 8
```

Cette étape génère :
- Les spectrogrammes originaux pour tous les splits (train/val/test)
- 10 variantes augmentées par échantillon pour le split train

### 3. Lancer l'entraînement avec les spectrogrammes pré-générés

```bash
python train_audio_large_scale.py \
    --index-db audio_index_large.db \
    --audio-root ~/NightScan/data/audio_data \
    --spectrogram-cache-dir ~/NightScan/data/spectrograms_cache \
    --num-classes 78 \
    --batch-size 32 \
    --num-workers 4 \
    --epochs 100
```

## Structure du cache

```
spectrograms_cache/
├── train/
│   ├── asio_flammeus/
│   │   ├── file_001.npy              # Spectrogramme original
│   │   ├── file_001_var001.npy       # Variante augmentée 1
│   │   ├── file_001_var002.npy       # Variante augmentée 2
│   │   └── ...
│   └── autre_classe/
├── val/
│   └── ... (seulement les originaux)
└── test/
    └── ... (seulement les originaux)
```

## Avantages

1. **Pas de problèmes de mémoire** : Les augmentations audio gourmandes (PitchShift, etc.) sont faites une seule fois
2. **Entraînement plus rapide** : Chargement direct des .npy au lieu de générer les spectrogrammes
3. **Reproductibilité** : Les mêmes augmentations à chaque epoch
4. **Scalabilité** : Peut gérer des millions d'échantillons

## Options importantes

- `--num-workers` : Nombre de processus parallèles pour la génération (8-16 recommandé)
- `--max-samples-per-class` : Limite le nombre d'échantillons par classe (défaut: 500)
- `--no-balance-classes` : Désactive l'équilibrage automatique des classes

## Monitoring

La pré-génération affiche :
- Progression en temps réel
- Statistiques par split
- Taille totale du cache

L'entraînement affiche :
- Utilisation du cache de spectrogrammes
- Performances de chargement
- Statistiques de validation