# Nouveau Workflow d'Équilibrage pour NightScan

## Vue d'ensemble

Ce nouveau workflow résout le problème d'augmentation excessive en appliquant l'équilibrage AVANT la division train/val/test.

### Ancien problème
- Division d'abord : 80% train, 10% val, 10% test
- Puis augmentation de chaque split jusqu'à 500
- Résultat : 1500 échantillons au lieu de 500 par classe !

### Nouvelle solution
- Augmentation d'abord jusqu'à 500 échantillons par classe
- Puis division : 80% train (400), 10% val (50), 10% test (50)
- Résultat : exactement 500 échantillons par classe au total

## Workflow en 3 étapes

### Étape 1 : Créer le pool augmenté

```bash
python create_augmented_pool.py \
    --audio-root ~/NightScan/data/audio_data \
    --output-dir ~/NightScan/data/augmented_pool \
    --target-samples 500
```

Ce script :
- Analyse chaque classe et calcule les augmentations nécessaires
- Crée des fichiers audio augmentés (time stretch, noise, volume)
- Génère un pool équilibré de 500 échantillons max par classe
- Sauvegarde les métadonnées dans `pool_metadata.json`

Exemple de calcul :
- 40 échantillons → 12 augmentations/échantillon = 520 total
- 100 échantillons → 4 augmentations/échantillon = 500 total
- 500+ échantillons → 0 augmentation

### Étape 2 : Créer l'index équilibré

```bash
python create_balanced_index.py \
    --pool-dir ~/NightScan/data/augmented_pool \
    --output-db balanced_audio_index.db \
    --train-split 0.8 \
    --val-split 0.1 \
    --test-split 0.1
```

Ce script :
- Lit le pool augmenté
- Groupe les fichiers par original (pour éviter la fuite de données)
- Divise chaque classe en 80/10/10
- Crée une base SQLite avec les assignations de splits
- Garantit un bon mélange originaux/augmentés dans chaque split

### Étape 3 : Pré-générer les spectrogrammes

```bash
python pregenerate_spectrograms_simple.py \
    --index-db balanced_audio_index.db \
    --pool-root ~/NightScan/data/augmented_pool \
    --output-dir ~/NightScan/data/spectrograms_balanced \
    --num-workers 8
```

Ce script :
- Lit l'index équilibré
- Génère UNIQUEMENT les spectrogrammes (pas d'augmentation)
- Crée ~5000 fichiers .npy (500 × 10 classes)
- Organise par split/classe

## Structure des données

```
data/
├── audio_data/                  # Fichiers originaux
│   ├── asio_flammeus/          # 39 fichiers
│   ├── airplane/               # 40 fichiers
│   └── ...
│
├── augmented_pool/             # Pool équilibré (500/classe)
│   ├── pool_metadata.json      # Métadonnées du pool
│   ├── asio_flammeus/          # ~500 fichiers
│   │   ├── file1_original.wav
│   │   ├── file1_aug001_time_stretch.wav
│   │   └── ...
│   └── ...
│
├── balanced_audio_index.db     # Index SQLite avec splits
│
└── spectrograms_balanced/      # Spectrogrammes finaux
    ├── train/                  # ~4000 fichiers (400×10)
    │   ├── asio_flammeus/      # ~400 .npy
    │   └── ...
    ├── val/                    # ~500 fichiers (50×10)
    └── test/                   # ~500 fichiers (50×10)
```

## Entraînement

Pour l'entraînement, utilisez le nouvel index SANS équilibrage :

```bash
python train_audio_large_scale.py \
    --index-db balanced_audio_index.db \
    --audio-root ~/NightScan/data/augmented_pool \
    --spectrogram-cache-dir ~/NightScan/data/spectrograms_balanced \
    --num-classes 10 \
    --model efficientnet-b1 \
    --batch-size 128 \
    --no-balance-classes \  # IMPORTANT: Pas d'équilibrage!
    --epochs 50
```

## Avantages

1. **Logique correcte** : 500 échantillons par classe au total, pas 1500
2. **Pas de fuite de données** : Un original et ses augmentations restent dans le même split
3. **Performance** : Validation aussi rapide que l'entraînement
4. **Simplicité** : Plus besoin d'équilibrage complexe pendant l'entraînement
5. **Reproductible** : Les mêmes augmentations pour tous les runs

## Notes importantes

- Les augmentations PitchShift ont été désactivées (trop lentes)
- Le pool utilise : time stretch, noise, volume, et combinaisons
- L'index garantit qu'un fichier original et ses augmentations sont dans le même split
- Tous les splits ont le même ratio originaux/augmentés

## Commande complète pour le serveur

```bash
# 1. Créer le pool augmenté (~30 min)
python create_augmented_pool.py \
    --audio-root ~/NightScan/data/audio_data \
    --output-dir ~/NightScan/data/augmented_pool

# 2. Créer l'index (instantané)
python create_balanced_index.py \
    --pool-dir ~/NightScan/data/augmented_pool \
    --output-db balanced_audio_index.db

# 3. Générer les spectrogrammes (~10 min)
python pregenerate_spectrograms_simple.py \
    --index-db balanced_audio_index.db \
    --pool-root ~/NightScan/data/augmented_pool \
    --output-dir ~/NightScan/data/spectrograms_balanced \
    --num-workers 8

# 4. Lancer l'entraînement
python train_audio_large_scale.py \
    --index-db balanced_audio_index.db \
    --audio-root ~/NightScan/data/augmented_pool \
    --spectrogram-cache-dir ~/NightScan/data/spectrograms_balanced \
    --num-classes 10 \
    --model efficientnet-b1 \
    --batch-size 128 \
    --no-balance-classes \
    --epochs 50
```