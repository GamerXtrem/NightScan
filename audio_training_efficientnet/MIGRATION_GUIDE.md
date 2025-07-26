# Guide de Migration vers le Nouveau Workflow Équilibré

## 🧹 Nettoyage des anciennes données

Avant de commencer avec le nouveau workflow, il est recommandé de nettoyer les anciennes données :

```bash
cd audio_training_efficientnet
./cleanup_old_data.sh
```

Cela supprimera :
- Les anciennes bases SQLite (`audio_index_balanced.db`, `audio_index.db`, etc.)
- Les anciens répertoires de spectrogrammes (`data/spectrograms_cache/`)
- Les checkpoints récents potentiellement corrompus

**Important** : Les données audio originales dans `~/NightScan/data/audio_data/` seront conservées.

## 📋 Différences principales

### Ancien workflow (problématique)
- Équilibrage fait À CHAQUE split → 1500 échantillons/classe au total
- Logique d'équilibrage complexe dans le dataset
- Augmentation à la volée pendant l'entraînement
- Utilisation intensive de la mémoire

### Nouveau workflow (corrigé)
- Équilibrage fait UNE SEULE FOIS → 500 échantillons/classe au total
- Dataset simple qui lit juste les fichiers
- Augmentation pré-calculée dans le pool
- Gestion optimisée de la mémoire

## 🚀 Nouveau workflow complet

### 1. Nettoyer les anciennes données
```bash
./cleanup_old_data.sh
```

### 2. Créer le pool augmenté (environ 30 minutes)
```bash
python create_augmented_pool.py \
    --audio-root ~/NightScan/data/audio_data \
    --output-dir ~/NightScan/data/augmented_pool \
    --batch-size 5  # Ajuster selon la RAM disponible
```

### 3. Créer l'index équilibré (instantané)
```bash
python create_balanced_index.py \
    --pool-dir ~/NightScan/data/augmented_pool \
    --output-db balanced_audio_index.db
```

### 4. Pré-générer les spectrogrammes (environ 10 minutes)
```bash
python pregenerate_spectrograms_simple.py \
    --index-db balanced_audio_index.db \
    --pool-root ~/NightScan/data/augmented_pool \
    --output-dir ~/NightScan/data/spectrograms_balanced \
    --num-workers 8
```

### 5. Vérifier que tout fonctionne
```bash
python test_balanced_pipeline.py \
    --index-db balanced_audio_index.db \
    --audio-root ~/NightScan/data/augmented_pool \
    --spectrogram-cache-dir ~/NightScan/data/spectrograms_balanced \
    --test-loading
```

### 6. Lancer l'entraînement
```bash
python train_audio_large_scale.py \
    --index-db balanced_audio_index.db \
    --audio-root ~/NightScan/data/augmented_pool \
    --spectrogram-cache-dir ~/NightScan/data/spectrograms_balanced \
    --num-classes 10 \
    --model efficientnet-b1 \
    --batch-size 128 \
    --epochs 50
```

## 🔧 Paramètres importants

- **`--batch-size` dans create_augmented_pool.py** : Réduire si le script est tué (OOM)
- **`--num-workers`** : Ajuster selon le nombre de CPU disponibles
- **Plus besoin de** : `--no-balance-classes` ou `--max-samples-per-class`

## 📊 Résultats attendus

Après le nouveau workflow, vous devriez avoir :
- **Pool augmenté** : ~5000 fichiers audio (500 par classe)
- **Spectrogrammes** : ~5000 fichiers .npy organisés en train/val/test
- **Distribution** : 80% train (400/classe), 10% val (50/classe), 10% test (50/classe)

## ⚠️ Problèmes courants

1. **"Killed" pendant create_augmented_pool.py**
   - Solution : Réduire `--batch-size` (essayer 3 ou même 1)

2. **Erreurs de chargement de fichiers audio**
   - Vérifier que le chemin `--audio-root` est correct
   - S'assurer que les fichiers audio sont valides

3. **Performance lente**
   - Augmenter `--num-workers` pour la génération de spectrogrammes
   - S'assurer d'utiliser le cache de spectrogrammes pendant l'entraînement