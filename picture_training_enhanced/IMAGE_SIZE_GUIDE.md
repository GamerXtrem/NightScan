# 📏 Guide des Tailles d'Images pour l'Entraînement

## ✅ Réponse Rapide

**Les images JPG de tailles non standardisées ne posent PAS de problème.** Le pipeline gère automatiquement toutes les tailles d'images et les standardise à 224x224 pixels pour EfficientNet.

## 🎯 Tailles Recommandées

### Optimal
- **Taille idéale**: 500-1500 pixels (largeur ou hauteur)
- **Format**: JPEG de bonne qualité (>85%)
- **Ratio d'aspect**: Entre 1:2 et 2:1

### Minimum Acceptable
- **Taille minimale**: 300x300 pixels
- **En dessous de 224px**: Sera agrandi (perte de qualité possible)

### Maximum Raisonnable
- **Taille maximale**: 3000x3000 pixels
- **Au-delà**: Consomme plus de mémoire sans gain de qualité

## 📊 Gestion Automatique des Tailles

### Comment le Pipeline Traite les Images

| Étape | Transformation | Exemple |
|-------|---------------|---------|
| 1. Chargement | Image originale | 1920x1080 (HD) |
| 2. Resize | Redimensionnement à ~256px | 455x256 |
| 3. Crop | Extraction du centre | 224x224 |
| 4. Tensor | Conversion pour le modèle | 3x224x224 |

### Transformations Appliquées

#### Entraînement
```python
RandomResizedCrop(224)  # Crop aléatoire + resize
# OU
Resize(256) → CenterCrop(224)  # Mode "light"
```

#### Validation/Test
```python
Resize(256) → CenterCrop(224)  # Toujours le même crop
```

## ⚠️ Cas Particuliers

### Images Très Petites (<224px)
- **Problème**: Agrandissement forcé → pixellisation
- **Solution**: Utiliser `preprocess_images.py` pour filtrer
- **Recommandation**: Minimum 300px

### Images Très Grandes (>3000px)
- **Problème**: Lenteur de chargement, consommation mémoire
- **Solution**: Prétraitement pour réduire à ~1500px
- **Commande**:
```bash
python preprocess_images.py \
  --input_dir images_originales/ \
  --output_dir images_optimisees/ \
  --max_size 1500
```

### Ratios d'Aspect Extrêmes
- **Panoramas** (ratio > 3:1): Perte d'information sur les côtés
- **Portraits étroits** (ratio < 1:3): Perte en haut/bas
- **Solution**: Padding ou crop intelligent avec `--preserve_aspect`

## 🛠️ Outils de Préparation

### 1. Script de Prétraitement
```bash
# Optimiser toutes les images
python preprocess_images.py \
  --input_dir data/raw/ \
  --output_dir data/processed/ \
  --max_size 1500 \
  --min_size 300 \
  --quality 95 \
  --check_blur
```

### 2. Analyse des Dimensions
```bash
# Préparer et analyser
python data_preparation.py \
  --input_dir data/images/ \
  --output_dir data/ready/
```
Affiche automatiquement :
- Distribution des tailles
- Ratios d'aspect
- Recommandations spécifiques

### 3. Vérification Rapide
```python
from PIL import Image
import os

def check_images(folder):
    sizes = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                img = Image.open(os.path.join(root, file))
                sizes.append(img.size)
    
    widths = [s[0] for s in sizes]
    heights = [s[1] for s in sizes]
    
    print(f"Largeur: {min(widths)}-{max(widths)}px")
    print(f"Hauteur: {min(heights)}-{max(heights)}px")
    print(f"Total: {len(sizes)} images")
```

## 📈 Impact sur les Performances

### Taille vs Qualité du Modèle

| Taille Image | Temps/Image | Qualité | Recommandation |
|-------------|-------------|---------|----------------|
| < 224px | Rapide | ⚠️ Dégradée | Éviter |
| 224-500px | Rapide | ✅ Bonne | Acceptable |
| 500-1500px | Normal | ✅ Optimale | **Recommandé** |
| 1500-3000px | Lent | ✅ Optimale | Acceptable |
| > 3000px | Très lent | = Pas mieux | Prétraiter |

### Consommation Mémoire

```
Batch Size 32:
- Images 500px: ~4 GB VRAM
- Images 1500px: ~8 GB VRAM  
- Images 3000px: ~16 GB VRAM
```

## 🚀 Workflow Recommandé

### 1. Analyse Initiale
```bash
# Voir la distribution des tailles
python data_preparation.py --input_dir data/raw --output_dir data/temp
```

### 2. Prétraitement (si nécessaire)
```bash
# Si beaucoup d'images > 2000px
python preprocess_images.py \
  --input_dir data/raw \
  --output_dir data/optimized \
  --max_size 1500
```

### 3. Préparation Finale
```bash
# Organiser en train/val/test
python data_preparation.py \
  --input_dir data/optimized \
  --output_dir data/ready
```

### 4. Entraînement
```bash
# Les images seront automatiquement redimensionnées à 224x224
python train_real_images.py \
  --data_dir data/ready \
  --config config.yaml
```

## 💡 Conseils Pratiques

### DO ✅
- Garder les images originales en backup
- Viser 500-1500px pour l'optimal
- Utiliser JPEG avec qualité 90-95%
- Vérifier les images floues avant l'entraînement

### DON'T ❌
- Ne pas utiliser d'images < 200px
- Ne pas mélanger des tailles trop variées (100px avec 4000px)
- Ne pas sur-compresser (JPEG qualité < 80%)
- Ne pas oublier de vérifier les corruptions

## 📊 Exemple de Rapport

Après `preprocess_images.py`:
```
📊 RÉSUMÉ DU PRÉTRAITEMENT
Total d'images: 5000
Images traitées: 4850
Images trop petites ignorées: 50
Images floues ignorées: 100

📐 Distribution finale:
  500-800px: 2000 images
  800-1200px: 2500 images
  1200-1500px: 350 images

💾 Compression:
  Taille originale: 2500 MB
  Après traitement: 850 MB
  Réduction: 66%
```

## 🔍 Diagnostic Rapide

Si vous avez des doutes sur vos images :

```bash
# 1. Analyser sans modifier
python data_preparation.py \
  --input_dir vos_images/ \
  --output_dir test/ \
  --train_ratio 0.8 \
  --val_ratio 0.1 \
  --test_ratio 0.1

# 2. Regarder le rapport
# Si recommandations → prétraiter
# Sinon → utiliser directement
```

## ✅ Conclusion

- **Les tailles variables sont gérées automatiquement**
- **Optimal**: 500-1500px
- **Minimum**: 300px
- **Prétraitement optionnel** mais recommandé pour > 2000px
- **Le modèle travaille toujours en 224x224** peu importe la taille d'origine