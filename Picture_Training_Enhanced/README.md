# Picture Training Enhanced - Advanced Wildlife Image Classification

Un système d'entraînement avancé pour la classification d'images de faune nocturne, conçu pour égaler la sophistication du système audio EfficientNet de NightScan.

## 🌟 Fonctionnalités

### Architecture Multi-Modèles
- **EfficientNet** (B0-B7) : Architecture optimisée pour l'efficacité
- **ResNet** (18/34/50/101) : Architecture éprouvée et robuste
- **Vision Transformer** (ViT-B/16, ViT-B/32) : Architecture transformer moderne

### Augmentation de Données Avancée
- **Augmentations géométriques** : Rotation, mise à l'échelle, retournement
- **Augmentations photométriques** : Luminosité, contraste, saturation
- **Techniques avancées** : CutMix, MixUp, AutoAugment
- **Spécialisations nocturnes** : Simulation vision nocturne et infrarouge

### Entraînement Avancé
- **Précision mixte** : Entraînement optimisé pour la mémoire
- **Planification de taux d'apprentissage** : Cosine, step, plateau
- **Arrêt précoce** : Prévention du surapprentissage
- **Gradient clipping** : Stabilité d'entraînement

### Métriques Complètes
- **Matrices de confusion** : Visualisation des performances par classe
- **Courbes ROC/PR** : Analyse approfondie des performances
- **Métriques par classe** : Précision, rappel, F1-score
- **Comparaison de modèles** : Évaluation multi-modèles

## 🚀 Installation

### Prérequis
```bash
# Python 3.8+
pip install torch torchvision torchaudio
pip install efficientnet-pytorch
pip install scikit-learn pandas numpy matplotlib seaborn
pip install Pillow opencv-python tqdm
```

### Structure des Dossiers
```
Picture_Training_Enhanced/
├── models/
│   ├── photo_config.py          # Configuration des modèles
│   └── data_augmentation.py     # Augmentation de données
├── utils/
│   ├── training_utils.py        # Utilitaires d'entraînement
│   └── metrics.py               # Métriques et visualisations
├── scripts/
│   ├── train_enhanced.py        # Script d'entraînement principal
│   ├── preprocess_enhanced.py   # Préprocessing des données
│   ├── evaluate_enhanced.py     # Évaluation des modèles
│   └── predict_enhanced.py      # Prédiction sur nouvelles images
├── configs/
│   ├── efficientnet_presets.json
│   ├── resnet_presets.json
│   ├── vit_presets.json
│   ├── augmentation_configs.json
│   └── class_mappings.json
└── README.md
```

## 📊 Utilisation

### 1. Préparation des Données

```bash
# Préprocessing et validation des données
python scripts/preprocess_enhanced.py \\
    --input_csv /path/to/dataset.csv \\
    --output_dir /path/to/processed \\
    --validate_images \\
    --create_splits \\
    --val_size 0.2 \\
    --test_size 0.1
```

### 2. Entraînement d'un Modèle

```bash
# Entraînement avec EfficientNet-B1
python scripts/train_enhanced.py \\
    --config efficientnet_b1_balanced \\
    --csv_dir /path/to/processed/splits \\
    --model_dir /path/to/model_output \\
    --epochs 100 \\
    --batch_size 32

# Entraînement avec ResNet-50
python scripts/train_enhanced.py \\
    --config resnet50_quality \\
    --csv_dir /path/to/processed/splits \\
    --model_dir /path/to/model_output
```

### 3. Évaluation des Modèles

```bash
# Évaluation d'un modèle unique
python scripts/evaluate_enhanced.py \\
    --models /path/to/model.pth \\
    --configs efficientnet_b1_balanced \\
    --test_csv /path/to/test.csv \\
    --output_dir /path/to/evaluation

# Comparaison de plusieurs modèles
python scripts/evaluate_enhanced.py \\
    --models model1.pth model2.pth model3.pth \\
    --model_names "EfficientNet-B1" "ResNet-50" "ViT-B/16" \\
    --configs efficientnet_b1_balanced resnet50_quality vit_b_16_balanced \\
    --test_csv /path/to/test.csv \\
    --output_dir /path/to/comparison
```

### 4. Prédiction sur Nouvelles Images

```bash
# Prédiction sur un dossier d'images
python scripts/predict_enhanced.py \\
    --model /path/to/best_model.pth \\
    --config efficientnet_b1_balanced \\
    --class_names bat owl raccoon opossum deer fox coyote unknown \\
    --input /path/to/images/ \\
    --output_dir /path/to/predictions \\
    --visualize \\
    --confidence_threshold 0.5
```

## ⚙️ Configurations Prédéfinies

### EfficientNet
- `efficientnet_b0_fast` : Entraînement rapide (50 epochs)
- `efficientnet_b1_balanced` : Équilibre performance/temps (100 epochs)
- `efficientnet_b2_quality` : Haute qualité (150 epochs)
- `efficientnet_b3_best` : Performance maximale (200 epochs)

### ResNet
- `resnet18_fast` : Léger et rapide
- `resnet50_quality` : Compromis qualité/performance
- `resnet101_best` : Performance maximale

### Vision Transformer
- `vit_b_16_balanced` : Équilibré avec fine-tuning
- `vit_b_16_quality` : Haute qualité
- `vit_b_32_fast` : Entraînement rapide

### Augmentation
- `minimal_augmentation` : Augmentations minimales
- `standard_augmentation` : Augmentations standard
- `aggressive_augmentation` : Augmentations intensives
- `nocturnal_specialized` : Spécialisé pour la faune nocturne

## 📈 Métriques et Visualisations

Le système génère automatiquement :
- **Matrices de confusion** normalisées et non-normalisées
- **Courbes ROC** multi-classes avec AUC
- **Courbes Précision-Rappel** avec Average Precision
- **Métriques par classe** (précision, rappel, F1-score)
- **Historique d'entraînement** (loss, accuracy, learning rate)
- **Comparaisons de modèles** (performance vs taille, temps d'inférence)

## 🔧 Personnalisation

### Nouvelle Architecture
```python
# Dans models/photo_config.py
def _create_custom_model(self, config):
    # Implémentation de votre architecture
    pass
```

### Augmentation Personnalisée
```python
# Dans models/data_augmentation.py
class CustomAugmentation:
    def __call__(self, image):
        # Votre augmentation personnalisée
        return processed_image
```

### Métriques Personnalisées
```python
# Dans utils/metrics.py
def custom_metric(y_true, y_pred):
    # Votre métrique personnalisée
    return metric_value
```

## 🏆 Avantages par Rapport au Système Basique

### Performance
- **Architectures modernes** : EfficientNet, ViT vs ResNet-18 seulement
- **Augmentation avancée** : CutMix, MixUp vs augmentations basiques
- **Précision mixte** : Entraînement 2x plus rapide avec moins de mémoire

### Robustesse
- **Validation automatique** : Détection d'images corrompues
- **Arrêt précoce** : Prévention du surapprentissage
- **Checkpointing** : Récupération après interruption

### Analyse
- **Métriques complètes** : 20+ métriques vs accuracy/loss seulement
- **Visualisations** : Graphiques automatiques pour l'analyse
- **Comparaison** : Évaluation multi-modèles

### Productivité
- **Configurations prêtes** : 15+ presets testés
- **Scripts complets** : Preprocessing, entraînement, évaluation
- **Documentation** : Guide détaillé d'utilisation

## 🐛 Dépannage

### Problèmes Courants
1. **Mémoire insuffisante** : Réduire `batch_size` ou utiliser `mixed_precision`
2. **Convergence lente** : Ajuster `learning_rate` ou utiliser `warmup`
3. **Surapprentissage** : Augmenter `dropout_rate` ou utiliser plus d'augmentation

### Logs et Monitoring
```bash
# Vérifier les logs d'entraînement
tail -f training.log

# Surveiller l'utilisation GPU
nvidia-smi -l 1
```

## 📚 Exemples d'Usage

### Entraînement pour Production
```bash
# Configuration optimale pour la production
python scripts/train_enhanced.py \\
    --config efficientnet_b2_quality \\
    --csv_dir data/splits \\
    --model_dir models/production \\
    --epochs 150 \\
    --mixed_precision
```

### Expérimentation Rapide
```bash
# Test rapide d'idées
python scripts/train_enhanced.py \\
    --config efficientnet_b0_fast \\
    --csv_dir data/splits \\
    --model_dir models/experiment \\
    --epochs 20
```

### Fine-tuning
```bash
# Fine-tuning d'un modèle existant
python scripts/train_enhanced.py \\
    --config efficientnet_b1_finetune \\
    --csv_dir data/splits \\
    --model_dir models/finetuned \\
    --resume models/base/best_model.pth
```

## 🤝 Contribution

Pour contribuer au système :
1. Créer une branche pour votre fonctionnalité
2. Ajouter des tests pour votre code
3. Documenter les nouvelles fonctionnalités
4. Soumettre une pull request

## 📄 Licence

Ce projet fait partie du système NightScan et suit la même licence.

## 🔗 Liens Utiles

- [Documentation EfficientNet](https://github.com/lukemelas/EfficientNet-PyTorch)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [Vision Transformer](https://pytorch.org/vision/stable/models.html#vision-transformer)
- [Augmentation Techniques](https://pytorch.org/vision/stable/transforms.html)

---

**Développé pour NightScan** - Système de surveillance de la faune nocturne