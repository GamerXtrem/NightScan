# 🖼️ NightScan Picture Training Enhanced

Module complet d'entraînement de modèles EfficientNet pour la classification d'images de faune sauvage.
Optimisé pour serveur GPU (NVIDIA L4) avec support de classes dynamiques et export multi-format.

## 📋 Table des Matières

- [Vue d'ensemble](#vue-densemble)
- [Fonctionnalités](#fonctionnalités)
- [Prérequis](#prérequis)
- [Installation](#installation)
- [Guide de Démarrage Rapide](#guide-de-démarrage-rapide)
- [Description des Scripts](#description-des-scripts)
- [Configuration](#configuration)
- [Utilisation Avancée](#utilisation-avancée)
- [Performance](#performance)
- [Troubleshooting](#troubleshooting)

## 🎯 Vue d'ensemble

Ce module permet d'entraîner des modèles de classification d'images avec :
- **Détection automatique** du nombre de classes depuis la structure des dossiers
- **Sélection intelligente** de l'architecture selon la complexité (B0 → B4)
- **Optimisations GPU** pour NVIDIA L4 (Mixed Precision, Gradient Accumulation)
- **Pipeline complet** de la préparation des données à l'export production

## ✨ Fonctionnalités

### Préparation des Données
- ✅ Découverte automatique des classes
- ✅ Split train/val/test configurable
- ✅ Validation et nettoyage des images corrompues
- ✅ Calcul automatique des statistiques pour normalisation

### Entraînement
- ✅ Architecture EfficientNet (B0-B4) avec sélection automatique
- ✅ Mixed Precision Training (FP16) pour GPU
- ✅ Augmentations adaptatives (3 niveaux)
- ✅ Learning rate scheduling (Cosine, OneCycle)
- ✅ Early stopping avec patience
- ✅ Checkpointing automatique

### Métriques et Visualisation
- ✅ Precision, Recall, F1-Score par classe
- ✅ Top-K accuracy (K=1,3,5)
- ✅ ROC-AUC multi-classes
- ✅ Matrice de confusion interactive
- ✅ Rapport HTML complet avec graphiques Plotly

### Export
- ✅ PyTorch (.pth)
- ✅ TorchScript (.pt)
- ✅ ONNX (.onnx)
- ✅ CoreML (.mlmodel) pour iOS
- ✅ TensorFlow Lite (.tflite) pour Android
- ✅ Quantification INT8

## 🔧 Prérequis

### Système
- Python 3.8+
- CUDA 11.8+ (pour GPU)
- 16 GB RAM minimum
- 50 GB espace disque

### GPU (Recommandé)
- NVIDIA GPU avec 8GB+ VRAM
- Driver NVIDIA 470+
- CUDA Toolkit 11.8+

## 📦 Installation

### 1. Configuration de l'environnement

```bash
# Cloner le repository
cd picture_training_enhanced

# Exécuter le script de setup
chmod +x setup_environment.sh
./setup_environment.sh
```

### 2. Installation manuelle (alternative)

```bash
# Créer environnement virtuel
python3 -m venv venv_training
source venv_training/bin/activate

# Installer PyTorch (GPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Installer dépendances
pip install -r requirements_training.txt
```

## 🚀 Guide de Démarrage Rapide

### Structure des Données Attendue

```
data/
├── classe1/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── classe2/
│   ├── image1.jpg
│   └── ...
└── classe3/
    └── ...
```

### Workflow Complet

```bash
# 1. Préparer les données
python data_preparation.py \
  --input_dir /chemin/vers/images/brutes \
  --output_dir ./data/processed \
  --train_ratio 0.7 \
  --val_ratio 0.15 \
  --test_ratio 0.15

# 2. Lancer l'entraînement
./run_training.sh ./data/processed ./outputs config.yaml 50 64

# 3. Évaluer le modèle
python evaluate_model.py \
  --checkpoint ./outputs/exp_*/best_model.pth \
  --test_dir ./data/processed/test \
  --output_dir ./evaluation_results

# 4. Exporter le modèle
python export_models.py \
  --checkpoint ./outputs/exp_*/best_model.pth \
  --output_dir ./exported_models \
  --formats pytorch torchscript onnx
```

## 📝 Description des Scripts

### Scripts Principaux

| Script | Description | Utilisation |
|--------|-------------|-------------|
| `data_preparation.py` | Prépare et organise les données | Division train/val/test, validation images |
| `train_real_images.py` | Script d'entraînement principal | Entraînement avec toutes les optimisations |
| `evaluate_model.py` | Évaluation complète | Métriques détaillées et test de robustesse |
| `export_models.py` | Export multi-format | Conversion pour production/mobile |

### Scripts de Support

| Script | Description |
|--------|-------------|
| `photo_dataset.py` | Gestion des datasets et augmentations |
| `photo_model_dynamic.py` | Architecture du modèle avec sélection auto |
| `metrics.py` | Calcul des métriques détaillées |
| `visualize_results.py` | Génération des graphiques et rapports |

### Scripts Bash

| Script | Description |
|--------|-------------|
| `setup_environment.sh` | Installation complète de l'environnement |
| `run_training.sh` | Lancement de l'entraînement avec monitoring |

## ⚙️ Configuration

### Fichier `config.yaml`

```yaml
data:
  data_dir: /path/to/data      # Dossier des données
  image_size: 224               # Taille des images
  augmentation_level: moderate  # light/moderate/heavy

model:
  model_name: null      # Auto-sélection si null
  pretrained: true      # Utiliser ImageNet
  dropout_rate: 0.3     # Régularisation
  use_attention: false  # Module d'attention

training:
  epochs: 50
  batch_size: 64
  learning_rate: 0.001
  optimizer: adamw      # adamw/sgd
  scheduler: cosine     # cosine/onecycle
  use_amp: true         # Mixed Precision
  patience: 10          # Early stopping

system:
  num_workers: 4        # DataLoader workers
  compile_model: false  # torch.compile (PyTorch 2.0+)
```

### Variables d'Environnement

```bash
export CUDA_VISIBLE_DEVICES=0     # GPU à utiliser
export OMP_NUM_THREADS=8          # Threads CPU
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

## 🔬 Utilisation Avancée

### Fine-tuning d'un Modèle Existant

```python
python train_real_images.py \
  --resume_from ./checkpoint.pth \
  --learning_rate 0.0001 \
  --epochs 20 \
  --differential_lr
```

### Entraînement avec Classes Déséquilibrées

```yaml
# Dans config.yaml
training:
  use_class_weights: true
  label_smoothing: 0.1
```

### Test de Robustesse

```bash
python evaluate_model.py \
  --checkpoint model.pth \
  --test_dir ./test_data \
  --test_robustness \
  --output_dir ./robustness_results
```

### Export Optimisé Mobile

```bash
# iOS (CoreML)
python export_models.py --checkpoint model.pth --formats coreml

# Android (TFLite quantifié)
python export_models.py --checkpoint model.pth --formats tflite --quantize
```

## 📊 Performance

### Benchmarks sur GPU L4

| Métrique | Valeur |
|----------|--------|
| Vitesse d'entraînement | ~500 images/sec |
| Utilisation VRAM (batch=64) | ~12 GB |
| Temps pour 50 epochs | ~2-4 heures |
| Vitesse d'inférence | ~1000 images/sec |

### Résultats Typiques

| Dataset | Classes | Accuracy | F1-Score |
|---------|---------|----------|----------|
| Petit (5K images) | 5 | 92-95% | 90-93% |
| Moyen (20K images) | 10 | 88-92% | 86-90% |
| Grand (100K images) | 50 | 85-90% | 83-88% |

## 🔧 Troubleshooting

### Problèmes Courants

#### Out of Memory (OOM) GPU
```bash
# Réduire batch size
--batch_size 32

# Ou utiliser gradient accumulation
--gradient_accumulation_steps 2
```

#### Images Corrompues
```bash
# Le script data_preparation.py les détecte automatiquement
# Vérifier les logs pour voir les images ignorées
```

#### Convergence Lente
```yaml
# Ajuster dans config.yaml
training:
  scheduler: onecycle  # Meilleur que cosine parfois
  learning_rate: 0.003  # Augmenter si trop lent
```

#### Overfitting
```yaml
model:
  dropout_rate: 0.5  # Augmenter dropout
data:
  augmentation_level: heavy  # Plus d'augmentation
```

### Logs et Debug

```bash
# Logs d'entraînement
tail -f outputs/logs/training_*.log

# TensorBoard
tensorboard --logdir outputs/exp_*/runs

# Vérifier GPU
nvidia-smi -l 1
```

## 📚 Ressources

- [Documentation PyTorch](https://pytorch.org/docs/)
- [EfficientNet Paper](https://arxiv.org/abs/1905.11946)
- [Mixed Precision Training](https://pytorch.org/docs/stable/amp.html)

## 🤝 Support

Pour toute question ou problème :
1. Vérifier ce README et les logs
2. Consulter le guide de troubleshooting
3. Ouvrir une issue sur le repository

## 📄 Licence

Voir le fichier LICENSE du projet principal NightScan.

---

**Dernière mise à jour:** Janvier 2025  
**Version:** 1.0.0  
**Auteur:** NightScan Team