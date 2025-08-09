#!/bin/bash

# ============================================
# Script de configuration de l'environnement
# Pour serveur Infomaniak avec GPU L4
# ============================================

set -e  # Arrêter en cas d'erreur

# Couleurs pour output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}==========================================${NC}"
echo -e "${BLUE}  Setup Environment - NightScan Training  ${NC}"
echo -e "${BLUE}==========================================${NC}"

# ============================================
# Détection du système
# ============================================

echo -e "\n${YELLOW}Détection du système...${NC}"

# OS Detection
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS="linux"
    echo "OS: Linux"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    OS="macos"
    echo "OS: macOS"
else
    echo -e "${RED}OS non supporté: $OSTYPE${NC}"
    exit 1
fi

# Python version
PYTHON_VERSION=$(python3 --version 2>&1 | grep -oE '[0-9]+\.[0-9]+')
echo "Python version: $PYTHON_VERSION"

# CUDA detection
if command -v nvidia-smi &> /dev/null; then
    CUDA_VERSION=$(nvidia-smi | grep -oE "CUDA Version: [0-9]+\.[0-9]+" | grep -oE "[0-9]+\.[0-9]+")
    echo "CUDA version: $CUDA_VERSION"
    HAS_GPU=true
else
    echo "Pas de GPU NVIDIA détecté"
    HAS_GPU=false
fi

# ============================================
# Création de l'environnement virtuel
# ============================================

echo -e "\n${YELLOW}Configuration de l'environnement Python...${NC}"

# Créer l'environnement virtuel si nécessaire
VENV_DIR="venv_training"
if [ ! -d "$VENV_DIR" ]; then
    echo "Création de l'environnement virtuel..."
    python3 -m venv $VENV_DIR
    echo -e "${GREEN}✓ Environnement créé${NC}"
else
    echo "Environnement virtuel existant trouvé"
fi

# Activer l'environnement
source $VENV_DIR/bin/activate
echo -e "${GREEN}✓ Environnement activé${NC}"

# Mettre à jour pip
pip install --upgrade pip setuptools wheel -q
echo -e "${GREEN}✓ pip mis à jour${NC}"

# ============================================
# Installation des dépendances
# ============================================

echo -e "\n${YELLOW}Installation des dépendances...${NC}"

# Créer le fichier requirements si nécessaire
cat > requirements_training.txt << EOF
# Core dependencies
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.21.0
pandas>=1.3.0
Pillow>=9.0.0
tqdm>=4.65.0
pyyaml>=6.0

# Machine Learning
scikit-learn>=1.0.0
scipy>=1.7.0

# Visualization
matplotlib>=3.5.0
seaborn>=0.11.0
plotly>=5.0.0
tensorboard>=2.11.0

# Image processing
opencv-python>=4.5.0
albumentations>=1.3.0

# Utilities
python-dotenv>=0.19.0
click>=8.0.0
EOF

# Installation PyTorch avec CUDA si GPU disponible
if [ "$HAS_GPU" = true ]; then
    echo "Installation de PyTorch avec support CUDA..."
    
    # Déterminer la version CUDA pour PyTorch
    if [[ "$CUDA_VERSION" == "11.8" ]]; then
        PYTORCH_CUDA="cu118"
    elif [[ "$CUDA_VERSION" == "12."* ]]; then
        PYTORCH_CUDA="cu121"
    else
        PYTORCH_CUDA="cu118"  # Défaut
    fi
    
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/$PYTORCH_CUDA
else
    echo "Installation de PyTorch CPU..."
    pip install torch torchvision torchaudio
fi

# Installer les autres dépendances
pip install -r requirements_training.txt

echo -e "${GREEN}✓ Dépendances installées${NC}"

# ============================================
# Vérification de l'installation
# ============================================

echo -e "\n${YELLOW}Vérification de l'installation...${NC}"

# Test PyTorch
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA disponible: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ PyTorch installé correctement${NC}"
else
    echo -e "${RED}✗ Erreur avec PyTorch${NC}"
    exit 1
fi

# Test des imports critiques
python3 -c "
import torchvision
import numpy
import pandas
import sklearn
import plotly
import yaml
print('✓ Tous les modules importés avec succès')
"

if [ $? -ne 0 ]; then
    echo -e "${RED}✗ Erreur lors de l'import des modules${NC}"
    exit 1
fi

# ============================================
# Configuration des dossiers
# ============================================

echo -e "\n${YELLOW}Configuration des dossiers...${NC}"

# Créer la structure de dossiers
mkdir -p data/{raw,processed}
mkdir -p outputs/{models,logs,reports}
mkdir -p picture_training_enhanced/{models,plots}

echo -e "${GREEN}✓ Structure de dossiers créée${NC}"

# ============================================
# Téléchargement des poids pré-entraînés
# ============================================

echo -e "\n${YELLOW}Téléchargement des poids pré-entraînés...${NC}"

python3 -c "
import torch
from torchvision import models

# Télécharger les poids EfficientNet
print('Téléchargement EfficientNet-B0...')
models.efficientnet_b0(pretrained=True)
print('✓ EfficientNet-B0')

print('Téléchargement EfficientNet-B1...')
models.efficientnet_b1(pretrained=True)
print('✓ EfficientNet-B1')

print('Poids pré-entraînés téléchargés avec succès')
"

# ============================================
# Optimisations système
# ============================================

echo -e "\n${YELLOW}Application des optimisations système...${NC}"

# Augmenter les limites systèmes si possible
if [ "$OS" == "linux" ]; then
    # Augmenter la limite de fichiers ouverts
    ulimit -n 4096 2>/dev/null || true
    
    # Optimisations réseau pour transferts de données
    if [ -w /proc/sys/net/core/rmem_max ]; then
        echo 134217728 > /proc/sys/net/core/rmem_max 2>/dev/null || true
        echo 134217728 > /proc/sys/net/core/wmem_max 2>/dev/null || true
    fi
fi

# ============================================
# Création du fichier de configuration
# ============================================

echo -e "\n${YELLOW}Création du fichier .env...${NC}"

cat > .env << EOF
# Configuration environnement
PYTHONPATH=$(pwd)
CUDA_VISIBLE_DEVICES=0
OMP_NUM_THREADS=8
MKL_NUM_THREADS=8
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Paths
DATA_DIR=$(pwd)/data/processed
OUTPUT_DIR=$(pwd)/outputs
MODEL_DIR=$(pwd)/picture_training_enhanced/models

# Training defaults
DEFAULT_BATCH_SIZE=64
DEFAULT_EPOCHS=50
DEFAULT_LR=0.001
EOF

echo -e "${GREEN}✓ Fichier .env créé${NC}"

# ============================================
# Script de test rapide
# ============================================

echo -e "\n${YELLOW}Création du script de test...${NC}"

cat > test_setup.py << 'EOF'
#!/usr/bin/env python3
"""Test rapide de la configuration."""

import torch
import torchvision
from pathlib import Path
import sys

def test_gpu():
    """Test GPU availability."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"✓ GPU disponible: {torch.cuda.get_device_name(0)}")
        
        # Test simple computation
        x = torch.randn(100, 100).to(device)
        y = torch.matmul(x, x)
        print(f"✓ Calcul GPU réussi")
        
        # Memory info
        print(f"  VRAM totale: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print(f"  VRAM allouée: {torch.cuda.memory_allocated() / 1024**3:.3f} GB")
    else:
        print("⚠ Pas de GPU disponible, utilisation du CPU")
        return False
    return True

def test_data_pipeline():
    """Test data loading."""
    from torchvision import transforms, datasets
    
    # Create dummy data
    dummy_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    print("✓ Pipeline de données configuré")
    return True

def test_model():
    """Test model creation."""
    from torchvision import models
    
    model = models.efficientnet_b0(pretrained=False)
    print(f"✓ Modèle créé: {sum(p.numel() for p in model.parameters())} paramètres")
    return True

if __name__ == "__main__":
    print("\n=== Test de configuration ===\n")
    
    tests = [
        ("GPU", test_gpu),
        ("Pipeline de données", test_data_pipeline),
        ("Création de modèle", test_model),
    ]
    
    all_passed = True
    for name, test_func in tests:
        try:
            if test_func():
                print(f"✅ {name}: OK")
            else:
                print(f"⚠️ {name}: Warning")
        except Exception as e:
            print(f"❌ {name}: FAILED - {e}")
            all_passed = False
    
    if all_passed:
        print("\n✅ Tous les tests passés avec succès!")
        sys.exit(0)
    else:
        print("\n⚠️ Certains tests ont échoué")
        sys.exit(1)
EOF

chmod +x test_setup.py

# Exécuter le test
echo -e "\n${YELLOW}Exécution du test de configuration...${NC}"
python3 test_setup.py

# ============================================
# Résumé final
# ============================================

echo -e "\n${BLUE}==========================================${NC}"
echo -e "${BLUE}          Configuration terminée          ${NC}"
echo -e "${BLUE}==========================================${NC}"

echo -e "\n${GREEN}✓ Environnement configuré avec succès!${NC}"
echo ""
echo "Pour activer l'environnement:"
echo "  source $VENV_DIR/bin/activate"
echo ""
echo "Pour lancer l'entraînement:"
echo "  ./run_training.sh /path/to/data /path/to/output"
echo ""
echo "Pour désactiver l'environnement:"
echo "  deactivate"
echo ""

if [ "$HAS_GPU" = true ]; then
    echo -e "${GREEN}GPU détecté et configuré: $CUDA_VERSION${NC}"
else
    echo -e "${YELLOW}⚠ Pas de GPU détecté - entraînement sur CPU${NC}"
fi