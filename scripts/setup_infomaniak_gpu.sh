#!/bin/bash
# Script de setup pour l'entraînement sur instance GPU Infomaniak Cloud
# Instance type: nvl4-a8-ram16-disk50-perf1 (NVIDIA L4)

set -e

echo "=== NightScan GPU Training Setup for Infomaniak Cloud ==="
echo "Instance: nvl4-a8-ram16-disk50-perf1"
echo "==========================================="

# Variables
PROJECT_DIR="$HOME/NightScan"
VENV_DIR="$PROJECT_DIR/venv"
DATA_DIR="$PROJECT_DIR/data"
CUDA_VERSION="118"  # CUDA 11.8 for L4 GPU

# Mise à jour système
echo "1. Mise à jour du système..."
sudo apt update && sudo apt upgrade -y

# Installation des dépendances système
echo "2. Installation des dépendances système..."
sudo apt install -y \
    python3-pip \
    python3-venv \
    python3-dev \
    git \
    ffmpeg \
    libsndfile1 \
    libsndfile1-dev \
    libportaudio2 \
    htop \
    tmux \
    screen \
    nvidia-utils-515

# Vérification NVIDIA
echo "3. Vérification GPU NVIDIA L4..."
nvidia-smi
if [ $? -ne 0 ]; then
    echo "ERREUR: GPU non détecté. Vérifiez que l'instance a bien une GPU."
    exit 1
fi

# Clone du projet (si pas déjà fait)
if [ ! -d "$PROJECT_DIR" ]; then
    echo "4. Clonage du projet NightScan..."
    cd $HOME
    git clone https://github.com/GamerXtrem/NightScan.git
else
    echo "4. Projet déjà cloné, mise à jour..."
    cd $PROJECT_DIR
    git pull
fi

# Création de l'environnement virtuel
echo "5. Création de l'environnement virtuel Python..."
cd $PROJECT_DIR
python3 -m venv $VENV_DIR
source $VENV_DIR/bin/activate

# Mise à jour pip
echo "6. Mise à jour de pip..."
pip install --upgrade pip wheel setuptools

# Installation PyTorch avec support CUDA
echo "7. Installation de PyTorch avec support CUDA 11.8..."
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu${CUDA_VERSION}

# Installation des dépendances du projet
echo "8. Installation des dépendances NightScan..."
pip install -r requirements.txt

# Création des répertoires nécessaires
echo "9. Création de la structure de répertoires..."
mkdir -p $DATA_DIR/audio_data
mkdir -p $DATA_DIR/processed/csv
mkdir -p $DATA_DIR/spectrograms_cache
mkdir -p $PROJECT_DIR/audio_training_efficientnet/models

# Test CUDA
echo "10. Test de CUDA..."
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB')
else:
    print('ATTENTION: CUDA non disponible!')
"

# Création du script de lancement
echo "11. Création du script de lancement..."
cat > $PROJECT_DIR/train_audio_cloud.sh << 'EOF'
#!/bin/bash
# Script de lancement de l'entraînement audio optimisé

# Activation de l'environnement
source ~/NightScan/venv/bin/activate
cd ~/NightScan

# Variables
DATA_DIR="${1:-data/audio_data}"
EPOCHS="${2:-50}"
BATCH_SIZE="${3:-64}"

echo "Configuration:"
echo "- Dataset: $DATA_DIR"
echo "- Epochs: $EPOCHS"
echo "- Batch size: $BATCH_SIZE"
echo "- GPU: $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")')"

# Lancement avec monitoring GPU
echo "Lancement de l'entraînement..."
python audio_training_efficientnet/train_audio_optimized.py \
    --data-dir "$DATA_DIR" \
    --csv-dir data/processed/csv \
    --output-dir audio_training_efficientnet/models \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --pregenerate \
    --num-workers 4

echo "Entraînement terminé!"
echo "Modèle sauvegardé dans: audio_training_efficientnet/models/"
EOF

chmod +x $PROJECT_DIR/train_audio_cloud.sh

# Script de monitoring GPU
echo "12. Création du script de monitoring..."
cat > $PROJECT_DIR/monitor_gpu.sh << 'EOF'
#!/bin/bash
# Monitoring GPU pendant l'entraînement
watch -n 1 'nvidia-smi; echo ""; ps aux | grep python | grep train'
EOF

chmod +x $PROJECT_DIR/monitor_gpu.sh

echo ""
echo "=== Setup terminé avec succès! ==="
echo ""
echo "Prochaines étapes:"
echo "1. Transférez votre dataset avec:"
echo "   scp -r dataset_audio.tar.gz ubuntu@$(hostname -I | awk '{print $1}'):~/NightScan/data/"
echo ""
echo "2. Décompressez le dataset:"
echo "   cd ~/NightScan/data && tar -xzf dataset_audio.tar.gz"
echo ""
echo "3. Lancez l'entraînement dans tmux/screen:"
echo "   tmux new -s training"
echo "   ~/NightScan/train_audio_cloud.sh"
echo ""
echo "4. Monitorer le GPU dans un autre terminal:"
echo "   ~/NightScan/monitor_gpu.sh"
echo ""
echo "5. Récupérer le modèle entraîné:"
echo "   scp ubuntu@$(hostname -I | awk '{print $1}'):~/NightScan/audio_training_efficientnet/models/best_model.pth ./"