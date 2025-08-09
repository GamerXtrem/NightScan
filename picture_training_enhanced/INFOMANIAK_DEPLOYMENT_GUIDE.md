# 🚀 Guide de Déploiement sur Serveur Infomaniak GPU

Guide spécifique pour l'entraînement de modèles sur les serveurs Infomaniak avec GPU NVIDIA L4.

## 📋 Spécifications du Serveur

### Configuration Matérielle
- **GPU:** NVIDIA L4 (24 GB VRAM)
- **CPU:** 8 cœurs
- **RAM:** 16 GB
- **Stockage:** SSD NVMe
- **Réseau:** 10 Gbps

### Environnement Logiciel
- **OS:** Ubuntu 22.04 LTS
- **CUDA:** 11.8 ou 12.1
- **Python:** 3.8+
- **Docker:** Disponible (optionnel)

## 🔧 Configuration Initiale

### 1. Connexion au Serveur

```bash
# SSH vers le serveur
ssh username@your-server.infomaniak.ch

# Vérifier GPU
nvidia-smi

# Vérifier CUDA
nvcc --version
```

### 2. Installation des Dépendances Système

```bash
# Mise à jour système
sudo apt update && sudo apt upgrade -y

# Outils essentiels
sudo apt install -y \
    build-essential \
    python3-pip \
    python3-venv \
    git \
    htop \
    nvtop \
    tmux \
    curl \
    wget

# Bibliothèques pour le traitement d'images
sudo apt install -y \
    libopencv-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev
```

### 3. Configuration CUDA et cuDNN

```bash
# Vérifier version CUDA
cat /usr/local/cuda/version.txt

# Ajouter CUDA au PATH (si nécessaire)
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# Tester CUDA
python3 -c "import torch; print(torch.cuda.is_available())"
```

## 📦 Installation du Projet

### 1. Cloner le Repository

```bash
# Créer dossier de travail
mkdir -p ~/projects
cd ~/projects

# Cloner NightScan
git clone https://github.com/your-repo/NightScan.git
cd NightScan/picture_training_enhanced
```

### 2. Configuration de l'Environnement Python

```bash
# Créer environnement virtuel
python3 -m venv venv_gpu
source venv_gpu/bin/activate

# Mettre à jour pip
pip install --upgrade pip setuptools wheel

# Installer PyTorch avec CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Installer autres dépendances
pip install -r requirements_training.txt
```

### 3. Optimisations Spécifiques L4

```bash
# Créer fichier de configuration optimisé
cat > infomaniak_env.sh << 'EOF'
#!/bin/bash

# GPU Configuration
export CUDA_VISIBLE_DEVICES=0
export CUDA_LAUNCH_BLOCKING=0

# Optimisations CPU
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

# Optimisations mémoire GPU L4
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# TensorFloat32 pour L4 (architecture Ada Lovelace)
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1
export NVIDIA_TF32_OVERRIDE=1

# Optimisations réseau
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_DISABLE=1

echo "Environment configuré pour GPU L4"
EOF

chmod +x infomaniak_env.sh
source infomaniak_env.sh
```

## 🎯 Configuration Optimale pour L4

### Fichier `config_infomaniak.yaml`

```yaml
# Configuration optimisée pour NVIDIA L4
data:
  data_dir: /data/processed
  image_size: 224
  augmentation_level: moderate

model:
  model_name: null  # Auto-sélection
  pretrained: true
  dropout_rate: 0.3
  use_attention: false  # Économiser VRAM

training:
  epochs: 50
  batch_size: 128  # L4 peut gérer des gros batches
  learning_rate: 0.001
  weight_decay: 0.0001
  optimizer: adamw
  scheduler: cosine
  
  # Optimisations L4
  use_amp: true  # Mixed Precision obligatoire
  gradient_accumulation_steps: 1
  gradient_clip: 1.0
  
  # Early stopping
  patience: 10
  
system:
  num_workers: 6  # 8 cœurs - 2 pour système
  pin_memory: true
  persistent_workers: true
  prefetch_factor: 2
  compile_model: false  # Désactiver si problèmes

monitoring:
  tensorboard: true
  save_frequency: 5
  keep_last_checkpoints: 3

# Spécifique Infomaniak
infomaniak:
  gpu_memory_fraction: 0.95
  cuda_visible_devices: "0"
  mixed_precision_backend: native
```

## 📊 Gestion des Données

### 1. Transfert de Données Efficace

```bash
# Depuis local vers serveur (rsync recommandé)
rsync -avzP --progress \
  /local/path/to/images/ \
  username@server.infomaniak.ch:~/data/raw/

# Compression pour gros volumes
tar -czf images.tar.gz images/
scp images.tar.gz username@server:~/data/
ssh username@server "cd ~/data && tar -xzf images.tar.gz"
```

### 2. Organisation sur le Serveur

```bash
# Structure recommandée
~/
├── data/
│   ├── raw/           # Images originales
│   ├── processed/     # Données préparées
│   └── cache/         # Cache des augmentations
├── models/            # Modèles entraînés
├── logs/              # Logs d'entraînement
└── exports/           # Modèles exportés
```

### 3. Stockage Optimisé

```bash
# Utiliser SSD local pour données d'entraînement
df -h  # Vérifier espace disponible

# Lien symbolique si stockage réseau
ln -s /mnt/network_storage/large_dataset ~/data/raw
```

## 🚀 Lancement de l'Entraînement

### 1. Session Persistante avec tmux

```bash
# Créer session tmux
tmux new -s training

# Dans la session tmux
source venv_gpu/bin/activate
source infomaniak_env.sh

# Lancer entraînement
./run_training.sh \
  ~/data/processed \
  ~/models \
  config_infomaniak.yaml \
  50 \
  128

# Détacher: Ctrl+B puis D
# Rattacher: tmux attach -t training
```

### 2. Script de Lancement Automatisé

```bash
cat > train_infomaniak.sh << 'EOF'
#!/bin/bash

# Configuration
DATA_DIR="$HOME/data/processed"
OUTPUT_DIR="$HOME/models/$(date +%Y%m%d_%H%M%S)"
CONFIG="config_infomaniak.yaml"

# Activation environnement
source $HOME/venv_gpu/bin/activate
source $HOME/infomaniak_env.sh

# Vérifications
echo "Checking GPU..."
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

# Lancement
python train_real_images.py \
  --data_dir $DATA_DIR \
  --output_dir $OUTPUT_DIR \
  --config $CONFIG \
  --use_amp \
  --num_workers 6 \
  2>&1 | tee $OUTPUT_DIR/training.log

# Notification fin
echo "Training completed at $(date)"
EOF

chmod +x train_infomaniak.sh
nohup ./train_infomaniak.sh > training_main.log 2>&1 &
```

## 📈 Monitoring et Optimisation

### 1. Monitoring en Temps Réel

```bash
# GPU utilisation
watch -n 1 nvidia-smi

# Ou avec nvtop (plus visuel)
nvtop

# CPU et RAM
htop

# Logs d'entraînement
tail -f ~/models/*/training.log

# TensorBoard
tensorboard --logdir ~/models --host 0.0.0.0 --port 6006
# Puis tunnel SSH: ssh -L 6006:localhost:6006 username@server
```

### 2. Optimisations Batch Size

```python
# Script pour trouver batch size optimal
cat > find_optimal_batch.py << 'EOF'
import torch
import gc

def find_optimal_batch_size(model, device, input_shape=(3, 224, 224)):
    batch_size = 2
    while True:
        try:
            # Clear cache
            torch.cuda.empty_cache()
            gc.collect()
            
            # Test batch
            batch = torch.randn(batch_size, *input_shape).to(device)
            output = model(batch)
            loss = output.sum()
            loss.backward()
            
            print(f"Batch size {batch_size}: OK")
            batch_size *= 2
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"Max batch size: {batch_size // 2}")
                break
            else:
                raise e
    
    return batch_size // 2
EOF
```

### 3. Profiling Performance

```python
# Profiler PyTorch
from torch.profiler import profile, ProfilerActivity

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    # Training step
    output = model(input_batch)
    loss = criterion(output, labels)
    loss.backward()

print(prof.key_averages().table(sort_by="cuda_time_total"))
```

## 🔥 Optimisations Avancées

### 1. Multi-GPU (si disponible)

```python
# DataParallel
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)

# Ou DistributedDataParallel (meilleur)
torch.distributed.init_process_group(backend='nccl')
model = nn.parallel.DistributedDataParallel(model)
```

### 2. Gradient Checkpointing

```python
# Pour économiser VRAM sur gros modèles
from torch.utils.checkpoint import checkpoint

class CheckpointedModel(nn.Module):
    def forward(self, x):
        x = checkpoint(self.layer1, x)
        x = checkpoint(self.layer2, x)
        return x
```

### 3. Cache des Augmentations

```python
# Pré-calculer augmentations lourdes
from functools import lru_cache

@lru_cache(maxsize=10000)
def cached_augmentation(image_path, seed):
    # Augmentation déterministe avec seed
    return augmented_image
```

## 🐛 Troubleshooting Spécifique Infomaniak

### Problème: CUDA Out of Memory

```bash
# Solution 1: Réduire batch size
--batch_size 64

# Solution 2: Gradient accumulation
--gradient_accumulation_steps 2 --batch_size 32

# Solution 3: Vider cache
python -c "import torch; torch.cuda.empty_cache()"
```

### Problème: Connexion SSH Perdue

```bash
# Utiliser tmux ou screen
tmux new -s training

# Ou nohup
nohup python train.py > log.txt 2>&1 &
```

### Problème: Performances Lentes

```bash
# Vérifier utilisation GPU
nvidia-smi dmon -s u

# Si < 90%, problème de data loading
# Augmenter num_workers ou utiliser cache
```

### Problème: Stockage Plein

```bash
# Nettoyer checkpoints anciens
find ~/models -name "checkpoint_epoch_*.pth" -mtime +7 -delete

# Compresser logs
gzip ~/logs/*.log
```

## 📊 Benchmarks Attendus sur L4

| Configuration | Images/sec | VRAM Usage | Temps 50 epochs |
|--------------|------------|------------|-----------------|
| B=32, AMP=ON | ~400 | 8 GB | 3h |
| B=64, AMP=ON | ~600 | 14 GB | 2h |
| B=128, AMP=ON | ~800 | 20 GB | 1.5h |

## 🔄 Workflow de Production

```bash
# 1. Préparer données
python data_preparation.py --input_dir ~/data/raw --output_dir ~/data/processed

# 2. Entraîner
./train_infomaniak.sh

# 3. Évaluer
python evaluate_model.py --checkpoint ~/models/best_model.pth --test_dir ~/data/processed/test

# 4. Exporter
python export_models.py --checkpoint ~/models/best_model.pth --formats onnx torchscript

# 5. Télécharger résultats
rsync -avz username@server:~/models/best_model.pth ./
rsync -avz username@server:~/models/evaluation_results/ ./eval/
```

## 📝 Checklist de Déploiement

- [ ] Serveur accessible via SSH
- [ ] GPU détecté (`nvidia-smi`)
- [ ] CUDA installé et fonctionnel
- [ ] Python environment créé
- [ ] PyTorch avec support GPU installé
- [ ] Données transférées
- [ ] Configuration optimisée créée
- [ ] Script de lancement testé
- [ ] Monitoring configuré
- [ ] Backup stratégie en place

## 🆘 Support

### Logs Importants

```bash
# Logs système
/var/log/syslog
journalctl -u nvidia-persistenced

# Logs GPU
nvidia-smi -q > gpu_info.txt

# Logs Python
~/.cache/pip/log/
```

### Contact Infomaniak

- Support technique: support@infomaniak.com
- Documentation: https://www.infomaniak.com/fr/support

---

**Dernière mise à jour:** Janvier 2025  
**Optimisé pour:** NVIDIA L4 24GB  
**Testé sur:** Ubuntu 22.04 LTS