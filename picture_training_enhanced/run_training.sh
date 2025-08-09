#!/bin/bash

# ============================================
# Script d'entraînement pour serveur Infomaniak
# GPU: NVIDIA L4, 8 cœurs, 16GB RAM
# ============================================

# Couleurs pour output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}==========================================${NC}"
echo -e "${GREEN}    NightScan Training Script - GPU L4    ${NC}"
echo -e "${GREEN}==========================================${NC}"

# ============================================
# Configuration environnement
# ============================================

# GPU Configuration
export CUDA_VISIBLE_DEVICES=0
export CUDA_LAUNCH_BLOCKING=0

# Optimisations CPU
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

# Optimisations mémoire GPU
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# TensorFlow32 pour GPU Ampere+ (L4)
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1

echo -e "${YELLOW}Configuration:${NC}"
echo "  GPU: CUDA device $CUDA_VISIBLE_DEVICES"
echo "  CPU threads: $OMP_NUM_THREADS"
echo "  Working directory: $(pwd)"

# ============================================
# Vérifications préliminaires
# ============================================

echo -e "\n${YELLOW}Vérifications système...${NC}"

# Vérifier GPU
if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${RED}Erreur: nvidia-smi non trouvé${NC}"
    exit 1
fi

echo "GPU disponible:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader

# Vérifier Python et PyTorch
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}')"
if [ $? -ne 0 ]; then
    echo -e "${RED}Erreur: PyTorch non installé${NC}"
    exit 1
fi

python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA non disponible'"
if [ $? -ne 0 ]; then
    echo -e "${RED}Erreur: CUDA non disponible${NC}"
    exit 1
fi

# ============================================
# Paramètres d'entraînement
# ============================================

# Valeurs par défaut
DATA_DIR="${1:-/data/processed}"
OUTPUT_DIR="${2:-./outputs}"
CONFIG_FILE="${3:-config.yaml}"
EPOCHS="${4:-50}"
BATCH_SIZE="${5:-64}"

echo -e "\n${YELLOW}Paramètres d'entraînement:${NC}"
echo "  Data directory: $DATA_DIR"
echo "  Output directory: $OUTPUT_DIR"
echo "  Config file: $CONFIG_FILE"
echo "  Epochs: $EPOCHS"
echo "  Batch size: $BATCH_SIZE"

# Vérifier que le dossier de données existe
if [ ! -d "$DATA_DIR" ]; then
    echo -e "${RED}Erreur: Dossier de données non trouvé: $DATA_DIR${NC}"
    exit 1
fi

# Créer le dossier de sortie
mkdir -p "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/logs"

# Pas de monitoring GPU en arrière-plan pour alléger le script

# ============================================
# Lancement de l'entraînement
# ============================================

echo -e "\n${GREEN}Démarrage de l'entraînement...${NC}"
echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S')"

# Créer un nom d'expérience unique
EXPERIMENT_NAME="exp_$(date +%Y%m%d_%H%M%S)"

# Commande d'entraînement avec tous les paramètres
python3 picture_training_enhanced/train_real_images.py \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUTPUT_DIR/$EXPERIMENT_NAME" \
    --config "$CONFIG_FILE" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --use_amp \
    --pretrained \
    --num_workers 4 \
    --augmentation_level moderate \
    --seed 42 \
    2>&1 | tee "$OUTPUT_DIR/logs/training_${EXPERIMENT_NAME}.log"

TRAINING_EXIT_CODE=$?

# ============================================
# Post-traitement
# ============================================

echo -e "\n${YELLOW}Post-traitement...${NC}"

if [ $TRAINING_EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}✓ Entraînement terminé avec succès${NC}"
    
    # Afficher le meilleur modèle
    BEST_MODEL="$OUTPUT_DIR/$EXPERIMENT_NAME/best_model.pth"
    if [ -f "$BEST_MODEL" ]; then
        echo -e "\n${GREEN}Meilleur modèle sauvegardé:${NC}"
        echo "  $BEST_MODEL"
        echo "  Taille: $(du -h "$BEST_MODEL" | cut -f1)"
    fi
    
    # Lien vers le rapport HTML
    REPORT_HTML="$OUTPUT_DIR/$EXPERIMENT_NAME/report_*.html"
    if ls $REPORT_HTML 1> /dev/null 2>&1; then
        echo -e "\n${GREEN}Rapport HTML généré:${NC}"
        ls -la $REPORT_HTML
    fi
    
else
    echo -e "${RED}✗ Erreur pendant l'entraînement (code: $TRAINING_EXIT_CODE)${NC}"
    echo "Consultez les logs pour plus de détails:"
    echo "  $OUTPUT_DIR/logs/training_${EXPERIMENT_NAME}.log"
fi

# ============================================
# Résumé final
# ============================================

echo -e "\n${GREEN}==========================================${NC}"
echo -e "${GREEN}              Résumé                      ${NC}"
echo -e "${GREEN}==========================================${NC}"
echo "Expérience: $EXPERIMENT_NAME"
echo "Durée: $SECONDS secondes"
echo "Dossier de sortie: $OUTPUT_DIR/$EXPERIMENT_NAME"
echo "Logs: $OUTPUT_DIR/logs/"
echo ""
echo "Pour visualiser avec TensorBoard:"
echo "  tensorboard --logdir=$OUTPUT_DIR/$EXPERIMENT_NAME/runs"
echo ""
echo -e "${GREEN}Fin du script$(NC)"