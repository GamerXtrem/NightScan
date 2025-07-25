#!/bin/bash
# Script pour préparer et transférer le dataset audio vers Infomaniak Cloud

set -e

echo "=== Préparation du dataset pour le transfert ==="

# Variables configurables
LOCAL_DATASET_PATH="/Volumes/dataset/petit dataset_segmented"
INSTANCE_IP="37.156.45.113"
INSTANCE_USER="ubuntu"
SSH_KEY="~/.ssh/NightScan"
COMPRESSION_LEVEL=6  # 1-9, 6 est un bon compromis vitesse/taille

# Vérification du dataset local
if [ ! -d "$LOCAL_DATASET_PATH" ]; then
    echo "ERREUR: Le dataset n'existe pas à: $LOCAL_DATASET_PATH"
    echo "Veuillez ajuster la variable LOCAL_DATASET_PATH dans ce script."
    exit 1
fi

# Calcul de la taille du dataset
echo "1. Analyse du dataset..."
DATASET_SIZE=$(du -sh "$LOCAL_DATASET_PATH" | cut -f1)
FILE_COUNT=$(find "$LOCAL_DATASET_PATH" -type f -name "*.wav" | wc -l | tr -d ' ')
echo "   - Taille: $DATASET_SIZE"
echo "   - Nombre de fichiers audio: $FILE_COUNT"

# Nom du fichier compressé avec timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
ARCHIVE_NAME="dataset_audio_${TIMESTAMP}.tar.gz"

# Compression du dataset
echo ""
echo "2. Compression du dataset..."
echo "   Niveau de compression: $COMPRESSION_LEVEL"
echo "   Fichier de sortie: $ARCHIVE_NAME"

# Création de l'archive avec barre de progression
tar -czf "$ARCHIVE_NAME" \
    --checkpoint=1000 \
    --checkpoint-action=dot \
    -C "$(dirname "$LOCAL_DATASET_PATH")" \
    "$(basename "$LOCAL_DATASET_PATH")"

ARCHIVE_SIZE=$(du -sh "$ARCHIVE_NAME" | cut -f1)
echo ""
echo "   Archive créée: $ARCHIVE_NAME ($ARCHIVE_SIZE)"

# Options de transfert
echo ""
echo "3. Options de transfert disponibles:"
echo ""
echo "   Option A - SCP (simple mais pas de reprise):"
echo "   scp -i $SSH_KEY $ARCHIVE_NAME $INSTANCE_USER@$INSTANCE_IP:~/"
echo ""
echo "   Option B - RSYNC (recommandé, avec reprise possible):"
echo "   rsync -avzP --progress -e \"ssh -i $SSH_KEY\" $ARCHIVE_NAME $INSTANCE_USER@$INSTANCE_IP:~/"
echo ""
echo "   Option C - Split + transfert (pour très gros fichiers):"
echo "   split -b 1G $ARCHIVE_NAME ${ARCHIVE_NAME}.part."
echo "   for f in ${ARCHIVE_NAME}.part.*; do scp -i $SSH_KEY \$f $INSTANCE_USER@$INSTANCE_IP:~/; done"
echo ""

# Demande de transfert automatique
read -p "Voulez-vous lancer le transfert maintenant avec rsync? (o/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Oo]$ ]]; then
    echo ""
    echo "4. Transfert en cours..."
    rsync -avzP --progress -e "ssh -i $SSH_KEY" "$ARCHIVE_NAME" "$INSTANCE_USER@$INSTANCE_IP:~/"
    
    echo ""
    echo "5. Commandes à exécuter sur l'instance:"
    echo ""
    echo "   # Connexion SSH:"
    echo "   ssh -i $SSH_KEY $INSTANCE_USER@$INSTANCE_IP"
    echo ""
    echo "   # Extraction du dataset:"
    echo "   cd ~/NightScan"
    echo "   tar -xzf ~/$ARCHIVE_NAME -C data/"
    echo "   mv \"data/$(basename "$LOCAL_DATASET_PATH")\" data/audio_data"
    echo ""
    echo "   # Vérification:"
    echo "   find data/audio_data -type f -name \"*.wav\" | wc -l"
    echo "   # Devrait afficher: $FILE_COUNT"
fi

# Script de nettoyage
cat > cleanup_local.sh << EOF
#!/bin/bash
# Nettoyer les fichiers temporaires après transfert réussi
rm -f $ARCHIVE_NAME
rm -f ${ARCHIVE_NAME}.part.*
echo "Fichiers temporaires supprimés"
EOF
chmod +x cleanup_local.sh

echo ""
echo "=== Préparation terminée ==="
echo "Archive prête: $ARCHIVE_NAME"
echo "Pour nettoyer après transfert: ./cleanup_local.sh"