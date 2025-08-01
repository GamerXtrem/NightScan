#!/bin/bash

# Script rapide pour télécharger uniquement les fichiers essentiels d'évaluation
# Usage: ./download_evaluation_quick.sh

# Configuration
SERVER="ubuntu@37.156.45.160"
SSH_KEY="~/.ssh/infomaniak_key"
REMOTE_DIR="/home/ubuntu/NightScan/audio_training_efficientnet/evaluation_results"
LOCAL_DIR="./eval_results_$(date +%Y%m%d_%H%M%S)"

# Créer le dossier local
mkdir -p "${LOCAL_DIR}"

echo "📥 Téléchargement des résultats d'évaluation..."

# Télécharger uniquement les fichiers importants (pas les gros PNG)
scp -i ${SSH_KEY} \
    "${SERVER}:${REMOTE_DIR}/evaluation_report.json" \
    "${SERVER}:${REMOTE_DIR}/evaluation_report.txt" \
    "${SERVER}:${REMOTE_DIR}/confusion_matrix.csv" \
    "${LOCAL_DIR}/" 2>/dev/null

if [ $? -eq 0 ]; then
    echo "✅ Téléchargement réussi!"
    echo "📁 Fichiers dans: ${LOCAL_DIR}"
    echo ""
    echo "📊 Résumé des performances:"
    grep -A 5 "MÉTRIQUES GLOBALES" "${LOCAL_DIR}/evaluation_report.txt" | tail -5
else
    echo "❌ Erreur lors du téléchargement"
fi