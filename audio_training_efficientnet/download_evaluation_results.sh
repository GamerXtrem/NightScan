#!/bin/bash

# Script pour télécharger les résultats d'évaluation depuis le serveur
# Usage: ./download_evaluation_results.sh [output_dir]

# Configuration
SERVER_USER="ubuntu"
SERVER_IP="37.156.45.160"
SSH_KEY="~/.ssh/infomaniak_key"
REMOTE_BASE_DIR="/home/ubuntu/NightScan/audio_training_efficientnet"
LOCAL_DIR="${1:-./evaluation_results_download}"

# Couleurs pour l'affichage
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== Téléchargement des Résultats d'Évaluation ===${NC}"
echo -e "Serveur: ${SERVER_USER}@${SERVER_IP}"
echo -e "Dossier local: ${LOCAL_DIR}"

# Créer le dossier local s'il n'existe pas
mkdir -p "${LOCAL_DIR}"

# Fonction pour télécharger un dossier
download_results() {
    local remote_path="$1"
    local local_path="$2"
    
    echo -e "\n${BLUE}Téléchargement depuis: ${remote_path}${NC}"
    
    # Utiliser rsync pour télécharger le dossier
    rsync -avz --progress \
        -e "ssh -i ${SSH_KEY}" \
        "${SERVER_USER}@${SERVER_IP}:${remote_path}/" \
        "${local_path}/"
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Téléchargement réussi${NC}"
        return 0
    else
        echo -e "${RED}✗ Erreur lors du téléchargement${NC}"
        return 1
    fi
}

# Fonction pour lister les dossiers de résultats disponibles
list_available_results() {
    echo -e "\n${BLUE}Recherche des résultats disponibles...${NC}"
    
    ssh -i ${SSH_KEY} ${SERVER_USER}@${SERVER_IP} \
        "find ${REMOTE_BASE_DIR} -type d -name 'evaluation_results*' -o -name 'eval_*' | sort"
}

# Option 1: Télécharger le dossier par défaut
if [ "$1" == "--auto" ]; then
    echo -e "\n${BLUE}Mode automatique: téléchargement de evaluation_results${NC}"
    download_results "${REMOTE_BASE_DIR}/evaluation_results" "${LOCAL_DIR}"
    
# Option 2: Mode interactif
else
    # Lister les résultats disponibles
    echo -e "\n${BLUE}Dossiers de résultats disponibles:${NC}"
    results_list=$(ssh -i ${SSH_KEY} ${SERVER_USER}@${SERVER_IP} \
        "find ${REMOTE_BASE_DIR} -type d -name 'evaluation_results*' -o -name 'eval_*' | sort")
    
    if [ -z "$results_list" ]; then
        echo -e "${RED}Aucun dossier de résultats trouvé${NC}"
        exit 1
    fi
    
    echo "$results_list" | nl -w2 -s'. '
    
    # Demander le choix
    echo -e "\n${BLUE}Entrez le numéro du dossier à télécharger (ou 'all' pour tout):${NC}"
    read -r choice
    
    if [ "$choice" == "all" ]; then
        # Télécharger tous les dossiers
        while IFS= read -r result_dir; do
            dir_name=$(basename "$result_dir")
            download_results "$result_dir" "${LOCAL_DIR}/${dir_name}"
        done <<< "$results_list"
    else
        # Télécharger le dossier sélectionné
        selected_dir=$(echo "$results_list" | sed -n "${choice}p")
        if [ -n "$selected_dir" ]; then
            dir_name=$(basename "$selected_dir")
            download_results "$selected_dir" "${LOCAL_DIR}/${dir_name}"
        else
            echo -e "${RED}Sélection invalide${NC}"
            exit 1
        fi
    fi
fi

# Afficher le contenu téléchargé
echo -e "\n${BLUE}=== Contenu Téléchargé ===${NC}"
find "${LOCAL_DIR}" -type f -name "*.json" -o -name "*.txt" -o -name "*.csv" -o -name "*.png" | sort

# Afficher les rapports texte s'ils existent
txt_files=$(find "${LOCAL_DIR}" -name "evaluation_report.txt" -type f)
if [ -n "$txt_files" ]; then
    echo -e "\n${BLUE}=== Résumé des Performances ===${NC}"
    for txt_file in $txt_files; do
        echo -e "\n${GREEN}$(dirname "$txt_file"):${NC}"
        # Extraire les métriques principales
        grep -E "Accuracy:|F1-score macro:|Top-3 accuracy:" "$txt_file" | head -5
    done
fi

echo -e "\n${GREEN}✓ Téléchargement terminé!${NC}"
echo -e "Les résultats sont dans: ${LOCAL_DIR}"

# Option pour ouvrir automatiquement les images
if command -v open &> /dev/null; then
    echo -e "\n${BLUE}Voulez-vous ouvrir les graphiques? (y/n)${NC}"
    read -r open_images
    if [ "$open_images" == "y" ]; then
        find "${LOCAL_DIR}" -name "*.png" -exec open {} \;
    fi
fi