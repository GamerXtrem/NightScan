#!/bin/bash
# Script pour créer le pool augmenté avec des paramètres de mémoire sûrs

echo "🔄 Création du pool augmenté avec gestion mémoire optimisée..."
echo ""

# Utiliser un batch size très petit pour éviter les problèmes de mémoire
python create_augmented_pool.py \
    --audio-root ~/NightScan/data/audio_data \
    --output-dir ~/NightScan/data/augmented_pool \
    --batch-size 1 \
    --target-samples 500 \
    --max-augmentations 20

echo ""
echo "✅ Si le script se termine avec succès, continuez avec :"
echo "python create_balanced_index.py --pool-dir ~/NightScan/data/augmented_pool --output-db balanced_audio_index.db"