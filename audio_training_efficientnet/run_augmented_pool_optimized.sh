#!/bin/bash
# Script optimisé pour créer le pool augmenté avec une utilisation mémoire minimale

echo "🚀 Création du pool augmenté avec optimisations mémoire..."
echo ""
echo "Configuration:"
echo "  - Batch size: 1 (traitement fichier par fichier)"
echo "  - Métadonnées: minimales (pas de liste détaillée)"
echo "  - Garbage collection: forcé après chaque classe"
echo ""

# Lancer avec les optimisations mémoire
python create_augmented_pool.py \
    --audio-root ~/NightScan/data/audio_data \
    --output-dir ~/NightScan/data/augmented_pool \
    --batch-size 1 \
    --target-samples 500 \
    --max-augmentations 20

# Note: on n'utilise PAS --save-detailed-metadata pour économiser la mémoire

echo ""
echo "✅ Si le script se termine avec succès, les métadonnées seront dans:"
echo "   ~/NightScan/data/augmented_pool/pool_metadata.json"
echo ""
echo "📊 Prochaine étape:"
echo "   python create_balanced_index.py --pool-dir ~/NightScan/data/augmented_pool --output-db balanced_audio_index.db"