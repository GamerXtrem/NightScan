#!/bin/bash
# Script pour nettoyer les anciennes données du système d'équilibrage

echo "🧹 Nettoyage des anciennes données..."

# Anciennes bases de données SQLite
echo "Suppression des anciennes bases de données..."
rm -f audio_index_balanced.db
rm -f audio_index.db
rm -f scalable_audio_index.db

# Anciens répertoires de spectrogrammes
echo "Suppression des anciens spectrogrammes..."
rm -rf data/spectrograms_cache/
rm -rf data/spectrograms/
rm -rf ~/NightScan/data/spectrograms_cache/

# Anciennes sauvegardes de modèles qui pourraient être corrompues
echo "Nettoyage des checkpoints potentiellement corrompus..."
find models_large_scale/ -name "*.pth" -mtime -7 -exec ls -la {} \; 2>/dev/null

echo ""
echo "⚠️  Les fichiers suivants vont être conservés :"
echo "  - ~/NightScan/data/audio_data/ (données audio originales)"
echo "  - models_large_scale/*.pth plus anciens que 7 jours"
echo ""
echo "📁 Les nouveaux fichiers seront créés dans :"
echo "  - ~/NightScan/data/augmented_pool/ (pool augmenté)"
echo "  - ~/NightScan/data/spectrograms_balanced/ (spectrogrammes)"
echo "  - balanced_audio_index.db (nouvel index)"
echo ""
echo "✅ Nettoyage terminé!"