#!/bin/bash
# Script pour créer le pool augmenté classe par classe
# Évite l'accumulation de mémoire en traitant une classe à la fois

AUDIO_ROOT="$HOME/NightScan/data/audio_data"
OUTPUT_DIR="$HOME/NightScan/data/augmented_pool"

echo "🎯 Création du pool augmenté classe par classe..."
echo "Source: $AUDIO_ROOT"
echo "Destination: $OUTPUT_DIR"
echo ""

# Créer le répertoire de sortie
mkdir -p "$OUTPUT_DIR"

# Lister toutes les classes
CLASSES=$(find "$AUDIO_ROOT" -mindepth 1 -maxdepth 1 -type d ! -name ".*" -exec basename {} \; | sort)
TOTAL_CLASSES=$(echo "$CLASSES" | wc -l | tr -d ' ')

echo "📊 Classes trouvées: $TOTAL_CLASSES"
echo ""

# Traiter chaque classe individuellement
CLASS_NUM=0
for CLASS in $CLASSES; do
    CLASS_NUM=$((CLASS_NUM + 1))
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "[$CLASS_NUM/$TOTAL_CLASSES] Traitement de: $CLASS"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    # Lancer le script pour cette classe uniquement
    python create_augmented_pool.py \
        --audio-root "$AUDIO_ROOT" \
        --output-dir "$OUTPUT_DIR" \
        --batch-size 1 \
        --target-samples 500 \
        --max-augmentations 20 \
        --specific-class "$CLASS"
    
    EXIT_CODE=$?
    
    if [ $EXIT_CODE -ne 0 ]; then
        echo "❌ Erreur lors du traitement de $CLASS (code: $EXIT_CODE)"
        echo "Voulez-vous continuer avec les autres classes? (y/n)"
        read -r CONTINUE
        if [ "$CONTINUE" != "y" ]; then
            echo "Arrêt du script."
            exit 1
        fi
    else
        echo "✅ $CLASS traité avec succès"
    fi
    
    # Pause entre les classes pour libérer la mémoire
    echo "⏸️  Pause de 2 secondes..."
    sleep 2
    echo ""
done

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ Traitement terminé!"
echo ""

# Créer le fichier de métadonnées global
echo "📝 Création du fichier de métadonnées global..."
python -c "
import json
from pathlib import Path
from datetime import datetime

output_dir = Path('$OUTPUT_DIR')
class_metadata_dir = output_dir / 'class_metadata'

if class_metadata_dir.exists():
    # Lire toutes les métadonnées de classe
    all_stats = {}
    total_original = 0
    total_created = 0
    
    for metadata_file in sorted(class_metadata_dir.glob('*.json')):
        with open(metadata_file) as f:
            data = json.load(f)
            class_name = data['class_name']
            stats = data['stats']
            all_stats[class_name] = stats
            total_original += stats['original_count']
            total_created += stats['total_count']
    
    # Créer le fichier global
    pool_metadata = {
        'timestamp': datetime.now().isoformat(),
        'target_samples_per_class': 500,
        'classes_processed': len(all_stats),
        'total_original': total_original,
        'total_created': total_created,
        'classes': all_stats
    }
    
    with open(output_dir / 'pool_metadata.json', 'w') as f:
        json.dump(pool_metadata, f, indent=2)
    
    print(f'✅ Métadonnées globales créées: {len(all_stats)} classes')
    print(f'   Fichiers originaux: {total_original}')
    print(f'   Fichiers dans le pool: {total_created}')
"

echo ""
echo "📊 Prochaine étape:"
echo "   python create_balanced_index.py --pool-dir $OUTPUT_DIR --output-db balanced_audio_index.db"