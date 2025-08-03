# NightScan Segments - Workflow en 2 passes (comme BirdNET)

Ce nouveau workflow suit exactement l'approche de BirdNET pour une meilleure sélection des segments audio.

## Vue d'ensemble

Le processus est maintenant divisé en 2 passes distinctes :

1. **Passe 1 - Analyse** : Détecte toutes les espèces dans les fichiers audio et génère des fichiers CSV de résultats
2. **Passe 2 - Extraction** : Lit les résultats, sélectionne les meilleurs segments et les extrait

## Installation

Les scripts utilisent les mêmes dépendances que le reste du projet NightScan.

## Utilisation

### Option 1 : Workflow complet (recommandé)

```bash
python nightscan_segments.py all \
    --audio-input /Volumes/dataset/NightScan_raw_audio_raw \
    --output /Volumes/dataset/NightScan_extracted_segments \
    --model "/path/to/models_balanced/best_model.pth" \
    --training-db "/path/to/balanced_audio_index.db" \
    --min-conf 0.25 \
    --max-segments 500 \
    --threads 4
```

### Option 2 : Exécuter les passes séparément

#### Passe 1 - Analyse seulement

```bash
python nightscan_segments.py analyze \
    --audio-input /Volumes/dataset/NightScan_raw_audio_raw \
    --output /Volumes/dataset/analysis_results \
    --model "/path/to/models_balanced/best_model.pth" \
    --training-db "/path/to/balanced_audio_index.db" \
    --min-conf 0.25 \
    --threads 4
```

#### Passe 2 - Extraction seulement

```bash
python nightscan_segments.py extract \
    --audio-input /Volumes/dataset/NightScan_raw_audio_raw \
    --results /Volumes/dataset/analysis_results \
    --output /Volumes/dataset/extracted_segments \
    --max-segments 500 \
    --threads 4
```

## Paramètres importants

### Paramètres d'analyse (Passe 1)
- `--min-conf` : Confiance minimale (défaut: 0.25, comme BirdNET)
- `--seg-length` : Longueur des segments en secondes (défaut: 3.0)
- `--threads` : Nombre de threads CPU pour le traitement parallèle

### Paramètres d'extraction (Passe 2)
- `--max-segments` : Nombre maximum de segments à extraire par fichier (défaut: 500)
- `--min-overlap` : Chevauchement minimum pour filtrer les doublons (défaut: 0.5)

## Structure de sortie

```
output_directory/
├── results/                    # Résultats de l'analyse (Passe 1)
│   ├── class_name_1/
│   │   ├── audio_file_1.csv
│   │   ├── audio_file_2.csv
│   │   └── ...
│   ├── class_name_2/
│   │   └── ...
│   └── analysis_summary.json
└── segments/                   # Segments extraits (Passe 2)
    ├── species_1/
    │   ├── audio_file_seg001500_conf85.wav
    │   ├── audio_file_seg003000_conf82.wav
    │   └── ...
    ├── species_2/
    │   └── ...
    └── extraction_summary.json
```

## Format des fichiers de résultats CSV

Les fichiers CSV suivent le format BirdNET :

```csv
Start (s),End (s),Scientific name,Common name,Confidence
1.5,4.5,species_name,Common Name,0.8523
3.0,6.0,species_name,Common Name,0.7891
```

## Avantages de cette approche

1. **Meilleure sélection** : Vue globale de toutes les détections avant de sélectionner
2. **Flexibilité** : Possibilité de ré-extraire avec différents paramètres sans refaire l'analyse
3. **Performance** : Séparation des tâches CPU-intensives (analyse) et I/O-intensives (extraction)
4. **Compatibilité** : Format de résultats compatible avec d'autres outils

## Exemple de commande complète

```bash
# Pour votre cas spécifique
python nightscan_segments.py all \
    --audio-input /Volumes/dataset/NightScan_raw_audio_raw \
    --output /Volumes/dataset/NightScan_segments_birdnet_style \
    --model "/Users/jonathanmaitrot/NightScan/Clone claude/NightScan/audio_training_efficientnet/models_balanced/best_model.pth" \
    --training-db "/Users/jonathanmaitrot/NightScan/Clone claude/NightScan/audio_training_efficientnet/balanced_audio_index.db" \
    --min-conf 0.25 \
    --max-segments 500 \
    --seg-length 3.0 \
    --threads 4
```

## Comparaison avec l'ancien workflow

| Aspect | Ancien workflow | Nouveau workflow (BirdNET-style) |
|--------|----------------|----------------------------------|
| Approche | 1 passe (détection + extraction) | 2 passes séparées |
| Sélection | Séquentielle | Globale (meilleurs segments) |
| Flexibilité | Doit tout refaire | Peut ré-extraire facilement |
| Performance | Variable | Optimisée par tâche |
| Résultats | Non sauvegardés | CSV réutilisables |