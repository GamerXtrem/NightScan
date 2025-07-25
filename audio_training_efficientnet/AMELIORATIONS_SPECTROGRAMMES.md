# Améliorations apportées à la génération des spectrogrammes

## Vue d'ensemble

J'ai créé un système unifié et optimisé pour la génération des spectrogrammes dans NightScan, basé sur les meilleures pratiques de la recherche en classification audio d'animaux sauvages.

## Fichiers créés/modifiés

### 1. `spectrogram_config.py` 
Configuration centralisée avec paramètres optimisés par type d'animal :
- **Général** : 22050 Hz, 128 mels, 50-11000 Hz
- **Chauves-souris** : 192000 Hz, 256 mels, 15000-100000 Hz (ultrasons)
- **Chouettes** : 16000 Hz, 128 mels, 200-4000 Hz (basses fréquences)
- **Oiseaux** : 22050 Hz, 128 mels, 1000-8000 Hz
- **Mammifères** : 22050 Hz, 128 mels, 100-5000 Hz
- **Amphibiens** : 16000 Hz, 128 mels, 200-5000 Hz
- **Insectes** : 44100 Hz, 128 mels, 2000-15000 Hz

### 2. `audio_augmentation.py`
Module complet d'augmentation des données :
- **AudioAugmentation** : Bruit, décalage temporel, changement de vitesse, pitch shift, SpecAugment
- **PreprocessingPipeline** : Pré-emphasis, suppression silence, normalisation, filtre passe-bande

### 3. `spectrogram_gen.py` (modifié)
- Utilise maintenant la configuration unifiée
- Support des paramètres par type d'animal
- Prétraitement intégré
- Logging amélioré

### 4. `audio_dataset.py` (modifié)
- Intégration complète de la configuration et de l'augmentation
- Génération de spectrogrammes cohérente avec spectrogram_gen.py
- Augmentation automatique pour le dataset d'entraînement
- Support du prétraitement

### 5. `analyze_spectrograms.ipynb`
Notebook Jupyter pour :
- Visualiser les différentes configurations
- Comparer les spectrogrammes
- Analyser l'effet des augmentations
- Générer des rapports d'analyse

## Paramètres clés optimisés

### Basé sur la recherche (2024) :
- **n_fft** : 2048 (résolution fréquentielle optimale)
- **hop_length** : 512 (n_fft/4, bon compromis temps/fréquence)
- **n_mels** : 128 (standard pour la classification audio)
- **window** : Hann (moins d'artefacts spectraux)
- **top_db** : 80 (plage dynamique appropriée)

### Prétraitement :
- **Pré-emphasis** : 0.97 (accentue les hautes fréquences)
- **Filtre passe-haut** : 50 Hz (élimine le bruit basse fréquence)
- **Normalisation** : Par spectrogramme (cohérence entre échantillons)
- **Suppression silence** : -40 dB (garde seulement le signal utile)

### Augmentation (améliore la généralisation) :
- **SpecAugment** : Masquage temps/fréquence (robustesse)
- **Bruit gaussien** : 0.5% (simule conditions réelles)
- **Décalage temporel** : ±20% (invariance temporelle)
- **Changement vitesse** : ±10% (simule distance)
- **Pitch shift** : ±2 demi-tons (variations naturelles)

## Utilisation

### Génération de spectrogrammes optimisés :
```bash
# Pour des oiseaux
python spectrogram_gen.py audio_dir/ spec_dir/ --animal-type bird_song

# Pour des chauves-souris (nécessite audio haute fréquence)
python spectrogram_gen.py audio_dir/ spec_dir/ --animal-type bat
```

### Entraînement avec configuration optimisée :
```python
from audio_dataset import create_data_loaders

loaders = create_data_loaders(
    csv_dir=Path("data/processed/csv"),
    audio_dir=Path("audio_data"),
    animal_type='bird_song',  # Configuration automatique
    augment_train=True        # Augmentation sur train uniquement
)
```

## Bénéfices

1. **Cohérence** : Mêmes paramètres entre capture et entraînement
2. **Performance** : Paramètres optimisés selon la recherche récente
3. **Flexibilité** : Adaptation automatique par type d'animal
4. **Robustesse** : Augmentation et prétraitement intégrés
5. **Maintenabilité** : Configuration centralisée

## Recommandations

1. **Commencer avec 'general'** pour la plupart des cas
2. **Utiliser l'augmentation** pour améliorer la généralisation
3. **Adapter le type d'animal** selon vos espèces cibles
4. **Vérifier avec le notebook** pour visualiser les résultats
5. **Ajuster fmin/fmax** si vous connaissez les fréquences exactes

## Résultats attendus

- Meilleure extraction des caractéristiques acoustiques
- Augmentation de la précision de classification
- Robustesse accrue aux variations environnementales
- Temps d'entraînement réduit grâce au cache des spectrogrammes