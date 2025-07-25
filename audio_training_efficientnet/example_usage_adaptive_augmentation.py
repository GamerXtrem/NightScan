#!/usr/bin/env python3
"""
Exemple d'utilisation du système d'augmentation adaptative dans l'entraînement
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from audio_dataset import create_data_loaders
from spectrogram_config import SpectrogramConfig

def main():
    """
    Exemple d'utilisation du système d'augmentation adaptative.
    """
    print("=== Exemple d'utilisation de l'Augmentation Adaptative ===")
    
    # Chemins des données
    csv_dir = Path("data/processed/csv")
    audio_dir = Path("/Volumes/dataset/NightScan_raw_audio_raw_segmented")
    spectrogram_dir = Path("data/spectrograms")
    
    # Configuration pour l'entraînement avec augmentation adaptative
    config = SpectrogramConfig(
        animal_type='general',
        use_augmentation=True,
        enable_minority_support=True  # Active le support des classes minoritaires
    )
    
    # Créer les dataloaders avec l'augmentation adaptative
    print("\nCréation des DataLoaders avec augmentation adaptative...")
    
    loaders = create_data_loaders(
        csv_dir=csv_dir,
        audio_dir=audio_dir,
        batch_size=32,
        num_workers=4,
        spectrogram_dir=spectrogram_dir,
        config=config,
        augment_train=True,           # Active l'augmentation pour l'entraînement
        adaptive_augment=True,        # Active l'augmentation adaptative
        enable_oversampling=True,     # Active l'oversampling pour classes < 500
    )
    
    # Afficher les informations sur les datasets
    print("\nInformations sur les datasets:")
    for split, loader in loaders.items():
        if loader:
            dataset = loader.dataset
            print(f"\n{split.upper()}:")
            print(f"  - Nombre total d'échantillons: {len(dataset)}")
            print(f"  - Nombre de classes: {dataset.num_classes}")
            print(f"  - Augmentation active: {dataset.augment}")
            
            if hasattr(dataset, 'class_counts') and split == 'train':
                print("\n  Distribution des classes:")
                for class_name in sorted(dataset.class_names):
                    count = dataset.class_counts.get(class_name, 0)
                    if hasattr(dataset, 'augmenter') and dataset.augmenter:
                        category = dataset.augmenter.get_class_category(count)
                        multiplier = dataset.augmenter.get_augmentation_multiplier(count)
                        print(f"    - {class_name}: {count} échantillons ({category}, {multiplier}x augmentation)")
                    else:
                        print(f"    - {class_name}: {count} échantillons")
    
    print("\n=== Configuration pour l'entraînement ===")
    print("\nDans votre script d'entraînement (train_audio.py), modifiez comme suit:")
    print("""
# Importer les modules nécessaires
from audio_dataset import create_data_loaders
from spectrogram_config import SpectrogramConfig

# Configuration avec support des classes minoritaires
config = SpectrogramConfig(
    animal_type='general',
    use_augmentation=True,
    enable_minority_support=True
)

# Créer les dataloaders
loaders = create_data_loaders(
    csv_dir=csv_dir,
    audio_dir=audio_dir,
    batch_size=args.batch_size,
    num_workers=args.num_workers,
    spectrogram_dir=spectrogram_dir,
    config=config,
    augment_train=True,          # Active l'augmentation
    adaptive_augment=True,       # Active l'augmentation adaptative
    enable_oversampling=True,    # Active l'oversampling
)

# Le reste du code d'entraînement reste identique
""")
    
    print("\n=== Avantages de l'Augmentation Adaptative ===")
    print("""
1. **Phase Shifting** : Nouvelle technique d'augmentation utilisant STFT/iSTFT
   - Décalage de phase de ±π/4 radians pour enrichir les variations

2. **Augmentation Adaptative** : Intensité ajustée selon la taille de classe
   - Classes < 50 : 5x augmentation avec paramètres agressifs
   - Classes 50-200 : 3x augmentation avec paramètres modérés
   - Classes 200-500 : 2x augmentation avec paramètres légers
   - Classes >= 500 : Augmentation standard

3. **Oversampling Intelligent** : Équilibrage automatique des classes
   - Duplication des échantillons pour approcher 500 par classe
   - Mélange aléatoire pour éviter les biais

4. **Logging Détaillé** : Suivi des statistiques d'augmentation
   - Affichage du nombre d'échantillons par classe
   - Catégorie et multiplicateur pour chaque classe
   - Paramètres d'augmentation utilisés
""")


if __name__ == "__main__":
    main()