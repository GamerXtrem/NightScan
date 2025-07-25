#!/usr/bin/env python3
"""
Script de pré-génération des spectrogrammes pour accélérer l'entraînement.
Génère et met en cache tous les spectrogrammes (originaux et augmentés).
"""

import argparse
import sys
from pathlib import Path
import logging
from tqdm import tqdm
import numpy as np
import pandas as pd

# Ajouter le chemin parent pour les imports
sys.path.append(str(Path(__file__).parent))

from audio_dataset import AudioSpectrogramDataset
from spectrogram_config import SpectrogramConfig, get_config_for_animal
from audio_augmentation import AdaptiveAudioAugmentation

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def pregenerate_spectrograms(csv_file: Path, 
                           audio_dir: Path,
                           spectrogram_dir: Path,
                           animal_type: str = 'general',
                           n_augmentations: int = 10,
                           skip_augmented: bool = False):
    """
    Pré-génère les spectrogrammes pour un dataset.
    
    Args:
        csv_file: Fichier CSV avec les données
        audio_dir: Répertoire des fichiers audio
        spectrogram_dir: Répertoire de sortie pour les spectrogrammes
        animal_type: Type d'animal pour la configuration
        n_augmentations: Nombre de versions augmentées à générer par échantillon
        skip_augmented: Si True, génère seulement les spectrogrammes originaux
    """
    logger.info(f"Pré-génération des spectrogrammes pour {csv_file}")
    
    # Créer la configuration
    config = get_config_for_animal(animal_type)
    
    # Créer le dataset SANS augmentation pour générer les originaux
    dataset_orig = AudioSpectrogramDataset(
        csv_file=csv_file,
        audio_dir=audio_dir,
        spectrogram_dir=spectrogram_dir,
        config=config,
        animal_type=animal_type,
        augment=False,
        cache_spectrograms=True
    )
    
    # Compter les échantillons par classe
    df = pd.read_csv(csv_file)
    class_counts = df['label'].value_counts().to_dict()
    
    logger.info(f"Génération des spectrogrammes originaux pour {len(dataset_orig)} échantillons...")
    
    # Générer tous les spectrogrammes originaux
    for idx in tqdm(range(len(dataset_orig)), desc="Spectrogrammes originaux"):
        # Le simple fait d'accéder à l'élément va le générer et le mettre en cache
        _ = dataset_orig[idx]
    
    if not skip_augmented:
        # Créer un augmenteur adaptatif
        augmenter = AdaptiveAudioAugmentation(dataset_orig.sample_rate)
        
        logger.info(f"Génération des spectrogrammes augmentés...")
        
        # Pour chaque classe, déterminer le nombre d'augmentations nécessaires
        for class_name, count in class_counts.items():
            category = augmenter.get_class_category(count)
            multiplier = augmenter.get_augmentation_multiplier(count)
            
            logger.info(f"Classe '{class_name}': {count} échantillons ({category}, {multiplier}x)")
            
            # Filtrer les échantillons de cette classe
            class_indices = df[df['label'] == class_name].index.tolist()
            
            # Nombre d'augmentations à générer pour cette classe
            if count < 250:  # Seulement pour les classes qui seront dupliquées
                n_aug_class = n_augmentations * multiplier
            else:
                n_aug_class = n_augmentations
            
            # Créer un dataset avec augmentation pour cette classe
            dataset_aug = AudioSpectrogramDataset(
                csv_file=csv_file,
                audio_dir=audio_dir,
                spectrogram_dir=spectrogram_dir,
                config=config,
                animal_type=animal_type,
                augment=True,
                adaptive_augment=True,
                cache_spectrograms=True
            )
            
            # Générer les augmentations pour chaque échantillon de la classe
            for idx in tqdm(class_indices, desc=f"Augmentations {class_name}"):
                for aug_idx in range(n_aug_class):
                    # Forcer un seed différent pour chaque augmentation
                    np.random.seed(idx * 1000 + aug_idx)
                    _ = dataset_aug[idx]
    
    logger.info("Pré-génération terminée!")
    
    # Afficher les statistiques
    if spectrogram_dir.exists():
        # Utiliser rglob pour chercher récursivement dans tous les sous-dossiers
        all_npy_files = list(spectrogram_dir.rglob("**/*.npy"))
        orig_files = [f for f in all_npy_files if "original" in str(f)]
        aug_files = [f for f in all_npy_files if "augmented" in str(f)]
        
        orig_count = len(orig_files)
        aug_count = len(aug_files)
        
        logger.info(f"\nStatistiques du cache:")
        logger.info(f"- Spectrogrammes originaux: {orig_count}")
        logger.info(f"- Spectrogrammes augmentés: {aug_count}")
        logger.info(f"- Total: {orig_count + aug_count}")
        
        # Calculer la taille du cache
        total_size = sum(f.stat().st_size for f in all_npy_files)
        logger.info(f"- Taille totale du cache: {total_size / 1024 / 1024:.1f} MB")
        
        # Afficher quelques exemples de fichiers générés
        if all_npy_files:
            logger.info(f"\nExemples de fichiers générés:")
            for f in all_npy_files[:5]:
                logger.info(f"  - {f.relative_to(spectrogram_dir)}")
            if len(all_npy_files) > 5:
                logger.info(f"  ... et {len(all_npy_files) - 5} autres fichiers")


def main():
    parser = argparse.ArgumentParser(
        description="Pré-génère les spectrogrammes pour accélérer l'entraînement"
    )
    parser.add_argument(
        '--csv-dir',
        type=Path,
        default=Path("data/processed/csv"),
        help="Répertoire contenant les fichiers CSV"
    )
    parser.add_argument(
        '--audio-dir',
        type=Path,
        required=True,
        help="Répertoire contenant les fichiers audio"
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path("data/spectrograms_cache"),
        help="Répertoire de sortie pour les spectrogrammes"
    )
    parser.add_argument(
        '--animal-type',
        type=str,
        default='general',
        help="Type d'animal pour la configuration"
    )
    parser.add_argument(
        '--n-augmentations',
        type=int,
        default=10,
        help="Nombre de versions augmentées par échantillon"
    )
    parser.add_argument(
        '--splits',
        nargs='+',
        default=['train'],
        help="Splits à traiter (train, val, test)"
    )
    parser.add_argument(
        '--skip-augmented',
        action='store_true',
        help="Générer seulement les spectrogrammes originaux"
    )
    
    args = parser.parse_args()
    
    # Créer le répertoire de sortie s'il n'existe pas
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Répertoire de cache créé/vérifié: {args.output_dir}")
    
    # Traiter chaque split
    for split in args.splits:
        csv_file = args.csv_dir / f"{split}.csv"
        
        if not csv_file.exists():
            logger.warning(f"Fichier {csv_file} non trouvé, ignoré")
            continue
        
        logger.info(f"\n{'='*50}")
        logger.info(f"Traitement du split: {split}")
        logger.info(f"{'='*50}")
        
        pregenerate_spectrograms(
            csv_file=csv_file,
            audio_dir=args.audio_dir,
            spectrogram_dir=args.output_dir,
            animal_type=args.animal_type,
            n_augmentations=args.n_augmentations,
            skip_augmented=(args.skip_augmented or split != 'train')  # Augmentations seulement pour train
        )
    
    logger.info("\n✅ Pré-génération complète terminée!")
    logger.info("\nPour utiliser le cache dans l'entraînement:")
    logger.info(f"python train_audio.py --spectrogram-dir {args.output_dir}")


if __name__ == "__main__":
    main()