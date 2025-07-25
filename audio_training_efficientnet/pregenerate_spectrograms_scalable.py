#!/usr/bin/env python3
"""
Script de pré-génération des spectrogrammes pour le système scalable.
Génère et sauvegarde tous les spectrogrammes (originaux et augmentés) en .npy.
"""

import os
import sys
import argparse
import sqlite3
import numpy as np
import torch
from pathlib import Path
import logging
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import gc
from typing import Tuple, Optional

# Ajouter le chemin parent
sys.path.append(str(Path(__file__).parent.parent))

from audio_dataset_scalable import AudioDatasetScalable
from spectrogram_config import get_config_for_animal

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def process_single_sample(args: Tuple[int, str, Path, str, str, Path, dict]) -> Optional[str]:
    """
    Traite un seul échantillon pour générer ses spectrogrammes.
    
    Args:
        args: Tuple contenant (sample_id, file_path, audio_root, class_name, split, output_dir, config_dict)
    
    Returns:
        Message de succès ou None si erreur
    """
    sample_id, file_path, audio_root, class_name, split, output_dir, config_dict = args
    
    try:
        # Recréer le dataset pour ce worker (nécessaire pour multiprocessing)
        from spectrogram_config import SpectrogramConfig
        config = SpectrogramConfig(**config_dict)
        
        # Créer un mini-dataset pour cet échantillon
        dataset = AudioDatasetScalable(
            index_db="dummy.db",  # Pas utilisé ici
            audio_root=audio_root,
            config=config,
            split=split,
            augment=False,  # On génère les augmentations manuellement
            balance_classes=False
        )
        
        # Chemins de sortie
        audio_path = audio_root / file_path
        base_name = Path(file_path).stem
        
        # Créer les répertoires de sortie
        class_output_dir = output_dir / split / class_name
        class_output_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Générer le spectrogramme original
        original_path = class_output_dir / f"{base_name}.npy"
        if not original_path.exists():
            spec = dataset._generate_spectrogram_memory_efficient(audio_path)
            np.save(original_path, spec.cpu().numpy())
        
        # 2. Générer les variantes augmentées (seulement pour train)
        if split == 'train':
            # Calculer combien de variantes on veut selon la classe
            # On génère jusqu'à 10 variantes pour les classes minoritaires
            for variant in range(1, 11):
                variant_path = class_output_dir / f"{base_name}_var{variant:03d}.npy"
                if not variant_path.exists():
                    spec = dataset._generate_augmented_spectrogram_variant(audio_path, variant)
                    np.save(variant_path, spec.cpu().numpy())
        
        # Libérer la mémoire
        del dataset
        gc.collect()
        
        return f"Traité: {file_path}"
        
    except Exception as e:
        logger.error(f"Erreur pour {file_path}: {e}")
        return None


def pregenerate_spectrograms(index_db: str, 
                           audio_root: Path,
                           output_dir: Path,
                           num_workers: int = 4,
                           animal_type: str = 'general'):
    """
    Pré-génère tous les spectrogrammes depuis la base SQLite.
    
    Args:
        index_db: Chemin vers la base SQLite
        audio_root: Répertoire racine des fichiers audio
        output_dir: Répertoire de sortie pour les spectrogrammes
        num_workers: Nombre de processus parallèles
        animal_type: Type d'animal pour la configuration
    """
    logger.info(f"Démarrage de la pré-génération des spectrogrammes")
    logger.info(f"Base de données: {index_db}")
    logger.info(f"Répertoire audio: {audio_root}")
    logger.info(f"Répertoire de sortie: {output_dir}")
    logger.info(f"Workers: {num_workers}")
    
    # Créer le répertoire de sortie
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Obtenir la configuration
    config = get_config_for_animal(animal_type)
    config_dict = config.__dict__
    
    # Se connecter à la base de données
    conn = sqlite3.connect(index_db)
    cursor = conn.cursor()
    
    # Récupérer tous les échantillons
    cursor.execute("""
        SELECT id, file_path, class_name, split 
        FROM audio_samples 
        ORDER BY split, class_name, file_path
    """)
    
    samples = cursor.fetchall()
    conn.close()
    
    logger.info(f"Nombre total d'échantillons: {len(samples)}")
    
    # Préparer les arguments pour le traitement parallèle
    args_list = []
    for sample_id, file_path, class_name, split in samples:
        args = (sample_id, file_path, audio_root, class_name, split, output_dir, config_dict)
        args_list.append(args)
    
    # Traiter en parallèle
    logger.info("Début du traitement...")
    
    if num_workers > 1:
        with Pool(num_workers) as pool:
            results = list(tqdm(
                pool.imap(process_single_sample, args_list),
                total=len(args_list),
                desc="Génération des spectrogrammes"
            ))
    else:
        # Traitement séquentiel pour le debug
        results = []
        for args in tqdm(args_list, desc="Génération des spectrogrammes"):
            result = process_single_sample(args)
            results.append(result)
    
    # Compter les succès
    successes = sum(1 for r in results if r is not None)
    logger.info(f"Génération terminée: {successes}/{len(samples)} échantillons traités avec succès")
    
    # Statistiques par split
    for split in ['train', 'val', 'test']:
        split_dir = output_dir / split
        if split_dir.exists():
            n_files = sum(1 for _ in split_dir.rglob("*.npy"))
            logger.info(f"  {split}: {n_files} fichiers .npy")


def main():
    parser = argparse.ArgumentParser(description="Pré-génération des spectrogrammes pour le système scalable")
    
    parser.add_argument('--index-db', type=str, required=True,
                       help='Base SQLite contenant l\'index du dataset')
    parser.add_argument('--audio-root', type=Path, required=True,
                       help='Répertoire racine des fichiers audio')
    parser.add_argument('--output-dir', type=Path, default=Path('data/spectrograms_cache'),
                       help='Répertoire de sortie pour les spectrogrammes')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Nombre de workers parallèles')
    parser.add_argument('--animal-type', type=str, default='general',
                       help='Type d\'animal pour la configuration')
    
    args = parser.parse_args()
    
    pregenerate_spectrograms(
        args.index_db,
        args.audio_root,
        args.output_dir,
        args.num_workers,
        args.animal_type
    )


if __name__ == "__main__":
    main()