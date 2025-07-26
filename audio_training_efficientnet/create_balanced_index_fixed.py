#!/usr/bin/env python3
"""
Version corrigée de create_balanced_index.py qui lit les métadonnées depuis class_metadata/
au lieu de pool_metadata.json
"""

import os
import sys
import argparse
import sqlite3
import json
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Tuple
import random
from collections import defaultdict

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_class_metadata(pool_dir: Path) -> Dict:
    """
    Charge les métadonnées depuis les fichiers class_metadata/*.json
    """
    class_metadata_dir = pool_dir / 'class_metadata'
    if not class_metadata_dir.exists():
        raise FileNotFoundError(f"Le dossier {class_metadata_dir} n'existe pas")
    
    pool_metadata = {'classes': {}}
    
    # Lire chaque fichier de métadonnées de classe
    for json_file in class_metadata_dir.glob('*.json'):
        with open(json_file, 'r') as f:
            class_data = json.load(f)
        
        class_name = class_data['class_name']
        stats = class_data['stats']
        
        # Reconstruire la structure attendue
        pool_metadata['classes'][class_name] = {
            'total_count': stats['total_count'],
            'files': []
        }
        
        # Si les fichiers détaillés sont disponibles
        if 'files' in stats:
            pool_metadata['classes'][class_name]['files'] = stats['files']
        else:
            # Sinon, scanner le répertoire de la classe
            class_dir = pool_dir / class_name
            if class_dir.exists():
                files = []
                for audio_file in class_dir.glob('*.wav'):
                    file_info = {
                        'path': str(audio_file.relative_to(pool_dir)),
                        'is_augmented': '_aug' in audio_file.stem,
                        'source': audio_file.stem.split('_aug')[0] if '_aug' in audio_file.stem else audio_file.stem
                    }
                    if file_info['is_augmented'] and '_' in audio_file.stem:
                        parts = audio_file.stem.split('_')
                        for i, part in enumerate(parts):
                            if part.startswith('aug'):
                                aug_type = '_'.join(parts[i+1:]) if i+1 < len(parts) else 'unknown'
                                file_info['augmentation_type'] = aug_type
                                break
                    files.append(file_info)
                pool_metadata['classes'][class_name]['files'] = files
                pool_metadata['classes'][class_name]['total_count'] = len(files)
    
    return pool_metadata


def create_balanced_index(
    pool_dir: Path,
    output_db: str,
    train_split: float = 0.8,
    val_split: float = 0.1,
    test_split: float = 0.1,
    random_seed: int = 42
) -> None:
    """
    Crée un index SQLite équilibré à partir du pool augmenté.
    
    Args:
        pool_dir: Répertoire contenant le pool augmenté
        output_db: Chemin de sortie pour la base SQLite
        train_split: Proportion pour l'entraînement
        val_split: Proportion pour la validation
        test_split: Proportion pour le test
        random_seed: Graine pour la reproductibilité
    """
    logger.info(f"Création de l'index équilibré")
    logger.info(f"Pool source: {pool_dir}")
    logger.info(f"Base de données: {output_db}")
    logger.info(f"Splits: train={train_split}, val={val_split}, test={test_split}")
    
    # Charger les métadonnées depuis class_metadata/
    pool_metadata = load_class_metadata(pool_dir)
    
    # Initialiser le générateur aléatoire
    np.random.seed(random_seed)
    random.seed(random_seed)
    
    # Créer la base de données
    conn = sqlite3.connect(output_db)
    cursor = conn.cursor()
    
    # Créer la table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS audio_samples (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_path TEXT NOT NULL,
            class_name TEXT NOT NULL,
            is_augmented INTEGER DEFAULT 0,
            original_file TEXT,
            augmentation_type TEXT,
            split TEXT NOT NULL,
            UNIQUE(file_path)
        )
    """)
    
    # Créer les index
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_class ON audio_samples(class_name)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_split ON audio_samples(split)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_augmented ON audio_samples(is_augmented)")
    
    # Statistiques globales
    global_stats = {
        'train': defaultdict(int),
        'val': defaultdict(int),
        'test': defaultdict(int)
    }
    
    # Traiter chaque classe
    for class_name, class_info in pool_metadata['classes'].items():
        logger.info(f"\nTraitement de {class_name}: {class_info['total_count']} échantillons")
        
        # Grouper par fichier original pour s'assurer qu'un même original
        # et ses augmentations ne se retrouvent pas dans différents splits
        files_by_original = defaultdict(list)
        
        for file_info in class_info['files']:
            source_key = file_info['source']
            files_by_original[source_key].append(file_info)
        
        # Mélanger les groupes
        groups = list(files_by_original.items())
        random.shuffle(groups)
        
        # Calculer les tailles des splits
        n_groups = len(groups)
        n_train = int(n_groups * train_split)
        n_val = int(n_groups * val_split)
        
        # Assigner les groupes aux splits
        train_groups = groups[:n_train]
        val_groups = groups[n_train:n_train+n_val]
        test_groups = groups[n_train+n_val:]
        
        # Insérer dans la base de données
        split_assignments = [
            ('train', train_groups),
            ('val', val_groups),
            ('test', test_groups)
        ]
        
        for split_name, split_groups in split_assignments:
            for original_name, files in split_groups:
                for file_info in files:
                    cursor.execute("""
                        INSERT OR REPLACE INTO audio_samples 
                        (file_path, class_name, is_augmented, original_file, augmentation_type, split)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (
                        file_info['path'],
                        class_name,
                        1 if file_info['is_augmented'] else 0,
                        original_name,
                        file_info.get('augmentation_type', None),
                        split_name
                    ))
                    
                    # Mettre à jour les statistiques
                    global_stats[split_name][class_name] += 1
                    if file_info['is_augmented']:
                        global_stats[split_name]['augmented'] += 1
                    else:
                        global_stats[split_name]['original'] += 1
        
        # Afficher les statistiques de la classe
        logger.info(f"  Répartition pour {class_name}:")
        for split_name in ['train', 'val', 'test']:
            count = global_stats[split_name][class_name]
            logger.info(f"    {split_name}: {count} échantillons")
    
    # Commit et fermer
    conn.commit()
    conn.close()
    
    # Afficher le résumé global
    logger.info("\nRésumé global:")
    total_samples = 0
    for split_name in ['train', 'val', 'test']:
        total = sum(v for k, v in global_stats[split_name].items() if k not in ['augmented', 'original'])
        aug_count = global_stats[split_name].get('augmented', 0)
        orig_count = global_stats[split_name].get('original', 0)
        total_samples += total
        logger.info(f"  {split_name}: {total} échantillons ({orig_count} originaux, {aug_count} augmentés)")
    
    logger.info(f"\nTotal: {total_samples} échantillons")
    logger.info(f"Base de données créée: {output_db}")


def main():
    parser = argparse.ArgumentParser(description="Créer un index SQLite équilibré depuis le pool augmenté")
    
    parser.add_argument('--pool-dir', type=Path, required=True,
                       help='Répertoire contenant le pool augmenté')
    parser.add_argument('--output-db', type=str, default='balanced_audio_index.db',
                       help='Fichier de sortie SQLite')
    parser.add_argument('--train-split', type=float, default=0.8,
                       help='Proportion pour train (défaut: 0.8)')
    parser.add_argument('--val-split', type=float, default=0.1,
                       help='Proportion pour val (défaut: 0.1)')
    parser.add_argument('--test-split', type=float, default=0.1,
                       help='Proportion pour test (défaut: 0.1)')
    parser.add_argument('--random-seed', type=int, default=42,
                       help='Graine aléatoire pour reproductibilité')
    
    args = parser.parse_args()
    
    # Vérifier les splits
    if abs(args.train_split + args.val_split + args.test_split - 1.0) > 0.001:
        parser.error("Les splits doivent sommer à 1.0")
    
    # Vérifier que le pool existe
    if not args.pool_dir.exists():
        parser.error(f"Le répertoire {args.pool_dir} n'existe pas")
    
    create_balanced_index(
        pool_dir=args.pool_dir,
        output_db=args.output_db,
        train_split=args.train_split,
        val_split=args.val_split,
        test_split=args.test_split,
        random_seed=args.random_seed
    )


if __name__ == "__main__":
    main()