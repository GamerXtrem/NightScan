#!/usr/bin/env python3
"""
Crée un index SQLite équilibré à partir du pool augmenté.
Divise chaque classe en train/val/test (80/10/10) en mélangeant originaux et augmentés.
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
    
    # Charger les métadonnées du pool
    metadata_path = pool_dir / 'pool_metadata.json'
    if not metadata_path.exists():
        raise FileNotFoundError(f"Métadonnées du pool non trouvées: {metadata_path}")
    
    with open(metadata_path, 'r') as f:
        pool_metadata = json.load(f)
    
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
        original_groups = list(files_by_original.values())
        np.random.shuffle(original_groups)
        
        # Calculer les indices de split
        n_groups = len(original_groups)
        n_train = int(n_groups * train_split)
        n_val = int(n_groups * val_split)
        # Le reste va dans test
        
        # Assigner les splits
        train_groups = original_groups[:n_train]
        val_groups = original_groups[n_train:n_train + n_val]
        test_groups = original_groups[n_train + n_val:]
        
        # S'assurer qu'on a au moins un groupe dans chaque split
        if len(val_groups) == 0 and n_groups >= 3:
            val_groups = [train_groups.pop()]
        if len(test_groups) == 0 and n_groups >= 3:
            test_groups = [train_groups.pop()]
        
        # Fonction pour insérer les fichiers d'un split
        def insert_split_files(groups, split_name):
            split_count = 0
            for group in groups:
                for file_info in group:
                    file_path = os.path.join(class_name, file_info['filename'])
                    is_augmented = 1 if file_info['type'] == 'augmented' else 0
                    original_file = file_info.get('source', '')
                    aug_type = file_info.get('augmentation', '')
                    
                    cursor.execute("""
                        INSERT OR IGNORE INTO audio_samples 
                        (file_path, class_name, is_augmented, original_file, augmentation_type, split)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (file_path, class_name, is_augmented, original_file, aug_type, split_name))
                    
                    split_count += 1
                    global_stats[split_name][class_name] += 1
            
            return split_count
        
        # Insérer les données
        train_count = insert_split_files(train_groups, 'train')
        val_count = insert_split_files(val_groups, 'val')
        test_count = insert_split_files(test_groups, 'test')
        
        logger.info(f"  Répartition: train={train_count}, val={val_count}, test={test_count}")
        
        # Vérifier que tous les fichiers ont été assignés
        total_assigned = train_count + val_count + test_count
        if total_assigned != class_info['total_count']:
            logger.warning(f"  Attention: {class_info['total_count'] - total_assigned} fichiers non assignés!")
    
    # Commit les changements
    conn.commit()
    
    # Afficher les statistiques finales
    logger.info("\n" + "="*50)
    logger.info("STATISTIQUES FINALES")
    logger.info("="*50)
    
    for split in ['train', 'val', 'test']:
        cursor.execute("SELECT COUNT(*) FROM audio_samples WHERE split = ?", (split,))
        total = cursor.fetchone()[0]
        
        cursor.execute("""
            SELECT COUNT(*) FROM audio_samples 
            WHERE split = ? AND is_augmented = 0
        """, (split,))
        original = cursor.fetchone()[0]
        augmented = total - original
        
        logger.info(f"\n{split.upper()}:")
        logger.info(f"  Total: {total} échantillons")
        logger.info(f"  Originaux: {original} ({original/max(total,1)*100:.1f}%)")
        logger.info(f"  Augmentés: {augmented} ({augmented/max(total,1)*100:.1f}%)")
        
        # Par classe
        cursor.execute("""
            SELECT class_name, COUNT(*) as count 
            FROM audio_samples 
            WHERE split = ?
            GROUP BY class_name
            ORDER BY class_name
        """, (split,))
        
        logger.info("  Par classe:")
        for row in cursor.fetchall():
            logger.info(f"    {row[0]}: {row[1]}")
    
    # Vérifier l'équilibre
    logger.info("\n" + "="*50)
    logger.info("VÉRIFICATION DE L'ÉQUILIBRE")
    logger.info("="*50)
    
    cursor.execute("""
        SELECT class_name, 
               SUM(CASE WHEN split = 'train' THEN 1 ELSE 0 END) as train,
               SUM(CASE WHEN split = 'val' THEN 1 ELSE 0 END) as val,
               SUM(CASE WHEN split = 'test' THEN 1 ELSE 0 END) as test,
               COUNT(*) as total
        FROM audio_samples
        GROUP BY class_name
        ORDER BY total DESC
    """)
    
    logger.info("Classe          | Train | Val  | Test | Total")
    logger.info("-" * 50)
    for row in cursor.fetchall():
        logger.info(f"{row[0]:15} | {row[1]:5} | {row[2]:4} | {row[3]:4} | {row[4]:5}")
    
    conn.close()
    
    logger.info(f"\n✅ Index équilibré créé avec succès: {output_db}")
    logger.info("\nProchaine étape: Utiliser cet index pour la pré-génération des spectrogrammes")


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
    
    # Vérifier que les splits somment à 1
    total_split = args.train_split + args.val_split + args.test_split
    if abs(total_split - 1.0) > 0.001:
        raise ValueError(f"Les splits doivent sommer à 1.0, actuellement: {total_split}")
    
    create_balanced_index(
        args.pool_dir,
        args.output_db,
        args.train_split,
        args.val_split,
        args.test_split,
        args.random_seed
    )


if __name__ == "__main__":
    main()