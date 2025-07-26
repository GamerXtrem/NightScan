#!/usr/bin/env python3
"""
Script de test pour vérifier le nouveau pipeline équilibré.
Vérifie que toutes les étapes fonctionnent correctement.
"""

import os
import sys
import argparse
import sqlite3
from pathlib import Path
import logging
import torch
from torch.utils.data import DataLoader

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Ajouter le chemin parent
sys.path.append(str(Path(__file__).parent.parent))
from audio_dataset_scalable import AudioDatasetScalable


def test_index_database(index_db: str):
    """Teste la base de données d'index."""
    logger.info(f"\n=== Test de la base de données: {index_db} ===")
    
    conn = sqlite3.connect(index_db)
    cursor = conn.cursor()
    
    # Vérifier la structure
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in cursor.fetchall()]
    logger.info(f"Tables trouvées: {tables}")
    
    # Statistiques par split
    logger.info("\nStatistiques par split:")
    cursor.execute("""
        SELECT split, COUNT(*) as count, COUNT(DISTINCT class_name) as classes
        FROM audio_samples
        GROUP BY split
        ORDER BY split
    """)
    
    for row in cursor.fetchall():
        logger.info(f"  {row[0]}: {row[1]} échantillons, {row[2]} classes")
    
    # Vérifier l'équilibre des classes
    logger.info("\nÉquilibre par classe (train):")
    cursor.execute("""
        SELECT class_name, COUNT(*) as count
        FROM audio_samples
        WHERE split = 'train'
        GROUP BY class_name
        ORDER BY count DESC
        LIMIT 10
    """)
    
    for row in cursor.fetchall():
        logger.info(f"  {row[0]}: {row[1]} échantillons")
    
    # Vérifier les chemins
    cursor.execute("SELECT file_path FROM audio_samples LIMIT 5")
    logger.info("\nExemples de chemins:")
    for row in cursor.fetchall():
        logger.info(f"  {row[0]}")
    
    conn.close()
    return True


def test_dataset_loading(index_db: str, audio_root: Path, spectrogram_cache_dir: Path = None):
    """Teste le chargement du dataset."""
    logger.info(f"\n=== Test du dataset ===")
    
    try:
        # Créer le dataset pour train
        dataset = AudioDatasetScalable(
            index_db=index_db,
            audio_root=audio_root,
            split='train',
            spectrogram_cache_dir=spectrogram_cache_dir
        )
        
        logger.info(f"Dataset créé avec succès!")
        logger.info(f"  Nombre d'échantillons: {len(dataset)}")
        logger.info(f"  Nombre de classes: {dataset.num_classes}")
        logger.info(f"  Classes: {dataset.class_names[:5]}... (affichage limité)")
        
        # Tester le chargement d'un échantillon
        logger.info("\nTest de chargement d'échantillons...")
        for i in range(min(3, len(dataset))):
            try:
                spec, label = dataset[i]
                logger.info(f"  Échantillon {i}: shape={spec.shape}, label={label} ({dataset.class_names[label]})")
            except Exception as e:
                logger.error(f"  Erreur échantillon {i}: {e}")
        
        # Créer un mini dataloader
        logger.info("\nTest du DataLoader...")
        loader = DataLoader(
            dataset,
            batch_size=4,
            shuffle=False,
            num_workers=0  # 0 pour éviter les problèmes de multiprocessing en test
        )
        
        # Charger un batch
        for batch_idx, (data, targets) in enumerate(loader):
            logger.info(f"  Batch {batch_idx}: data.shape={data.shape}, targets.shape={targets.shape}")
            break
        
        dataset.close()
        return True
        
    except Exception as e:
        logger.error(f"Erreur lors du test du dataset: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_spectrogram_cache(spectrogram_cache_dir: Path):
    """Vérifie la structure du cache de spectrogrammes."""
    logger.info(f"\n=== Test du cache de spectrogrammes ===")
    
    if not spectrogram_cache_dir.exists():
        logger.warning(f"Le répertoire de cache n'existe pas: {spectrogram_cache_dir}")
        return False
    
    # Compter les fichiers par split
    for split in ['train', 'val', 'test']:
        split_dir = spectrogram_cache_dir / split
        if split_dir.exists():
            n_files = sum(1 for _ in split_dir.rglob("*.npy"))
            n_classes = len(list(split_dir.iterdir()))
            logger.info(f"  {split}: {n_files} fichiers .npy dans {n_classes} classes")
            
            # Exemples de fichiers
            examples = list(split_dir.rglob("*.npy"))[:3]
            if examples:
                logger.info(f"    Exemples:")
                for ex in examples:
                    relative = ex.relative_to(spectrogram_cache_dir)
                    logger.info(f"      {relative}")
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Test du nouveau pipeline équilibré")
    
    parser.add_argument('--index-db', type=str, required=True,
                       help='Base SQLite créée par create_balanced_index.py')
    parser.add_argument('--audio-root', type=Path, required=True,
                       help='Répertoire racine des fichiers audio (pool augmenté)')
    parser.add_argument('--spectrogram-cache-dir', type=Path, default=None,
                       help='Répertoire des spectrogrammes pré-générés')
    parser.add_argument('--test-loading', action='store_true',
                       help='Tester le chargement complet du dataset')
    
    args = parser.parse_args()
    
    logger.info("=== Test du nouveau pipeline équilibré ===")
    logger.info(f"Index DB: {args.index_db}")
    logger.info(f"Audio root: {args.audio_root}")
    if args.spectrogram_cache_dir:
        logger.info(f"Spectrogram cache: {args.spectrogram_cache_dir}")
    
    # Test 1: Vérifier la base de données
    if not Path(args.index_db).exists():
        logger.error(f"La base de données n'existe pas: {args.index_db}")
        return
    
    test_index_database(args.index_db)
    
    # Test 2: Vérifier le cache de spectrogrammes
    if args.spectrogram_cache_dir:
        test_spectrogram_cache(args.spectrogram_cache_dir)
    
    # Test 3: Tester le chargement du dataset
    if args.test_loading:
        if not args.audio_root.exists():
            logger.error(f"Le répertoire audio n'existe pas: {args.audio_root}")
            return
        
        success = test_dataset_loading(args.index_db, args.audio_root, args.spectrogram_cache_dir)
        if success:
            logger.info("\n✅ Tous les tests ont réussi!")
        else:
            logger.error("\n❌ Certains tests ont échoué")
    else:
        logger.info("\n💡 Utilisez --test-loading pour tester le chargement complet du dataset")
    
    logger.info("\n=== Commande d'entraînement suggérée ===")
    logger.info(f"""python train_audio_large_scale.py \\
    --index-db {args.index_db} \\
    --audio-root {args.audio_root} \\
    --spectrogram-cache-dir {args.spectrogram_cache_dir or 'data/spectrograms_balanced'} \\
    --num-classes 10 \\
    --model efficientnet-b1 \\
    --batch-size 128 \\
    --epochs 50""")


if __name__ == "__main__":
    main()