#!/usr/bin/env python3
"""
Version simplifiée du script de pré-génération des spectrogrammes.
Utilise l'index équilibré créé par create_balanced_index.py.
Ne fait AUCUNE augmentation - génère seulement les spectrogrammes.
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
from multiprocessing import Pool
import gc
from typing import Tuple, Dict
import torchaudio
import torchaudio.transforms as T

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Ajouter le chemin parent
sys.path.append(str(Path(__file__).parent.parent))
from spectrogram_config import get_config_for_animal


def generate_spectrogram(audio_path: Path, config: dict) -> torch.Tensor:
    """
    Génère un spectrogramme sans augmentation.
    """
    # Charger l'audio
    waveform, sr = torchaudio.load(str(audio_path))
    
    # Convertir en mono
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    
    # Rééchantillonner si nécessaire
    if sr != config['sample_rate']:
        resampler = T.Resample(sr, config['sample_rate'])
        waveform = resampler(waveform)
    
    # Ajuster la durée
    target_length = int(config['sample_rate'] * config['duration'])
    if waveform.shape[1] < target_length:
        pad = target_length - waveform.shape[1]
        waveform = torch.nn.functional.pad(waveform, (0, pad))
    elif waveform.shape[1] > target_length:
        waveform = waveform[:, :target_length]
    
    # Créer le spectrogramme mel
    mel_transform = T.MelSpectrogram(
        sample_rate=config['sample_rate'],
        n_fft=config['n_fft'],
        hop_length=config['hop_length'],
        n_mels=config['n_mels'],
        f_min=config['fmin'],
        f_max=config['fmax']
    )
    
    mel_spec = mel_transform(waveform)
    
    # Convertir en dB
    db_transform = T.AmplitudeToDB(top_db=config['top_db'])
    mel_spec_db = db_transform(mel_spec)
    
    # Normaliser
    mel_spec_db = (mel_spec_db - mel_spec_db.mean()) / (mel_spec_db.std() + 1e-6)
    
    # Convertir en 3 canaux
    if mel_spec_db.dim() == 3:
        mel_spec_db = mel_spec_db.squeeze(0)
    mel_spec_db = mel_spec_db.unsqueeze(0).repeat(3, 1, 1)
    
    return mel_spec_db


def process_single_sample(args: Tuple[int, str, Path, str, str, Path, dict]) -> Dict[str, any]:
    """
    Traite un seul échantillon pour générer son spectrogramme.
    """
    sample_id, file_path, pool_root, class_name, split, output_dir, config_dict = args
    
    try:
        # Chemins
        audio_path = pool_root / file_path
        
        # Vérifier que le fichier existe
        if not audio_path.exists():
            return {
                'status': 'error',
                'file': str(file_path),
                'error': 'Fichier audio introuvable'
            }
        
        # Créer les répertoires de sortie
        class_output_dir = output_dir / split / class_name
        class_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Générer le nom de sortie
        output_name = Path(file_path).stem + ".npy"
        output_path = class_output_dir / output_name
        
        # Générer le spectrogramme si nécessaire
        if not output_path.exists():
            spec = generate_spectrogram(audio_path, config_dict)
            np.save(output_path, spec.detach().cpu().numpy())
            
            return {
                'status': 'success',
                'file': str(file_path),
                'generated': True
            }
        else:
            return {
                'status': 'success',
                'file': str(file_path),
                'generated': False
            }
        
    except Exception as e:
        logger.error(f"Erreur pour {file_path}: {type(e).__name__}: {str(e)}")
        return {
            'status': 'error',
            'file': str(file_path),
            'error': f"{type(e).__name__}: {str(e)}"
        }


def pregenerate_spectrograms(
    index_db: str,
    pool_root: Path,
    output_dir: Path,
    num_workers: int = 4,
    animal_type: str = 'general'
):
    """
    Pré-génère tous les spectrogrammes depuis l'index équilibré.
    """
    logger.info(f"Pré-génération des spectrogrammes")
    logger.info(f"Index: {index_db}")
    logger.info(f"Pool audio: {pool_root}")
    logger.info(f"Sortie: {output_dir}")
    logger.info(f"Workers: {num_workers}")
    
    # Créer le répertoire de sortie
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Obtenir la configuration
    config = get_config_for_animal(animal_type)
    config_dict = {
        'sample_rate': config.sample_rate,
        'n_fft': config.n_fft,
        'hop_length': config.hop_length,
        'n_mels': config.n_mels,
        'fmin': config.fmin,
        'fmax': config.fmax,
        'duration': config.duration,
        'top_db': config.top_db
    }
    
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
    
    # Préparer les arguments
    args_list = []
    for sample_id, file_path, class_name, split in samples:
        args = (sample_id, file_path, pool_root, class_name, split, output_dir, config_dict)
        args_list.append(args)
    
    # Traiter en parallèle
    logger.info("Début du traitement...")
    
    results = []
    errors = []
    
    if num_workers > 1:
        with Pool(num_workers) as pool:
            for result in tqdm(
                pool.imap_unordered(process_single_sample, args_list, chunksize=10),
                total=len(args_list),
                desc="Génération des spectrogrammes"
            ):
                results.append(result)
                if result['status'] == 'error':
                    errors.append(result)
    else:
        for args in tqdm(args_list, desc="Génération des spectrogrammes"):
            result = process_single_sample(args)
            results.append(result)
            if result['status'] == 'error':
                errors.append(result)
    
    # Statistiques
    successes = sum(1 for r in results if r['status'] == 'success')
    generated = sum(1 for r in results if r.get('generated', False))
    
    logger.info(f"\nGénération terminée:")
    logger.info(f"  Échantillons traités: {successes}/{len(samples)}")
    logger.info(f"  Nouveaux spectrogrammes: {generated}")
    logger.info(f"  Erreurs: {len(errors)}")
    
    if errors:
        logger.warning(f"\nErreurs détectées:")
        for err in errors[:10]:
            logger.warning(f"  {err['file']}: {err['error']}")
        if len(errors) > 10:
            logger.warning(f"  ... et {len(errors) - 10} autres")
    
    # Statistiques par split
    for split in ['train', 'val', 'test']:
        split_dir = output_dir / split
        if split_dir.exists():
            n_files = sum(1 for _ in split_dir.rglob("*.npy"))
            logger.info(f"  {split}: {n_files} fichiers .npy")


def main():
    parser = argparse.ArgumentParser(description="Pré-génération simplifiée des spectrogrammes")
    
    parser.add_argument('--index-db', type=str, required=True,
                       help='Base SQLite créée par create_balanced_index.py')
    parser.add_argument('--pool-root', type=Path, required=True,
                       help='Répertoire racine du pool augmenté')
    parser.add_argument('--output-dir', type=Path, default=Path('data/spectrograms_balanced'),
                       help='Répertoire de sortie pour les spectrogrammes')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Nombre de workers parallèles')
    parser.add_argument('--animal-type', type=str, default='general',
                       help='Type d\'animal pour la configuration')
    
    args = parser.parse_args()
    
    pregenerate_spectrograms(
        args.index_db,
        args.pool_root,
        args.output_dir,
        args.num_workers,
        args.animal_type
    )


if __name__ == "__main__":
    main()