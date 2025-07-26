#!/usr/bin/env python3
"""
Crée un pool augmenté équilibré pour chaque classe AVANT la division train/val/test.
Chaque classe aura exactement 500 échantillons (ou moins si pas assez d'originaux).
"""

import os
import sys
import argparse
import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T
from pathlib import Path
import logging
from tqdm import tqdm
import json
from datetime import datetime
import shutil
from typing import Dict, List, Tuple
import random

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Ajouter le chemin parent
sys.path.append(str(Path(__file__).parent.parent))
from spectrogram_config import get_config_for_animal


def apply_audio_augmentation(waveform: torch.Tensor, sr: int, augmentation_type: str, strength: float = 1.0) -> torch.Tensor:
    """
    Applique une augmentation audio spécifique.
    
    Args:
        waveform: Signal audio
        sr: Sample rate
        augmentation_type: Type d'augmentation ('time_stretch', 'noise', 'volume')
        strength: Force de l'augmentation (0-1)
    
    Returns:
        Signal augmenté
    """
    waveform = waveform.clone()
    
    if augmentation_type == 'time_stretch':
        # Étirement temporel
        rate = 1.0 + (strength - 0.5) * 0.4  # 0.8x à 1.2x
        if rate != 1.0:
            waveform = T.Resample(sr, int(sr * rate))(waveform)
            waveform = T.Resample(int(sr * rate), sr)(waveform)
    
    elif augmentation_type == 'noise':
        # Ajout de bruit
        noise_level = 0.002 * strength
        noise = torch.randn_like(waveform) * noise_level
        waveform = waveform + noise
    
    elif augmentation_type == 'volume':
        # Variation de volume
        volume_factor = 0.5 + strength * 0.5  # 0.5x à 1.0x
        waveform = waveform * volume_factor
    
    elif augmentation_type == 'combined':
        # Combinaison de plusieurs augmentations
        # Time stretch léger
        rate = 1.0 + (strength - 0.5) * 0.2  # 0.9x à 1.1x
        if rate != 1.0:
            waveform = T.Resample(sr, int(sr * rate))(waveform)
            waveform = T.Resample(int(sr * rate), sr)(waveform)
        
        # Bruit léger
        noise_level = 0.001 * strength
        noise = torch.randn_like(waveform) * noise_level
        waveform = waveform + noise
        
        # Volume
        volume_factor = 0.7 + strength * 0.3  # 0.7x à 1.0x
        waveform = waveform * volume_factor
    
    return waveform


def create_augmented_pool(
    audio_root: Path,
    output_dir: Path,
    target_samples_per_class: int = 500,
    max_augmentations_per_sample: int = 20
) -> Dict:
    """
    Crée un pool augmenté équilibré pour toutes les classes.
    
    Args:
        audio_root: Répertoire racine des fichiers audio originaux
        output_dir: Répertoire de sortie pour le pool augmenté
        target_samples_per_class: Nombre cible d'échantillons par classe
        max_augmentations_per_sample: Maximum d'augmentations par échantillon
    
    Returns:
        Dictionnaire avec les statistiques du pool créé
    """
    logger.info(f"Création du pool augmenté")
    logger.info(f"Source: {audio_root}")
    logger.info(f"Destination: {output_dir}")
    logger.info(f"Cible: {target_samples_per_class} échantillons par classe")
    
    # Créer le répertoire de sortie
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Scanner les classes
    classes = [d for d in audio_root.iterdir() if d.is_dir() and not d.name.startswith('.')]
    logger.info(f"Nombre de classes trouvées: {len(classes)}")
    
    pool_stats = {
        'timestamp': datetime.now().isoformat(),
        'target_samples_per_class': target_samples_per_class,
        'classes': {}
    }
    
    # Types d'augmentation disponibles
    augmentation_types = ['time_stretch', 'noise', 'volume', 'combined']
    
    # Traiter chaque classe
    for class_dir in sorted(classes):
        class_name = class_dir.name
        logger.info(f"\nTraitement de la classe: {class_name}")
        
        # Créer le répertoire de sortie pour cette classe
        class_output_dir = output_dir / class_name
        class_output_dir.mkdir(exist_ok=True)
        
        # Lister tous les fichiers audio
        audio_files = list(class_dir.glob("*.wav")) + list(class_dir.glob("*.mp3"))
        original_count = len(audio_files)
        
        if original_count == 0:
            logger.warning(f"  Aucun fichier audio trouvé pour {class_name}")
            continue
        
        # Calculer le nombre d'augmentations nécessaires
        if original_count >= target_samples_per_class:
            # Pas besoin d'augmentation
            augmentations_per_sample = 0
            total_target = original_count
            logger.info(f"  {original_count} échantillons - Pas d'augmentation nécessaire")
        else:
            # Calculer le multiplicateur nécessaire
            multiplier = target_samples_per_class / original_count
            augmentations_per_sample = min(int(multiplier) - 1, max_augmentations_per_sample)
            
            # Si on n'atteint pas exactement la cible, ajouter quelques augmentations supplémentaires
            total_with_aug = original_count * (augmentations_per_sample + 1)
            if total_with_aug < target_samples_per_class:
                extra_needed = target_samples_per_class - total_with_aug
                extra_per_sample = extra_needed / original_count
                if extra_per_sample > 0.5:  # Si on a besoin de plus de 0.5 aug par sample
                    augmentations_per_sample += 1
            
            total_target = original_count * (augmentations_per_sample + 1)
            logger.info(f"  {original_count} échantillons → {augmentations_per_sample} augmentations/échantillon = ~{total_target} total")
        
        # Copier les originaux et créer les augmentations
        created_files = []
        
        for idx, audio_file in enumerate(tqdm(audio_files, desc=f"  {class_name}")):
            # 1. Copier l'original
            output_name = f"{audio_file.stem}_original{audio_file.suffix}"
            output_path = class_output_dir / output_name
            shutil.copy2(audio_file, output_path)
            created_files.append({
                'filename': output_name,
                'type': 'original',
                'source': audio_file.name
            })
            
            # 2. Créer les augmentations si nécessaire
            if augmentations_per_sample > 0:
                # Charger l'audio une seule fois
                waveform, sr = torchaudio.load(str(audio_file))
                
                # Convertir en mono si nécessaire
                if waveform.shape[0] > 1:
                    waveform = waveform.mean(dim=0, keepdim=True)
                
                for aug_idx in range(augmentations_per_sample):
                    # Sélectionner le type d'augmentation
                    aug_type = augmentation_types[aug_idx % len(augmentation_types)]
                    
                    # Varier la force de l'augmentation
                    strength = 0.3 + (aug_idx / max(augmentations_per_sample - 1, 1)) * 0.7
                    
                    # Appliquer l'augmentation
                    aug_waveform = apply_audio_augmentation(waveform, sr, aug_type, strength)
                    
                    # Sauvegarder
                    aug_name = f"{audio_file.stem}_aug{aug_idx+1:03d}_{aug_type}.wav"
                    aug_path = class_output_dir / aug_name
                    torchaudio.save(str(aug_path), aug_waveform, sr)
                    
                    created_files.append({
                        'filename': aug_name,
                        'type': 'augmented',
                        'source': audio_file.name,
                        'augmentation': aug_type,
                        'strength': strength
                    })
        
        # Statistiques pour cette classe
        pool_stats['classes'][class_name] = {
            'original_count': original_count,
            'augmentations_per_sample': augmentations_per_sample,
            'total_count': len(created_files),
            'files': created_files
        }
        
        logger.info(f"  Créé {len(created_files)} fichiers pour {class_name}")
    
    # Sauvegarder les métadonnées
    metadata_path = output_dir / 'pool_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(pool_stats, f, indent=2)
    
    logger.info(f"\nPool augmenté créé avec succès!")
    logger.info(f"Métadonnées sauvegardées dans: {metadata_path}")
    
    # Résumé
    total_original = sum(stats['original_count'] for stats in pool_stats['classes'].values())
    total_created = sum(stats['total_count'] for stats in pool_stats['classes'].values())
    logger.info(f"\nRésumé:")
    logger.info(f"  Classes traitées: {len(pool_stats['classes'])}")
    logger.info(f"  Fichiers originaux: {total_original}")
    logger.info(f"  Fichiers dans le pool: {total_created}")
    logger.info(f"  Ratio d'augmentation moyen: {total_created/max(total_original, 1):.1f}x")
    
    return pool_stats


def main():
    parser = argparse.ArgumentParser(description="Créer un pool augmenté équilibré avant division train/val/test")
    
    parser.add_argument('--audio-root', type=Path, required=True,
                       help='Répertoire racine des fichiers audio originaux')
    parser.add_argument('--output-dir', type=Path, default=Path('data/augmented_pool'),
                       help='Répertoire de sortie pour le pool augmenté')
    parser.add_argument('--target-samples', type=int, default=500,
                       help='Nombre cible d\'échantillons par classe')
    parser.add_argument('--max-augmentations', type=int, default=20,
                       help='Maximum d\'augmentations par échantillon')
    
    args = parser.parse_args()
    
    create_augmented_pool(
        args.audio_root,
        args.output_dir,
        args.target_samples,
        args.max_augmentations
    )


if __name__ == "__main__":
    main()