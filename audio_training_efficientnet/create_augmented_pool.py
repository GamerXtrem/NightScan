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
from typing import Dict, List, Tuple, Optional
import random
import gc
import psutil
import time

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
    # Travailler sur une copie pour éviter de modifier l'original
    waveform = waveform.detach().clone()
    
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
    max_augmentations_per_sample: int = 20,
    batch_size: int = 5,
    save_detailed_metadata: bool = False,
    specific_class: Optional[str] = None,
    low_memory: bool = False
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
    logger.info(f"Batch size: {batch_size} fichiers à la fois")
    if low_memory:
        logger.info("Mode FAIBLE MÉMOIRE activé")
    
    # Vérifier la mémoire initiale
    memory_info = psutil.virtual_memory()
    logger.info(f"Mémoire disponible: {memory_info.available / (1024**3):.1f} GB / {memory_info.total / (1024**3):.1f} GB")
    
    # Créer le répertoire de sortie
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Scanner les classes
    if specific_class:
        class_dir = audio_root / specific_class
        if not class_dir.exists() or not class_dir.is_dir():
            logger.error(f"Classe spécifique '{specific_class}' non trouvée dans {audio_root}")
            return pool_stats
        classes = [class_dir]
        logger.info(f"Traitement de la classe spécifique: {specific_class}")
    else:
        classes = [d for d in audio_root.iterdir() if d.is_dir() and not d.name.startswith('.')]
        logger.info(f"Nombre de classes trouvées: {len(classes)}")
    
    pool_stats = {
        'timestamp': datetime.now().isoformat(),
        'target_samples_per_class': target_samples_per_class,
        'classes_processed': 0,
        'total_original': 0,
        'total_created': 0
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
        try:
            wav_files = list(class_dir.glob("*.wav"))
            mp3_files = list(class_dir.glob("*.mp3"))
            audio_files = wav_files + mp3_files
            logger.debug(f"  Trouvé: {len(wav_files)} WAV, {len(mp3_files)} MP3")
        except Exception as e:
            logger.error(f"  Erreur lors du listage des fichiers: {e}")
            continue
            
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
        created_files = [] if save_detailed_metadata else None
        files_created_count = 0
        
        # Traiter par batch pour gérer la mémoire
        for batch_start in range(0, len(audio_files), batch_size):
            batch_end = min(batch_start + batch_size, len(audio_files))
            batch_files = audio_files[batch_start:batch_end]
            
            # Vérifier la mémoire disponible
            memory_percent = psutil.virtual_memory().percent
            if memory_percent > 85:
                logger.warning(f"  Mémoire à {memory_percent}%, pause de 5 secondes...")
                gc.collect()
                time.sleep(5)
            
            for file_idx, audio_file in enumerate(tqdm(batch_files, desc=f"  {class_name} (batch {batch_start//batch_size + 1})")):
                if file_idx == 0 and batch_start == 0:
                    logger.debug(f"  Traitement du premier fichier: {audio_file.name}")
                # 1. Copier l'original
                try:
                    output_name = f"{audio_file.stem}_original{audio_file.suffix}"
                    output_path = class_output_dir / output_name
                    shutil.copy2(audio_file, output_path)
                    files_created_count += 1
                except Exception as e:
                    logger.error(f"  Erreur copie {audio_file.name}: {e}")
                    continue
                if save_detailed_metadata:
                    created_files.append({
                        'filename': output_name,
                        'type': 'original',
                        'source': audio_file.name
                    })
                
                # 2. Créer les augmentations si nécessaire
                if augmentations_per_sample > 0:
                    try:
                        # Charger l'audio avec gestion mémoire
                        # Utiliser backend sox_io pour une meilleure gestion mémoire
                        try:
                            waveform, sr = torchaudio.load(str(audio_file))
                        except Exception as load_error:
                            logger.error(f"  Impossible de charger {audio_file.name}: {load_error}")
                            continue
                        
                        # Convertir en mono si nécessaire
                        if waveform.shape[0] > 1:
                            waveform = waveform.mean(dim=0, keepdim=True)
                        
                        for aug_idx in range(augmentations_per_sample):
                            # Sélectionner le type d'augmentation
                            aug_type = augmentation_types[aug_idx % len(augmentation_types)]
                            
                            # Varier la force de l'augmentation
                            strength = 0.3 + (aug_idx / max(augmentations_per_sample - 1, 1)) * 0.7
                            
                            if low_memory:
                                # Mode faible mémoire: recharger l'audio à chaque fois
                                waveform_aug, sr_aug = torchaudio.load(str(audio_file))
                                if waveform_aug.shape[0] > 1:
                                    waveform_aug = waveform_aug.mean(dim=0, keepdim=True)
                                aug_waveform = apply_audio_augmentation(waveform_aug, sr_aug, aug_type, strength)
                                del waveform_aug
                            else:
                                # Mode normal: utiliser le waveform déjà chargé
                                aug_waveform = apply_audio_augmentation(waveform, sr, aug_type, strength)
                            
                            # Sauvegarder
                            aug_name = f"{audio_file.stem}_aug{aug_idx+1:03d}_{aug_type}.wav"
                            aug_path = class_output_dir / aug_name
                            torchaudio.save(str(aug_path), aug_waveform, sr if not low_memory else sr_aug)
                            files_created_count += 1
                            
                            if save_detailed_metadata:
                                created_files.append({
                                    'filename': aug_name,
                                    'type': 'augmented',
                                    'source': audio_file.name,
                                    'augmentation': aug_type,
                                    'strength': strength
                                })
                            
                            # Libérer la mémoire de l'augmentation
                            del aug_waveform
                            
                            # En mode faible mémoire, forcer le gc après chaque augmentation
                            if low_memory:
                                gc.collect()
                        
                        # Libérer la mémoire du waveform original
                        del waveform
                        gc.collect()
                        
                    except Exception as e:
                        logger.error(f"  Erreur lors du traitement de {audio_file.name}: {e}")
                        continue
            
            # Forcer la libération de mémoire après chaque batch
            gc.collect()
        
        # Statistiques pour cette classe
        class_stats = {
            'original_count': original_count,
            'augmentations_per_sample': augmentations_per_sample,
            'total_count': files_created_count
        }
        
        if save_detailed_metadata:
            class_stats['files'] = created_files
            
        # Sauvegarder les métadonnées de la classe immédiatement
        class_metadata_path = output_dir / 'class_metadata' / f'{class_name}.json'
        class_metadata_path.parent.mkdir(exist_ok=True)
        with open(class_metadata_path, 'w') as f:
            json.dump({
                'class_name': class_name,
                'stats': class_stats,
                'timestamp': datetime.now().isoformat()
            }, f, indent=2)
        
        # Mettre à jour les statistiques globales
        pool_stats['classes_processed'] += 1
        pool_stats['total_original'] += original_count
        pool_stats['total_created'] += files_created_count
        
        logger.info(f"  Créé {files_created_count} fichiers pour {class_name}")
        
        # Libérer la mémoire
        if save_detailed_metadata and created_files:
            created_files.clear()
        del class_stats
        
        # Forcer la libération de la mémoire
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Pause pour permettre au GC de nettoyer
        time.sleep(0.5)
        
        # Afficher l'utilisation mémoire
        memory_info = psutil.virtual_memory()
        logger.info(f"  Mémoire utilisée: {memory_info.percent:.1f}% ({memory_info.used / (1024**3):.1f}GB / {memory_info.total / (1024**3):.1f}GB)")
    
    # Sauvegarder les métadonnées
    metadata_path = output_dir / 'pool_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(pool_stats, f, indent=2)
    
    logger.info(f"\nPool augmenté créé avec succès!")
    logger.info(f"Métadonnées sauvegardées dans: {metadata_path}")
    
    # Résumé
    logger.info(f"\nRésumé:")
    logger.info(f"  Classes traitées: {pool_stats['classes_processed']}")
    logger.info(f"  Fichiers originaux: {pool_stats['total_original']}")
    logger.info(f"  Fichiers dans le pool: {pool_stats['total_created']}")
    logger.info(f"  Ratio d'augmentation moyen: {pool_stats['total_created']/max(pool_stats['total_original'], 1):.1f}x")
    
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
    parser.add_argument('--batch-size', type=int, default=5,
                       help='Nombre de fichiers à traiter simultanément')
    parser.add_argument('--save-detailed-metadata', action='store_true',
                       help='Sauvegarder les métadonnées détaillées (liste de tous les fichiers)')
    parser.add_argument('--specific-class', type=str, default=None,
                       help='Traiter uniquement cette classe spécifique')
    parser.add_argument('--low-memory', action='store_true',
                       help='Mode faible mémoire: recharge l\'audio pour chaque augmentation')
    
    args = parser.parse_args()
    
    create_augmented_pool(
        args.audio_root,
        args.output_dir,
        args.target_samples,
        args.max_augmentations,
        args.batch_size,
        args.save_detailed_metadata,
        args.specific_class,
        args.low_memory
    )


if __name__ == "__main__":
    main()