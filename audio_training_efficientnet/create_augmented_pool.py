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
# Désactiver MKLDNN pour éviter les fuites mémoire avec torchaudio
# Voir https://github.com/pytorch/audio/issues/2338
from torch.backends import mkldnn
mkldnn.m.set_flags(False)

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
from multiprocessing import Pool
from functools import partial
import sys
# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)




def count_torch_tensors():
    """Compte le nombre de tensors PyTorch actifs en mémoire."""
    import gc
    count = 0
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj):
                count += 1
        except:
            pass
    return count


def log_memory_state(context: str, verbose: bool = False):
    """Log l'état détaillé de la mémoire."""
    gc.collect()
    
    memory_info = psutil.virtual_memory()
    process = psutil.Process()
    process_memory = process.memory_info()
    
    logger.info(f"[MEM {context}]")
    logger.info(f"  Système: {memory_info.percent:.1f}% ({memory_info.used/(1024**3):.2f}/{memory_info.total/(1024**3):.2f} GB)")
    logger.info(f"  Processus: RSS={process_memory.rss/(1024**3):.2f} GB, VMS={process_memory.vms/(1024**3):.2f} GB")
    
    if verbose:
        # Compter les tensors
        tensor_count = count_torch_tensors()
        logger.info(f"  Tensors PyTorch actifs: {tensor_count}")
        
        # Obtenir les plus gros objets
        import sys
        all_objects = gc.get_objects()
        logger.info(f"  Total objets Python: {len(all_objects)}")
        
        # Chercher les gros objets
        big_objects = []
        for obj in all_objects:
            try:
                size = sys.getsizeof(obj)
                if size > 1024 * 1024:  # Plus de 1 MB
                    big_objects.append((size, type(obj).__name__))
            except:
                pass
        
        if big_objects:
            big_objects.sort(reverse=True)
            logger.info("  Gros objets (>1MB):")
            for size, typename in big_objects[:5]:
                logger.info(f"    {typename}: {size/(1024**2):.1f} MB")

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


def process_file_low_memory(args):
    """
    Traite un fichier audio en mode low_memory (utilise subprocess).
    Conçu pour être utilisé avec multiprocessing.Pool.
    """
    audio_file, class_output_dir, augmentations_per_sample, augmentation_types, save_detailed_metadata = args
    
    import subprocess
    import json as json_module
    
    files_created = 0
    metadata_entries = []
    
    # 1. Copier l'original
    try:
        output_name = f"{audio_file.stem}_original{audio_file.suffix}"
        output_path = class_output_dir / output_name
        shutil.copy2(str(audio_file), str(output_path))
        files_created += 1
        
        if save_detailed_metadata:
            metadata_entries.append({
                'filename': output_name,
                'type': 'original',
                'source': audio_file.name
            })
    except Exception as e:
        return {
            'success': False,
            'error': f"Erreur copie {audio_file.name}: {e}",
            'files_created': 0,
            'metadata': []
        }
    
    # 2. Créer les augmentations si nécessaire
    if augmentations_per_sample > 0:
        # Préparer toutes les augmentations
        augmentations = []
        for aug_idx in range(augmentations_per_sample):
            aug_type = augmentation_types[aug_idx % len(augmentation_types)]
            strength = 0.3 + (aug_idx / max(augmentations_per_sample - 1, 1)) * 0.7
            aug_name = f"{audio_file.stem}_aug{aug_idx+1:03d}_{aug_type}.wav"
            aug_path = class_output_dir / aug_name
            
            augmentations.append({
                'output_path': str(aug_path),
                'aug_type': aug_type,
                'strength': strength
            })
        
        # Préparer les paramètres pour le subprocess
        params = {
            'input_path': str(audio_file),
            'augmentations': augmentations
        }
        
        # Appeler le processeur avec toutes les augmentations
        processor_path = Path(__file__).parent / 'process_single_augmentation.py'
        
        # Ajouter retry pour gérer les erreurs transitoires
        max_retries = 2
        for retry in range(max_retries):
            try:
                result = subprocess.run(
                    [sys.executable, str(processor_path)],
                    input=json_module.dumps(params),
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                
                if result.returncode != 0:
                    error_msg = result.stderr.strip() if result.stderr else "Code de retour non-zéro"
                    if retry < max_retries - 1:
                        time.sleep(1)
                        continue
                    else:
                        return {
                            'success': False,
                            'error': f"Erreur subprocess après {max_retries} essais: {error_msg}",
                            'files_created': files_created,
                            'metadata': metadata_entries
                        }
                
                # Vérifier le résultat
                try:
                    response = json_module.loads(result.stdout)
                    if response['status'] == 'success':
                        # Compter les succès
                        for result_item in response.get('results', []):
                            if result_item['status'] == 'success':
                                files_created += 1
                                
                                if save_detailed_metadata:
                                    aug_filename = Path(result_item['output_path']).name
                                    parts = aug_filename.replace(audio_file.stem + '_', '').replace('.wav', '').split('_')
                                    aug_type_from_name = '_'.join(parts[1:]) if len(parts) > 1 else 'unknown'
                                    
                                    metadata_entries.append({
                                        'filename': aug_filename,
                                        'type': 'augmented',
                                        'source': audio_file.name,
                                        'augmentation': aug_type_from_name
                                    })
                        break
                    else:
                        return {
                            'success': False,
                            'error': f"Erreur batch augmentation: {response.get('error', 'Unknown')}",
                            'files_created': files_created,
                            'metadata': metadata_entries
                        }
                except json_module.JSONDecodeError:
                    if retry < max_retries - 1:
                        time.sleep(1)
                        continue
                    return {
                        'success': False,
                        'error': f"Réponse subprocess invalide",
                        'files_created': files_created,
                        'metadata': metadata_entries
                    }
                break
                
            except subprocess.TimeoutExpired:
                return {
                    'success': False,
                    'error': "Timeout subprocess après 30 secondes",
                    'files_created': files_created,
                    'metadata': metadata_entries
                }
            except Exception as e:
                return {
                    'success': False,
                    'error': f"Erreur subprocess inattendue: {e}",
                    'files_created': files_created,
                    'metadata': metadata_entries
                }
    
    return {
        'success': True,
        'files_created': files_created,
        'metadata': metadata_entries
    }


def create_augmented_pool(
    audio_root: Path,
    output_dir: Path,
    target_samples_per_class: int = 500,
    max_augmentations_per_sample: int = 20,
    batch_size: int = 5,
    save_detailed_metadata: bool = False,
    specific_class: Optional[str] = None,
    low_memory: bool = False,
    debug: bool = False,
    workers: int = 1,
    ultra_debug: bool = False
) -> Dict:
    """
    Crée un pool augmenté équilibré pour toutes les classes.
    
    Args:
        audio_root: Répertoire racine des fichiers audio originaux
        output_dir: Répertoire de sortie pour le pool augmenté
        target_samples_per_class: Nombre cible d'échantillons par classe
        max_augmentations_per_sample: Maximum d'augmentations par échantillon
        batch_size: Nombre de fichiers à traiter par batch
        save_detailed_metadata: Sauvegarder les métadonnées détaillées
        specific_class: Traiter uniquement cette classe
        low_memory: Mode faible mémoire (utilise subprocess)
        debug: Mode debug avec logs détaillés
        workers: Nombre de processus parallèles (seulement en mode low_memory)
        ultra_debug: Mode ultra debug avec monitoring mémoire intensif
    
    Returns:
        Dictionnaire avec les statistiques du pool créé
    """
    logger.info(f"Création du pool augmenté")
    logger.info(f"Source: {audio_root}")
    logger.info(f"Destination: {output_dir}")
    logger.info(f"Cible: {target_samples_per_class} échantillons par classe")
    logger.info(f"Batch size: {batch_size} fichiers à la fois")
    if low_memory:
        logger.info(f"Mode FAIBLE MÉMOIRE activé avec {workers} workers")
    if debug:
        logger.info("Mode DEBUG activé")
        logging.getLogger().setLevel(logging.DEBUG)
    if ultra_debug:
        logger.info("Mode ULTRA DEBUG activé - Monitoring mémoire intensif")
    
    # Vérifier la mémoire initiale
    memory_info = psutil.virtual_memory()
    logger.info(f"Mémoire disponible: {memory_info.available / (1024**3):.1f} GB / {memory_info.total / (1024**3):.1f} GB")
    
    # Créer le répertoire de sortie
    output_dir.mkdir(parents=True, exist_ok=True)
    
    pool_stats = {
        'timestamp': datetime.now().isoformat(),
        'target_samples_per_class': target_samples_per_class,
        'classes_processed': 0,
        'total_original': 0,
        'total_created': 0
    }
    
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
    
    # Types d'augmentation disponibles
    augmentation_types = ['time_stretch', 'noise', 'volume', 'combined']
    
    # Traiter chaque classe
    for class_dir in sorted(classes):
        class_name = class_dir.name
        logger.info(f"\nTraitement de la classe: {class_name}")
        
        if ultra_debug:
            log_memory_state(f"Début classe {class_name}", verbose=True)
        elif debug:
            memory_info = psutil.virtual_memory()
            logger.debug(f"  [DEBUG] Mémoire avant traitement classe: {memory_info.percent:.1f}% ({memory_info.used / (1024**3):.1f} GB)")
        
        # Créer le répertoire de sortie pour cette classe
        class_output_dir = output_dir / class_name
        if debug:
            logger.debug(f"  [DEBUG] Création du répertoire: {class_output_dir}")
        class_output_dir.mkdir(exist_ok=True)
        
        # Lister tous les fichiers audio
        if debug:
            logger.debug(f"  [DEBUG] Début du listage des fichiers dans: {class_dir}")
        
        try:
            wav_files = list(class_dir.glob("*.wav"))
            if debug:
                logger.debug(f"  [DEBUG] Fichiers WAV trouvés: {len(wav_files)}")
            
            mp3_files = list(class_dir.glob("*.mp3"))
            if debug:
                logger.debug(f"  [DEBUG] Fichiers MP3 trouvés: {len(mp3_files)}")
            
            audio_files = wav_files + mp3_files
            if debug:
                logger.debug(f"  [DEBUG] Total fichiers audio: {len(audio_files)}")
                if audio_files:
                    logger.debug(f"  [DEBUG] Premier fichier: {audio_files[0].name}")
        except Exception as e:
            logger.error(f"  Erreur lors du listage des fichiers: {e}")
            if debug:
                import traceback
                logger.debug(f"  [DEBUG] Traceback: {traceback.format_exc()}")
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
        
        # Si mode low_memory avec workers > 1, utiliser multiprocessing
        if low_memory and workers > 1:
            logger.info(f"  Utilisation de {workers} processus parallèles")
            
            # Préparer les arguments pour multiprocessing
            process_args = [
                (audio_file, class_output_dir, augmentations_per_sample, augmentation_types, save_detailed_metadata)
                for audio_file in audio_files
            ]
            
            # Traiter avec Pool
            with Pool(processes=workers) as pool:
                try:
                    results = list(tqdm(
                        pool.imap(process_file_low_memory, process_args),
                        total=len(audio_files),
                        desc=f"  {class_name} (parallèle)"
                    ))
                    
                    # Compiler les résultats
                    for result in results:
                        if result['success']:
                            files_created_count += result['files_created']
                            if save_detailed_metadata:
                                created_files.extend(result['metadata'])
                        else:
                            logger.error(f"  {result['error']}")
                            
                except Exception as e:
                    logger.error(f"  Erreur multiprocessing: {e}")
                    
            # Forcer le gc après le traitement parallèle
            gc.collect()
            
        else:
            # Mode séquentiel (original)
            # Traiter par batch pour gérer la mémoire
            if debug:
                logger.debug(f"  [DEBUG] Début du traitement par batch (batch_size={batch_size})")
            
            for batch_start in range(0, len(audio_files), batch_size):
                batch_end = min(batch_start + batch_size, len(audio_files))
                batch_files = audio_files[batch_start:batch_end]
                
                if debug:
                    logger.debug(f"  [DEBUG] Batch {batch_start//batch_size + 1}: indices {batch_start}-{batch_end}, {len(batch_files)} fichiers")
                
                # Vérifier la mémoire disponible
                memory_percent = psutil.virtual_memory().percent
                if memory_percent > 85:
                    logger.warning(f"  Mémoire à {memory_percent}%, pause de 5 secondes...")
                    gc.collect()
                    time.sleep(5)
            
            if debug:
                logger.debug(f"  [DEBUG] Création de la barre de progression tqdm")
            
            try:
                progress_bar = tqdm(batch_files, desc=f"  {class_name} (batch {batch_start//batch_size + 1})")
            except Exception as e:
                logger.error(f"  Erreur création tqdm: {e}")
                if debug:
                    import traceback
                    logger.debug(f"  [DEBUG] Traceback: {traceback.format_exc()}")
                progress_bar = batch_files
            
            for file_idx, audio_file in enumerate(progress_bar):
                if ultra_debug:
                    log_memory_state(f"Avant fichier {file_idx}: {audio_file.name}", verbose=True)
                elif debug and file_idx == 0:
                    logger.debug(f"  [DEBUG] Début traitement fichier {file_idx}: {audio_file.name}")
                    memory_info = psutil.virtual_memory()
                    logger.debug(f"  [DEBUG] Mémoire: {memory_info.percent:.1f}% ({memory_info.used / (1024**3):.1f} GB)")
                # 1. Copier l'original
                try:
                    if debug:
                        logger.debug(f"  [DEBUG] Avant création nom de sortie pour: {audio_file.name}")
                        logger.debug(f"  [DEBUG] audio_file.stem: {audio_file.stem}")
                        logger.debug(f"  [DEBUG] audio_file.suffix: {audio_file.suffix}")
                    
                    output_name = f"{audio_file.stem}_original{audio_file.suffix}"
                    
                    if debug:
                        logger.debug(f"  [DEBUG] Nom de sortie créé: {output_name}")
                        logger.debug(f"  [DEBUG] class_output_dir: {class_output_dir}")
                    
                    output_path = class_output_dir / output_name
                    
                    if debug:
                        logger.debug(f"  [DEBUG] Chemin de sortie créé: {output_path}")
                        logger.debug(f"  [DEBUG] Fichier source existe: {audio_file.exists()}")
                        logger.debug(f"  [DEBUG] Répertoire destination existe: {class_output_dir.exists()}")
                        logger.debug(f"  [DEBUG] Avant shutil.copy2...")
                    
                    shutil.copy2(str(audio_file), str(output_path))
                    
                    if debug:
                        logger.debug(f"  [DEBUG] Copie réussie!")
                    
                    files_created_count += 1
                    
                    if debug:
                        logger.debug(f"  [DEBUG] files_created_count incrémenté: {files_created_count}")
                    
                except Exception as e:
                    logger.error(f"  Erreur copie {audio_file.name}: {e}")
                    if debug:
                        import traceback
                        logger.debug(f"  [DEBUG] Traceback complet: {traceback.format_exc()}")
                    continue
                
                if debug:
                    logger.debug(f"  [DEBUG] Avant check save_detailed_metadata: {save_detailed_metadata}")
                
                if save_detailed_metadata:
                    created_files.append({
                        'filename': output_name,
                        'type': 'original',
                        'source': audio_file.name
                    })
                
                if debug:
                    logger.debug(f"  [DEBUG] Avant check augmentations_per_sample: {augmentations_per_sample}")
                
                # 2. Créer les augmentations si nécessaire
                if augmentations_per_sample > 0:
                    if debug:
                        logger.debug(f"  [DEBUG] Début création des augmentations")
                    
                    try:
                        if low_memory:
                            # Mode faible mémoire: préparer toutes les augmentations pour un seul subprocess
                            if debug:
                                logger.debug(f"  [DEBUG] Mode low_memory: préparation batch d'augmentations")
                            
                            import subprocess
                            import json as json_module
                            
                            # Préparer toutes les augmentations
                            augmentations = []
                            for aug_idx in range(augmentations_per_sample):
                                aug_type = augmentation_types[aug_idx % len(augmentation_types)]
                                strength = 0.3 + (aug_idx / max(augmentations_per_sample - 1, 1)) * 0.7
                                aug_name = f"{audio_file.stem}_aug{aug_idx+1:03d}_{aug_type}.wav"
                                aug_path = class_output_dir / aug_name
                                
                                augmentations.append({
                                    'output_path': str(aug_path),
                                    'aug_type': aug_type,
                                    'strength': strength
                                })
                            
                            # Préparer les paramètres pour le subprocess
                            params = {
                                'input_path': str(audio_file),
                                'augmentations': augmentations
                            }
                            
                            # Appeler le processeur avec toutes les augmentations
                            processor_path = Path(__file__).parent / 'process_single_augmentation.py'
                            
                            # Ajouter retry pour gérer les erreurs transitoires
                            max_retries = 2
                            for retry in range(max_retries):
                                try:
                                    result = subprocess.run(
                                        [sys.executable, str(processor_path)],
                                        input=json_module.dumps(params),
                                        capture_output=True,
                                        text=True,
                                        timeout=30  # Timeout de 30 secondes
                                    )
                                    
                                    if result.returncode != 0:
                                        error_msg = result.stderr.strip() if result.stderr else "Code de retour non-zéro"
                                        if retry < max_retries - 1:
                                            logger.warning(f"  Retry {retry+1}/{max_retries} après erreur: {error_msg}")
                                            time.sleep(1)
                                            continue
                                        else:
                                            logger.error(f"  Erreur subprocess après {max_retries} essais: {error_msg}")
                                            break
                                    
                                    # Vérifier le résultat
                                    try:
                                        response = json_module.loads(result.stdout)
                                        if response['status'] == 'success':
                                            # Compter les succès
                                            for result_item in response.get('results', []):
                                                if result_item['status'] == 'success':
                                                    files_created_count += 1
                                                    
                                                    if save_detailed_metadata:
                                                        # Extraire les infos du nom de fichier
                                                        aug_filename = Path(result_item['output_path']).name
                                                        # Parser le type et l'index depuis le nom
                                                        parts = aug_filename.replace(audio_file.stem + '_', '').replace('.wav', '').split('_')
                                                        aug_type_from_name = '_'.join(parts[1:]) if len(parts) > 1 else 'unknown'
                                                        
                                                        created_files.append({
                                                            'filename': aug_filename,
                                                            'type': 'augmented',
                                                            'source': audio_file.name,
                                                            'augmentation': aug_type_from_name
                                                        })
                                                else:
                                                    logger.warning(f"  Échec augmentation: {result_item.get('error', 'Unknown')}")
                                            break  # Sortir de la boucle retry
                                        else:
                                            logger.error(f"  Erreur batch augmentation: {response.get('error', 'Unknown')}")
                                            break
                                    except json_module.JSONDecodeError:
                                        logger.error(f"  Réponse subprocess invalide: {result.stdout[:200]}")
                                        if retry < max_retries - 1:
                                            time.sleep(1)
                                            continue
                                    break  # Sortir si succès
                                    
                                except subprocess.TimeoutExpired:
                                    logger.error(f"  Timeout subprocess après 30 secondes")
                                    break
                                except Exception as e:
                                    logger.error(f"  Erreur subprocess inattendue: {e}")
                                    break
                            
                            # Forcer le gc après le batch
                            gc.collect()
                            
                        else:
                            # Mode normal: charger l'audio et traiter en mémoire
                            if debug:
                                logger.debug(f"  [DEBUG] Mode normal: chargement audio")
                            
                            # NOTE: gc.collect() et time.sleep() causent une explosion du VMS
                            # de 3.37GB à 7.79GB sur certains systèmes. Désactivé pour éviter OOM.
                            # Voir test_vms_gc_issue.py pour plus de détails.
                            
                            if ultra_debug:
                                log_memory_state(f"Avant load {audio_file.name}", verbose=True)
                            
                            try:
                                waveform, sr = torchaudio.load(str(audio_file))
                                if ultra_debug:
                                    logger.info(f"  Waveform shape: {waveform.shape}, dtype: {waveform.dtype}")
                                    log_memory_state(f"Après load {audio_file.name}", verbose=True)
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
                                
                                # Définir le nom de sortie
                                aug_name = f"{audio_file.stem}_aug{aug_idx+1:03d}_{aug_type}.wav"
                                aug_path = class_output_dir / aug_name
                                
                                # Appliquer l'augmentation
                                aug_waveform = apply_audio_augmentation(waveform, sr, aug_type, strength)
                                
                                # Sauvegarder
                                torchaudio.save(str(aug_path), aug_waveform, sr)
                                
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
                            
                            # Libérer la mémoire du waveform original
                            del waveform
                            gc.collect()
                        
                    except Exception as e:
                        logger.error(f"  Erreur lors du traitement de {audio_file.name}: {e}")
                        if ultra_debug:
                            log_memory_state(f"Après erreur {audio_file.name}", verbose=True)
                        continue
                
                # Log mémoire après chaque fichier en ultra_debug
                if ultra_debug:
                    log_memory_state(f"Après fichier {file_idx}: {audio_file.name}", verbose=True)
            
            # Forcer la libération de mémoire après chaque batch
            gc.collect()
            
            if ultra_debug:
                logger.info(f"  [ULTRA DEBUG] Fin batch {batch_start//batch_size + 1}")
                logger.info(f"  [ULTRA DEBUG] files_created_count: {files_created_count}")
                if created_files is not None:
                    logger.info(f"  [ULTRA DEBUG] len(created_files): {len(created_files)}")
                    # Calculer la taille approximative
                    try:
                        size_estimate = sys.getsizeof(created_files) / (1024**2)
                        logger.info(f"  [ULTRA DEBUG] created_files size: ~{size_estimate:.1f} MB")
                    except:
                        pass
                log_memory_state(f"Fin batch {batch_start//batch_size + 1}", verbose=True)
        
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
        # NOTE: torch.cuda.empty_cache() cause une explosion du VMS (+4.4GB)
        # même sur les systèmes sans GPU. Désactivé pour éviter OOM.
        
        # Pause pour permettre au GC de nettoyer
        gc.collect()
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
    parser.add_argument('--debug', action='store_true',
                       help='Activer les logs de debug détaillés')
    parser.add_argument('--workers', type=int, default=1,
                       help='Nombre de processus parallèles (seulement en mode --low-memory)')
    parser.add_argument('--ultra-debug', action='store_true',
                       help='Mode ultra debug avec monitoring mémoire intensif')
    
    args = parser.parse_args()
    
    create_augmented_pool(
        args.audio_root,
        args.output_dir,
        args.target_samples,
        args.max_augmentations,
        args.batch_size,
        args.save_detailed_metadata,
        args.specific_class,
        args.low_memory,
        args.debug,
        args.workers,
        args.ultra_debug
    )


if __name__ == "__main__":
    main()