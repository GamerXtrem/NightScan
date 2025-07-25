#!/usr/bin/env python3
"""
Script de test pour diagnostiquer les problèmes de mémoire lors du chargement du dataset.
"""

import os
import sys
import psutil
import gc
from pathlib import Path
import time

# Ajouter le chemin parent pour les imports
sys.path.append(str(Path(__file__).parent.parent))

def print_memory_usage(step_name):
    """Affiche l'utilisation mémoire actuelle."""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    print(f"\n[{step_name}]")
    print(f"  RSS (Resident Set Size): {mem_info.rss / 1024 / 1024:.1f} MB")
    print(f"  VMS (Virtual Memory Size): {mem_info.vms / 1024 / 1024:.1f} MB")
    print(f"  Available system memory: {psutil.virtual_memory().available / 1024 / 1024:.1f} MB")
    gc.collect()

def test_dataset_loading():
    """Test le chargement du dataset étape par étape."""
    
    print("=== Test de consommation mémoire du dataset ===")
    print_memory_usage("Démarrage")
    
    # 1. Import des modules
    print("\n1. Import des modules...")
    from audio_dataset import AudioSpectrogramDataset
    print_memory_usage("Après imports")
    
    # 2. Paramètres
    csv_file = Path("data/processed/csv/train.csv")
    audio_dir = Path("data/audio_data")
    
    # 3. Chargement sans augmentation
    print("\n2. Chargement du dataset SANS augmentation...")
    try:
        dataset_no_aug = AudioSpectrogramDataset(
            csv_file=csv_file,
            audio_dir=audio_dir,
            augment=False,
            enable_oversampling=False,
            cache_spectrograms=False
        )
        print(f"Dataset chargé: {len(dataset_no_aug)} échantillons")
        print_memory_usage("Dataset sans augmentation")
        
        # Tester le chargement d'un échantillon
        print("\n3. Test de chargement d'un échantillon...")
        sample = dataset_no_aug[0]
        print(f"Échantillon chargé: spectrogramme shape = {sample[0].shape}")
        print_memory_usage("Après chargement 1 échantillon")
        
        del dataset_no_aug
        gc.collect()
        
    except Exception as e:
        print(f"ERREUR lors du chargement sans augmentation: {e}")
        return
    
    # 4. Chargement avec augmentation mais sans oversampling
    print("\n4. Chargement du dataset AVEC augmentation, SANS oversampling...")
    try:
        dataset_aug_no_oversample = AudioSpectrogramDataset(
            csv_file=csv_file,
            audio_dir=audio_dir,
            augment=True,
            adaptive_augment=True,
            enable_oversampling=False,
            cache_spectrograms=False
        )
        print(f"Dataset chargé: {len(dataset_aug_no_oversample)} échantillons")
        print_memory_usage("Dataset avec augmentation, sans oversampling")
        
        del dataset_aug_no_oversample
        gc.collect()
        
    except Exception as e:
        print(f"ERREUR lors du chargement avec augmentation: {e}")
        return
    
    # 5. Chargement complet (comme dans train_audio.py)
    print("\n5. Chargement du dataset COMPLET (augmentation + oversampling)...")
    try:
        dataset_full = AudioSpectrogramDataset(
            csv_file=csv_file,
            audio_dir=audio_dir,
            augment=True,
            adaptive_augment=True,
            enable_oversampling=True,
            cache_spectrograms=True,
            spectrogram_dir=Path("data/spectrograms_cache")
        )
        print(f"Dataset chargé: {len(dataset_full)} échantillons")
        print_memory_usage("Dataset complet")
        
        # Compter les échantillons par classe
        print("\nÉchantillons par classe après oversampling:")
        class_counts = {}
        for i in range(len(dataset_full)):
            if i % 100 == 0:
                print(f"  Traitement: {i}/{len(dataset_full)}...", end='\r')
            row = dataset_full.data_df.iloc[i]
            label = row['label']
            class_counts[label] = class_counts.get(label, 0) + 1
        
        print("\n")
        for label, count in sorted(class_counts.items()):
            print(f"  {label}: {count} échantillons")
        
        print_memory_usage("Après comptage")
        
    except Exception as e:
        print(f"ERREUR lors du chargement complet: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n=== Test terminé ===")

if __name__ == "__main__":
    # Se placer dans le bon répertoire
    os.chdir(Path(__file__).parent.parent)
    test_dataset_loading()