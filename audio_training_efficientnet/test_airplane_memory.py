#!/usr/bin/env python3
"""
Script de diagnostic pour comprendre le problème de mémoire avec la classe airplane.
"""

import torchaudio
import torchaudio.transforms as T
import torch
import gc
import psutil
import sys
from pathlib import Path

def monitor_memory(label):
    """Affiche l'utilisation mémoire actuelle."""
    gc.collect()  # Force le garbage collection avant la mesure
    mem = psutil.virtual_memory()
    process = psutil.Process()
    process_mem = process.memory_info().rss / (1024**3)  # GB
    print(f"{label}:")
    print(f"  Système: {mem.percent:.1f}% ({mem.used/(1024**3):.2f}/{mem.total/(1024**3):.2f} GB)")
    print(f"  Processus: {process_mem:.2f} GB")
    print()

def test_single_augmentation(waveform, sr, aug_type):
    """Teste une augmentation spécifique et retourne le résultat."""
    print(f"Test augmentation: {aug_type}")
    
    if aug_type == "time_stretch":
        # Time stretch 1.2x
        monitor_memory("  Avant time stretch")
        stretched = T.Resample(sr, int(sr * 1.2))(waveform)
        monitor_memory("  Après premier resample")
        result = T.Resample(int(sr * 1.2), sr)(stretched)
        del stretched
        monitor_memory("  Après second resample")
        return result
        
    elif aug_type == "noise":
        monitor_memory("  Avant ajout bruit")
        result = waveform + torch.randn_like(waveform) * 0.002
        monitor_memory("  Après ajout bruit")
        return result
        
    elif aug_type == "volume":
        monitor_memory("  Avant changement volume")
        result = waveform * 0.8
        monitor_memory("  Après changement volume")
        return result
        
    elif aug_type == "combined":
        monitor_memory("  Avant combined")
        # Time stretch léger
        temp = T.Resample(sr, int(sr * 1.1))(waveform)
        temp = T.Resample(int(sr * 1.1), sr)(temp)
        # Bruit
        temp = temp + torch.randn_like(temp) * 0.001
        # Volume
        result = temp * 0.9
        del temp
        monitor_memory("  Après combined")
        return result

def main():
    if len(sys.argv) > 1:
        audio_file = sys.argv[1]
    else:
        audio_file = "/home/ubuntu/NightScan/data/audio_data/airplane/5-235956-A-47.wav"
    
    print(f"Test de mémoire pour: {audio_file}")
    print("=" * 60)
    
    monitor_memory("DÉBUT")
    
    # 1. Charger le fichier
    print("Chargement du fichier...")
    try:
        waveform, sr = torchaudio.load(audio_file)
        print(f"Chargé: shape={waveform.shape}, sr={sr}")
        monitor_memory("Après chargement")
        
        # Convertir en mono si nécessaire
        if waveform.shape[0] > 1:
            print("Conversion en mono...")
            waveform = waveform.mean(dim=0, keepdim=True)
            monitor_memory("Après conversion mono")
        
        # 2. Tester chaque type d'augmentation individuellement
        aug_types = ["time_stretch", "noise", "volume", "combined"]
        
        for aug_type in aug_types:
            print("\n" + "=" * 40)
            try:
                aug_result = test_single_augmentation(waveform, sr, aug_type)
                print(f"  Résultat: shape={aug_result.shape}")
                
                # Simuler la sauvegarde
                output_path = f"/tmp/test_{aug_type}.wav"
                torchaudio.save(output_path, aug_result, sr)
                print(f"  Sauvegardé: {output_path}")
                
                del aug_result
                gc.collect()
                monitor_memory(f"Après nettoyage {aug_type}")
                
            except Exception as e:
                print(f"ERREUR avec {aug_type}: {e}")
                monitor_memory(f"Après erreur {aug_type}")
        
        # 3. Tester plusieurs augmentations en séquence
        print("\n" + "=" * 60)
        print("Test de 11 augmentations en séquence...")
        
        for i in range(11):
            aug_type = aug_types[i % len(aug_types)]
            print(f"\nAugmentation {i+1}/11 ({aug_type})")
            
            try:
                aug_result = test_single_augmentation(waveform, sr, aug_type)
                del aug_result
                gc.collect()
                
            except Exception as e:
                print(f"ERREUR à l'augmentation {i+1}: {e}")
                monitor_memory("Après erreur")
                break
        
        # Nettoyage final
        del waveform
        gc.collect()
        monitor_memory("FINAL (après nettoyage)")
        
    except Exception as e:
        print(f"ERREUR FATALE: {e}")
        import traceback
        traceback.print_exc()
        monitor_memory("Après erreur fatale")

    print("\n" + "=" * 60)
    print("Test terminé!")

if __name__ == "__main__":
    main()