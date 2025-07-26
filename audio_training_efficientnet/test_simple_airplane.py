#!/usr/bin/env python3
"""
Test simple pour voir si le problème vient du chargement du fichier.
"""

import torchaudio
import psutil
import gc

def show_memory():
    mem = psutil.virtual_memory()
    print(f"Mémoire: {mem.percent:.1f}% ({mem.used/(1024**3):.2f} GB utilisés)")

print("Test simple de chargement airplane")
print("-" * 40)

show_memory()

try:
    print("\nChargement du fichier...")
    waveform, sr = torchaudio.load("/home/ubuntu/NightScan/data/audio_data/airplane/5-235956-A-47.wav")
    print(f"Succès! Shape: {waveform.shape}, SR: {sr}")
    show_memory()
    
    print("\nTest d'une augmentation simple (volume)...")
    aug = waveform * 0.8
    print(f"Augmentation créée: {aug.shape}")
    show_memory()
    
    print("\nSauvegarde test...")
    torchaudio.save("/tmp/test_airplane.wav", aug, sr)
    print("Sauvegarde réussie!")
    show_memory()
    
    del aug
    del waveform
    gc.collect()
    
    print("\nAprès nettoyage:")
    show_memory()
    
except Exception as e:
    print(f"\nERREUR: {e}")
    import traceback
    traceback.print_exc()
    show_memory()

print("\nTest terminé!")