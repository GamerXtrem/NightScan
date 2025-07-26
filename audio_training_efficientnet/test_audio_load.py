#!/usr/bin/env python3
"""
Script de test pour identifier le problème de chargement audio
"""

import sys
import os
import gc
import psutil
from pathlib import Path

def test_torchaudio():
    print("\n=== Test avec torchaudio ===")
    try:
        import torch
        import torchaudio
        
        # Forcer le backend
        try:
            torchaudio.set_audio_backend("sox_io")
            print(f"Backend audio: {torchaudio.get_audio_backend()}")
        except:
            print("Impossible de définir le backend audio")
        
        # Test de chargement
        test_file = Path(sys.argv[1])
        print(f"Chargement de: {test_file}")
        
        # Nettoyer avant
        gc.collect()
        
        # Mémoire avant
        mem = psutil.virtual_memory()
        print(f"Mémoire avant: {mem.percent}% ({mem.used/(1024**3):.2f} GB)")
        
        # Charger
        waveform, sr = torchaudio.load(str(test_file))
        print(f"✓ Chargé avec succès: shape={waveform.shape}, sr={sr}")
        
        # Mémoire après
        mem = psutil.virtual_memory()
        print(f"Mémoire après: {mem.percent}% ({mem.used/(1024**3):.2f} GB)")
        
        del waveform
        gc.collect()
        
    except Exception as e:
        print(f"✗ Erreur torchaudio: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()


def test_soundfile():
    print("\n=== Test avec soundfile ===")
    try:
        import soundfile as sf
        import numpy as np
        
        test_file = Path(sys.argv[1])
        print(f"Chargement de: {test_file}")
        
        # Mémoire avant
        mem = psutil.virtual_memory()
        print(f"Mémoire avant: {mem.percent}% ({mem.used/(1024**3):.2f} GB)")
        
        # Charger
        data, sr = sf.read(str(test_file))
        print(f"✓ Chargé avec succès: shape={data.shape}, sr={sr}")
        
        # Mémoire après
        mem = psutil.virtual_memory()
        print(f"Mémoire après: {mem.percent}% ({mem.used/(1024**3):.2f} GB)")
        
        del data
        gc.collect()
        
    except Exception as e:
        print(f"✗ Erreur soundfile: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()


def test_librosa():
    print("\n=== Test avec librosa ===")
    try:
        import librosa
        
        test_file = Path(sys.argv[1])
        print(f"Chargement de: {test_file}")
        
        # Mémoire avant
        mem = psutil.virtual_memory()
        print(f"Mémoire avant: {mem.percent}% ({mem.used/(1024**3):.2f} GB)")
        
        # Charger
        y, sr = librosa.load(str(test_file), sr=None)
        print(f"✓ Chargé avec succès: shape={y.shape}, sr={sr}")
        
        # Mémoire après
        mem = psutil.virtual_memory()
        print(f"Mémoire après: {mem.percent}% ({mem.used/(1024**3):.2f} GB)")
        
        del y
        gc.collect()
        
    except Exception as e:
        print(f"✗ Erreur librosa: {type(e).__name__}: {e}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python test_audio_load.py <fichier_audio>")
        sys.exit(1)
    
    print(f"Test de chargement audio pour: {sys.argv[1]}")
    print(f"PID: {os.getpid()}")
    
    # Tester chaque méthode
    test_torchaudio()
    test_soundfile()
    test_librosa()
    
    print("\n=== Tests terminés ===")


if __name__ == "__main__":
    main()