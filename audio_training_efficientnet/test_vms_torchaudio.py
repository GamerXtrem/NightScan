#!/usr/bin/env python3
"""Test VMS avec torchaudio"""
import torch
import torchaudio
import psutil
import gc

def show_mem(label):
    p = psutil.Process()
    m = p.memory_info()
    print(f'{label}: RSS={m.rss/(1024**3):.2f}GB, VMS={m.vms/(1024**3):.2f}GB')

print("Test torchaudio sur airplane")
print("-" * 40)

show_mem("Initial")

# Import et initialisation
print("\nAprès imports:")
show_mem("Après imports")

# Vérifier le backend
try:
    backend = torchaudio.get_audio_backend()
    print(f"Backend audio: {backend}")
except:
    print("Pas de backend audio défini")

# Charger le fichier
file_path = "/home/ubuntu/NightScan/data/audio_data/airplane/5-235956-A-47.wav"
show_mem("Avant load")

waveform, sr = torchaudio.load(file_path)
print(f"Loaded: shape={waveform.shape}, sr={sr}, dtype={waveform.dtype}")
show_mem("Après load")

# Nettoyer
del waveform
gc.collect()
show_mem("Après del + gc")

print("\nTest de plusieurs fichiers:")
for i in range(5):
    waveform, sr = torchaudio.load(file_path)
    print(f"  Load {i+1}: {waveform.shape}")
    show_mem(f"  Après load {i+1}")
    del waveform
    gc.collect()

show_mem("Final")

# Test avec différents backends si disponibles
print("\nTest des backends disponibles:")
for backend in ['sox_io', 'soundfile']:
    try:
        torchaudio.set_audio_backend(backend)
        print(f"\nBackend: {backend}")
        show_mem(f"  Avant load ({backend})")
        waveform, sr = torchaudio.load(file_path)
        show_mem(f"  Après load ({backend})")
        del waveform
        gc.collect()
    except Exception as e:
        print(f"  Backend {backend} non disponible: {e}")