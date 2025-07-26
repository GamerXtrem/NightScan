#!/usr/bin/env python3
"""Test pour isoler le problème VMS avec gc.collect()"""
import psutil
import gc
import torch
import torchaudio
import time
import sys
from pathlib import Path

def show_mem(label):
    p = psutil.Process()
    m = p.memory_info()
    print(f'{label}: RSS={m.rss/(1024**3):.2f}GB, VMS={m.vms/(1024**3):.2f}GB')

print("Test VMS explosion avec gc.collect()")
print("-" * 40)

show_mem('Initial')

# Importer spectrogram_config
sys.path.append(str(Path(__file__).parent.parent))
try:
    from spectrogram_config import get_config_for_animal
    show_mem('Après spectrogram_config')
except:
    print('Pas de spectrogram_config')

# Test 1: gc.collect() seul
print("\nTest 1: gc.collect() seul")
show_mem('Avant gc.collect()')
gc.collect()
show_mem('Après gc.collect()')

# Test 2: torch.cuda.empty_cache() seul
print("\nTest 2: torch.cuda.empty_cache() seul")
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    show_mem('Après cuda.empty_cache()')
else:
    print("Pas de CUDA")

# Test 3: time.sleep() seul
print("\nTest 3: time.sleep(0.1) seul")
time.sleep(0.1)
show_mem('Après time.sleep()')

# Test 4: Combinaison complète
print("\nTest 4: Combinaison gc + cuda + sleep")
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
time.sleep(0.1)
show_mem('Après combinaison complète')

# Test 5: Load après tout ça
print("\nTest 5: torchaudio.load()")
waveform, sr = torchaudio.load('/home/ubuntu/NightScan/data/audio_data/airplane/5-235956-A-47.wav')
show_mem('Après load')
print(f"Loaded: shape={waveform.shape}")

# Test 6: Un autre load
print("\nTest 6: Deuxième load")
waveform2, sr2 = torchaudio.load('/home/ubuntu/NightScan/data/audio_data/airplane/4-161099-A-47.wav')
show_mem('Après 2ème load')

print("\nFinal")
show_mem('Final')