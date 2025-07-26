#!/usr/bin/env python3
"""Test VMS dans le contexte de create_augmented_pool"""
import sys
import psutil
import gc
from pathlib import Path

# Ajouter le chemin parent
sys.path.append(str(Path(__file__).parent.parent))

def show_mem(label):
    p = psutil.Process()
    m = p.memory_info()
    print(f'{label}: RSS={m.rss/(1024**3):.2f}GB, VMS={m.vms/(1024**3):.2f}GB')

print("Test VMS dans contexte create_augmented_pool")
print("-" * 40)

show_mem("Initial")

# Importer les modules dans l'ordre de create_augmented_pool
print("\nImport des modules...")
import os
show_mem("Après os")

import numpy as np
show_mem("Après numpy")

import torch
show_mem("Après torch")

import torchaudio
show_mem("Après torchaudio")

import torchaudio.transforms as T
show_mem("Après transforms")

from tqdm import tqdm
show_mem("Après tqdm")

import json
from datetime import datetime
import shutil
from typing import Dict, List, Tuple, Optional
import random
import time
from multiprocessing import Pool
from functools import partial
show_mem("Après tous les imports standard")

# Importer spectrogram_config
try:
    from spectrogram_config import get_config_for_animal
    show_mem("Après spectrogram_config")
except:
    print("Impossible d'importer spectrogram_config")

# Tester le chargement
print("\nTest de chargement:")
file_path = "/home/ubuntu/NightScan/data/audio_data/airplane/5-235956-A-47.wav"

# Simuler exactement ce que fait create_augmented_pool
show_mem("Avant gc.collect()")
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

import time
time.sleep(0.1)
show_mem("Après gc + sleep")

# Charger
waveform, sr = torchaudio.load(str(file_path))
print(f"Loaded: shape={waveform.shape}, sr={sr}")
show_mem("Après load")

# Conversion mono
if waveform.shape[0] > 1:
    waveform = waveform.mean(dim=0, keepdim=True)
    show_mem("Après conversion mono")

# Test d'une augmentation
print("\nTest augmentation time_stretch:")
show_mem("Avant time_stretch")
stretched = T.Resample(sr, int(sr * 1.2))(waveform)
show_mem("Après premier resample")
result = T.Resample(int(sr * 1.2), sr)(stretched)
show_mem("Après second resample")

del stretched
del result
del waveform
gc.collect()
show_mem("Final après nettoyage")