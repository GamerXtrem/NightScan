#!/usr/bin/env python3
"""Test VMS avec soundfile"""
import soundfile as sf
import psutil
import numpy as np

def show_mem(label):
    p = psutil.Process()
    m = p.memory_info()
    print(f'{label}: RSS={m.rss/(1024**3):.2f}GB, VMS={m.vms/(1024**3):.2f}GB')

print("Test soundfile sur airplane")
print("-" * 40)

show_mem("Initial")

# Charger le fichier
file_path = "/home/ubuntu/NightScan/data/audio_data/airplane/5-235956-A-47.wav"
show_mem("Avant read")

data, sr = sf.read(file_path)
print(f"Loaded: shape={data.shape}, sr={sr}, dtype={data.dtype}")
show_mem("Après read")

# Convertir en torch tensor pour comparaison
import torch
tensor = torch.from_numpy(data).float()
if len(tensor.shape) == 1:
    tensor = tensor.unsqueeze(0)
else:
    tensor = tensor.T  # soundfile retourne (samples, channels)
print(f"Tensor: shape={tensor.shape}, dtype={tensor.dtype}")
show_mem("Après conversion torch")

# Nettoyer
del data
del tensor
show_mem("Après del")

print("\nTest de plusieurs fichiers:")
for i in range(5):
    data, sr = sf.read(file_path)
    print(f"  Load {i+1}: {data.shape}")
    show_mem(f"  Après load {i+1}")
    del data

show_mem("Final")