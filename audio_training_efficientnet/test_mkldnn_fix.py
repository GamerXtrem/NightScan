#!/usr/bin/env python3
"""Test si la désactivation de MKLDNN résout le problème VMS"""
import torch
from torch.backends import mkldnn
import torchaudio
import psutil
import gc

def show_mem(label):
    gc.collect()
    p = psutil.Process()
    m = p.memory_info()
    print(f'{label}: RSS={m.rss/(1024**3):.2f}GB, VMS={m.vms/(1024**3):.2f}GB')

print("Test MKLDNN fix pour airplane")
print("-" * 40)

# Test 1: Avec MKLDNN activé (par défaut)
print("\n1. AVEC MKLDNN (par défaut):")
show_mem("Initial")

file_path = "/home/ubuntu/NightScan/data/audio_data/airplane/5-235956-A-47.wav"
waveform, sr = torchaudio.load(file_path)
print(f"Loaded: {waveform.shape}")
show_mem("Après load avec MKLDNN")

del waveform
gc.collect()

# Test 2: Avec MKLDNN désactivé
print("\n2. SANS MKLDNN (désactivé):")
mkldnn.m.set_flags(False)
show_mem("Après désactivation MKLDNN")

waveform, sr = torchaudio.load(file_path)
print(f"Loaded: {waveform.shape}")
show_mem("Après load sans MKLDNN")

# Test plusieurs chargements
print("\nTest de 5 chargements successifs:")
for i in range(5):
    waveform, sr = torchaudio.load(file_path)
    show_mem(f"  Load {i+1}")
    del waveform
    gc.collect()

show_mem("Final")