#!/usr/bin/env python3
"""Test de mémoire pour les augmentations airplane"""
import torch
from torch.backends import mkldnn
mkldnn.m.set_flags(False)
import torchaudio
import torchaudio.transforms as T
import psutil
import gc

def show_mem(label):
    p = psutil.Process()
    m = p.memory_info()
    print(f'{label}: RSS={m.rss/(1024**3):.2f}GB, VMS={m.vms/(1024**3):.2f}GB')

print("Test augmentations sur airplane")
print("-" * 40)

# Charger un fichier
file_path = '/home/ubuntu/NightScan/data/audio_data/airplane/5-235956-A-47.wav'
print(f"Chargement de {file_path}")
show_mem('Initial')

waveform, sr = torchaudio.load(file_path)
print(f"Loaded: shape={waveform.shape}, sr={sr}")
show_mem('Après load')

# Tester chaque type d'augmentation
augmentation_types = ['time_stretch', 'noise', 'volume', 'combined']

for i in range(11):
    aug_type = augmentation_types[i % 4]
    strength = 0.3 + (i / 10) * 0.7
    
    print(f'\nAugmentation {i+1}/11 ({aug_type}, strength={strength:.2f}):')
    
    if aug_type == 'time_stretch':
        rate = 1.0 + (strength - 0.5) * 0.4
        print(f"  Time stretch rate: {rate:.2f}")
        stretched = T.Resample(sr, int(sr * rate))(waveform)
        result = T.Resample(int(sr * rate), sr)(stretched)
        del stretched
        
    elif aug_type == 'noise':
        noise_level = 0.002 * strength
        print(f"  Noise level: {noise_level:.5f}")
        result = waveform + torch.randn_like(waveform) * noise_level
        
    elif aug_type == 'volume':
        volume_factor = 0.5 + strength * 0.5
        print(f"  Volume factor: {volume_factor:.2f}")
        result = waveform * volume_factor
        
    else:  # combined
        # Time stretch léger
        rate = 1.0 + (strength - 0.5) * 0.2
        temp = T.Resample(sr, int(sr * rate))(waveform)
        temp = T.Resample(int(sr * rate), sr)(temp)
        # Bruit
        noise_level = 0.001 * strength
        temp = temp + torch.randn_like(temp) * noise_level
        # Volume
        volume_factor = 0.7 + strength * 0.3
        result = temp * volume_factor
        del temp
        print(f"  Combined: rate={rate:.2f}, noise={noise_level:.5f}, vol={volume_factor:.2f}")
    
    show_mem(f'  Après création aug {i+1}')
    
    # Sauvegarder
    output_path = f'/tmp/test_aug_{i:02d}_{aug_type}.wav'
    torchaudio.save(output_path, result, sr)
    print(f"  Sauvegardé: {output_path}")
    show_mem(f'  Après save aug {i+1}')
    
    # Nettoyer
    del result
    gc.collect()
    show_mem(f'  Après del+gc aug {i+1}')

# Nettoyer waveform original
del waveform
gc.collect()

print("\n" + "-" * 40)
show_mem('Final')