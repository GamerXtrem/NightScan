#!/usr/bin/env python3
"""
Processeur isolé pour créer des augmentations.
Utilisé par create_augmented_pool.py via subprocess pour éviter les fuites de mémoire.
Peut traiter une seule augmentation ou plusieurs en batch.
"""

import sys
import json
import torch
# Désactiver MKLDNN pour éviter les fuites mémoire avec torchaudio
from torch.backends import mkldnn
mkldnn.m.set_flags(False)

import torchaudio
import torchaudio.transforms as T
from pathlib import Path



def apply_audio_augmentation(waveform: torch.Tensor, sr: int, augmentation_type: str, strength: float = 1.0) -> torch.Tensor:
    """Applique une augmentation audio spécifique."""
    waveform = waveform.detach().clone()
    
    if augmentation_type == 'time_stretch':
        rate = 1.0 + (strength - 0.5) * 0.4  # 0.8x à 1.2x
        if rate != 1.0:
            # Optimisation mémoire: éviter le double resample pour les taux proches de 1
            if abs(rate - 1.0) < 0.05:  # Pour des taux entre 0.95 et 1.05
                # Utiliser une interpolation directe plus économe
                target_length = int(waveform.shape[-1] / rate)
                waveform = torch.nn.functional.interpolate(
                    waveform.unsqueeze(0), 
                    size=target_length, 
                    mode='linear', 
                    align_corners=False
                ).squeeze(0)
            else:
                # Pour des taux plus extrêmes, utiliser le resample classique
                # mais forcer la libération mémoire entre les opérations
                target_sr = int(sr * rate)
                stretched = T.Resample(sr, target_sr)(waveform)
                del waveform  # Libérer la mémoire de l'original
                waveform = T.Resample(target_sr, sr)(stretched)
                del stretched  # Libérer la mémoire intermédiaire
    
    elif augmentation_type == 'noise':
        noise_level = 0.002 * strength
        noise = torch.randn_like(waveform) * noise_level
        waveform = waveform + noise
    
    elif augmentation_type == 'volume':
        volume_factor = 0.5 + strength * 0.5  # 0.5x à 1.0x
        waveform = waveform * volume_factor
    
    elif augmentation_type == 'combined':
        # Time stretch léger
        rate = 1.0 + (strength - 0.5) * 0.2  # 0.9x à 1.1x
        if rate != 1.0:
            # Toujours utiliser l'interpolation pour combined (taux légers)
            target_length = int(waveform.shape[-1] / rate)
            waveform = torch.nn.functional.interpolate(
                waveform.unsqueeze(0), 
                size=target_length, 
                mode='linear', 
                align_corners=False
            ).squeeze(0)
        
        # Bruit léger
        noise_level = 0.001 * strength
        noise = torch.randn_like(waveform) * noise_level
        waveform = waveform + noise
        
        # Volume
        volume_factor = 0.7 + strength * 0.3  # 0.7x à 1.0x
        waveform = waveform * volume_factor
    
    return waveform


def main():
    # Lire les arguments depuis stdin (JSON)
    params = json.loads(sys.stdin.read())
    
    input_path = Path(params['input_path'])
    
    # Support pour une seule augmentation (rétrocompatibilité)
    if 'output_path' in params:
        augmentations = [{
            'output_path': params['output_path'],
            'aug_type': params['aug_type'],
            'strength': params['strength']
        }]
    else:
        # Support pour plusieurs augmentations
        augmentations = params['augmentations']
    
    try:
        # Charger l'audio une seule fois
        waveform, sr = torchaudio.load(str(input_path))
        
        # Convertir en mono si nécessaire
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # Traiter toutes les augmentations
        results = []
        for aug in augmentations:
            try:
                output_path = Path(aug['output_path'])
                aug_type = aug['aug_type']
                strength = aug['strength']
                
                # Appliquer l'augmentation
                aug_waveform = apply_audio_augmentation(waveform, sr, aug_type, strength)
                
                # Sauvegarder
                torchaudio.save(str(output_path), aug_waveform, sr)
                
                results.append({
                    'output_path': str(output_path),
                    'status': 'success'
                })
                
            except Exception as e:
                results.append({
                    'output_path': str(aug.get('output_path', 'unknown')),
                    'status': 'error',
                    'error': str(e)
                })
        
        # Retourner tous les résultats
        print(json.dumps({
            'status': 'success',
            'results': results
        }))
        
    except Exception as e:
        # Erreur lors du chargement du fichier
        print(json.dumps({
            'status': 'error',
            'error': str(e)
        }))
        sys.exit(1)


if __name__ == "__main__":
    main()