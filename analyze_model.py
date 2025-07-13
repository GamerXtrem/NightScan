#!/usr/bin/env python3
import torch

# Charger le modèle
checkpoint = torch.load('models/resnet18/best_model.pth', map_location='cpu', weights_only=False)

print('=== ANALYSE DU MODÈLE EXISTANT ===')
print('Métadonnées du checkpoint:')
for k, v in checkpoint.items():
    if k not in ['model_state_dict', 'optimizer_state_dict']:
        print(f'  {k}: {v}')

print('\nAnalyse du state_dict:')
state_dict = checkpoint['model_state_dict']
print(f'  Nombre total de paramètres: {len(state_dict)}')

# Analyser l'architecture
conv_layers = [k for k in state_dict.keys() if 'conv' in k and 'weight' in k]
fc_layers = [k for k in state_dict.keys() if 'fc' in k and 'weight' in k]

print(f'  Couches convolutionnelles: {len(conv_layers)}')
print(f'  Couches fully connected: {len(fc_layers)}')

# Analyser la couche finale pour le nombre de classes
for k, v in state_dict.items():
    if 'fc' in k and 'weight' in k:
        print(f'  Couche finale {k}: {v.shape} -> {v.shape[0]} classes de sortie')
        
# Analyser la première couche pour les canaux d'entrée  
for k, v in state_dict.items():
    if 'conv1' in k and 'weight' in k:
        print(f'  Première couche {k}: {v.shape} -> {v.shape[1]} canaux d'entrée')
        break

print('\n=== TAILLE RÉELLE DU MODÈLE ===')
total_params = sum(p.numel() for p in state_dict.values())
print(f'  Paramètres totaux: {total_params:,}')
print(f'  Taille approximative: {total_params * 4 / 1024 / 1024:.1f} MB (float32)')