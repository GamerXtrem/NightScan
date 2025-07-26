#!/usr/bin/env python3
"""Test de la solution pour la fuite mémoire"""
import subprocess
import json
import psutil
from pathlib import Path

def show_mem(label):
    p = psutil.Process()
    m = p.memory_info()
    print(f'{label}: RSS={m.rss/(1024**3):.2f}GB, VMS={m.vms/(1024**3):.2f}GB')

print("Test de la solution time_stretch optimisée")
print("-" * 40)

show_mem("Initial")

# Préparer les augmentations de test
augmentations = []
for i in range(11):
    aug_type = ['time_stretch', 'noise', 'volume', 'combined'][i % 4]
    strength = 0.3 + (i / 10) * 0.7
    augmentations.append({
        'output_path': f'/tmp/test_fixed_{i:02d}_{aug_type}.wav',
        'aug_type': aug_type,
        'strength': strength
    })

# Paramètres pour le subprocess
params = {
    'input_path': '/home/ubuntu/NightScan/data/audio_data/airplane/5-235956-A-47.wav',
    'augmentations': augmentations
}

print(f"\nLancement du subprocess avec {len(augmentations)} augmentations...")
show_mem("Avant subprocess")

# Lancer le subprocess
result = subprocess.run(
    ['python', 'process_single_augmentation.py'],
    input=json.dumps(params),
    capture_output=True,
    text=True
)

show_mem("Après subprocess")

if result.returncode == 0:
    response = json.loads(result.stdout)
    if response['status'] == 'success':
        print(f"\n✓ Succès! {len(response['results'])} augmentations créées")
        for i, res in enumerate(response['results']):
            if res['status'] == 'success':
                print(f"  Aug {i+1}: ✓")
            else:
                print(f"  Aug {i+1}: ✗ {res.get('error', 'Unknown error')}")
    else:
        print(f"\n✗ Erreur: {response.get('error', 'Unknown error')}")
else:
    print(f"\n✗ Subprocess a échoué (code {result.returncode})")
    print("STDERR:", result.stderr)

show_mem("Final")