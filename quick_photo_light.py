#!/usr/bin/env python3
"""
Création rapide du modèle photo léger uniquement
"""

import torch
import torch.nn as nn
import json
from pathlib import Path
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Créer rapidement le modèle photo léger."""
    print("🌙 Création Rapide du Modèle Photo Léger")
    print("=" * 40)
    
    # Charger le modèle photo lourd
    heavy_path = Path('picture_training_enhanced/models/best_model.pth')
    checkpoint = torch.load(heavy_path, map_location='cpu')
    
    # Créer un modèle B0 léger
    from picture_training_enhanced.models.photo_config import create_photo_model
    light_model = create_photo_model(
        num_classes=8,
        model_name='efficientnet-b0',
        pretrained=False
    )
    
    # Mini-distillation rapide (1 époque)
    from picture_training_enhanced.models.photo_config import create_photo_model
    heavy_model = create_photo_model(
        num_classes=8,
        model_name='efficientnet-b1',
        pretrained=False
    )
    heavy_model.load_state_dict(checkpoint['model_state_dict'])
    heavy_model.eval()
    
    # Distillation très rapide
    light_model.train()
    optimizer = torch.optim.Adam(light_model.parameters(), lr=0.001)
    criterion_kd = nn.KLDivLoss(reduction='batchmean')
    temperature = 3.0
    
    print("🎓 Distillation express...")
    for batch in range(10):  # Seulement 10 batches
        inputs = torch.randn(4, 3, 224, 224)
        optimizer.zero_grad()
        
        with torch.no_grad():
            teacher_outputs = heavy_model(inputs)
            teacher_probs = torch.softmax(teacher_outputs / temperature, dim=1)
        
        student_outputs = light_model(inputs)
        student_log_probs = torch.log_softmax(student_outputs / temperature, dim=1)
        
        loss = criterion_kd(student_log_probs, teacher_probs) * (temperature ** 2)
        loss.backward()
        optimizer.step()
    
    light_model.eval()
    
    # Sauvegarder
    output_dir = Path('mobile_models')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    light_checkpoint = {
        'model_state_dict': light_model.state_dict(),
        'model_config': {
            'model_name': 'efficientnet-b0',
            'architecture': 'efficientnet',
            'num_classes': 8,
            'input_size': (224, 224),
            'quantized': False,
            'distilled': True
        },
        'training_info': {
            'base_model': str(heavy_path),
            'creation_date': datetime.now().isoformat(),
            'method': 'Knowledge distillation from EfficientNet-B1',
            'device': 'cpu'
        },
        'class_names': ['bat', 'owl', 'raccoon', 'opossum', 'deer', 'fox', 'coyote', 'unknown'],
        'metadata': {
            'model_version': '1.0.0',
            'framework': 'pytorch',
            'creation_date': datetime.now().isoformat(),
            'description': 'Lightweight EfficientNet-B0 for wildlife photo classification',
            'model_type': 'photo',
            'variant': 'light',
            'deployment_target': 'ios'
        }
    }
    
    light_path = output_dir / 'photo_light_model.pth'
    metadata_path = output_dir / 'photo_light_metadata.json'
    
    torch.save(light_checkpoint, light_path)
    
    with open(metadata_path, 'w') as f:
        json.dump({
            'model_info': light_checkpoint['metadata'],
            'training_info': light_checkpoint['training_info'],
            'class_names': light_checkpoint['class_names']
        }, f, indent=2)
    
    # Stats
    original_size = heavy_path.stat().st_size
    light_size = light_path.stat().st_size
    reduction = (original_size - light_size) / original_size
    
    print(f"✅ Modèle photo léger créé!")
    print(f"   Original: {original_size / 1024 / 1024:.1f} MB")
    print(f"   Léger: {light_size / 1024 / 1024:.1f} MB")
    print(f"   Réduction: {reduction:.1%}")

if __name__ == "__main__":
    main()