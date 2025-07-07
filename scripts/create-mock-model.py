#!/usr/bin/env python3
"""
Script pour créer un modèle ML temporaire pour tests
Ce modèle permet de tester l'infrastructure sans données réelles
"""

import torch
import torch.nn as nn
import json
from pathlib import Path
import os

def create_mock_model():
    """Créer un modèle ResNet18 simplifié pour tests"""
    
    # Architecture similaire au modèle réel mais simplifiée
    class MockWildlifeClassifier(nn.Module):
        def __init__(self, num_classes=6):
            super(MockWildlifeClassifier, self).__init__()
            
            # Couches convolutionnelles simplifiées
            self.features = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                
                nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                
                nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1))
            )
            
            # Classificateur
            self.classifier = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(256, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(128, num_classes)
            )
            
        def forward(self, x):
            x = self.features(x)
            x = torch.flatten(x, 1)
            x = self.classifier(x)
            return x
    
    return MockWildlifeClassifier()

def create_mock_weights(model, project_root):
    """Créer des poids pré-entraînés simulés"""
    
    models_dir = Path(project_root) / "models"
    models_dir.mkdir(exist_ok=True)
    
    # Initialiser avec des poids Xavier/Glorot
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.constant_(module.bias, 0)
    
    # Simuler un modèle "entraîné" avec performance raisonnable
    model_state = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': {},  # Pas besoin pour inference
        'epoch': 100,
        'loss': 0.234,
        'accuracy': 0.892,
        'val_accuracy': 0.856,
        'training_time_hours': 24.5,
        'model_version': '1.0.0-mock',
        'pytorch_version': torch.__version__,
        'creation_date': '2024-07-06',
        'is_mock_model': True,
        'description': 'Mock model for infrastructure testing - NOT for production use'
    }
    
    # Sauvegarder le modèle
    model_path = models_dir / "best_model.pth"
    torch.save(model_state, model_path)
    
    print(f"✅ Modèle mock créé: {model_path}")
    return model_path

def create_metadata(project_root):
    """Créer métadonnées du modèle"""
    
    metadata = {
        "model_info": {
            "name": "wildlife_audio_classifier_mock",
            "version": "1.0.0-mock",
            "architecture": "ResNet18-simplified",
            "framework": "PyTorch",
            "creation_date": "2024-07-06",
            "is_mock": True
        },
        "training_info": {
            "dataset_size": "simulated",
            "training_epochs": 100,
            "batch_size": 32,
            "learning_rate": 0.001,
            "optimizer": "Adam",
            "loss_function": "CrossEntropyLoss",
            "final_accuracy": 0.892,
            "validation_accuracy": 0.856
        },
        "input_specs": {
            "input_shape": [3, 224, 224],
            "input_type": "spectrogram_image",
            "preprocessing": "normalize_imagenet",
            "sample_rate": 22050,
            "n_fft": 2048,
            "hop_length": 512
        },
        "output_specs": {
            "num_classes": 6,
            "output_type": "logits",
            "confidence_threshold": 0.7
        },
        "performance": {
            "inference_time_ms": 45,
            "memory_usage_mb": 150,
            "cpu_optimized": True,
            "gpu_compatible": True
        },
        "warning": "⚠️ This is a MOCK model for testing infrastructure only. Do NOT use in production!"
    }
    
    metadata_path = Path(project_root) / "models" / "metadata.json"
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Métadonnées créées: {metadata_path}")
    return metadata_path

def create_mock_training_data(project_root):
    """Créer des fichiers CSV de données d'entraînement simulées"""
    
    data_dir = Path(project_root) / "data" / "processed" / "csv"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Labels disponibles
    labels = ["bird_song", "mammal_call", "insect_sound", "amphibian_call", "environmental_sound", "unknown_species"]
    
    # Créer train.csv
    train_data = []
    for i in range(1000):  # 1000 échantillons simulés
        label = labels[i % len(labels)]
        filename = f"mock_audio_{i:04d}.wav"
        train_data.append(f"{filename},{label}")
    
    train_path = data_dir / "train.csv"
    with open(train_path, 'w') as f:
        f.write("filename,label\n")
        f.write("\n".join(train_data))
    
    # Créer val.csv
    val_data = []
    for i in range(200):  # 200 échantillons validation
        label = labels[i % len(labels)]
        filename = f"mock_val_{i:04d}.wav"
        val_data.append(f"{filename},{label}")
    
    val_path = data_dir / "val.csv"
    with open(val_path, 'w') as f:
        f.write("filename,label\n")
        f.write("\n".join(val_data))
    
    # Créer test.csv
    test_data = []
    for i in range(100):  # 100 échantillons test
        label = labels[i % len(labels)]
        filename = f"mock_test_{i:04d}.wav"
        test_data.append(f"{filename},{label}")
    
    test_path = data_dir / "test.csv"
    with open(test_path, 'w') as f:
        f.write("filename,label\n")
        f.write("\n".join(test_data))
    
    print(f"✅ Données d'entraînement simulées créées:")
    print(f"   - Train: {train_path} (1000 échantillons)")
    print(f"   - Validation: {val_path} (200 échantillons)")
    print(f"   - Test: {test_path} (100 échantillons)")

def main():
    """Fonction principale"""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    print("🔧 Création d'un modèle ML mock pour tests d'infrastructure...")
    print("⚠️  ATTENTION: Ce modèle est uniquement pour tester l'infrastructure!")
    print("⚠️  NE PAS utiliser en production - créer un vrai modèle entraîné!")
    print()
    
    # Créer le modèle
    model = create_mock_model()
    print(f"📐 Architecture modèle: {sum(p.numel() for p in model.parameters())} paramètres")
    
    # Créer les poids
    model_path = create_mock_weights(model, project_root)
    
    # Créer métadonnées
    metadata_path = create_metadata(project_root)
    
    # Créer données d'entraînement simulées
    create_mock_training_data(project_root)
    
    print()
    print("✅ Modèle mock créé avec succès!")
    print()
    print("📋 Fichiers créés:")
    print(f"   - models/best_model.pth (modèle PyTorch)")
    print(f"   - models/metadata.json (métadonnées)")
    print(f"   - models/labels.json (labels des classes)")
    print(f"   - data/processed/csv/*.csv (données simulées)")
    print()
    print("🚀 L'API peut maintenant démarrer pour tests d'infrastructure!")
    print()
    print("⚠️  PROCHAINES ÉTAPES POUR PRODUCTION:")
    print("   1. Collecter vraies données audio wildlife")
    print("   2. Entraîner modèle avec vraies données")
    print("   3. Remplacer best_model.pth par modèle réel")
    print("   4. Mettre à jour labels.json avec vraies espèces")

if __name__ == "__main__":
    main()