#!/usr/bin/env python3
"""
Audio Training Script for NightScan EfficientNet
Creates a specialized audio model for spectrogram classification
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
from datetime import datetime
import json
import argparse
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from models.efficientnet_config import create_audio_model, print_model_info
from audio_dataset import create_data_loaders

def load_class_info(csv_dir: Path):
    """Charge les informations sur les classes depuis le fichier JSON."""
    classes_json = csv_dir / 'classes.json'
    if not classes_json.exists():
        raise FileNotFoundError(f"Fichier classes.json non trouvé dans {csv_dir}")
    
    with open(classes_json, 'r', encoding='utf-8') as f:
        class_info = json.load(f)
    
    return class_info['class_names'], class_info['num_classes']

def train_model(model, train_loader, val_loader, num_epochs, device, learning_rate=0.001, weight_decay=0.01, use_amp=True):
    """Entraîne le modèle avec validation."""
    print(f"Entraînement du modèle pour {num_epochs} epochs sur {device}...")
    print(f"Optimiseur: AdamW (lr={learning_rate}, weight_decay={weight_decay})")
    print(f"Mixed Precision Training: {'Activé' if use_amp and device.type == 'cuda' else 'Désactivé'}")
    
    # Obtenir les poids de classe pour gérer le déséquilibre
    if hasattr(train_loader.dataset, 'get_class_weights'):
        class_weights = train_loader.dataset.get_class_weights().to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()
    
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    
    # Configurer l'AMP (Automatic Mixed Precision) pour accélérer l'entraînement
    scaler = GradScaler() if use_amp and device.type == 'cuda' else None
    
    best_val_acc = 0
    best_model_state = None
    training_history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(num_epochs):
        # Phase d'entraînement
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for data, targets in train_pbar:
            data, targets = data.to(device), targets.to(device)
            
            optimizer.zero_grad()
            
            # Utiliser autocast pour mixed precision si disponible
            if scaler is not None:
                with autocast():
                    outputs = model(data)
                    loss = criterion(outputs, targets)
                
                # Backward pass avec gradient scaling
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(data)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
            
            train_loss += loss.item() * data.size(0)
            _, predicted = torch.max(outputs.data, 1)
            train_total += targets.size(0)
            train_correct += (predicted == targets).sum().item()
            
            # Mettre à jour la barre de progression
            train_pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*train_correct/train_total:.2f}%'
            })
        
        # Calculer les métriques d'entraînement
        avg_train_loss = train_loss / train_total
        train_accuracy = 100. * train_correct / train_total
        
        # Phase de validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
            for data, targets in val_pbar:
                data, targets = data.to(device), targets.to(device)
                
                # Utiliser autocast pour la validation aussi si AMP est activé
                if scaler is not None:
                    with autocast():
                        outputs = model(data)
                        loss = criterion(outputs, targets)
                else:
                    outputs = model(data)
                    loss = criterion(outputs, targets)
                
                val_loss += loss.item() * data.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_total += targets.size(0)
                val_correct += (predicted == targets).sum().item()
                
                val_pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100.*val_correct/val_total:.2f}%'
                })
        
        # Calculer les métriques de validation
        avg_val_loss = val_loss / val_total
        val_accuracy = 100. * val_correct / val_total
        
        # Sauvegarder l'historique
        training_history['train_loss'].append(avg_train_loss)
        training_history['train_acc'].append(train_accuracy)
        training_history['val_loss'].append(avg_val_loss)
        training_history['val_acc'].append(val_accuracy)
        
        # Afficher le résumé de l'epoch
        print(f'\nEpoch {epoch+1} Summary:')
        print(f'  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}%')
        print(f'  Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%')
        
        # Ajuster le learning rate
        scheduler.step(avg_val_loss)
        
        # Sauvegarder le meilleur modèle
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            best_model_state = model.state_dict().copy()
            print(f'  ✅ Nouveau meilleur modèle! Val Acc: {val_accuracy:.2f}%')
    
    # Restaurer le meilleur modèle
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, training_history, best_val_acc

def main():
    """Fonction principale d'entraînement."""
    parser = argparse.ArgumentParser(description="Entraînement du modèle audio NightScan")
    parser.add_argument('--data-dir', type=Path, required=True,
                       help="Répertoire contenant les fichiers audio")
    parser.add_argument('--csv-dir', type=Path, default=Path('data/processed/csv'),
                       help="Répertoire contenant les CSV (défaut: data/processed/csv)")
    parser.add_argument('--spectrogram-dir', type=Path, default=None,
                       help="Répertoire pour sauvegarder/charger les spectrogrammes")
    parser.add_argument('--output-dir', type=Path, default=Path('audio_training_efficientnet/models'),
                       help="Répertoire de sortie pour le modèle")
    parser.add_argument('--epochs', type=int, default=50,
                       help="Nombre d'epochs (défaut: 50)")
    parser.add_argument('--batch-size', type=int, default=32,
                       help="Taille du batch (défaut: 32)")
    parser.add_argument('--learning-rate', type=float, default=0.001,
                       help="Learning rate (défaut: 0.001)")
    parser.add_argument('--weight-decay', type=float, default=0.01,
                       help="Weight decay pour AdamW (défaut: 0.01)")
    parser.add_argument('--model-name', type=str, default='efficientnet-b1',
                       help="Modèle à utiliser (défaut: efficientnet-b1)")
    parser.add_argument('--dropout', type=float, default=0.3,
                       help="Taux de dropout (défaut: 0.3)")
    parser.add_argument('--num-workers', type=int, default=8,
                       help="Nombre de workers pour le chargement des données (défaut: 8)")
    parser.add_argument('--no-amp', action='store_true',
                       help="Désactiver l'Automatic Mixed Precision (AMP)")
    parser.add_argument('--persistent-workers', action='store_true', default=True,
                       help="Utiliser des workers persistants pour réduire l'overhead (défaut: True)")
    parser.add_argument('--prefetch-factor', type=int, default=2,
                       help="Nombre de batches à précharger par worker (défaut: 2)")
    parser.add_argument('--pregenerate-spectrograms', action='store_true',
                       help="Prégénérer tous les spectrogrammes avant l'entraînement")
    parser.add_argument('--use-mock-data', action='store_true',
                       help="Utiliser des données simulées (pour les tests)")
    
    args = parser.parse_args()
    
    print("🌙 NightScan Audio EfficientNet Training")
    print("="*50)
    
    # Configurer le device
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Device utilisé: {device}")
    
    # Vérifier si AMP est supporté
    use_amp = not args.no_amp and device.type == 'cuda'
    if device.type == 'cuda':
        print(f"GPU détecté: {torch.cuda.get_device_name(0)}")
        print(f"Mémoire GPU disponible: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    if args.use_mock_data:
        # Mode test avec données simulées
        print("\n⚠️  Mode test avec données simulées")
        num_classes = 6
        class_names = ['bird_song', 'mammal_call', 'insect_sound', 
                      'amphibian_call', 'environmental_sound', 'unknown_species']
    else:
        # Charger les informations sur les classes
        print(f"\n📂 Chargement des données depuis: {args.csv_dir}")
        try:
            class_names, num_classes = load_class_info(args.csv_dir)
            print(f"Classes détectées ({num_classes}): {', '.join(class_names)}")
        except FileNotFoundError as e:
            print(f"\n❌ Erreur: {e}")
            print("Avez-vous exécuté prepare_audio_data.py pour créer les fichiers CSV?")
            return 1
    
    # Créer le modèle
    print(f"\n📦 Création du modèle {args.model_name}...")
    model = create_audio_model(
        num_classes=num_classes,
        model_name=args.model_name,
        pretrained=True,
        dropout_rate=args.dropout
    )
    model.to(device)
    print_model_info(model)
    
    if args.use_mock_data:
        # Créer des données simulées
        print("\n🎵 Génération des données simulées...")
        from models.efficientnet_config import get_audio_classes
        X_train, y_train = create_mock_audio_data(num_samples=800, num_classes=6)
        X_val, y_val = create_mock_audio_data(num_samples=200, num_classes=6)
        
        train_dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(X_train), torch.LongTensor(y_train)
        )
        val_dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(X_val), torch.LongTensor(y_val)
        )
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False
        )
    else:
        # Créer les DataLoaders avec paramètres optimisés
        print("\n📊 Création des DataLoaders...")
        loaders = create_data_loaders(
            csv_dir=args.csv_dir,
            audio_dir=args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            spectrogram_dir=args.spectrogram_dir,
            persistent_workers=args.persistent_workers,
            prefetch_factor=args.prefetch_factor
        )
        
        if 'train' not in loaders or 'val' not in loaders:
            print("\n❌ Erreur: DataLoaders train et val requis")
            return 1
        
        train_loader = loaders['train']
        val_loader = loaders['val']
    
    # Prégénérer les spectrogrammes si demandé
    if args.pregenerate_spectrograms and args.spectrogram_dir:
        print("\n🎵 Prégénération des spectrogrammes...")
        print("Cette étape peut prendre du temps mais accélérera significativement l'entraînement.")
        
        # Importer et exécuter la prégénération
        from pregenerate_spectrograms import pregenerate_spectrograms
        
        for split in ['train', 'val']:
            csv_file = args.csv_dir / f"{split}.csv"
            if csv_file.exists():
                pregenerate_spectrograms(
                    csv_file=csv_file,
                    audio_dir=args.data_dir,
                    spectrogram_dir=args.spectrogram_dir,
                    n_augmentations=5 if split == 'train' else 0,
                    skip_augmented=(split != 'train')
                )
    
    # Entraîner le modèle
    print("\n🚀 Démarrage de l'entraînement...")
    trained_model, history, best_val_acc = train_model(
        model, train_loader, val_loader, 
        num_epochs=args.epochs, 
        device=device,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        use_amp=use_amp
    )
    
    # Créer le répertoire de sortie
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Sauvegarder le modèle
    model_path = args.output_dir / "best_model.pth"
    
    # Créer un checkpoint complet
    checkpoint = {
        'model_state_dict': trained_model.state_dict(),
        'model_config': {
            'model_name': args.model_name,
            'num_classes': num_classes,
            'input_size': (128, 128),
            'pretrained': True,
            'dropout_rate': args.dropout
        },
        'training_info': {
            'epochs': args.epochs,
            'best_val_accuracy': best_val_acc,
            'final_train_accuracy': history['train_acc'][-1] if history['train_acc'] else 0,
            'final_train_loss': history['train_loss'][-1] if history['train_loss'] else 0,
            'device': str(device),
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'weight_decay': args.weight_decay,
            'optimizer': 'AdamW',
            'mixed_precision': use_amp
        },
        'class_names': class_names,
        'training_history': history,
        'metadata': {
            'model_version': '2.0.0',
            'framework': 'pytorch',
            'creation_date': datetime.now().isoformat(),
            'description': f'{args.model_name} specialized for audio spectrogram classification',
            'model_type': 'audio',
            'variant': 'heavy',
            'data_source': 'mock' if args.use_mock_data else str(args.data_dir)
        }
    }
    
    torch.save(checkpoint, model_path)
    print(f"\n✅ Modèle sauvegardé dans: {model_path}")
    
    # Sauvegarder les métadonnées séparément
    metadata_path = args.output_dir / "metadata.json"
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump({
            'model_info': checkpoint['metadata'],
            'training_info': checkpoint['training_info'],
            'class_names': checkpoint['class_names'],
            'num_classes': num_classes
        }, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Métadonnées sauvegardées dans: {metadata_path}")
    
    # Sauvegarder l'historique d'entraînement
    history_path = args.output_dir / "training_history.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"✅ Historique d'entraînement sauvegardé dans: {history_path}")
    
    # Tester le modèle final
    print("\n🧪 Test du modèle entraîné...")
    trained_model.eval()
    with torch.no_grad():
        test_input = torch.randn(1, 3, 128, 128).to(device)
        output = trained_model(test_input)
        probabilities = torch.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1)
        
        print(f"Forme de l'entrée test: {test_input.shape}")
        print(f"Forme de la sortie: {output.shape}")
        print(f"Classe prédite: {class_names[predicted_class.item()]}")
        print(f"Confiance: {probabilities.max().item():.3f}")
    
    print(f"\n🎉 Entraînement terminé avec succès!")
    print(f"Meilleure précision validation: {best_val_acc:.2f}%")
    print(f"Taille du fichier modèle: {model_path.stat().st_size / 1024 / 1024:.1f} MB")
    
    return 0

# Fonction pour créer des données simulées (conservée pour les tests)
def create_mock_audio_data(num_samples=1000, num_classes=6):
    """Crée des spectrogrammes audio simulés pour les tests."""
    print(f"Génération de {num_samples} spectrogrammes simulés...")
    
    X = []
    y = []
    
    for i in range(num_samples):
        # Créer un spectrogramme aléatoire
        base_freq = np.random.rand(128, 128)
        
        # Ajouter des patterns selon la classe
        class_id = i % num_classes
        
        if class_id == 0:  # bird_song
            freq_pattern = np.sin(np.linspace(0, 20, 40)) * 0.3
            base_freq[60:100, :] += freq_pattern.reshape(-1, 1)
        elif class_id == 1:  # mammal_call
            freq_pattern = np.sin(np.linspace(0, 10, 40)) * 0.4
            base_freq[40:80, :] += freq_pattern.reshape(-1, 1)
        elif class_id == 2:  # insect_sound
            for j in range(0, 128, 20):
                base_freq[80:120, j:j+5] += 0.5
        elif class_id == 3:  # amphibian_call
            freq_pattern = np.sin(np.linspace(0, 5, 40)) * 0.3
            base_freq[10:50, :] += freq_pattern.reshape(-1, 1)
        elif class_id == 4:  # environmental_sound
            base_freq += np.random.rand(128, 128) * 0.2
        else:  # unknown_species
            base_freq += np.random.rand(128, 128) * 0.1
        
        # Convertir en RGB
        spectrogram = np.stack([base_freq, base_freq, base_freq], axis=0)
        
        # Normaliser
        spectrogram = (spectrogram - spectrogram.mean()) / (spectrogram.std() + 1e-8)
        
        X.append(spectrogram)
        y.append(class_id)
    
    return np.array(X), np.array(y)


if __name__ == "__main__":
    sys.exit(main())