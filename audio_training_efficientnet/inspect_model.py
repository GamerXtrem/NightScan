#!/usr/bin/env python3
"""
Script pour inspecter un checkpoint de modèle et afficher ses informations
"""

import torch
import argparse
import json
from pathlib import Path

def inspect_checkpoint(checkpoint_path):
    """Inspecte un checkpoint et affiche toutes les informations disponibles."""
    
    print(f"\n📦 Inspection du checkpoint: {checkpoint_path}")
    print("=" * 80)
    
    # Charger le checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Afficher les clés disponibles
    print("\n🔑 Clés disponibles dans le checkpoint:")
    for key in checkpoint.keys():
        if key == 'model_state_dict':
            print(f"  - {key} (modèle)")
        elif key == 'history':
            print(f"  - {key} (historique d'entraînement)")
        else:
            print(f"  - {key}")
    
    # Informations sur le modèle
    print("\n🤖 Informations sur le modèle:")
    print(f"  - Nombre de classes: {checkpoint.get('num_classes', 'Non spécifié')}")
    print(f"  - Architecture: {checkpoint.get('model_name', 'Non spécifié')}")
    
    # Classes
    if 'class_names' in checkpoint:
        class_names = checkpoint['class_names']
        print(f"\n📋 Classes ({len(class_names)} au total):")
        # Afficher les premières et dernières classes
        if len(class_names) <= 20:
            for i, name in enumerate(class_names):
                print(f"  {i:3d}: {name}")
        else:
            print("  Premières 10 classes:")
            for i in range(10):
                print(f"  {i:3d}: {class_names[i]}")
            print("  ...")
            print("  Dernières 10 classes:")
            for i in range(len(class_names)-10, len(class_names)):
                print(f"  {i:3d}: {class_names[i]}")
    else:
        print("\n⚠️  Les noms de classes ne sont PAS sauvegardés dans ce checkpoint!")
        print("  Le modèle devra charger les classes depuis une base SQLite externe")
    
    # Performances
    if 'best_val_acc' in checkpoint:
        print(f"\n📊 Performances:")
        print(f"  - Meilleure précision validation: {checkpoint['best_val_acc']:.2f}%")
        print(f"  - Epoch: {checkpoint.get('epoch', 'Non spécifié')}")
        print(f"  - Train Loss: {checkpoint.get('train_loss', 'Non spécifié'):.4f}" if 'train_loss' in checkpoint else "")
        print(f"  - Train Acc: {checkpoint.get('train_acc', 'Non spécifié'):.2f}%" if 'train_acc' in checkpoint else "")
        print(f"  - Val Loss: {checkpoint.get('val_loss', 'Non spécifié'):.4f}" if 'val_loss' in checkpoint else "")
        print(f"  - Val Acc: {checkpoint.get('val_acc', 'Non spécifié'):.2f}%" if 'val_acc' in checkpoint else "")
    
    # Arguments d'entraînement
    if 'args' in checkpoint:
        print(f"\n⚙️  Arguments d'entraînement:")
        args = checkpoint['args']
        print(f"  - Batch size: {getattr(args, 'batch_size', 'Non spécifié')}")
        print(f"  - Learning rate: {getattr(args, 'lr', 'Non spécifié')}")
        print(f"  - Epochs: {getattr(args, 'epochs', 'Non spécifié')}")
        print(f"  - Model: {getattr(args, 'model', 'Non spécifié')}")
    
    # Taille du modèle
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        total_params = sum(p.numel() for p in state_dict.values() if p.dtype == torch.float32)
        model_size_mb = total_params * 4 / 1024 / 1024
        print(f"\n💾 Taille du modèle:")
        print(f"  - Paramètres: {total_params:,}")
        print(f"  - Taille estimée: {model_size_mb:.2f} MB")
    
    # Historique d'entraînement
    if 'history' in checkpoint:
        history = checkpoint['history']
        print(f"\n📈 Historique d'entraînement:")
        if 'train_loss' in history and history['train_loss']:
            print(f"  - Epochs entraînés: {len(history['train_loss'])}")
            print(f"  - Dernière train loss: {history['train_loss'][-1]:.4f}")
            print(f"  - Dernière train acc: {history['train_acc'][-1]:.2f}%")
            if 'val_loss' in history and history['val_loss']:
                print(f"  - Dernière val loss: {history['val_loss'][-1]:.4f}")
                print(f"  - Dernière val acc: {history['val_acc'][-1]:.2f}%")

def main():
    parser = argparse.ArgumentParser(description="Inspecte un checkpoint de modèle NightScan")
    parser.add_argument('checkpoint', type=str, help="Chemin vers le checkpoint (.pth)")
    parser.add_argument('--export-classes', type=str, help="Exporter les noms de classes vers un fichier texte")
    
    args = parser.parse_args()
    
    if not Path(args.checkpoint).exists():
        print(f"❌ Erreur: Le fichier {args.checkpoint} n'existe pas")
        return
    
    # Inspecter le checkpoint
    inspect_checkpoint(args.checkpoint)
    
    # Exporter les classes si demandé
    if args.export_classes:
        checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        if 'class_names' in checkpoint:
            with open(args.export_classes, 'w') as f:
                for class_name in checkpoint['class_names']:
                    f.write(f"{class_name}\n")
            print(f"\n✅ Classes exportées vers: {args.export_classes}")
        else:
            print(f"\n❌ Pas de classes à exporter dans ce checkpoint")

if __name__ == "__main__":
    main()