#!/usr/bin/env python3
"""
Script d'entraînement optimisé pour les datasets à grande échelle (1500+ classes)
Conçu pour gérer efficacement des millions d'échantillons audio
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from pathlib import Path
import logging
import json
import gc
from datetime import datetime
from tqdm import tqdm
import psutil

# Ajouter le chemin parent
sys.path.append(str(Path(__file__).parent.parent))

from audio_dataset_scalable import AudioDatasetScalable, create_index_database, create_scalable_data_loaders
from models.efficientnet_config import create_audio_model

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_memory_usage():
    """Retourne l'utilisation mémoire actuelle en MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, epoch, 
                   accumulation_steps=4, log_interval=100):
    """
    Entraîne le modèle pour une époque avec gradient accumulation.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Barre de progression
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    
    optimizer.zero_grad()
    
    for batch_idx, (data, targets) in enumerate(pbar):
        # Transférer sur GPU
        data = data.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
        # Mixed precision forward pass
        with autocast():
            outputs = model(data)
            loss = criterion(outputs, targets)
            loss = loss / accumulation_steps  # Normaliser pour l'accumulation
        
        # Backward pass
        scaler.scale(loss).backward()
        
        # Mise à jour des poids après accumulation_steps
        if (batch_idx + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        # Statistiques
        running_loss += loss.item() * accumulation_steps
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        # Mise à jour de la barre de progression
        if batch_idx % log_interval == 0:
            accuracy = 100. * correct / total
            avg_loss = running_loss / (batch_idx + 1)
            memory_mb = get_memory_usage()
            
            pbar.set_postfix({
                'Loss': f'{avg_loss:.4f}',
                'Acc': f'{accuracy:.2f}%',
                'Mem': f'{memory_mb:.0f}MB'
            })
        
        # Nettoyage mémoire périodique
        if batch_idx % 500 == 0:
            gc.collect()
            torch.cuda.empty_cache()
    
    return running_loss / len(train_loader), 100. * correct / total


def validate(model, val_loader, criterion, device):
    """Évalue le modèle sur l'ensemble de validation."""
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, targets in tqdm(val_loader, desc='Validation'):
            data = data.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            
            with autocast():
                outputs = model(data)
                loss = criterion(outputs, targets)
            
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    accuracy = 100. * correct / total
    avg_loss = val_loss / len(val_loader)
    
    return avg_loss, accuracy


def main():
    parser = argparse.ArgumentParser(description='Entraînement NightScan Large Scale')
    
    # Données
    parser.add_argument('--index-db', type=str, required=True,
                       help='Base SQLite contenant l\'index du dataset')
    parser.add_argument('--audio-root', type=str, required=True,
                       help='Répertoire racine des fichiers audio')
    parser.add_argument('--create-index', action='store_true',
                       help='Créer l\'index SQLite avant l\'entraînement')
    
    # Modèle
    parser.add_argument('--model', type=str, default='efficientnet-b3',
                       choices=['efficientnet-b1', 'efficientnet-b2', 'efficientnet-b3', 
                               'efficientnet-b4', 'efficientnet-b5'],
                       help='Architecture du modèle (défaut: efficientnet-b3)')
    parser.add_argument('--num-classes', type=int, default=1500,
                       help='Nombre de classes (défaut: 1500)')
    parser.add_argument('--pretrained', action='store_true', default=True,
                       help='Utiliser les poids pré-entraînés')
    
    # Entraînement
    parser.add_argument('--epochs', type=int, default=100,
                       help='Nombre d\'epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Taille du batch par GPU')
    parser.add_argument('--accumulation-steps', type=int, default=4,
                       help='Gradient accumulation steps (batch effectif = batch-size * accumulation)')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate initial')
    parser.add_argument('--weight-decay', type=float, default=0.01,
                       help='Weight decay')
    
    # Optimisation
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Nombre de workers pour le chargement')
    parser.add_argument('--max-samples-per-class', type=int, default=500,
                       help='Maximum d\'échantillons par classe')
    parser.add_argument('--no-balance-classes', action='store_true',
                       help='Désactiver l\'équilibrage des classes')
    parser.add_argument('--spectrogram-cache-dir', type=Path, default=None,
                       help='Répertoire contenant les spectrogrammes pré-générés')
    
    # Sauvegarde
    parser.add_argument('--output-dir', type=str, default='models_large_scale',
                       help='Répertoire de sauvegarde des modèles')
    parser.add_argument('--save-interval', type=int, default=10,
                       help='Sauvegarder le modèle tous les N epochs')
    
    args = parser.parse_args()
    
    # Configuration
    logger.info("=== NightScan Large Scale Training ===")
    logger.info(f"Configuration:")
    logger.info(f"  Model: {args.model}")
    logger.info(f"  Classes: {args.num_classes}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Effective batch: {args.batch_size * args.accumulation_steps}")
    logger.info(f"  Epochs: {args.epochs}")
    logger.info(f"  Balance classes: {not args.no_balance_classes}")
    logger.info(f"  Max samples per class: {args.max_samples_per_class}")
    if args.spectrogram_cache_dir:
        logger.info(f"  Spectrogram cache: {args.spectrogram_cache_dir}")
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")
    if device.type == 'cuda':
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Créer l'index si demandé
    if args.create_index:
        logger.info("Création de l'index SQLite...")
        create_index_database(
            audio_dir=Path(args.audio_root),
            output_db=args.index_db
        )
    
    # Créer les datasets
    logger.info("Chargement des datasets...")
    loaders = create_scalable_data_loaders(
        index_db=args.index_db,
        audio_root=Path(args.audio_root),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_samples_per_class=args.max_samples_per_class,
        balance_classes=not args.no_balance_classes,
        spectrogram_cache_dir=args.spectrogram_cache_dir
    )
    
    if 'train' not in loaders:
        logger.error("Aucun dataset d'entraînement trouvé!")
        return
    
    train_loader = loaders['train']
    val_loader = loaders.get('val', None)
    
    logger.info(f"Dataset train chargé: {len(train_loader)} batches")
    if val_loader:
        logger.info(f"Dataset val chargé: {len(val_loader)} batches")
    
    # Créer le modèle
    logger.info(f"Création du modèle {args.model}...")
    model = create_audio_model(
        num_classes=args.num_classes,
        model_name=args.model,
        pretrained=args.pretrained,
        dropout_rate=0.3
    )
    model = model.to(device)
    
    # Optimiseur et scheduler
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        epochs=args.epochs,
        steps_per_epoch=len(train_loader) // args.accumulation_steps,
        pct_start=0.1,
        anneal_strategy='cos'
    )
    
    # Loss avec label smoothing pour les nombreuses classes
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # Scaler pour mixed precision
    scaler = GradScaler()
    
    # Créer le répertoire de sortie
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Boucle d'entraînement
    best_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(1, args.epochs + 1):
        logger.info(f"\nEpoch {epoch}/{args.epochs}")
        logger.info(f"Learning rate: {scheduler.get_last_lr()[0]:.6f}")
        
        # Entraînement
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler,
            device, epoch, args.accumulation_steps
        )
        
        # Mise à jour du scheduler
        scheduler.step()
        
        # Validation si disponible
        val_loss = None
        val_acc = None
        if val_loader:
            val_loss, val_acc = validate(model, val_loader, criterion, device)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            # Sauvegarder le meilleur modèle
            if val_acc > best_acc:
                best_acc = val_acc
                best_model_path = output_dir / 'best_model.pth'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'scaler_state_dict': scaler.state_dict(),
                    'best_val_acc': best_acc,
                    'train_loss': train_loss,
                    'train_acc': train_acc,
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                    'args': args
                }, best_model_path)
                logger.info(f"✅ Nouveau meilleur modèle sauvegardé! Val Acc: {val_acc:.2f}%")
        
        # Log
        logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        if val_loader:
            logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        logger.info(f"Memory Usage: {get_memory_usage():.0f} MB")
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        
        # Sauvegarde périodique
        if epoch % args.save_interval == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'train_loss': train_loss,
                'train_acc': train_acc,
                'args': args,
                'history': history
            }
            
            checkpoint_path = output_dir / f'checkpoint_epoch_{epoch}.pth'
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"Checkpoint sauvegardé: {checkpoint_path}")
        
        # Nettoyage mémoire
        gc.collect()
        torch.cuda.empty_cache()
    
    # Sauvegarde finale
    final_model_path = output_dir / 'final_model.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'num_classes': args.num_classes,
        'model_name': args.model,
        'history': history
    }, final_model_path)
    logger.info(f"\nModèle final sauvegardé: {final_model_path}")
    
    # Sauvegarder l'historique
    history_path = output_dir / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    logger.info("\nEntraînement terminé!")
    if val_loader and best_acc > 0:
        logger.info(f"Meilleure précision validation: {best_acc:.2f}%")


if __name__ == "__main__":
    main()