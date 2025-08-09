#!/usr/bin/env python3
"""
Script d'entraînement principal pour images réelles avec optimisations GPU L4.
Supporte Mixed Precision Training, gradient accumulation, et monitoring avancé.
"""

import os
import sys
import argparse
import json
import yaml
from pathlib import Path
from datetime import datetime
import time
from typing import Dict, Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import logging

# Importer les modules locaux
from photo_dataset import PhotoDataset
from photo_model_dynamic import create_dynamic_model, estimate_model_size
from metrics import MetricsTracker, compute_confusion_matrix
from visualize_results import ResultsVisualizer

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Aplatit la configuration hiérarchique en configuration plate.
    
    Args:
        config: Configuration hiérarchique (depuis YAML)
        
    Returns:
        Configuration aplatie compatible avec le code existant
    """
    flat_config = {}
    
    # Mapper la configuration hiérarchique vers une configuration plate
    if 'data' in config:
        flat_config['data_dir'] = config['data'].get('data_dir')
        flat_config['metadata_path'] = config['data'].get('metadata_path')
        flat_config['image_size'] = config['data'].get('image_size', 224)
        flat_config['augmentation_level'] = config['data'].get('augmentation_level', 'moderate')
    
    if 'model' in config:
        flat_config['model_name'] = config['model'].get('model_name')
        flat_config['pretrained'] = config['model'].get('pretrained', True)
        flat_config['dropout_rate'] = config['model'].get('dropout_rate', 0.3)
        flat_config['use_attention'] = config['model'].get('use_attention', False)
        flat_config['differential_lr'] = config['model'].get('differential_lr', False)
    
    if 'training' in config:
        flat_config['epochs'] = config['training'].get('epochs', 50)
        flat_config['batch_size'] = config['training'].get('batch_size', 64)
        flat_config['learning_rate'] = config['training'].get('learning_rate', 0.001)
        flat_config['weight_decay'] = config['training'].get('weight_decay', 0.0001)
        flat_config['optimizer'] = config['training'].get('optimizer', 'adamw')
        flat_config['scheduler'] = config['training'].get('scheduler', 'cosine')
        flat_config['min_lr'] = config['training'].get('min_lr', 1e-6)
        flat_config['use_amp'] = config['training'].get('use_amp', True)
        flat_config['gradient_accumulation_steps'] = config['training'].get('gradient_accumulation_steps', 1)
        flat_config['gradient_clip'] = config['training'].get('gradient_clip', 1.0)
        flat_config['use_class_weights'] = config['training'].get('use_class_weights', False)
        flat_config['label_smoothing'] = config['training'].get('label_smoothing', 0.0)
        flat_config['patience'] = config['training'].get('patience', 10)
    
    if 'system' in config:
        flat_config['num_workers'] = config['system'].get('num_workers', 4)
        flat_config['compile_model'] = config['system'].get('compile_model', False)
    
    if 'monitoring' in config:
        flat_config['keep_last_checkpoints'] = config['monitoring'].get('keep_last_checkpoints', 5)
    
    if 'output' in config:
        flat_config['output_dir'] = config['output'].get('output_dir', './outputs')
    
    if 'advanced' in config:
        flat_config['seed'] = config['advanced'].get('seed', 42)
    
    # Ajouter les valeurs qui pourraient venir des arguments CLI
    for key, value in config.items():
        if key not in ['data', 'model', 'training', 'system', 'monitoring', 'output', 'advanced', 'validation', 'infomaniak']:
            flat_config[key] = value
    
    return flat_config


class Trainer:
    """Classe principale pour l'entraînement du modèle."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialise le trainer.
        
        Args:
            config: Configuration complète de l'entraînement
        """
        self.config = config
        self.original_config = config.copy()  # Garder la config originale pour les checkpoints
        self.device = self._setup_device()
        self.start_epoch = 0
        self.best_val_acc = 0.0
        self.patience_counter = 0
        
        # Créer les dossiers de sortie
        self.output_dir = Path(config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.output_dir / 'checkpoints'
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # TensorBoard
        self.writer = SummaryWriter(self.output_dir / 'runs' / datetime.now().strftime('%Y%m%d_%H%M%S'))
        
        # Charger le dataset
        self.dataset = PhotoDataset(
            data_dir=config['data_dir'],
            metadata_path=config.get('metadata_path')
        )
        
        # Créer les dataloaders
        self.dataloaders = self._create_dataloaders()
        
        # Créer le modèle
        self.model = self._create_model()
        
        # Critère et optimiseur
        self.criterion = self._create_criterion()
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # Mixed Precision Training
        self.use_amp = config.get('use_amp', True) and self.device.type == 'cuda'
        self.scaler = GradScaler() if self.use_amp else None
        
        # Gradient accumulation
        self.accumulation_steps = config.get('gradient_accumulation_steps', 1)
        
        # Métriques
        self.metrics_tracker = MetricsTracker(
            num_classes=self.dataset.num_classes,
            class_names=self.dataset.classes if hasattr(self.dataset, 'classes') else []
        )
        
        # Charger un checkpoint si demandé
        if config.get('resume_from'):
            self.load_checkpoint(config['resume_from'])
        
        logger.info(f"Trainer initialisé - Device: {self.device}")
        logger.info(f"Mixed Precision: {self.use_amp}")
        logger.info(f"Gradient Accumulation Steps: {self.accumulation_steps}")
    
    def _setup_device(self) -> torch.device:
        """Configure le device (GPU/CPU)."""
        if torch.cuda.is_available():
            device = torch.device('cuda')
            # Optimisations pour GPU NVIDIA L4
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            logger.info(f"GPU détecté: {torch.cuda.get_device_name(0)}")
            logger.info(f"VRAM disponible: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        else:
            device = torch.device('cpu')
            logger.warning("Pas de GPU disponible, utilisation du CPU")
        
        return device
    
    def _create_dataloaders(self) -> Dict[str, torch.utils.data.DataLoader]:
        """Crée les dataloaders optimisés."""
        batch_size = self.config['batch_size']
        num_workers = self.config.get('num_workers', 4)
        
        # Ajuster le batch size selon la VRAM disponible
        if self.device.type == 'cuda':
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            if vram_gb < 16:
                batch_size = min(batch_size, 32)
                logger.info(f"Batch size ajusté à {batch_size} pour VRAM {vram_gb:.1f}GB")
        
        return self.dataset.create_all_dataloaders(
            batch_size=batch_size,
            num_workers=num_workers,
            image_size=self.config.get('image_size', 224),
            augmentation_level=self.config.get('augmentation_level', 'moderate')
        )
    
    def _create_model(self) -> nn.Module:
        """Crée et configure le modèle."""
        # Estimer les ressources
        estimates = estimate_model_size(self.dataset.num_classes)
        logger.info(f"Estimation du modèle: {estimates}")
        
        # Créer le modèle
        model = create_dynamic_model(
            num_classes=self.dataset.num_classes,
            model_name=self.config.get('model_name'),  # None pour auto-sélection
            pretrained=self.config.get('pretrained', True),
            dropout_rate=self.config.get('dropout_rate', 0.3),
            use_attention=self.config.get('use_attention', False)
        )
        
        model = model.to(self.device)
        
        # Compiler le modèle avec torch.compile si disponible (PyTorch 2.0+)
        if hasattr(torch, 'compile') and self.config.get('compile_model', False):
            model = torch.compile(model, mode='reduce-overhead')
            logger.info("Modèle compilé avec torch.compile")
        
        return model
    
    def _create_criterion(self) -> nn.Module:
        """Crée la fonction de perte."""
        # Obtenir les poids des classes si déséquilibre
        class_weights = None
        if self.config.get('use_class_weights', False):
            class_weights = self.dataset.get_class_weights().to(self.device)
            logger.info(f"Utilisation des poids de classe: {class_weights}")
        
        # Label smoothing si demandé
        label_smoothing = self.config.get('label_smoothing', 0.0)
        
        return nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=label_smoothing
        )
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Crée l'optimiseur."""
        optimizer_name = self.config.get('optimizer', 'adamw')
        lr = self.config.get('learning_rate', 0.001)
        weight_decay = self.config.get('weight_decay', 0.0001)
        
        # Obtenir les groupes de paramètres pour fine-tuning différencié
        if self.config.get('differential_lr', False):
            param_groups = self.model.get_param_groups(base_lr=lr)
        else:
            param_groups = self.model.parameters()
        
        if optimizer_name.lower() == 'adamw':
            optimizer = optim.AdamW(
                param_groups,
                lr=lr,
                weight_decay=weight_decay,
                betas=(0.9, 0.999),
                eps=1e-8
            )
        elif optimizer_name.lower() == 'sgd':
            optimizer = optim.SGD(
                param_groups,
                lr=lr,
                momentum=0.9,
                weight_decay=weight_decay,
                nesterov=True
            )
        else:
            raise ValueError(f"Optimiseur non supporté: {optimizer_name}")
        
        return optimizer
    
    def _create_scheduler(self) -> Optional[object]:
        """Crée le scheduler de learning rate."""
        scheduler_name = self.config.get('scheduler', 'cosine')
        
        if scheduler_name == 'cosine':
            scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=self.config['epochs'],
                eta_min=self.config.get('min_lr', 1e-6)
            )
        elif scheduler_name == 'onecycle':
            steps_per_epoch = len(self.dataloaders['train'])
            scheduler = OneCycleLR(
                self.optimizer,
                max_lr=self.config['learning_rate'],
                epochs=self.config['epochs'],
                steps_per_epoch=steps_per_epoch,
                pct_start=0.3,
                anneal_strategy='cos'
            )
        else:
            scheduler = None
        
        return scheduler
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Entraîne le modèle pour une epoch.
        
        Args:
            epoch: Numéro de l'epoch
            
        Returns:
            Métriques de l'epoch
        """
        self.model.train()
        dataloader = self.dataloaders['train']
        
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Barre de progression
        pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{self.config["epochs"]} [Train]')
        
        for batch_idx, (images, labels) in enumerate(pbar):
            images, labels = images.to(self.device), labels.to(self.device)
            
            # Mixed precision training
            if self.use_amp:
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                    loss = loss / self.accumulation_steps
                
                self.scaler.scale(loss).backward()
                
                # Gradient accumulation
                if (batch_idx + 1) % self.accumulation_steps == 0:
                    # Gradient clipping
                    if self.config.get('gradient_clip', 0) > 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.config['gradient_clip']
                        )
                    
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss = loss / self.accumulation_steps
                loss.backward()
                
                if (batch_idx + 1) % self.accumulation_steps == 0:
                    if self.config.get('gradient_clip', 0) > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.config['gradient_clip']
                        )
                    
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            
            # Métriques
            running_loss += loss.item() * self.accumulation_steps
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Mise à jour de la barre de progression
            current_acc = 100. * correct / total
            pbar.set_postfix({
                'loss': f'{running_loss/(batch_idx+1):.4f}',
                'acc': f'{current_acc:.2f}%',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
            })
            
            # Mise à jour du scheduler si OneCycle
            if isinstance(self.scheduler, OneCycleLR):
                self.scheduler.step()
        
        epoch_loss = running_loss / len(dataloader)
        epoch_acc = 100. * correct / total
        
        return {'loss': epoch_loss, 'accuracy': epoch_acc}
    
    def validate(self, epoch: int) -> Dict[str, float]:
        """
        Valide le modèle.
        
        Args:
            epoch: Numéro de l'epoch
            
        Returns:
            Métriques de validation
        """
        self.model.eval()
        dataloader = self.dataloaders['val']
        
        running_loss = 0.0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{self.config["epochs"]} [Val]')
            
            for images, labels in pbar:
                images, labels = images.to(self.device), labels.to(self.device)
                
                if self.use_amp:
                    with autocast():
                        outputs = self.model(images)
                        loss = self.criterion(outputs, labels)
                else:
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculer les métriques
        val_loss = running_loss / len(dataloader)
        metrics = self.metrics_tracker.compute_metrics(all_labels, all_predictions)
        metrics['loss'] = val_loss
        
        return metrics
    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], is_best: bool = False):
        """Sauvegarde un checkpoint."""
        # Préparer la configuration pour la sérialisation
        config_to_save = {}
        for key, value in self.config.items():
            if isinstance(value, (str, int, float, bool, list, dict, type(None))):
                config_to_save[key] = value
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            'metrics': metrics,
            'config': config_to_save,
            'original_config': self.original_config if hasattr(self, 'original_config') else config_to_save,
            'dataset_info': self.dataset.get_data_info()
        }
        
        # Sauvegarder le checkpoint régulier
        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # Sauvegarder le meilleur modèle
        if is_best:
            best_path = self.output_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            logger.info(f"✅ Nouveau meilleur modèle sauvegardé: {metrics['accuracy']:.2f}%")
        
        # Garder seulement les N derniers checkpoints
        keep_last = self.config.get('keep_last_checkpoints', 5)
        checkpoints = sorted(self.checkpoint_dir.glob('checkpoint_epoch_*.pth'))
        if len(checkpoints) > keep_last:
            for old_checkpoint in checkpoints[:-keep_last]:
                old_checkpoint.unlink()
    
    def load_checkpoint(self, checkpoint_path: str):
        """Charge un checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint.get('scheduler_state_dict'):
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if self.scaler and checkpoint.get('scaler_state_dict'):
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_val_acc = checkpoint['metrics'].get('accuracy', 0)
        
        logger.info(f"Checkpoint chargé: epoch {checkpoint['epoch']}, acc {self.best_val_acc:.2f}%")
    
    def train(self):
        """Lance l'entraînement complet."""
        logger.info("🚀 Début de l'entraînement")
        
        for epoch in range(self.start_epoch, self.config['epochs']):
            start_time = time.time()
            
            # Entraînement
            train_metrics = self.train_epoch(epoch)
            
            # Validation
            val_metrics = self.validate(epoch)
            
            # Mise à jour du scheduler
            if self.scheduler and not isinstance(self.scheduler, OneCycleLR):
                self.scheduler.step()
            
            # Temps de l'epoch
            epoch_time = time.time() - start_time
            
            # Logging
            logger.info(f"\nEpoch {epoch+1}/{self.config['epochs']} - {epoch_time:.1f}s")
            logger.info(f"Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.2f}%")
            logger.info(f"Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.2f}%")
            
            # TensorBoard
            self.writer.add_scalars('Loss', {
                'train': train_metrics['loss'],
                'val': val_metrics['loss']
            }, epoch)
            
            self.writer.add_scalars('Accuracy', {
                'train': train_metrics['accuracy'],
                'val': val_metrics['accuracy']
            }, epoch)
            
            self.writer.add_scalar('Learning_Rate', 
                                  self.optimizer.param_groups[0]['lr'], 
                                  epoch)
            
            # Sauvegarder le checkpoint
            is_best = val_metrics['accuracy'] > self.best_val_acc
            if is_best:
                self.best_val_acc = val_metrics['accuracy']
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            self.save_checkpoint(epoch, val_metrics, is_best)
            
            # Early stopping
            if self.patience_counter >= self.config.get('patience', 10):
                logger.info(f"Early stopping après {epoch+1} epochs")
                break
        
        logger.info("✅ Entraînement terminé!")
        logger.info(f"Meilleure accuracy: {self.best_val_acc:.2f}%")
        
        # Fermer TensorBoard
        self.writer.close()


def load_config(config_path: str) -> Dict[str, Any]:
    """Charge la configuration depuis un fichier YAML ou JSON."""
    config_file = Path(config_path)
    
    if config_file.suffix == '.yaml' or config_file.suffix == '.yml':
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
    elif config_file.suffix == '.json':
        with open(config_file, 'r') as f:
            config = json.load(f)
    else:
        raise ValueError(f"Format de configuration non supporté: {config_file.suffix}")
    
    return config


def main():
    """Fonction principale."""
    parser = argparse.ArgumentParser(description="Entraîner un modèle EfficientNet sur des images réelles")
    
    # Arguments principaux
    parser.add_argument('--config', type=str, help='Fichier de configuration (YAML ou JSON)')
    parser.add_argument('--data_dir', type=str, help='Dossier des données')
    parser.add_argument('--output_dir', type=str, default='./output', help='Dossier de sortie')
    
    # Hyperparamètres
    parser.add_argument('--epochs', type=int, default=50, help='Nombre d\'epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Taille du batch')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='Weight decay')
    
    # Options du modèle
    parser.add_argument('--model_name', type=str, help='Nom du modèle EfficientNet')
    parser.add_argument('--pretrained', action='store_true', help='Utiliser les poids pré-entraînés')
    parser.add_argument('--dropout_rate', type=float, default=0.3, help='Taux de dropout')
    
    # Options d'entraînement
    parser.add_argument('--use_amp', action='store_true', help='Utiliser Mixed Precision')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='Steps d\'accumulation')
    parser.add_argument('--num_workers', type=int, default=4, help='Nombre de workers')
    parser.add_argument('--augmentation_level', type=str, default='moderate', 
                       choices=['light', 'moderate', 'heavy'], help='Niveau d\'augmentation')
    
    # Autres options
    parser.add_argument('--resume_from', type=str, help='Reprendre depuis un checkpoint')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Charger la configuration
    if args.config:
        config = load_config(args.config)
        # Aplatir la configuration hiérarchique
        config = parse_config(config)
        # Les arguments CLI écrasent la configuration
        for key, value in vars(args).items():
            if value is not None and key != 'config':
                config[key] = value
    else:
        config = vars(args)
    
    # Vérifier les arguments requis
    if not config.get('data_dir'):
        parser.error("--data_dir est requis")
    
    # Set random seeds
    torch.manual_seed(config.get('seed', 42))
    np.random.seed(config.get('seed', 42))
    
    # Créer et lancer le trainer
    trainer = Trainer(config)
    trainer.train()


if __name__ == "__main__":
    main()