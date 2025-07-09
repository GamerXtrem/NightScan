"""
EfficientNet Training Script for Wildlife Audio Classification

Advanced training script with mixed precision, data augmentation, 
comprehensive metrics, and robust checkpointing.
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import json
import csv

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import torchvision.transforms as transforms
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from models.efficientnet_config import EfficientNetConfig, create_model, get_config
from models.data_augmentation import AudioAugmentationPipeline, SpecAugment
from utils.training_utils import (
    setup_training_environment,
    TrainingHistory,
    MixedPrecisionTrainer
)
from utils.metrics import MetricsCalculator
from utils.cross_validation import KFoldCrossValidator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class EfficientNetDataset(Dataset):
    """
    Dataset for loading spectrograms for EfficientNet training.
    """
    
    def __init__(
        self,
        csv_file: Path,
        augmentation_pipeline: Optional[AudioAugmentationPipeline] = None,
        use_mixup: bool = False,
        mixup_alpha: float = 0.2
    ):
        self.samples = []
        self.augmentation_pipeline = augmentation_pipeline
        self.use_mixup = use_mixup
        self.mixup_alpha = mixup_alpha
        
        # Load samples from CSV
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.samples.append((Path(row['path']), int(row['label'])))
        
        logger.info(f"Loaded {len(self.samples)} samples from {csv_file}")
        
        # Basic transforms
        self.basic_transform = transforms.Compose([
            transforms.Lambda(lambda x: (x - x.min()) / (x.max() - x.min() + 1e-9)),
            transforms.Resize((224, 224)),
        ])
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        path, label = self.samples[idx]
        
        # Load spectrogram
        mel = np.load(path)
        mel = torch.tensor(mel, dtype=torch.float32)
        
        # Convert to 3 channels for EfficientNet
        if mel.dim() == 2:
            mel = mel.unsqueeze(0).repeat(3, 1, 1)
        
        # Apply basic transforms
        mel = self.basic_transform(mel)
        
        # Apply augmentations if available
        if self.augmentation_pipeline and self.training:
            mel = self.augmentation_pipeline.augment_spectrogram(mel)
        
        # Handle mixup
        if self.use_mixup and self.training and hasattr(self, 'mixup_target_idx'):
            # Get second sample for mixup
            path2, label2 = self.samples[self.mixup_target_idx]
            mel2 = np.load(path2)
            mel2 = torch.tensor(mel2, dtype=torch.float32)
            
            if mel2.dim() == 2:
                mel2 = mel2.unsqueeze(0).repeat(3, 1, 1)
            
            mel2 = self.basic_transform(mel2)
            if self.augmentation_pipeline:
                mel2 = self.augmentation_pipeline.augment_spectrogram(mel2)
            
            # Apply mixup
            mixed_mel, mixed_label = self.augmentation_pipeline.apply_mixup(
                mel, mel2, label, label2, num_classes=6
            )
            
            return mixed_mel, mixed_label
        
        # Convert label to one-hot for consistency
        label_onehot = torch.zeros(6)
        label_onehot[label] = 1.0
        
        return mel, label_onehot
    
    def set_training_mode(self, training: bool) -> None:
        """Set training mode for augmentations."""
        self.training = training


def create_data_loaders(
    train_csv: Path,
    val_csv: Path,
    config: EfficientNetConfig,
    augmentation_pipeline: Optional[AudioAugmentationPipeline] = None
) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation data loaders.
    
    Args:
        train_csv: Path to training CSV file
        val_csv: Path to validation CSV file
        config: EfficientNet configuration
        augmentation_pipeline: Optional augmentation pipeline
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Create datasets
    train_dataset = EfficientNetDataset(
        train_csv,
        augmentation_pipeline=augmentation_pipeline,
        use_mixup=config.use_augmentation
    )
    train_dataset.set_training_mode(True)
    
    val_dataset = EfficientNetDataset(val_csv)
    val_dataset.set_training_mode(False)
    
    # Determine device for pin_memory
    pin_memory = config.device != "cpu"
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=pin_memory,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=pin_memory
    )
    
    return train_loader, val_loader


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    trainer: MixedPrecisionTrainer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int
) -> Dict[str, float]:
    """
    Train model for one epoch.
    
    Args:
        model: Model to train
        train_loader: Training data loader
        trainer: Mixed precision trainer
        criterion: Loss function
        device: Device to use
        epoch: Current epoch number
        
    Returns:
        Training metrics
    """
    model.train()
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}")
    
    for batch_idx, (inputs, targets) in enumerate(progress_bar):
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Handle both regular labels and one-hot encoded labels
        if targets.dim() == 2:  # One-hot encoded (from mixup)
            target_labels = targets.argmax(dim=1)
        else:  # Regular labels
            target_labels = targets
            targets = torch.zeros(targets.size(0), 6, device=device)
            targets.scatter_(1, target_labels.unsqueeze(1), 1.0)
        
        # Training step
        loss = trainer.train_step((inputs, targets), criterion, device)
        total_loss += loss
        
        # Calculate accuracy
        with torch.no_grad():
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += target_labels.size(0)
            correct += (predicted == target_labels).sum().item()
        
        # Update progress bar
        progress_bar.set_postfix({
            'Loss': f'{loss:.4f}',
            'Acc': f'{100. * correct / total:.2f}%'
        })
    
    avg_loss = total_loss / len(train_loader)
    accuracy = 100. * correct / total
    
    return {
        'train_loss': avg_loss,
        'train_accuracy': accuracy
    }


def validate_epoch(
    model: nn.Module,
    val_loader: DataLoader,
    trainer: MixedPrecisionTrainer,
    criterion: nn.Module,
    device: torch.device,
    metrics_calculator: MetricsCalculator
) -> Dict[str, float]:
    """
    Validate model for one epoch.
    
    Args:
        model: Model to validate
        val_loader: Validation data loader
        trainer: Mixed precision trainer
        criterion: Loss function
        device: Device to use
        metrics_calculator: Metrics calculator
        
    Returns:
        Validation metrics
    """
    model.eval()
    
    total_loss = 0.0
    metrics_calculator.reset()
    
    with torch.no_grad():
        for inputs, targets in tqdm(val_loader, desc="Validation"):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Handle both regular labels and one-hot encoded labels
            if targets.dim() == 2:  # One-hot encoded
                target_labels = targets.argmax(dim=1)
            else:  # Regular labels
                target_labels = targets
                targets = torch.zeros(targets.size(0), 6, device=device)
                targets.scatter_(1, target_labels.unsqueeze(1), 1.0)
            
            # Validation step
            loss, outputs, _ = trainer.validate_step((inputs, targets), criterion, device)
            total_loss += loss
            
            # Update metrics
            _, predicted = torch.max(outputs, 1)
            probabilities = torch.softmax(outputs, dim=1)
            metrics_calculator.update(predicted, target_labels, probabilities)
    
    avg_loss = total_loss / len(val_loader)
    val_metrics = metrics_calculator.compute_basic_metrics()
    
    return {
        'val_loss': avg_loss,
        'val_accuracy': val_metrics.get('accuracy', 0) * 100,
        'val_f1_macro': val_metrics.get('f1_macro', 0),
        'val_f1_weighted': val_metrics.get('f1_weighted', 0),
        'val_precision_macro': val_metrics.get('precision_macro', 0),
        'val_recall_macro': val_metrics.get('recall_macro', 0)
    }


def train_model(
    config: EfficientNetConfig,
    train_csv: Path,
    val_csv: Path,
    model_dir: Path,
    resume_from: Optional[Path] = None,
    use_cross_validation: bool = False
) -> Dict[str, Any]:
    """
    Train EfficientNet model.
    
    Args:
        config: EfficientNet configuration
        train_csv: Path to training CSV file
        val_csv: Path to validation CSV file
        model_dir: Directory to save model
        resume_from: Optional checkpoint to resume from
        use_cross_validation: Whether to use cross-validation
        
    Returns:
        Training results
    """
    # Setup
    device = torch.device(config.device)
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    config_path = model_dir / "config.json"
    with open(config_path, 'w') as f:
        json.dump(config.to_dict(), f, indent=2)
    
    # Create model
    model = create_model(config)
    model.to(device)
    
    logger.info(f"Model created: {config.model_name}")
    logger.info(f"Device: {device}")
    logger.info(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create augmentation pipeline
    augmentation_pipeline = None
    if config.use_augmentation:
        augmentation_config = {
            "use_spec_augment": True,
            "spec_augment_config": {
                "freq_mask_param": config.spec_augment_freq_mask,
                "time_mask_param": config.spec_augment_time_mask
            },
            "use_mixup": True,
            "mixup_alpha": config.mixup_alpha
        }
        augmentation_pipeline = AudioAugmentationPipeline(
            sample_rate=22050,
            **augmentation_config
        )
    
    # Cross-validation mode
    if use_cross_validation:
        logger.info("Starting cross-validation...")
        
        # Load all data for cross-validation
        all_spectrograms = []
        all_labels = []
        
        for csv_file in [train_csv, val_csv]:
            with open(csv_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    all_spectrograms.append(Path(row['path']))
                    all_labels.append(int(row['label']))
        
        # Create cross-validator
        def model_factory():
            return create_model(config)
        
        cv = KFoldCrossValidator(
            model_factory=model_factory,
            config=config.to_dict(),
            class_names=["bird_song", "mammal_call", "insect_sound", 
                        "amphibian_call", "environmental_sound", "unknown_species"],
            k_folds=5,
            save_dir=model_dir / "cross_validation"
        )
        
        # Run cross-validation
        cv_results = cv.run_cross_validation(all_spectrograms, all_labels)
        cv.print_summary()
        
        return cv_results
    
    # Regular training mode
    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        train_csv, val_csv, config, augmentation_pipeline
    )
    
    # Setup training environment
    optimizer, scheduler, checkpoint, early_stopping, trainer = setup_training_environment(
        model, config.to_dict(), device, model_dir
    )
    
    # Loss function with class weights
    class_weights = None
    if config.use_class_weights and config.class_weights:
        weights = [config.class_weights[class_name] for class_name in 
                  ["bird_song", "mammal_call", "insect_sound", 
                   "amphibian_call", "environmental_sound", "unknown_species"]]
        class_weights = torch.tensor(weights, dtype=torch.float32, device=device)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Training history and metrics
    history = TrainingHistory(model_dir)
    metrics_calculator = MetricsCalculator(
        class_names=["bird_song", "mammal_call", "insect_sound", 
                    "amphibian_call", "environmental_sound", "unknown_species"],
        save_dir=model_dir / "metrics"
    )
    
    # Resume from checkpoint if provided
    start_epoch = 0
    if resume_from and resume_from.exists():
        logger.info(f"Resuming from checkpoint: {resume_from}")
        checkpoint_data = torch.load(resume_from, map_location=device)
        model.load_state_dict(checkpoint_data["model_state_dict"])
        optimizer.load_state_dict(checkpoint_data["optimizer_state_dict"])
        start_epoch = checkpoint_data["epoch"] + 1
        logger.info(f"Resumed from epoch {start_epoch}")
    
    # Training loop
    logger.info("Starting training...")
    best_val_acc = 0.0
    training_start_time = time.time()
    
    for epoch in range(start_epoch, config.epochs):
        epoch_start_time = time.time()
        
        # Train epoch
        train_metrics = train_epoch(
            model, train_loader, trainer, criterion, device, epoch + 1
        )
        
        # Validate epoch
        val_metrics = validate_epoch(
            model, val_loader, trainer, criterion, device, metrics_calculator
        )
        
        # Combine metrics
        epoch_metrics = {**train_metrics, **val_metrics}
        epoch_metrics['learning_rate'] = scheduler.get_lr()[0]
        epoch_metrics['epoch_time'] = time.time() - epoch_start_time
        
        # Update history
        history.update(epoch_metrics)
        
        # Log progress
        logger.info(
            f"Epoch {epoch + 1}/{config.epochs} - "
            f"Train Loss: {train_metrics['train_loss']:.4f}, "
            f"Train Acc: {train_metrics['train_accuracy']:.2f}%, "
            f"Val Loss: {val_metrics['val_loss']:.4f}, "
            f"Val Acc: {val_metrics['val_accuracy']:.2f}%, "
            f"Val F1: {val_metrics['val_f1_macro']:.4f}"
        )
        
        # Save checkpoint
        checkpoint(epoch, model, optimizer, scheduler, epoch_metrics)
        
        # Early stopping
        if early_stopping(val_metrics['val_loss'], model):
            logger.info(f"Early stopping at epoch {epoch + 1}")
            break
        
        # Track best accuracy
        if val_metrics['val_accuracy'] > best_val_acc:
            best_val_acc = val_metrics['val_accuracy']
        
        # Step scheduler
        scheduler.step(val_metrics['val_loss'])
    
    # Training completed
    total_training_time = time.time() - training_start_time
    logger.info(f"Training completed in {total_training_time:.2f} seconds")
    logger.info(f"Best validation accuracy: {best_val_acc:.2f}%")
    
    # Save final results
    history.save("training_history.json")
    metrics_calculator.save_comprehensive_report("final_metrics.json")
    metrics_calculator.create_summary_plots()
    
    # Save model metadata
    metadata = {
        "model_name": config.model_name,
        "best_val_accuracy": best_val_acc,
        "total_training_time": total_training_time,
        "epochs_trained": epoch + 1,
        "config": config.to_dict(),
        "final_metrics": val_metrics
    }
    
    with open(model_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return {
        "best_val_accuracy": best_val_acc,
        "total_training_time": total_training_time,
        "epochs_trained": epoch + 1,
        "final_metrics": val_metrics
    }


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train EfficientNet on wildlife audio spectrograms")
    
    # Required arguments
    parser.add_argument("--train_csv", type=Path, required=True, 
                       help="Path to training CSV file")
    parser.add_argument("--val_csv", type=Path, required=True,
                       help="Path to validation CSV file")
    parser.add_argument("--model_dir", type=Path, required=True,
                       help="Directory to save trained model")
    
    # Model configuration
    parser.add_argument("--config", type=str, default="efficientnet_b1_balanced",
                       help="Model configuration name")
    parser.add_argument("--model_name", type=str, default=None,
                       help="EfficientNet model name (overrides config)")
    parser.add_argument("--batch_size", type=int, default=None,
                       help="Batch size (overrides config)")
    parser.add_argument("--epochs", type=int, default=None,
                       help="Number of epochs (overrides config)")
    parser.add_argument("--learning_rate", type=float, default=None,
                       help="Learning rate (overrides config)")
    
    # Training options
    parser.add_argument("--resume_from", type=Path, default=None,
                       help="Checkpoint to resume from")
    parser.add_argument("--cross_validation", action="store_true",
                       help="Use k-fold cross-validation")
    parser.add_argument("--no_augmentation", action="store_true",
                       help="Disable data augmentation")
    
    args = parser.parse_args()
    
    # Get configuration
    config = get_config(args.config)
    
    # Override configuration with command line arguments
    if args.model_name:
        config.model_name = args.model_name
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.epochs:
        config.epochs = args.epochs
    if args.learning_rate:
        config.learning_rate = args.learning_rate
    if args.no_augmentation:
        config.use_augmentation = False
    
    # Validate inputs
    if not args.train_csv.exists():
        raise FileNotFoundError(f"Training CSV not found: {args.train_csv}")
    if not args.val_csv.exists():
        raise FileNotFoundError(f"Validation CSV not found: {args.val_csv}")
    
    # Start training
    logger.info("Starting EfficientNet training...")
    logger.info(f"Configuration: {args.config}")
    logger.info(f"Model: {config.model_name}")
    logger.info(f"Batch size: {config.batch_size}")
    logger.info(f"Epochs: {config.epochs}")
    logger.info(f"Learning rate: {config.learning_rate}")
    
    try:
        results = train_model(
            config=config,
            train_csv=args.train_csv,
            val_csv=args.val_csv,
            model_dir=args.model_dir,
            resume_from=args.resume_from,
            use_cross_validation=args.cross_validation
        )
        
        logger.info("Training completed successfully!")
        if not args.cross_validation:
            logger.info(f"Best validation accuracy: {results['best_val_accuracy']:.2f}%")
            logger.info(f"Model saved to: {args.model_dir}")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()