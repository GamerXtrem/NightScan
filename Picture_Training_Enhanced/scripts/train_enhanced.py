"""
Enhanced Photo Training Script for Wildlife Image Classification

Advanced training script with support for multiple architectures, mixed precision,
comprehensive data augmentation, and detailed metrics tracking.
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
from PIL import Image

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from models.photo_config import PhotoConfig, create_model, get_config, get_model_info
from models.data_augmentation import AugmentationManager
from utils.training_utils import TrainingManager
from utils.metrics import MetricsCalculator, evaluate_model

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


class PhotoDataset(Dataset):
    """Enhanced dataset for wildlife photo classification."""
    
    def __init__(self, csv_file: Path, augmentation_manager: AugmentationManager, 
                 is_training: bool = True, class_to_idx: Optional[Dict[str, int]] = None):
        """
        Initialize photo dataset.
        
        Args:
            csv_file: Path to CSV file with 'path' and 'label' columns
            augmentation_manager: Augmentation manager
            is_training: Whether this is training data
            class_to_idx: Mapping from class names to indices
        """
        self.samples = []
        self.class_names = []
        self.class_to_idx = class_to_idx or {}
        self.augmentation_manager = augmentation_manager
        self.is_training = is_training
        
        # Read CSV file
        with open(csv_file, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                path = Path(row["path"])
                if not path.exists():
                    logger.warning(f"Image not found: {path}")
                    continue
                
                # Handle both string labels and numeric labels
                label = row["label"]
                if label.isdigit():
                    label_idx = int(label)
                else:
                    if label not in self.class_to_idx:
                        if not class_to_idx:  # Only build mapping if not provided
                            self.class_to_idx[label] = len(self.class_to_idx)
                        else:
                            logger.warning(f"Unknown class: {label}")
                            continue
                    label_idx = self.class_to_idx[label]
                
                self.samples.append((path, label_idx))
        
        # Build class names list
        if not class_to_idx:
            self.class_names = [k for k, v in sorted(self.class_to_idx.items(), key=lambda x: x[1])]
        else:
            self.class_names = [k for k, v in sorted(class_to_idx.items(), key=lambda x: x[1])]
        
        logger.info(f"Loaded {len(self.samples)} samples from {csv_file}")
        logger.info(f"Classes: {self.class_names}")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        
        try:
            # Load image
            image = Image.open(path).convert("RGB")
            
            # Apply augmentations
            image_tensor = self.augmentation_manager.image_augmentation(image, self.is_training)
            
            return image_tensor, label
        
        except Exception as e:
            logger.error(f"Error loading image {path}: {e}")
            # Return a blank image with the same dimensions
            blank_image = Image.new('RGB', (224, 224), color='black')
            image_tensor = self.augmentation_manager.image_augmentation(blank_image, False)
            return image_tensor, label


def create_data_loaders(config: PhotoConfig, csv_dir: Path, 
                       augmentation_manager: AugmentationManager) -> Tuple[DataLoader, DataLoader, List[str]]:
    """Create train and validation data loaders."""
    
    # Create train dataset first to get class mapping
    train_dataset = PhotoDataset(
        csv_dir / "train.csv",
        augmentation_manager,
        is_training=True
    )
    
    # Use the same class mapping for validation
    val_dataset = PhotoDataset(
        csv_dir / "val.csv",
        augmentation_manager,
        is_training=False,
        class_to_idx=train_dataset.class_to_idx
    )
    
    # Verify consistent class mapping
    assert len(train_dataset.class_names) == len(val_dataset.class_names), \
        "Train and validation datasets have different number of classes"
    
    # Update config with actual number of classes
    config.num_classes = len(train_dataset.class_names)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, val_loader, train_dataset.class_names


def train_epoch(model: nn.Module, train_loader: DataLoader, optimizer, scheduler,
               criterion: nn.Module, device: torch.device, trainer: TrainingManager,
               augmentation_manager: AugmentationManager, epoch: int) -> Tuple[float, float]:
    """Train one epoch with advanced features."""
    
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    # Unfreeze backbone if needed
    if hasattr(model, 'config') and model.config.freeze_backbone:
        if epoch >= model.config.unfreeze_at_epoch:
            model.unfreeze_backbone()
            logger.info(f"Unfroze backbone at epoch {epoch}")
    
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)
    
    for batch_idx, (inputs, targets) in enumerate(progress_bar):
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Apply batch-level augmentations (CutMix, MixUp)
        inputs, label_info = augmentation_manager.apply_batch_augmentation(inputs, targets)
        
        # Mixed precision forward pass
        with autocast(enabled=trainer.mixed_precision.enabled):
            outputs = model(inputs)
            loss = augmentation_manager.compute_loss(outputs, label_info, criterion)
            loss = trainer.gradient_accumulator.scale_loss(loss)
        
        # Backward pass
        trainer.mixed_precision.backward(loss)
        
        # Gradient clipping
        if trainer.config.get("gradient_clipping", 0) > 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 
                trainer.config["gradient_clipping"]
            )
        
        # Optimizer step
        if trainer.gradient_accumulator.should_update():
            trainer.mixed_precision.step(optimizer)
            optimizer.zero_grad()
        
        # Statistics (only for non-mixed labels)
        if not isinstance(label_info, dict):
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        total_loss += loss.item()
        
        # Update progress bar
        if not isinstance(label_info, dict):
            progress_bar.set_postfix({
                "Loss": f"{loss.item():.4f}",
                "Acc": f"{100.*correct/total:.2f}%"
            })
        else:
            progress_bar.set_postfix({
                "Loss": f"{loss.item():.4f}",
                "Aug": label_info["type"]
            })
    
    # Step scheduler if not plateau-based
    if trainer.config["scheduler"] != "reduce_on_plateau":
        scheduler.step()
    
    avg_loss = total_loss / len(train_loader)
    accuracy = 100. * correct / total if total > 0 else 0.0
    
    return avg_loss, accuracy


def validate_epoch(model: nn.Module, val_loader: DataLoader, criterion: nn.Module,
                  device: torch.device, epoch: int) -> Tuple[float, float]:
    """Validate one epoch."""
    
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc=f"Validation {epoch}", leave=False)
        
        for inputs, targets in progress_bar:
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            progress_bar.set_postfix({
                "Loss": f"{loss.item():.4f}",
                "Acc": f"{100.*correct/total:.2f}%"
            })
    
    avg_loss = total_loss / len(val_loader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Enhanced Photo Training for Wildlife Classification")
    parser.add_argument("--config", type=str, default="efficientnet_b1_balanced",
                       help="Configuration preset name")
    parser.add_argument("--csv_dir", type=Path, required=True,
                       help="Directory containing train.csv and val.csv")
    parser.add_argument("--model_dir", type=Path, required=True,
                       help="Directory to save model and results")
    parser.add_argument("--resume", type=Path, help="Path to checkpoint to resume from")
    parser.add_argument("--evaluate_only", action="store_true",
                       help="Only evaluate the model")
    parser.add_argument("--test_csv", type=Path, help="Test CSV file for evaluation")
    
    # Override config parameters
    parser.add_argument("--epochs", type=int, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, help="Batch size")
    parser.add_argument("--learning_rate", type=float, help="Learning rate")
    parser.add_argument("--model_name", type=str, help="Model name")
    parser.add_argument("--architecture", type=str, help="Architecture type")
    
    args = parser.parse_args()
    
    # Load configuration
    config = get_config(args.config)
    
    # Override config with command line arguments
    if args.epochs:
        config.epochs = args.epochs
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.learning_rate:
        config.learning_rate = args.learning_rate
    if args.model_name:
        config.model_name = args.model_name
    if args.architecture:
        config.architecture = args.architecture
    
    # Setup device
    device = torch.device(config.device)
    logger.info(f"Using device: {device}")
    
    # Create model directory
    args.model_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize training manager
    trainer = TrainingManager(config.to_dict(), args.model_dir)
    
    # Initialize augmentation manager
    augmentation_manager = AugmentationManager(config.to_dict())
    
    # Create data loaders
    train_loader, val_loader, class_names = create_data_loaders(
        config, args.csv_dir, augmentation_manager
    )
    
    logger.info(f"Training samples: {len(train_loader.dataset)}")
    logger.info(f"Validation samples: {len(val_loader.dataset)}")
    logger.info(f"Classes: {class_names}")
    
    # Create model
    model = create_model(config)
    model.to(device)
    
    # Log model info
    model_info = get_model_info(model)
    logger.info(f"Model: {model_info}")
    
    # Setup optimizer and scheduler
    optimizer, scheduler = trainer.setup_optimizer_and_scheduler(model)
    
    # Loss function with class weights
    if config.use_class_weights and config.class_weights:
        # Convert class weights to tensor
        weight_tensor = torch.ones(config.num_classes)
        for i, class_name in enumerate(class_names):
            if class_name in config.class_weights:
                weight_tensor[i] = config.class_weights[class_name]
        criterion = nn.CrossEntropyLoss(weight=weight_tensor.to(device))
    else:
        criterion = nn.CrossEntropyLoss()
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        checkpoint_info = trainer.load_checkpoint(args.resume, model, optimizer)
        start_epoch = checkpoint_info["epoch"] + 1
        trainer.history.history = checkpoint_info["history"]
        logger.info(f"Resumed from epoch {start_epoch}")
    
    # Evaluation only mode
    if args.evaluate_only:
        test_csv = args.test_csv or args.csv_dir / "val.csv"
        if not test_csv.exists():
            logger.error(f"Test CSV not found: {test_csv}")
            return
        
        # Create test dataset
        test_dataset = PhotoDataset(
            test_csv,
            augmentation_manager,
            is_training=False,
            class_to_idx={name: i for i, name in enumerate(class_names)}
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        # Evaluate model
        metrics = evaluate_model(model, test_loader, class_names, device, 
                               args.model_dir / "evaluation")
        
        logger.info(f"Evaluation complete. Accuracy: {metrics['accuracy']:.4f}")
        return
    
    # Training loop
    logger.info("Starting training...")
    best_val_acc = 0.0
    
    for epoch in range(start_epoch, config.epochs):
        epoch_start_time = time.time()
        
        # Training phase
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, scheduler, criterion, device,
            trainer, augmentation_manager, epoch
        )
        
        # Validation phase
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device, epoch)
        
        epoch_time = time.time() - epoch_start_time
        
        # Get current learning rate
        current_lr = scheduler.get_last_lr()[0]
        
        # Update training history
        trainer.history.update(
            epoch, train_loss, val_loss, train_acc, val_acc, current_lr, epoch_time
        )
        
        # Log progress
        logger.info(f"Epoch {epoch:3d}/{config.epochs}: "
                   f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                   f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, "
                   f"LR: {current_lr:.2e}, Time: {epoch_time:.1f}s")
        
        # Step scheduler if plateau-based
        if trainer.config["scheduler"] == "reduce_on_plateau":
            scheduler.step(val_acc)
        
        # Save checkpoint
        trainer.checkpoint(val_acc, model, optimizer, epoch, trainer.history)
        
        # Early stopping
        if trainer.early_stopping(val_acc, model):
            logger.info(f"Early stopping at epoch {epoch}")
            break
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), args.model_dir / "best_model.pth")
    
    # Final evaluation
    logger.info("Training complete. Evaluating final model...")
    
    # Load best model
    model.load_state_dict(torch.load(args.model_dir / "best_model.pth"))
    
    # Evaluate on validation set
    metrics = evaluate_model(model, val_loader, class_names, device, 
                           args.model_dir / "final_evaluation")
    
    # Save training artifacts
    trainer.save_training_artifacts(model)
    
    # Plot training history
    metrics_calc = MetricsCalculator(class_names, args.model_dir / "plots")
    metrics_calc.plot_training_history(trainer.history.history)
    
    logger.info(f"Training complete! Best validation accuracy: {best_val_acc:.4f}")
    logger.info(f"Final evaluation accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"Results saved to: {args.model_dir}")


if __name__ == "__main__":
    main()