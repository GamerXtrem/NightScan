"""
Training Utilities for EfficientNet Audio Classification

Advanced training utilities including mixed precision, learning rate scheduling,
checkpointing, and early stopping.
"""

import os
import json
import time
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import logging

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import (
    CosineAnnealingLR, 
    StepLR, 
    ReduceLROnPlateau,
    OneCycleLR
)
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)


class EarlyStopping:
    """
    Early stopping utility to stop training when validation loss stops improving.
    """
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        restore_best_weights: bool = True,
        verbose: bool = True
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.verbose = verbose
        
        self.best_loss = np.inf
        self.counter = 0
        self.best_weights = None
        self.early_stop = False
    
    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        """
        Check if training should be stopped.
        
        Args:
            val_loss: Current validation loss
            model: The model being trained
            
        Returns:
            True if training should be stopped
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
            if self.verbose:
                logger.info(f"Validation loss improved to {val_loss:.6f}")
        else:
            self.counter += 1
            if self.verbose:
                logger.info(f"EarlyStopping counter: {self.counter}/{self.patience}")
            
            if self.counter >= self.patience:
                self.early_stop = True
                if self.restore_best_weights and self.best_weights is not None:
                    model.load_state_dict(self.best_weights)
                    if self.verbose:
                        logger.info("Restored best weights")
                return True
        
        return False


class ModelCheckpoint:
    """
    Save model checkpoints during training.
    """
    
    def __init__(
        self,
        checkpoint_dir: Path,
        monitor: str = "val_loss",
        mode: str = "min",
        save_best_only: bool = True,
        save_weights_only: bool = False,
        verbose: bool = True
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.verbose = verbose
        
        self.best_metric = np.inf if mode == "min" else -np.inf
        self.best_epoch = 0
    
    def __call__(
        self,
        epoch: int,
        model: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: Optional[Any],
        metrics: Dict[str, float]
    ) -> None:
        """
        Save checkpoint if metric has improved.
        
        Args:
            epoch: Current epoch
            model: Model to save
            optimizer: Optimizer state
            scheduler: Learning rate scheduler
            metrics: Training metrics
        """
        if self.monitor not in metrics:
            logger.warning(f"Metric '{self.monitor}' not found in metrics")
            return
        
        current_metric = metrics[self.monitor]
        
        # Check if metric has improved
        is_best = False
        if self.mode == "min":
            is_best = current_metric < self.best_metric
        else:
            is_best = current_metric > self.best_metric
        
        if is_best:
            self.best_metric = current_metric
            self.best_epoch = epoch
            
            # Save best model
            if self.save_weights_only:
                torch.save(
                    model.state_dict(),
                    self.checkpoint_dir / "best_model.pth"
                )
            else:
                checkpoint = {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
                    "best_metric": self.best_metric,
                    "metrics": metrics
                }
                torch.save(checkpoint, self.checkpoint_dir / "best_checkpoint.pth")
            
            if self.verbose:
                logger.info(f"Saved best model at epoch {epoch} with {self.monitor}={current_metric:.6f}")
        
        # Save latest checkpoint (if not save_best_only)
        if not self.save_best_only:
            if self.save_weights_only:
                torch.save(
                    model.state_dict(),
                    self.checkpoint_dir / f"model_epoch_{epoch}.pth"
                )
            else:
                checkpoint = {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
                    "metrics": metrics
                }
                torch.save(checkpoint, self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pth")


class LearningRateScheduler:
    """
    Learning rate scheduler wrapper with multiple scheduling strategies.
    """
    
    def __init__(
        self,
        optimizer: optim.Optimizer,
        scheduler_type: str = "cosine",
        scheduler_params: Optional[Dict[str, Any]] = None,
        warmup_epochs: int = 0,
        warmup_factor: float = 0.1
    ):
        self.optimizer = optimizer
        self.scheduler_type = scheduler_type
        self.warmup_epochs = warmup_epochs
        self.warmup_factor = warmup_factor
        self.current_epoch = 0
        
        if scheduler_params is None:
            scheduler_params = {}
        
        # Create scheduler
        if scheduler_type == "cosine":
            self.scheduler = CosineAnnealingLR(
                optimizer,
                T_max=scheduler_params.get("T_max", 100),
                eta_min=scheduler_params.get("eta_min", 0)
            )
        elif scheduler_type == "step":
            self.scheduler = StepLR(
                optimizer,
                step_size=scheduler_params.get("step_size", 30),
                gamma=scheduler_params.get("gamma", 0.1)
            )
        elif scheduler_type == "reduce_on_plateau":
            self.scheduler = ReduceLROnPlateau(
                optimizer,
                mode=scheduler_params.get("mode", "min"),
                factor=scheduler_params.get("factor", 0.1),
                patience=scheduler_params.get("patience", 10),
                verbose=scheduler_params.get("verbose", True)
            )
        elif scheduler_type == "one_cycle":
            self.scheduler = OneCycleLR(
                optimizer,
                max_lr=scheduler_params.get("max_lr", 1e-3),
                total_steps=scheduler_params.get("total_steps", 1000),
                pct_start=scheduler_params.get("pct_start", 0.3)
            )
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")
    
    def step(self, metric: Optional[float] = None) -> None:
        """
        Step the learning rate scheduler.
        
        Args:
            metric: Metric value for ReduceLROnPlateau scheduler
        """
        if self.current_epoch < self.warmup_epochs:
            # Warmup phase
            warmup_lr = self.warmup_factor + (1 - self.warmup_factor) * (
                self.current_epoch / self.warmup_epochs
            )
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = param_group['initial_lr'] * warmup_lr
        else:
            # Normal scheduling
            if self.scheduler_type == "reduce_on_plateau":
                if metric is not None:
                    self.scheduler.step(metric)
            else:
                self.scheduler.step()
        
        self.current_epoch += 1
    
    def get_lr(self) -> List[float]:
        """Get current learning rates."""
        return [param_group['lr'] for param_group in self.optimizer.param_groups]
    
    def state_dict(self) -> Dict[str, Any]:
        """Get scheduler state dict."""
        return {
            "scheduler_state_dict": self.scheduler.state_dict(),
            "current_epoch": self.current_epoch,
            "scheduler_type": self.scheduler_type,
            "warmup_epochs": self.warmup_epochs,
            "warmup_factor": self.warmup_factor
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load scheduler state dict."""
        self.scheduler.load_state_dict(state_dict["scheduler_state_dict"])
        self.current_epoch = state_dict["current_epoch"]


class MixedPrecisionTrainer:
    """
    Mixed precision training wrapper for faster training and reduced memory usage.
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: Optional[LearningRateScheduler] = None,
        gradient_clipping: Optional[float] = None,
        accumulation_steps: int = 1
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.gradient_clipping = gradient_clipping
        self.accumulation_steps = accumulation_steps
        
        # Initialize GradScaler for mixed precision
        self.scaler = GradScaler()
        self.step_count = 0
    
    def train_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        criterion: nn.Module,
        device: torch.device
    ) -> float:
        """
        Perform a single training step with mixed precision.
        
        Args:
            batch: Input batch (inputs, targets)
            criterion: Loss function
            device: Device to use
            
        Returns:
            Loss value
        """
        inputs, targets = batch
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Forward pass with autocast
        with autocast():
            outputs = self.model(inputs)
            loss = criterion(outputs, targets)
            # Scale loss for gradient accumulation
            loss = loss / self.accumulation_steps
        
        # Backward pass
        self.scaler.scale(loss).backward()
        
        # Gradient accumulation
        if (self.step_count + 1) % self.accumulation_steps == 0:
            # Gradient clipping
            if self.gradient_clipping is not None:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.gradient_clipping
                )
            
            # Optimizer step
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
            
            # Scheduler step
            if self.scheduler is not None:
                self.scheduler.step()
        
        self.step_count += 1
        return loss.item() * self.accumulation_steps
    
    def validate_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        criterion: nn.Module,
        device: torch.device
    ) -> Tuple[float, torch.Tensor, torch.Tensor]:
        """
        Perform a single validation step.
        
        Args:
            batch: Input batch (inputs, targets)
            criterion: Loss function
            device: Device to use
            
        Returns:
            Loss value, predictions, and targets
        """
        inputs, targets = batch
        inputs, targets = inputs.to(device), targets.to(device)
        
        with torch.no_grad():
            with autocast():
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
        
        return loss.item(), outputs, targets


class TrainingHistory:
    """
    Track and save training history.
    """
    
    def __init__(self, save_dir: Path):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
            "learning_rate": [],
            "epoch_time": []
        }
    
    def update(self, epoch_metrics: Dict[str, float]) -> None:
        """
        Update history with epoch metrics.
        
        Args:
            epoch_metrics: Dictionary of metrics for the epoch
        """
        for key, value in epoch_metrics.items():
            if key in self.history:
                self.history[key].append(value)
    
    def save(self, filename: str = "training_history.json") -> None:
        """
        Save training history to JSON file.
        
        Args:
            filename: Name of the file to save
        """
        history_path = self.save_dir / filename
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        logger.info(f"Training history saved to {history_path}")
    
    def load(self, filename: str = "training_history.json") -> None:
        """
        Load training history from JSON file.
        
        Args:
            filename: Name of the file to load
        """
        history_path = self.save_dir / filename
        if history_path.exists():
            with open(history_path, 'r') as f:
                self.history = json.load(f)
            logger.info(f"Training history loaded from {history_path}")
        else:
            logger.warning(f"History file not found: {history_path}")


def setup_training_environment(
    model: nn.Module,
    config: Dict[str, Any],
    device: torch.device,
    checkpoint_dir: Path
) -> Tuple[optim.Optimizer, LearningRateScheduler, ModelCheckpoint, EarlyStopping, MixedPrecisionTrainer]:
    """
    Set up complete training environment.
    
    Args:
        model: Model to train
        config: Training configuration
        device: Device to use
        checkpoint_dir: Directory to save checkpoints
        
    Returns:
        Optimizer, scheduler, checkpoint callback, early stopping, and mixed precision trainer
    """
    # Create optimizer
    if config["optimizer"] == "adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=config["learning_rate"],
            weight_decay=config.get("weight_decay", 0.0)
        )
    elif config["optimizer"] == "adamw":
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config["learning_rate"],
            weight_decay=config.get("weight_decay", 0.0)
        )
    elif config["optimizer"] == "sgd":
        optimizer = optim.SGD(
            model.parameters(),
            lr=config["learning_rate"],
            momentum=config.get("momentum", 0.9),
            weight_decay=config.get("weight_decay", 0.0)
        )
    else:
        raise ValueError(f"Unknown optimizer: {config['optimizer']}")
    
    # Create scheduler
    scheduler = LearningRateScheduler(
        optimizer,
        scheduler_type=config.get("scheduler", "cosine"),
        scheduler_params=config.get("scheduler_params", {}),
        warmup_epochs=config.get("warmup_epochs", 0)
    )
    
    # Create checkpoint callback
    checkpoint = ModelCheckpoint(
        checkpoint_dir,
        monitor=config.get("monitor", "val_loss"),
        mode=config.get("mode", "min"),
        save_best_only=config.get("save_best_only", True)
    )
    
    # Create early stopping
    early_stopping = EarlyStopping(
        patience=config.get("patience", 10),
        min_delta=config.get("min_delta", 0.0),
        verbose=config.get("verbose", True)
    )
    
    # Create mixed precision trainer
    mixed_precision_trainer = MixedPrecisionTrainer(
        model,
        optimizer,
        scheduler,
        gradient_clipping=config.get("gradient_clipping"),
        accumulation_steps=config.get("accumulation_steps", 1)
    )
    
    return optimizer, scheduler, checkpoint, early_stopping, mixed_precision_trainer


if __name__ == "__main__":
    # Test training utilities
    print("Testing training utilities...")
    
    # Create dummy model and config
    model = nn.Linear(10, 2)
    config = {
        "optimizer": "adamw",
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "scheduler": "cosine",
        "patience": 5,
        "gradient_clipping": 1.0
    }
    
    device = torch.device("cpu")
    checkpoint_dir = Path("./test_checkpoints")
    
    # Setup training environment
    optimizer, scheduler, checkpoint, early_stopping, trainer = setup_training_environment(
        model, config, device, checkpoint_dir
    )
    
    print("Training environment setup successful!")
    
    # Clean up
    if checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)
    
    print("All tests passed!")