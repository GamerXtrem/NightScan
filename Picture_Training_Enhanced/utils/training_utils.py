"""
Training Utilities for Photo Classification

Advanced training utilities including mixed precision, learning rate scheduling,
checkpointing, early stopping, and comprehensive training management.
"""

import os
import json
import time
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import logging
import pickle

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


class TrainingHistory:
    """Track and manage training history."""
    
    def __init__(self):
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
            "learning_rate": [],
            "epoch_time": [],
            "best_val_acc": 0.0,
            "best_epoch": 0
        }
    
    def update(self, epoch: int, train_loss: float, val_loss: float, 
               train_acc: float, val_acc: float, lr: float, epoch_time: float):
        """Update training history."""
        self.history["train_loss"].append(train_loss)
        self.history["val_loss"].append(val_loss)
        self.history["train_acc"].append(train_acc)
        self.history["val_acc"].append(val_acc)
        self.history["learning_rate"].append(lr)
        self.history["epoch_time"].append(epoch_time)
        
        if val_acc > self.history["best_val_acc"]:
            self.history["best_val_acc"] = val_acc
            self.history["best_epoch"] = epoch
    
    def save(self, path: Path):
        """Save training history."""
        with open(path, 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def load(self, path: Path):
        """Load training history."""
        with open(path, 'r') as f:
            self.history = json.load(f)
    
    def get_best_metrics(self) -> Dict[str, Any]:
        """Get best validation metrics."""
        if not self.history["val_acc"]:
            return {"best_val_acc": 0.0, "best_epoch": 0}
        
        return {
            "best_val_acc": self.history["best_val_acc"],
            "best_epoch": self.history["best_epoch"],
            "best_val_loss": self.history["val_loss"][self.history["best_epoch"]],
            "best_train_acc": self.history["train_acc"][self.history["best_epoch"]],
            "best_train_loss": self.history["train_loss"][self.history["best_epoch"]]
        }


class EarlyStopping:
    """Early stopping utility."""
    
    def __init__(self, patience: int = 15, min_delta: float = 0.001, 
                 restore_best_weights: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_score = None
        self.counter = 0
        self.best_weights = None
    
    def __call__(self, val_score: float, model: nn.Module) -> bool:
        """Check if training should stop."""
        if self.best_score is None:
            self.best_score = val_score
            self.best_weights = model.state_dict().copy()
        elif val_score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                if self.restore_best_weights:
                    model.load_state_dict(self.best_weights)
                return True
        else:
            self.best_score = val_score
            self.counter = 0
            self.best_weights = model.state_dict().copy()
        
        return False


class ModelCheckpoint:
    """Model checkpointing utility."""
    
    def __init__(self, filepath: Path, monitor: str = "val_acc", 
                 save_best_only: bool = True, save_weights_only: bool = False):
        self.filepath = filepath
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.best_score = None
        
        # Create directory if it doesn't exist
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
    
    def __call__(self, current_score: float, model: nn.Module, optimizer: optim.Optimizer,
                 epoch: int, history: TrainingHistory):
        """Save model checkpoint if conditions are met."""
        should_save = False
        
        if not self.save_best_only:
            should_save = True
        else:
            if self.best_score is None or current_score > self.best_score:
                self.best_score = current_score
                should_save = True
        
        if should_save:
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "history": history.history,
                "best_score": current_score
            }
            
            if self.save_weights_only:
                checkpoint = {"model_state_dict": model.state_dict()}
            
            torch.save(checkpoint, self.filepath)
            logger.info(f"Checkpoint saved: {self.filepath}")


class LearningRateScheduler:
    """Learning rate scheduler wrapper."""
    
    def __init__(self, optimizer: optim.Optimizer, scheduler_type: str, 
                 scheduler_params: Dict[str, Any]):
        self.optimizer = optimizer
        self.scheduler_type = scheduler_type
        self.scheduler_params = scheduler_params
        self.scheduler = self._create_scheduler()
    
    def _create_scheduler(self):
        """Create appropriate scheduler."""
        if self.scheduler_type == "cosine":
            return CosineAnnealingLR(
                self.optimizer, 
                T_max=self.scheduler_params.get("T_max", 50),
                eta_min=self.scheduler_params.get("eta_min", 1e-6)
            )
        elif self.scheduler_type == "step":
            return StepLR(
                self.optimizer,
                step_size=self.scheduler_params.get("step_size", 30),
                gamma=self.scheduler_params.get("gamma", 0.1)
            )
        elif self.scheduler_type == "reduce_on_plateau":
            return ReduceLROnPlateau(
                self.optimizer,
                mode="max",
                factor=self.scheduler_params.get("factor", 0.5),
                patience=self.scheduler_params.get("patience", 10),
                min_lr=self.scheduler_params.get("min_lr", 1e-6)
            )
        elif self.scheduler_type == "one_cycle":
            return OneCycleLR(
                self.optimizer,
                max_lr=self.scheduler_params.get("max_lr", 1e-3),
                steps_per_epoch=self.scheduler_params.get("steps_per_epoch", 100),
                epochs=self.scheduler_params.get("epochs", 50)
            )
        else:
            raise ValueError(f"Unknown scheduler type: {self.scheduler_type}")
    
    def step(self, metric: Optional[float] = None):
        """Step the scheduler."""
        if self.scheduler_type == "reduce_on_plateau":
            self.scheduler.step(metric)
        else:
            self.scheduler.step()
    
    def get_last_lr(self) -> List[float]:
        """Get last learning rate."""
        return self.scheduler.get_last_lr()


class MixedPrecisionTrainer:
    """Mixed precision training wrapper."""
    
    def __init__(self, enabled: bool = True):
        self.enabled = enabled and torch.cuda.is_available()
        self.scaler = GradScaler() if self.enabled else None
    
    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """Scale loss for mixed precision."""
        if self.enabled:
            return self.scaler.scale(loss)
        return loss
    
    def step(self, optimizer: optim.Optimizer) -> bool:
        """Step optimizer with mixed precision."""
        if self.enabled:
            self.scaler.step(optimizer)
            self.scaler.update()
            return True
        else:
            optimizer.step()
            return False
    
    def backward(self, loss: torch.Tensor):
        """Backward pass with mixed precision."""
        if self.enabled:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()


class GradientAccumulator:
    """Gradient accumulation utility."""
    
    def __init__(self, accumulation_steps: int = 1):
        self.accumulation_steps = accumulation_steps
        self.step_count = 0
    
    def should_update(self) -> bool:
        """Check if gradients should be updated."""
        self.step_count += 1
        return self.step_count % self.accumulation_steps == 0
    
    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """Scale loss for gradient accumulation."""
        return loss / self.accumulation_steps


class WarmupScheduler:
    """Warmup learning rate scheduler."""
    
    def __init__(self, optimizer: optim.Optimizer, warmup_epochs: int, 
                 base_lr: float, target_lr: float):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.base_lr = base_lr
        self.target_lr = target_lr
        self.current_epoch = 0
    
    def step(self, epoch: int):
        """Step the warmup scheduler."""
        if epoch < self.warmup_epochs:
            lr = self.base_lr + (self.target_lr - self.base_lr) * (epoch / self.warmup_epochs)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        self.current_epoch = epoch


def setup_training_environment(config: Dict[str, Any], model_dir: Path) -> Dict[str, Any]:
    """Setup training environment with directories and logging."""
    # Create directories
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "checkpoints").mkdir(exist_ok=True)
    (model_dir / "logs").mkdir(exist_ok=True)
    (model_dir / "plots").mkdir(exist_ok=True)
    
    # Setup logging
    log_file = model_dir / "logs" / "training.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    # Save configuration
    config_file = model_dir / "config.json"
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    return {
        "model_dir": model_dir,
        "checkpoint_dir": model_dir / "checkpoints",
        "log_dir": model_dir / "logs",
        "plot_dir": model_dir / "plots",
        "config_file": config_file
    }


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Count model parameters."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        "total": total_params,
        "trainable": trainable_params,
        "non_trainable": total_params - trainable_params
    }


def save_model_summary(model: nn.Module, config: Dict[str, Any], 
                      save_path: Path):
    """Save model summary information."""
    param_count = count_parameters(model)
    
    summary = {
        "model_config": config,
        "parameter_count": param_count,
        "model_size_mb": param_count["total"] * 4 / (1024 * 1024),
        "architecture": str(model)
    }
    
    with open(save_path, 'w') as f:
        json.dump(summary, f, indent=2)


def load_checkpoint(checkpoint_path: Path, model: nn.Module, 
                   optimizer: optim.Optimizer = None) -> Dict[str, Any]:
    """Load model checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    model.load_state_dict(checkpoint["model_state_dict"])
    
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    return {
        "epoch": checkpoint.get("epoch", 0),
        "history": checkpoint.get("history", {}),
        "best_score": checkpoint.get("best_score", 0.0)
    }


def get_device_info() -> Dict[str, Any]:
    """Get device information."""
    device_info = {
        "device": "cpu",
        "device_name": "CPU",
        "memory_total": 0,
        "memory_available": 0
    }
    
    if torch.cuda.is_available():
        device_info["device"] = "cuda"
        device_info["device_name"] = torch.cuda.get_device_name(0)
        device_info["memory_total"] = torch.cuda.get_device_properties(0).total_memory
        device_info["memory_available"] = torch.cuda.memory_reserved(0)
    elif torch.backends.mps.is_available():
        device_info["device"] = "mps"
        device_info["device_name"] = "Apple Silicon GPU"
    
    return device_info


class TrainingManager:
    """Comprehensive training management."""
    
    def __init__(self, config: Dict[str, Any], model_dir: Path):
        self.config = config
        self.model_dir = model_dir
        self.env = setup_training_environment(config, model_dir)
        
        # Initialize training utilities
        self.history = TrainingHistory()
        self.early_stopping = EarlyStopping(
            patience=config.get("patience", 15),
            min_delta=config.get("min_delta", 0.001)
        )
        self.checkpoint = ModelCheckpoint(
            filepath=self.env["checkpoint_dir"] / "best_model.pth",
            monitor="val_acc"
        )
        self.mixed_precision = MixedPrecisionTrainer(
            enabled=config.get("mixed_precision", True)
        )
        self.gradient_accumulator = GradientAccumulator(
            accumulation_steps=config.get("gradient_accumulation_steps", 1)
        )
        
        # Device info
        self.device_info = get_device_info()
        logger.info(f"Training on: {self.device_info['device_name']}")
    
    def setup_optimizer_and_scheduler(self, model: nn.Module) -> Tuple[optim.Optimizer, Any]:
        """Setup optimizer and scheduler."""
        # Optimizer
        if self.config["optimizer"] == "adam":
            optimizer = optim.Adam(
                model.parameters(),
                lr=self.config["learning_rate"],
                weight_decay=self.config["weight_decay"]
            )
        elif self.config["optimizer"] == "adamw":
            optimizer = optim.AdamW(
                model.parameters(),
                lr=self.config["learning_rate"],
                weight_decay=self.config["weight_decay"]
            )
        elif self.config["optimizer"] == "sgd":
            optimizer = optim.SGD(
                model.parameters(),
                lr=self.config["learning_rate"],
                weight_decay=self.config["weight_decay"],
                momentum=0.9
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config['optimizer']}")
        
        # Scheduler
        scheduler_params = {
            "T_max": self.config["epochs"],
            "eta_min": 1e-6,
            "step_size": self.config["epochs"] // 3,
            "gamma": 0.1,
            "factor": 0.5,
            "patience": 10,
            "min_lr": 1e-6
        }
        
        scheduler = LearningRateScheduler(
            optimizer, 
            self.config["scheduler"], 
            scheduler_params
        )
        
        return optimizer, scheduler
    
    def train_epoch(self, model: nn.Module, train_loader, optimizer: optim.Optimizer, 
                   criterion: nn.Module, device: torch.device) -> Tuple[float, float]:
        """Train one epoch."""
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        progress_bar = tqdm(train_loader, desc="Training", leave=False)
        
        for batch_idx, (inputs, targets) in enumerate(progress_bar):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Mixed precision forward pass
            with autocast(enabled=self.mixed_precision.enabled):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss = self.gradient_accumulator.scale_loss(loss)
            
            # Backward pass
            self.mixed_precision.backward(loss)
            
            # Gradient clipping
            if self.config.get("gradient_clipping", 0) > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 
                    self.config["gradient_clipping"]
                )
            
            # Optimizer step
            if self.gradient_accumulator.should_update():
                self.mixed_precision.step(optimizer)
                optimizer.zero_grad()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Update progress bar
            progress_bar.set_postfix({
                "Loss": f"{loss.item():.4f}",
                "Acc": f"{100.*correct/total:.2f}%"
            })
        
        return total_loss / len(train_loader), 100. * correct / total
    
    def validate_epoch(self, model: nn.Module, val_loader, criterion: nn.Module, 
                      device: torch.device) -> Tuple[float, float]:
        """Validate one epoch."""
        model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in tqdm(val_loader, desc="Validation", leave=False):
                inputs, targets = inputs.to(device), targets.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        return total_loss / len(val_loader), 100. * correct / total
    
    def save_training_artifacts(self, model: nn.Module):
        """Save training artifacts."""
        # Save history
        self.history.save(self.env["log_dir"] / "history.json")
        
        # Save model summary
        save_model_summary(model, self.config, self.env["log_dir"] / "model_summary.json")
        
        # Save final model
        torch.save(model.state_dict(), self.env["model_dir"] / "final_model.pth")
        
        logger.info("Training artifacts saved successfully")


if __name__ == "__main__":
    # Test training utilities
    config = {
        "learning_rate": 1e-4,
        "weight_decay": 1e-4,
        "optimizer": "adamw",
        "scheduler": "cosine",
        "epochs": 100,
        "patience": 15,
        "mixed_precision": True,
        "gradient_clipping": 1.0
    }
    
    model_dir = Path("./test_training")
    trainer = TrainingManager(config, model_dir)
    print("Training Manager initialized successfully!")
    
    # Test device info
    device_info = get_device_info()
    print(f"Device info: {device_info}")
    
    # Test parameter counting
    dummy_model = nn.Linear(100, 10)
    param_count = count_parameters(dummy_model)
    print(f"Parameter count: {param_count}")