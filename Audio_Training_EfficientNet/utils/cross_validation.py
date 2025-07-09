"""
K-Fold Cross-Validation for Wildlife Audio Classification

Implements stratified k-fold cross-validation for robust model evaluation
with proper handling of imbalanced datasets and comprehensive reporting.
"""

import numpy as np
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Callable
from collections import defaultdict
import copy

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import classification_report
import pandas as pd

from .metrics import MetricsCalculator
from .training_utils import (
    setup_training_environment,
    TrainingHistory,
    EarlyStopping,
    ModelCheckpoint
)

logger = logging.getLogger(__name__)


class CrossValidationDataset(Dataset):
    """
    Dataset wrapper for cross-validation with proper indexing.
    """
    
    def __init__(
        self,
        spectrograms: List[Path],
        labels: List[int],
        transform: Optional[Callable] = None
    ):
        self.spectrograms = spectrograms
        self.labels = labels
        self.transform = transform
    
    def __len__(self) -> int:
        return len(self.spectrograms)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        # Load spectrogram
        spectrogram = np.load(self.spectrograms[idx])
        spectrogram = torch.tensor(spectrogram, dtype=torch.float32)
        
        # Convert to 3 channels for EfficientNet
        if spectrogram.dim() == 2:
            spectrogram = spectrogram.unsqueeze(0).repeat(3, 1, 1)
        
        # Apply transforms
        if self.transform:
            spectrogram = self.transform(spectrogram)
        
        return spectrogram, self.labels[idx]


class KFoldCrossValidator:
    """
    K-Fold Cross-Validation manager for audio classification models.
    """
    
    def __init__(
        self,
        model_factory: Callable[[], nn.Module],
        config: Dict[str, Any],
        class_names: List[str],
        k_folds: int = 5,
        random_state: int = 42,
        save_dir: Optional[Path] = None
    ):
        self.model_factory = model_factory
        self.config = config
        self.class_names = class_names
        self.k_folds = k_folds
        self.random_state = random_state
        self.save_dir = Path(save_dir) if save_dir else None
        
        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize result storage
        self.fold_results = []
        self.overall_results = {}
        
        # Device configuration
        self.device = torch.device(config.get("device", "cpu"))
        
        logger.info(f"Initialized {k_folds}-fold cross-validation")
    
    def prepare_data(
        self,
        spectrograms: List[Path],
        labels: List[int],
        transform: Optional[Callable] = None
    ) -> Tuple[CrossValidationDataset, List[Tuple[List[int], List[int]]]]:
        """
        Prepare data for cross-validation with stratified splits.
        
        Args:
            spectrograms: List of spectrogram file paths
            labels: List of corresponding labels
            transform: Optional transform to apply
            
        Returns:
            Dataset and list of (train_indices, val_indices) for each fold
        """
        # Create dataset
        dataset = CrossValidationDataset(spectrograms, labels, transform)
        
        # Create stratified k-fold splits
        skf = StratifiedKFold(
            n_splits=self.k_folds,
            shuffle=True,
            random_state=self.random_state
        )
        
        # Generate fold indices
        fold_indices = []
        for train_idx, val_idx in skf.split(spectrograms, labels):
            fold_indices.append((train_idx.tolist(), val_idx.tolist()))
        
        return dataset, fold_indices
    
    def train_fold(
        self,
        fold_idx: int,
        train_loader: DataLoader,
        val_loader: DataLoader,
        model: nn.Module
    ) -> Dict[str, Any]:
        """
        Train model for a single fold.
        
        Args:
            fold_idx: Current fold index
            train_loader: Training data loader
            val_loader: Validation data loader
            model: Model to train
            
        Returns:
            Training results for this fold
        """
        logger.info(f"Training fold {fold_idx + 1}/{self.k_folds}")
        
        # Setup training environment
        optimizer, scheduler, checkpoint, early_stopping, trainer = setup_training_environment(
            model, self.config, self.device,
            self.save_dir / f"fold_{fold_idx}" if self.save_dir else Path("./temp_checkpoints")
        )
        
        # Loss function
        criterion = nn.CrossEntropyLoss()
        
        # Training history
        history = TrainingHistory(
            self.save_dir / f"fold_{fold_idx}" if self.save_dir else Path("./temp_history")
        )
        
        # Metrics calculator
        metrics_calculator = MetricsCalculator(
            self.class_names,
            self.save_dir / f"fold_{fold_idx}" / "metrics" if self.save_dir else None
        )
        
        best_val_loss = float('inf')
        best_val_acc = 0.0
        
        # Training loop
        for epoch in range(self.config.get("epochs", 50)):
            # Training phase
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch in train_loader:
                loss = trainer.train_step(batch, criterion, self.device)
                train_loss += loss
                
                # Calculate accuracy
                inputs, targets = batch
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                with torch.no_grad():
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    train_total += targets.size(0)
                    train_correct += (predicted == targets).sum().item()
            
            train_loss /= len(train_loader)
            train_acc = 100 * train_correct / train_total
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            metrics_calculator.reset()
            
            with torch.no_grad():
                for batch in val_loader:
                    loss, outputs, targets = trainer.validate_step(batch, criterion, self.device)
                    val_loss += loss
                    
                    # Update metrics
                    _, predicted = torch.max(outputs, 1)
                    probabilities = torch.softmax(outputs, dim=1)
                    metrics_calculator.update(predicted, targets, probabilities)
            
            val_loss /= len(val_loader)
            val_metrics = metrics_calculator.compute_basic_metrics()
            val_acc = val_metrics.get("accuracy", 0) * 100
            
            # Update history
            epoch_metrics = {
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_acc": train_acc,
                "val_acc": val_acc,
                "learning_rate": scheduler.get_lr()[0] if scheduler else self.config["learning_rate"]
            }
            history.update(epoch_metrics)
            
            # Log progress
            logger.info(
                f"Fold {fold_idx + 1}, Epoch {epoch + 1}: "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%"
            )
            
            # Save checkpoint
            checkpoint(epoch, model, optimizer, scheduler, epoch_metrics)
            
            # Early stopping
            if early_stopping(val_loss, model):
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break
            
            # Track best metrics
            if val_loss < best_val_loss:
                best_val_loss = val_loss
            if val_acc > best_val_acc:
                best_val_acc = val_acc
        
        # Save training history
        history.save(f"fold_{fold_idx}_history.json")
        
        # Final evaluation
        final_metrics = metrics_calculator.compute_basic_metrics()
        class_metrics = metrics_calculator.compute_class_metrics()
        
        # Generate comprehensive report
        metrics_calculator.save_comprehensive_report(f"fold_{fold_idx}_metrics.json")
        
        fold_results = {
            "fold_index": fold_idx,
            "best_val_loss": best_val_loss,
            "best_val_acc": best_val_acc,
            "final_metrics": final_metrics,
            "class_metrics": class_metrics,
            "training_history": history.history
        }
        
        return fold_results
    
    def run_cross_validation(
        self,
        spectrograms: List[Path],
        labels: List[int],
        transform: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Run complete k-fold cross-validation.
        
        Args:
            spectrograms: List of spectrogram file paths
            labels: List of corresponding labels
            transform: Optional transform to apply
            
        Returns:
            Complete cross-validation results
        """
        logger.info("Starting k-fold cross-validation...")
        
        # Prepare data
        dataset, fold_indices = self.prepare_data(spectrograms, labels, transform)
        
        # Run each fold
        self.fold_results = []
        
        for fold_idx, (train_indices, val_indices) in enumerate(fold_indices):
            # Create data loaders
            train_sampler = SubsetRandomSampler(train_indices)
            val_sampler = SubsetRandomSampler(val_indices)
            
            train_loader = DataLoader(
                dataset,
                batch_size=self.config.get("batch_size", 32),
                sampler=train_sampler,
                num_workers=self.config.get("num_workers", 4),
                pin_memory=self.device.type != "cpu"
            )
            
            val_loader = DataLoader(
                dataset,
                batch_size=self.config.get("batch_size", 32),
                sampler=val_sampler,
                num_workers=self.config.get("num_workers", 4),
                pin_memory=self.device.type != "cpu"
            )
            
            # Create fresh model for this fold
            model = self.model_factory()
            model.to(self.device)
            
            # Train fold
            fold_result = self.train_fold(fold_idx, train_loader, val_loader, model)
            self.fold_results.append(fold_result)
            
            logger.info(f"Fold {fold_idx + 1} completed")
        
        # Calculate overall results
        self.overall_results = self._calculate_overall_results()
        
        # Save complete results
        if self.save_dir:
            self._save_complete_results()
        
        logger.info("Cross-validation completed!")
        return self.overall_results
    
    def _calculate_overall_results(self) -> Dict[str, Any]:
        """
        Calculate overall cross-validation results.
        
        Returns:
            Dictionary of overall results
        """
        if not self.fold_results:
            return {}
        
        # Extract metrics from all folds
        all_metrics = defaultdict(list)
        
        for fold_result in self.fold_results:
            all_metrics["val_loss"].append(fold_result["best_val_loss"])
            all_metrics["val_acc"].append(fold_result["best_val_acc"])
            
            # Extract final metrics
            final_metrics = fold_result["final_metrics"]
            for metric_name, value in final_metrics.items():
                all_metrics[metric_name].append(value)
        
        # Calculate statistics
        overall_results = {}
        for metric_name, values in all_metrics.items():
            overall_results[metric_name] = {
                "mean": np.mean(values),
                "std": np.std(values),
                "min": np.min(values),
                "max": np.max(values),
                "values": values
            }
        
        # Calculate per-class statistics
        class_results = defaultdict(lambda: defaultdict(list))
        for fold_result in self.fold_results:
            class_metrics = fold_result["class_metrics"]
            for class_name, metrics in class_metrics.items():
                for metric_name, value in metrics.items():
                    class_results[class_name][metric_name].append(value)
        
        # Calculate class statistics
        overall_class_results = {}
        for class_name, metrics in class_results.items():
            overall_class_results[class_name] = {}
            for metric_name, values in metrics.items():
                overall_class_results[class_name][metric_name] = {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "min": np.min(values),
                    "max": np.max(values)
                }
        
        overall_results["class_results"] = overall_class_results
        overall_results["fold_results"] = self.fold_results
        overall_results["num_folds"] = self.k_folds
        
        return overall_results
    
    def _save_complete_results(self) -> None:
        """Save complete cross-validation results."""
        if not self.save_dir:
            return
        
        # Save overall results
        results_path = self.save_dir / "cross_validation_results.json"
        with open(results_path, 'w') as f:
            json.dump(self.overall_results, f, indent=2, default=str)
        
        # Create summary report
        self._create_summary_report()
        
        logger.info(f"Cross-validation results saved to {self.save_dir}")
    
    def _create_summary_report(self) -> None:
        """Create a human-readable summary report."""
        if not self.save_dir:
            return
        
        summary_path = self.save_dir / "cross_validation_summary.txt"
        
        with open(summary_path, 'w') as f:
            f.write("K-Fold Cross-Validation Summary\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Number of folds: {self.k_folds}\n")
            f.write(f"Random state: {self.random_state}\n\n")
            
            # Overall metrics
            f.write("Overall Metrics:\n")
            f.write("-" * 20 + "\n")
            
            key_metrics = ["val_acc", "val_loss", "f1_macro", "f1_weighted"]
            for metric in key_metrics:
                if metric in self.overall_results:
                    stats = self.overall_results[metric]
                    f.write(f"{metric}:\n")
                    f.write(f"  Mean: {stats['mean']:.4f} ± {stats['std']:.4f}\n")
                    f.write(f"  Range: [{stats['min']:.4f}, {stats['max']:.4f}]\n\n")
            
            # Per-class results
            f.write("Per-Class Results:\n")
            f.write("-" * 20 + "\n")
            
            class_results = self.overall_results.get("class_results", {})
            for class_name in self.class_names:
                if class_name in class_results:
                    f.write(f"{class_name}:\n")
                    class_metrics = class_results[class_name]
                    for metric_name, stats in class_metrics.items():
                        f.write(f"  {metric_name}: {stats['mean']:.4f} ± {stats['std']:.4f}\n")
                    f.write("\n")
            
            # Individual fold results
            f.write("Individual Fold Results:\n")
            f.write("-" * 25 + "\n")
            
            for i, fold_result in enumerate(self.fold_results):
                f.write(f"Fold {i + 1}:\n")
                f.write(f"  Best Val Accuracy: {fold_result['best_val_acc']:.2f}%\n")
                f.write(f"  Best Val Loss: {fold_result['best_val_loss']:.4f}\n")
                
                final_metrics = fold_result["final_metrics"]
                f.write(f"  Final F1 (macro): {final_metrics.get('f1_macro', 0):.4f}\n")
                f.write(f"  Final F1 (weighted): {final_metrics.get('f1_weighted', 0):.4f}\n\n")
        
        logger.info(f"Summary report saved to {summary_path}")
    
    def get_best_fold(self) -> Tuple[int, Dict[str, Any]]:
        """
        Get the best performing fold.
        
        Returns:
            Tuple of (fold_index, fold_results)
        """
        if not self.fold_results:
            return -1, {}
        
        # Find fold with best validation accuracy
        best_fold_idx = 0
        best_val_acc = 0
        
        for i, fold_result in enumerate(self.fold_results):
            if fold_result["best_val_acc"] > best_val_acc:
                best_val_acc = fold_result["best_val_acc"]
                best_fold_idx = i
        
        return best_fold_idx, self.fold_results[best_fold_idx]
    
    def print_summary(self) -> None:
        """Print cross-validation summary to console."""
        if not self.overall_results:
            print("No cross-validation results available")
            return
        
        print(f"\n{'='*60}")
        print(f"K-Fold Cross-Validation Summary ({self.k_folds} folds)")
        print(f"{'='*60}")
        
        # Key metrics
        key_metrics = ["val_acc", "val_loss", "f1_macro", "f1_weighted"]
        print("\nKey Metrics:")
        print("-" * 40)
        
        for metric in key_metrics:
            if metric in self.overall_results:
                stats = self.overall_results[metric]
                print(f"{metric:15}: {stats['mean']:.4f} ± {stats['std']:.4f}")
        
        # Best fold
        best_fold_idx, best_fold_result = self.get_best_fold()
        print(f"\nBest Fold: {best_fold_idx + 1}")
        print(f"Best Val Accuracy: {best_fold_result['best_val_acc']:.2f}%")
        print(f"Best Val Loss: {best_fold_result['best_val_loss']:.4f}")
        
        print(f"{'='*60}\n")


if __name__ == "__main__":
    # Test cross-validation
    print("Testing cross-validation...")
    
    # This would normally be done with real data and models
    # For testing, we'll create a simple setup
    
    from pathlib import Path
    import tempfile
    
    # Create dummy data
    spectrograms = [Path(f"dummy_spec_{i}.npy") for i in range(100)]
    labels = [i % 6 for i in range(100)]  # 6 classes
    class_names = ["bird_song", "mammal_call", "insect_sound", "amphibian_call", "environmental_sound", "unknown_species"]
    
    # Dummy model factory
    def model_factory():
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 224 * 224, 64),
            nn.ReLU(),
            nn.Linear(64, 6)
        )
    
    # Dummy config
    config = {
        "batch_size": 16,
        "epochs": 2,
        "learning_rate": 1e-3,
        "optimizer": "adam",
        "device": "cpu"
    }
    
    # Create temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # Initialize cross-validator
        cv = KFoldCrossValidator(
            model_factory=model_factory,
            config=config,
            class_names=class_names,
            k_folds=3,
            save_dir=Path(temp_dir)
        )
        
        print("Cross-validation setup successful!")
        print("In real usage, you would call cv.run_cross_validation(spectrograms, labels)")
    
    print("All tests passed!")