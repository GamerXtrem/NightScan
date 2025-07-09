"""
Advanced Metrics for Wildlife Audio Classification

Comprehensive metrics including confusion matrix, F1-score, precision, recall,
ROC curves, and classification reports with visualization capabilities.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import json
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score
)
from sklearn.preprocessing import label_binarize
import pandas as pd

logger = logging.getLogger(__name__)


class MetricsCalculator:
    """
    Comprehensive metrics calculator for multi-class classification.
    """
    
    def __init__(self, class_names: List[str], save_dir: Optional[Path] = None):
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.save_dir = Path(save_dir) if save_dir else None
        
        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize accumulators
        self.reset()
    
    def reset(self) -> None:
        """Reset all accumulated metrics."""
        self.all_predictions = []
        self.all_targets = []
        self.all_probabilities = []
    
    def update(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        probabilities: Optional[torch.Tensor] = None
    ) -> None:
        """
        Update metrics with new batch results.
        
        Args:
            predictions: Model predictions (batch_size,)
            targets: Ground truth labels (batch_size,)
            probabilities: Class probabilities (batch_size, num_classes)
        """
        # Convert to numpy
        predictions = predictions.cpu().numpy()
        targets = targets.cpu().numpy()
        
        self.all_predictions.extend(predictions)
        self.all_targets.extend(targets)
        
        if probabilities is not None:
            probabilities = probabilities.cpu().numpy()
            self.all_probabilities.extend(probabilities)
    
    def compute_basic_metrics(self) -> Dict[str, float]:
        """
        Compute basic classification metrics.
        
        Returns:
            Dictionary of basic metrics
        """
        if not self.all_predictions:
            return {}
        
        y_true = np.array(self.all_targets)
        y_pred = np.array(self.all_predictions)
        
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "f1_macro": f1_score(y_true, y_pred, average="macro"),
            "f1_micro": f1_score(y_true, y_pred, average="micro"),
            "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
            "precision_macro": precision_score(y_true, y_pred, average="macro"),
            "precision_micro": precision_score(y_true, y_pred, average="micro"),
            "precision_weighted": precision_score(y_true, y_pred, average="weighted"),
            "recall_macro": recall_score(y_true, y_pred, average="macro"),
            "recall_micro": recall_score(y_true, y_pred, average="micro"),
            "recall_weighted": recall_score(y_true, y_pred, average="weighted"),
        }
        
        return metrics
    
    def compute_class_metrics(self) -> Dict[str, Dict[str, float]]:
        """
        Compute per-class metrics.
        
        Returns:
            Dictionary of per-class metrics
        """
        if not self.all_predictions:
            return {}
        
        y_true = np.array(self.all_targets)
        y_pred = np.array(self.all_predictions)
        
        # Per-class metrics
        f1_per_class = f1_score(y_true, y_pred, average=None)
        precision_per_class = precision_score(y_true, y_pred, average=None)
        recall_per_class = recall_score(y_true, y_pred, average=None)
        
        class_metrics = {}
        for i, class_name in enumerate(self.class_names):
            class_metrics[class_name] = {
                "f1_score": f1_per_class[i],
                "precision": precision_per_class[i],
                "recall": recall_per_class[i]
            }
        
        return class_metrics
    
    def compute_confusion_matrix(self) -> np.ndarray:
        """
        Compute confusion matrix.
        
        Returns:
            Confusion matrix as numpy array
        """
        if not self.all_predictions:
            return np.array([])
        
        y_true = np.array(self.all_targets)
        y_pred = np.array(self.all_predictions)
        
        return confusion_matrix(y_true, y_pred)
    
    def plot_confusion_matrix(
        self,
        normalize: bool = True,
        save_path: Optional[Path] = None,
        figsize: Tuple[int, int] = (10, 8)
    ) -> None:
        """
        Plot confusion matrix.
        
        Args:
            normalize: Whether to normalize the confusion matrix
            save_path: Path to save the plot
            figsize: Figure size
        """
        cm = self.compute_confusion_matrix()
        if cm.size == 0:
            logger.warning("No predictions available for confusion matrix")
            return
        
        if normalize:
            cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
            title = "Normalized Confusion Matrix"
        else:
            title = "Confusion Matrix"
        
        plt.figure(figsize=figsize)
        sns.heatmap(
            cm,
            annot=True,
            fmt=".2f" if normalize else "d",
            cmap="Blues",
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            cbar_kws={"label": "Proportion" if normalize else "Count"}
        )
        plt.title(title)
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Confusion matrix saved to {save_path}")
        
        if self.save_dir:
            save_path = self.save_dir / f"confusion_matrix{'_normalized' if normalize else ''}.png"
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        
        plt.show()
    
    def compute_roc_metrics(self) -> Dict[str, Any]:
        """
        Compute ROC curves and AUC scores.
        
        Returns:
            Dictionary containing ROC metrics
        """
        if not self.all_probabilities:
            logger.warning("No probabilities available for ROC computation")
            return {}
        
        y_true = np.array(self.all_targets)
        y_proba = np.array(self.all_probabilities)
        
        # Binarize labels for multi-class ROC
        y_true_bin = label_binarize(y_true, classes=range(self.num_classes))
        
        roc_metrics = {}
        
        # Compute ROC curve and AUC for each class
        for i, class_name in enumerate(self.class_names):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
            roc_auc = auc(fpr, tpr)
            
            roc_metrics[class_name] = {
                "fpr": fpr.tolist(),
                "tpr": tpr.tolist(),
                "auc": roc_auc
            }
        
        # Compute macro-average ROC
        all_fpr = np.unique(np.concatenate([
            roc_metrics[class_name]["fpr"] for class_name in self.class_names
        ]))
        
        mean_tpr = np.zeros_like(all_fpr)
        for class_name in self.class_names:
            fpr = np.array(roc_metrics[class_name]["fpr"])
            tpr = np.array(roc_metrics[class_name]["tpr"])
            mean_tpr += np.interp(all_fpr, fpr, tpr)
        
        mean_tpr /= self.num_classes
        macro_auc = auc(all_fpr, mean_tpr)
        
        roc_metrics["macro_average"] = {
            "fpr": all_fpr.tolist(),
            "tpr": mean_tpr.tolist(),
            "auc": macro_auc
        }
        
        return roc_metrics
    
    def plot_roc_curves(
        self,
        save_path: Optional[Path] = None,
        figsize: Tuple[int, int] = (10, 8)
    ) -> None:
        """
        Plot ROC curves for all classes.
        
        Args:
            save_path: Path to save the plot
            figsize: Figure size
        """
        roc_metrics = self.compute_roc_metrics()
        if not roc_metrics:
            return
        
        plt.figure(figsize=figsize)
        
        # Plot ROC curve for each class
        for class_name in self.class_names:
            if class_name in roc_metrics:
                fpr = roc_metrics[class_name]["fpr"]
                tpr = roc_metrics[class_name]["tpr"]
                auc_score = roc_metrics[class_name]["auc"]
                
                plt.plot(
                    fpr, tpr,
                    label=f"{class_name} (AUC = {auc_score:.2f})",
                    linewidth=2
                )
        
        # Plot macro-average ROC
        if "macro_average" in roc_metrics:
            fpr = roc_metrics["macro_average"]["fpr"]
            tpr = roc_metrics["macro_average"]["tpr"]
            auc_score = roc_metrics["macro_average"]["auc"]
            
            plt.plot(
                fpr, tpr,
                label=f"Macro-average (AUC = {auc_score:.2f})",
                linewidth=2,
                linestyle="--",
                color="black"
            )
        
        # Plot diagonal line
        plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curves - Wildlife Audio Classification")
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"ROC curves saved to {save_path}")
        
        if self.save_dir:
            save_path = self.save_dir / "roc_curves.png"
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        
        plt.show()
    
    def compute_precision_recall_metrics(self) -> Dict[str, Any]:
        """
        Compute precision-recall curves and average precision scores.
        
        Returns:
            Dictionary containing precision-recall metrics
        """
        if not self.all_probabilities:
            logger.warning("No probabilities available for precision-recall computation")
            return {}
        
        y_true = np.array(self.all_targets)
        y_proba = np.array(self.all_probabilities)
        
        # Binarize labels for multi-class precision-recall
        y_true_bin = label_binarize(y_true, classes=range(self.num_classes))
        
        pr_metrics = {}
        
        # Compute precision-recall curve for each class
        for i, class_name in enumerate(self.class_names):
            precision, recall, _ = precision_recall_curve(
                y_true_bin[:, i], y_proba[:, i]
            )
            avg_precision = average_precision_score(
                y_true_bin[:, i], y_proba[:, i]
            )
            
            pr_metrics[class_name] = {
                "precision": precision.tolist(),
                "recall": recall.tolist(),
                "average_precision": avg_precision
            }
        
        # Compute macro-average precision
        macro_avg_precision = np.mean([
            pr_metrics[class_name]["average_precision"]
            for class_name in self.class_names
        ])
        
        pr_metrics["macro_average"] = {
            "average_precision": macro_avg_precision
        }
        
        return pr_metrics
    
    def plot_precision_recall_curves(
        self,
        save_path: Optional[Path] = None,
        figsize: Tuple[int, int] = (10, 8)
    ) -> None:
        """
        Plot precision-recall curves for all classes.
        
        Args:
            save_path: Path to save the plot
            figsize: Figure size
        """
        pr_metrics = self.compute_precision_recall_metrics()
        if not pr_metrics:
            return
        
        plt.figure(figsize=figsize)
        
        # Plot precision-recall curve for each class
        for class_name in self.class_names:
            if class_name in pr_metrics:
                precision = pr_metrics[class_name]["precision"]
                recall = pr_metrics[class_name]["recall"]
                avg_precision = pr_metrics[class_name]["average_precision"]
                
                plt.plot(
                    recall, precision,
                    label=f"{class_name} (AP = {avg_precision:.2f})",
                    linewidth=2
                )
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curves - Wildlife Audio Classification")
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Precision-recall curves saved to {save_path}")
        
        if self.save_dir:
            save_path = self.save_dir / "precision_recall_curves.png"
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        
        plt.show()
    
    def generate_classification_report(self) -> str:
        """
        Generate detailed classification report.
        
        Returns:
            Classification report as string
        """
        if not self.all_predictions:
            return "No predictions available"
        
        y_true = np.array(self.all_targets)
        y_pred = np.array(self.all_predictions)
        
        return classification_report(
            y_true, y_pred,
            target_names=self.class_names,
            digits=4
        )
    
    def save_comprehensive_report(self, filename: str = "metrics_report.json") -> None:
        """
        Save comprehensive metrics report to JSON file.
        
        Args:
            filename: Name of the output file
        """
        if not self.save_dir:
            logger.warning("No save directory specified")
            return
        
        report = {
            "basic_metrics": self.compute_basic_metrics(),
            "class_metrics": self.compute_class_metrics(),
            "confusion_matrix": self.compute_confusion_matrix().tolist(),
            "roc_metrics": self.compute_roc_metrics(),
            "precision_recall_metrics": self.compute_precision_recall_metrics(),
            "classification_report": self.generate_classification_report(),
            "class_names": self.class_names,
            "num_samples": len(self.all_predictions)
        }
        
        report_path = self.save_dir / filename
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Comprehensive metrics report saved to {report_path}")
    
    def create_summary_plots(self) -> None:
        """
        Create and save all summary plots.
        """
        logger.info("Creating summary plots...")
        
        # Plot confusion matrix
        self.plot_confusion_matrix(normalize=False)
        self.plot_confusion_matrix(normalize=True)
        
        # Plot ROC curves
        self.plot_roc_curves()
        
        # Plot precision-recall curves
        self.plot_precision_recall_curves()
        
        logger.info("Summary plots created successfully")


def calculate_top_k_accuracy(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    k: int = 3
) -> float:
    """
    Calculate top-k accuracy.
    
    Args:
        predictions: Model predictions (batch_size, num_classes)
        targets: Ground truth labels (batch_size,)
        k: Number of top predictions to consider
        
    Returns:
        Top-k accuracy
    """
    with torch.no_grad():
        batch_size = targets.size(0)
        _, top_k_pred = predictions.topk(k, 1, True, True)
        top_k_pred = top_k_pred.t()
        correct = top_k_pred.eq(targets.view(1, -1).expand_as(top_k_pred))
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        
        return correct_k.mul_(100.0 / batch_size).item()


def compute_class_weights(
    class_counts: Dict[str, int],
    method: str = "balanced"
) -> Dict[str, float]:
    """
    Compute class weights for imbalanced datasets.
    
    Args:
        class_counts: Dictionary of class names and their counts
        method: Weighting method ("balanced" or "inverse_frequency")
        
    Returns:
        Dictionary of class weights
    """
    total_samples = sum(class_counts.values())
    num_classes = len(class_counts)
    
    if method == "balanced":
        # Balanced weighting: total_samples / (num_classes * class_count)
        weights = {
            class_name: total_samples / (num_classes * count)
            for class_name, count in class_counts.items()
        }
    elif method == "inverse_frequency":
        # Inverse frequency weighting: 1 / (class_count / total_samples)
        weights = {
            class_name: total_samples / count
            for class_name, count in class_counts.items()
        }
    else:
        raise ValueError(f"Unknown weighting method: {method}")
    
    return weights


if __name__ == "__main__":
    # Test metrics calculator
    print("Testing metrics calculator...")
    
    # Create dummy data
    class_names = ["bird_song", "mammal_call", "insect_sound", "amphibian_call", "environmental_sound", "unknown_species"]
    num_samples = 1000
    num_classes = len(class_names)
    
    # Generate dummy predictions and targets
    np.random.seed(42)
    targets = np.random.randint(0, num_classes, num_samples)
    predictions = np.random.randint(0, num_classes, num_samples)
    probabilities = np.random.rand(num_samples, num_classes)
    
    # Normalize probabilities
    probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
    
    # Create metrics calculator
    calculator = MetricsCalculator(class_names, save_dir=Path("./test_metrics"))
    
    # Update with dummy data
    calculator.update(
        torch.tensor(predictions),
        torch.tensor(targets),
        torch.tensor(probabilities)
    )
    
    # Compute metrics
    basic_metrics = calculator.compute_basic_metrics()
    class_metrics = calculator.compute_class_metrics()
    
    print("Basic metrics:")
    for metric, value in basic_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    print("\nClass metrics:")
    for class_name, metrics in class_metrics.items():
        print(f"  {class_name}:")
        for metric, value in metrics.items():
            print(f"    {metric}: {value:.4f}")
    
    # Test classification report
    report = calculator.generate_classification_report()
    print("\nClassification report:")
    print(report)
    
    # Clean up
    import shutil
    if Path("./test_metrics").exists():
        shutil.rmtree("./test_metrics")
    
    print("\nAll tests passed!")