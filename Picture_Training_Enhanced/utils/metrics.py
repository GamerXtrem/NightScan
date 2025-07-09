"""
Advanced Metrics for Wildlife Photo Classification

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
    average_precision_score,
    top_k_accuracy_score
)
from sklearn.preprocessing import label_binarize
import pandas as pd

logger = logging.getLogger(__name__)


class MetricsCalculator:
    """Comprehensive metrics calculator for image classification."""
    
    def __init__(self, class_names: List[str], save_dir: Optional[Path] = None):
        """
        Initialize metrics calculator.
        
        Args:
            class_names: List of class names
            save_dir: Directory to save visualizations
        """
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.save_dir = save_dir
        
        if save_dir:
            save_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                         y_proba: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Calculate comprehensive metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Prediction probabilities (optional)
            
        Returns:
            Dictionary containing all metrics
        """
        metrics = {}
        
        # Basic metrics
        metrics["accuracy"] = accuracy_score(y_true, y_pred)
        metrics["precision_macro"] = precision_score(y_true, y_pred, average='macro', zero_division=0)
        metrics["recall_macro"] = recall_score(y_true, y_pred, average='macro', zero_division=0)
        metrics["f1_macro"] = f1_score(y_true, y_pred, average='macro', zero_division=0)
        
        metrics["precision_weighted"] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics["recall_weighted"] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics["f1_weighted"] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # Per-class metrics
        precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
        recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
        
        metrics["per_class"] = {
            "precision": {self.class_names[i]: precision_per_class[i] for i in range(self.num_classes)},
            "recall": {self.class_names[i]: recall_per_class[i] for i in range(self.num_classes)},
            "f1": {self.class_names[i]: f1_per_class[i] for i in range(self.num_classes)}
        }
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics["confusion_matrix"] = cm.tolist()
        
        # Classification report
        report = classification_report(y_true, y_pred, target_names=self.class_names, output_dict=True)
        metrics["classification_report"] = report
        
        # Top-k accuracy if probabilities provided
        if y_proba is not None:
            if self.num_classes > 2:
                metrics["top_2_accuracy"] = top_k_accuracy_score(y_true, y_proba, k=2)
            if self.num_classes > 3:
                metrics["top_3_accuracy"] = top_k_accuracy_score(y_true, y_proba, k=3)
            
            # ROC and PR curves for multi-class
            if self.num_classes > 2:
                roc_metrics = self._calculate_multiclass_roc(y_true, y_proba)
                pr_metrics = self._calculate_multiclass_pr(y_true, y_proba)
                metrics.update(roc_metrics)
                metrics.update(pr_metrics)
            else:
                # Binary classification
                roc_metrics = self._calculate_binary_roc(y_true, y_proba[:, 1])
                pr_metrics = self._calculate_binary_pr(y_true, y_proba[:, 1])
                metrics.update(roc_metrics)
                metrics.update(pr_metrics)
        
        return metrics
    
    def _calculate_multiclass_roc(self, y_true: np.ndarray, y_proba: np.ndarray) -> Dict[str, Any]:
        """Calculate ROC metrics for multi-class classification."""
        # Binarize labels
        y_true_bin = label_binarize(y_true, classes=range(self.num_classes))
        
        roc_auc = {}
        fpr = {}
        tpr = {}
        
        for i in range(self.num_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        # Compute micro-average ROC curve
        fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_proba.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        
        # Compute macro-average ROC curve
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(self.num_classes)]))
        mean_tpr = np.zeros_like(all_fpr)
        
        for i in range(self.num_classes):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
        
        mean_tpr /= self.num_classes
        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
        
        return {
            "roc_auc": {self.class_names[i]: roc_auc[i] for i in range(self.num_classes)},
            "roc_auc_micro": roc_auc["micro"],
            "roc_auc_macro": roc_auc["macro"],
            "fpr": fpr,
            "tpr": tpr
        }
    
    def _calculate_binary_roc(self, y_true: np.ndarray, y_proba: np.ndarray) -> Dict[str, Any]:
        """Calculate ROC metrics for binary classification."""
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = auc(fpr, tpr)
        
        return {
            "roc_auc": roc_auc,
            "fpr": fpr,
            "tpr": tpr
        }
    
    def _calculate_multiclass_pr(self, y_true: np.ndarray, y_proba: np.ndarray) -> Dict[str, Any]:
        """Calculate Precision-Recall metrics for multi-class classification."""
        # Binarize labels
        y_true_bin = label_binarize(y_true, classes=range(self.num_classes))
        
        precision = {}
        recall = {}
        ap = {}
        
        for i in range(self.num_classes):
            precision[i], recall[i], _ = precision_recall_curve(y_true_bin[:, i], y_proba[:, i])
            ap[i] = average_precision_score(y_true_bin[:, i], y_proba[:, i])
        
        # Compute micro-average PR curve
        precision["micro"], recall["micro"], _ = precision_recall_curve(
            y_true_bin.ravel(), y_proba.ravel()
        )
        ap["micro"] = average_precision_score(y_true_bin, y_proba, average="micro")
        
        # Macro-average AP
        ap["macro"] = np.mean([ap[i] for i in range(self.num_classes)])
        
        return {
            "average_precision": {self.class_names[i]: ap[i] for i in range(self.num_classes)},
            "average_precision_micro": ap["micro"],
            "average_precision_macro": ap["macro"],
            "precision": precision,
            "recall": recall
        }
    
    def _calculate_binary_pr(self, y_true: np.ndarray, y_proba: np.ndarray) -> Dict[str, Any]:
        """Calculate Precision-Recall metrics for binary classification."""
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        ap = average_precision_score(y_true, y_proba)
        
        return {
            "average_precision": ap,
            "precision": precision,
            "recall": recall
        }
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, 
                             normalize: bool = False, title: str = "Confusion Matrix") -> plt.Figure:
        """Plot confusion matrix."""
        cm = confusion_matrix(y_true, y_pred)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
        else:
            fmt = 'd'
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', 
                   xticklabels=self.class_names, yticklabels=self.class_names, ax=ax)
        
        ax.set_title(title)
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        
        plt.tight_layout()
        
        if self.save_dir:
            filename = title.lower().replace(' ', '_') + '.png'
            fig.savefig(self.save_dir / filename, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_roc_curves(self, y_true: np.ndarray, y_proba: np.ndarray, 
                       title: str = "ROC Curves") -> plt.Figure:
        """Plot ROC curves."""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        if self.num_classes == 2:
            fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
        else:
            # Multi-class ROC
            y_true_bin = label_binarize(y_true, classes=range(self.num_classes))
            
            for i in range(self.num_classes):
                fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
                roc_auc = auc(fpr, tpr)
                ax.plot(fpr, tpr, label=f'{self.class_names[i]} (AUC = {roc_auc:.2f})')
            
            # Micro-average ROC
            fpr_micro, tpr_micro, _ = roc_curve(y_true_bin.ravel(), y_proba.ravel())
            roc_auc_micro = auc(fpr_micro, tpr_micro)
            ax.plot(fpr_micro, tpr_micro, 
                   label=f'Micro-average (AUC = {roc_auc_micro:.2f})',
                   linestyle='--', linewidth=2)
        
        ax.plot([0, 1], [0, 1], 'k--', label='Random')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(title)
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if self.save_dir:
            filename = title.lower().replace(' ', '_') + '.png'
            fig.savefig(self.save_dir / filename, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_precision_recall_curves(self, y_true: np.ndarray, y_proba: np.ndarray,
                                   title: str = "Precision-Recall Curves") -> plt.Figure:
        """Plot Precision-Recall curves."""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        if self.num_classes == 2:
            precision, recall, _ = precision_recall_curve(y_true, y_proba[:, 1])
            ap = average_precision_score(y_true, y_proba[:, 1])
            ax.plot(recall, precision, label=f'PR curve (AP = {ap:.2f})')
        else:
            # Multi-class PR
            y_true_bin = label_binarize(y_true, classes=range(self.num_classes))
            
            for i in range(self.num_classes):
                precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_proba[:, i])
                ap = average_precision_score(y_true_bin[:, i], y_proba[:, i])
                ax.plot(recall, precision, label=f'{self.class_names[i]} (AP = {ap:.2f})')
            
            # Micro-average PR
            precision_micro, recall_micro, _ = precision_recall_curve(
                y_true_bin.ravel(), y_proba.ravel()
            )
            ap_micro = average_precision_score(y_true_bin, y_proba, average="micro")
            ax.plot(recall_micro, precision_micro, 
                   label=f'Micro-average (AP = {ap_micro:.2f})',
                   linestyle='--', linewidth=2)
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title(title)
        ax.legend(loc="lower left")
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if self.save_dir:
            filename = title.lower().replace(' ', '_') + '.png'
            fig.savefig(self.save_dir / filename, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_per_class_metrics(self, metrics: Dict[str, Any], 
                             title: str = "Per-Class Metrics") -> plt.Figure:
        """Plot per-class metrics bar chart."""
        per_class = metrics["per_class"]
        
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        
        classes = list(per_class["precision"].keys())
        precision_values = list(per_class["precision"].values())
        recall_values = list(per_class["recall"].values())
        f1_values = list(per_class["f1"].values())
        
        x = np.arange(len(classes))
        width = 0.8
        
        # Precision
        ax1.bar(x, precision_values, width, color='skyblue', alpha=0.8)
        ax1.set_title('Precision by Class')
        ax1.set_xlabel('Class')
        ax1.set_ylabel('Precision')
        ax1.set_xticks(x)
        ax1.set_xticklabels(classes, rotation=45, ha='right')
        ax1.set_ylim([0, 1])
        ax1.grid(True, alpha=0.3)
        
        # Recall
        ax2.bar(x, recall_values, width, color='lightcoral', alpha=0.8)
        ax2.set_title('Recall by Class')
        ax2.set_xlabel('Class')
        ax2.set_ylabel('Recall')
        ax2.set_xticks(x)
        ax2.set_xticklabels(classes, rotation=45, ha='right')
        ax2.set_ylim([0, 1])
        ax2.grid(True, alpha=0.3)
        
        # F1-Score
        ax3.bar(x, f1_values, width, color='lightgreen', alpha=0.8)
        ax3.set_title('F1-Score by Class')
        ax3.set_xlabel('Class')
        ax3.set_ylabel('F1-Score')
        ax3.set_xticks(x)
        ax3.set_xticklabels(classes, rotation=45, ha='right')
        ax3.set_ylim([0, 1])
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if self.save_dir:
            filename = title.lower().replace(' ', '_') + '.png'
            fig.savefig(self.save_dir / filename, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_training_history(self, history: Dict[str, List], 
                            title: str = "Training History") -> plt.Figure:
        """Plot training history."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        epochs = range(1, len(history["train_loss"]) + 1)
        
        # Loss
        ax1.plot(epochs, history["train_loss"], 'b-', label='Training Loss')
        ax1.plot(epochs, history["val_loss"], 'r-', label='Validation Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Accuracy
        ax2.plot(epochs, history["train_acc"], 'b-', label='Training Accuracy')
        ax2.plot(epochs, history["val_acc"], 'r-', label='Validation Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Learning Rate
        ax3.plot(epochs, history["learning_rate"], 'g-')
        ax3.set_title('Learning Rate')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Learning Rate')
        ax3.set_yscale('log')
        ax3.grid(True, alpha=0.3)
        
        # Epoch Time
        ax4.plot(epochs, history["epoch_time"], 'm-')
        ax4.set_title('Epoch Time')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Time (seconds)')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if self.save_dir:
            filename = title.lower().replace(' ', '_') + '.png'
            fig.savefig(self.save_dir / filename, dpi=300, bbox_inches='tight')
        
        return fig
    
    def generate_report(self, metrics: Dict[str, Any], 
                       save_path: Optional[Path] = None) -> str:
        """Generate comprehensive text report."""
        report = []
        report.append("=" * 60)
        report.append("WILDLIFE PHOTO CLASSIFICATION REPORT")
        report.append("=" * 60)
        report.append("")
        
        # Overall metrics
        report.append("OVERALL METRICS:")
        report.append(f"  Accuracy: {metrics['accuracy']:.4f}")
        report.append(f"  Precision (macro): {metrics['precision_macro']:.4f}")
        report.append(f"  Recall (macro): {metrics['recall_macro']:.4f}")
        report.append(f"  F1-Score (macro): {metrics['f1_macro']:.4f}")
        report.append("")
        
        # Per-class metrics
        report.append("PER-CLASS METRICS:")
        report.append(f"{'Class':<15} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
        report.append("-" * 45)
        
        for class_name in self.class_names:
            precision = metrics["per_class"]["precision"][class_name]
            recall = metrics["per_class"]["recall"][class_name]
            f1 = metrics["per_class"]["f1"][class_name]
            report.append(f"{class_name:<15} {precision:<10.4f} {recall:<10.4f} {f1:<10.4f}")
        
        report.append("")
        
        # ROC AUC if available
        if "roc_auc" in metrics:
            report.append("ROC AUC SCORES:")
            if isinstance(metrics["roc_auc"], dict):
                for class_name in self.class_names:
                    if class_name in metrics["roc_auc"]:
                        score = metrics["roc_auc"][class_name]
                        report.append(f"  {class_name}: {score:.4f}")
            else:
                report.append(f"  Overall: {metrics['roc_auc']:.4f}")
            report.append("")
        
        # Top-k accuracy if available
        if "top_2_accuracy" in metrics:
            report.append("TOP-K ACCURACY:")
            report.append(f"  Top-2 Accuracy: {metrics['top_2_accuracy']:.4f}")
            if "top_3_accuracy" in metrics:
                report.append(f"  Top-3 Accuracy: {metrics['top_3_accuracy']:.4f}")
            report.append("")
        
        report_text = "\n".join(report)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
        
        return report_text
    
    def save_metrics(self, metrics: Dict[str, Any], file_path: Path):
        """Save metrics to JSON file."""
        # Convert numpy arrays to lists for JSON serialization
        serializable_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, np.ndarray):
                serializable_metrics[key] = value.tolist()
            elif isinstance(value, dict):
                serializable_metrics[key] = {}
                for k, v in value.items():
                    if isinstance(v, np.ndarray):
                        serializable_metrics[key][k] = v.tolist()
                    else:
                        serializable_metrics[key][k] = v
            else:
                serializable_metrics[key] = value
        
        with open(file_path, 'w') as f:
            json.dump(serializable_metrics, f, indent=2)


def evaluate_model(model: nn.Module, test_loader, class_names: List[str], 
                  device: torch.device, save_dir: Optional[Path] = None) -> Dict[str, Any]:
    """
    Comprehensive model evaluation.
    
    Args:
        model: Trained model
        test_loader: Test data loader
        class_names: List of class names
        device: Computation device
        save_dir: Directory to save results
        
    Returns:
        Dictionary containing all evaluation metrics
    """
    model.eval()
    
    all_predictions = []
    all_probabilities = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            probabilities = F.softmax(outputs, dim=1)
            predictions = torch.argmax(outputs, dim=1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Convert to numpy arrays
    y_true = np.array(all_labels)
    y_pred = np.array(all_predictions)
    y_proba = np.array(all_probabilities)
    
    # Calculate metrics
    metrics_calc = MetricsCalculator(class_names, save_dir)
    metrics = metrics_calc.calculate_metrics(y_true, y_pred, y_proba)
    
    # Generate visualizations
    if save_dir:
        metrics_calc.plot_confusion_matrix(y_true, y_pred)
        metrics_calc.plot_confusion_matrix(y_true, y_pred, normalize=True, 
                                         title="Normalized Confusion Matrix")
        metrics_calc.plot_roc_curves(y_true, y_proba)
        metrics_calc.plot_precision_recall_curves(y_true, y_proba)
        metrics_calc.plot_per_class_metrics(metrics)
        
        # Save metrics and report
        metrics_calc.save_metrics(metrics, save_dir / "metrics.json")
        report = metrics_calc.generate_report(metrics, save_dir / "evaluation_report.txt")
        
        logger.info(f"Evaluation complete. Results saved to {save_dir}")
    
    return metrics


if __name__ == "__main__":
    # Test metrics calculator
    class_names = ["bat", "owl", "raccoon", "opossum", "deer", "fox", "coyote", "unknown"]
    
    # Generate dummy data
    np.random.seed(42)
    n_samples = 1000
    y_true = np.random.randint(0, len(class_names), n_samples)
    y_pred = np.random.randint(0, len(class_names), n_samples)
    y_proba = np.random.dirichlet(np.ones(len(class_names)), n_samples)
    
    # Test metrics calculation
    metrics_calc = MetricsCalculator(class_names)
    metrics = metrics_calc.calculate_metrics(y_true, y_pred, y_proba)
    
    print("Metrics calculated successfully!")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"F1-Score (macro): {metrics['f1_macro']:.4f}")
    
    # Test visualizations
    fig = metrics_calc.plot_confusion_matrix(y_true, y_pred)
    print("Confusion matrix plot created successfully!")
    
    plt.close('all')  # Close all figures
    
    # Test report generation
    report = metrics_calc.generate_report(metrics)
    print("Report generated successfully!")
    print(report[:500] + "...")  # Print first 500 characters