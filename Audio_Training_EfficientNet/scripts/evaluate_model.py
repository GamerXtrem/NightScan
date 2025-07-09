"""
Model Evaluation and Comparison Script

Comprehensive evaluation script for comparing EfficientNet vs ResNet18 models
with detailed metrics, performance analysis, and visualization.
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import sys
import csv

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from models.efficientnet_config import EfficientNetConfig, create_model as create_efficientnet_model
from scripts.predict_efficientnet import EfficientNetPredictor
from scripts.train_efficientnet import EfficientNetDataset
from utils.metrics import MetricsCalculator
from utils.cross_validation import CrossValidationDataset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Comprehensive model evaluator for comparing different architectures.
    """
    
    def __init__(self, save_dir: Path):
        self.save_dir = save_dir
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Class names
        self.class_names = [
            "bird_song", "mammal_call", "insect_sound",
            "amphibian_call", "environmental_sound", "unknown_species"
        ]
        
        # Results storage
        self.evaluation_results = {}
        self.comparison_results = {}
        
        logger.info(f"Model evaluator initialized. Results will be saved to: {save_dir}")
    
    def evaluate_efficientnet(
        self,
        model_path: Path,
        config_path: Optional[Path],
        test_csv: Path,
        batch_size: int = 32,
        device: Optional[torch.device] = None
    ) -> Dict[str, Any]:
        """
        Evaluate EfficientNet model.
        
        Args:
            model_path: Path to EfficientNet model
            config_path: Path to model configuration
            test_csv: Path to test CSV file
            batch_size: Batch size for evaluation
            device: Device to use
            
        Returns:
            Evaluation results
        """
        logger.info("Evaluating EfficientNet model...")
        
        # Initialize predictor
        predictor = EfficientNetPredictor(
            model_path=model_path,
            config_path=config_path,
            device=device
        )
        
        # Load test data
        test_dataset = EfficientNetDataset(test_csv)
        test_dataset.set_training_mode(False)
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4
        )
        
        # Evaluate
        results = self._evaluate_model(
            predictor.model,
            test_loader,
            device or predictor.device,
            "EfficientNet"
        )
        
        # Add model-specific information
        results["model_info"] = predictor.get_model_info()
        results["model_type"] = "EfficientNet"
        
        return results
    
    def evaluate_resnet18(
        self,
        model_path: Path,
        test_csv: Path,
        batch_size: int = 32,
        device: Optional[torch.device] = None
    ) -> Dict[str, Any]:
        """
        Evaluate ResNet18 model.
        
        Args:
            model_path: Path to ResNet18 model
            test_csv: Path to test CSV file
            batch_size: Batch size for evaluation
            device: Device to use
            
        Returns:
            Evaluation results
        """
        logger.info("Evaluating ResNet18 model...")
        
        # Load ResNet18 model
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create ResNet18 model (simplified for comparison)
        from torchvision import models
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, len(self.class_names))
        
        # Load weights
        state_dict = torch.load(model_path, map_location=device)
        if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
            model.load_state_dict(state_dict["model_state_dict"])
        else:
            model.load_state_dict(state_dict)
        
        model.to(device)
        model.eval()
        
        # Load test data (compatible with ResNet18 format)
        test_dataset = EfficientNetDataset(test_csv)
        test_dataset.set_training_mode(False)
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4
        )
        
        # Evaluate
        results = self._evaluate_model(model, test_loader, device, "ResNet18")
        
        # Add model-specific information
        total_params = sum(p.numel() for p in model.parameters())
        results["model_info"] = {
            "model_name": "ResNet18",
            "total_parameters": total_params,
            "model_size_mb": total_params * 4 / (1024 * 1024),
            "device": str(device)
        }
        results["model_type"] = "ResNet18"
        
        return results
    
    def _evaluate_model(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        device: torch.device,
        model_name: str
    ) -> Dict[str, Any]:
        """
        Evaluate a model on test data.
        
        Args:
            model: Model to evaluate
            test_loader: Test data loader
            device: Device to use
            model_name: Name of the model
            
        Returns:
            Evaluation results
        """
        model.eval()
        
        # Initialize metrics calculator
        metrics_calculator = MetricsCalculator(
            self.class_names,
            save_dir=self.save_dir / f"{model_name.lower()}_metrics"
        )
        
        # Tracking variables
        all_predictions = []
        all_targets = []
        all_probabilities = []
        total_time = 0
        total_samples = 0
        
        # Evaluation loop
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(test_loader, desc=f"Evaluating {model_name}")):
                inputs, targets = batch
                inputs = inputs.to(device)
                
                # Handle both regular labels and one-hot encoded labels
                if targets.dim() == 2:  # One-hot encoded
                    target_labels = targets.argmax(dim=1)
                else:  # Regular labels
                    target_labels = targets
                
                target_labels = target_labels.to(device)
                
                # Time inference
                start_time = time.time()
                outputs = model(inputs)
                inference_time = time.time() - start_time
                
                # Get predictions
                probabilities = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)
                
                # Update metrics
                metrics_calculator.update(predicted, target_labels, probabilities)
                
                # Store results
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(target_labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                
                # Update timing
                total_time += inference_time
                total_samples += inputs.size(0)
        
        # Calculate metrics
        basic_metrics = metrics_calculator.compute_basic_metrics()
        class_metrics = metrics_calculator.compute_class_metrics()
        confusion_mat = metrics_calculator.compute_confusion_matrix()
        
        # Generate comprehensive report
        metrics_calculator.save_comprehensive_report(f"{model_name.lower()}_evaluation.json")
        metrics_calculator.create_summary_plots()
        
        # Performance metrics
        avg_inference_time = total_time / total_samples
        throughput = total_samples / total_time
        
        # Compile results
        results = {
            "model_name": model_name,
            "basic_metrics": basic_metrics,
            "class_metrics": class_metrics,
            "confusion_matrix": confusion_mat.tolist(),
            "performance_metrics": {
                "avg_inference_time_ms": avg_inference_time * 1000,
                "throughput_samples_per_sec": throughput,
                "total_samples": total_samples,
                "total_time_sec": total_time
            },
            "predictions": all_predictions,
            "targets": all_targets,
            "probabilities": all_probabilities
        }
        
        return results
    
    def compare_models(
        self,
        efficientnet_results: Dict[str, Any],
        resnet18_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Compare two model evaluation results.
        
        Args:
            efficientnet_results: EfficientNet evaluation results
            resnet18_results: ResNet18 evaluation results
            
        Returns:
            Comparison results
        """
        logger.info("Comparing model results...")
        
        comparison = {
            "models_compared": [
                efficientnet_results["model_name"],
                resnet18_results["model_name"]
            ],
            "metric_comparisons": {},
            "performance_comparisons": {},
            "class_performance_comparisons": {},
            "model_size_comparison": {},
            "winner": {}
        }
        
        # Compare basic metrics
        eff_metrics = efficientnet_results["basic_metrics"]
        res_metrics = resnet18_results["basic_metrics"]
        
        for metric_name in eff_metrics.keys():
            eff_value = eff_metrics[metric_name]
            res_value = res_metrics[metric_name]
            
            comparison["metric_comparisons"][metric_name] = {
                "EfficientNet": eff_value,
                "ResNet18": res_value,
                "difference": eff_value - res_value,
                "percentage_improvement": ((eff_value - res_value) / res_value * 100) if res_value != 0 else 0,
                "winner": "EfficientNet" if eff_value > res_value else "ResNet18"
            }
        
        # Compare performance metrics
        eff_perf = efficientnet_results["performance_metrics"]
        res_perf = resnet18_results["performance_metrics"]
        
        for metric_name in eff_perf.keys():
            if metric_name in res_perf:
                eff_value = eff_perf[metric_name]
                res_value = res_perf[metric_name]
                
                # For inference time, lower is better
                if "time" in metric_name:
                    winner = "EfficientNet" if eff_value < res_value else "ResNet18"
                else:
                    winner = "EfficientNet" if eff_value > res_value else "ResNet18"
                
                comparison["performance_comparisons"][metric_name] = {
                    "EfficientNet": eff_value,
                    "ResNet18": res_value,
                    "difference": eff_value - res_value,
                    "winner": winner
                }
        
        # Compare class performance
        eff_class = efficientnet_results["class_metrics"]
        res_class = resnet18_results["class_metrics"]
        
        for class_name in self.class_names:
            if class_name in eff_class and class_name in res_class:
                comparison["class_performance_comparisons"][class_name] = {}
                
                for metric_name in ["f1_score", "precision", "recall"]:
                    eff_value = eff_class[class_name][metric_name]
                    res_value = res_class[class_name][metric_name]
                    
                    comparison["class_performance_comparisons"][class_name][metric_name] = {
                        "EfficientNet": eff_value,
                        "ResNet18": res_value,
                        "difference": eff_value - res_value,
                        "winner": "EfficientNet" if eff_value > res_value else "ResNet18"
                    }
        
        # Compare model sizes
        eff_info = efficientnet_results["model_info"]
        res_info = resnet18_results["model_info"]
        
        comparison["model_size_comparison"] = {
            "EfficientNet": {
                "parameters": eff_info["total_parameters"],
                "size_mb": eff_info["model_size_mb"]
            },
            "ResNet18": {
                "parameters": res_info["total_parameters"],
                "size_mb": res_info["model_size_mb"]
            },
            "parameter_ratio": eff_info["total_parameters"] / res_info["total_parameters"],
            "size_ratio": eff_info["model_size_mb"] / res_info["model_size_mb"]
        }
        
        # Determine overall winner
        key_metrics = ["accuracy", "f1_macro", "f1_weighted"]
        eff_wins = 0
        res_wins = 0
        
        for metric in key_metrics:
            if metric in comparison["metric_comparisons"]:
                winner = comparison["metric_comparisons"][metric]["winner"]
                if winner == "EfficientNet":
                    eff_wins += 1
                else:
                    res_wins += 1
        
        comparison["winner"] = {
            "overall": "EfficientNet" if eff_wins > res_wins else "ResNet18",
            "EfficientNet_wins": eff_wins,
            "ResNet18_wins": res_wins,
            "key_metrics_evaluated": key_metrics
        }
        
        return comparison
    
    def create_comparison_visualizations(
        self,
        efficientnet_results: Dict[str, Any],
        resnet18_results: Dict[str, Any],
        comparison_results: Dict[str, Any]
    ) -> None:
        """
        Create visualization plots for model comparison.
        
        Args:
            efficientnet_results: EfficientNet evaluation results
            resnet18_results: ResNet18 evaluation results
            comparison_results: Comparison results
        """
        logger.info("Creating comparison visualizations...")
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        
        # 1. Metrics comparison bar chart
        self._plot_metrics_comparison(comparison_results)
        
        # 2. Confusion matrices comparison
        self._plot_confusion_matrices_comparison(efficientnet_results, resnet18_results)
        
        # 3. Per-class performance comparison
        self._plot_class_performance_comparison(comparison_results)
        
        # 4. Performance metrics comparison
        self._plot_performance_comparison(comparison_results)
        
        # 5. Model size comparison
        self._plot_model_size_comparison(comparison_results)
        
        logger.info("Comparison visualizations saved")
    
    def _plot_metrics_comparison(self, comparison_results: Dict[str, Any]) -> None:
        """Plot basic metrics comparison."""
        metrics = comparison_results["metric_comparisons"]
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        metric_names = list(metrics.keys())
        eff_values = [metrics[m]["EfficientNet"] for m in metric_names]
        res_values = [metrics[m]["ResNet18"] for m in metric_names]
        
        x = np.arange(len(metric_names))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, eff_values, width, label='EfficientNet', alpha=0.8)
        bars2 = ax.bar(x + width/2, res_values, width, label='ResNet18', alpha=0.8)
        
        ax.set_xlabel('Metrics')
        ax.set_ylabel('Values')
        ax.set_title('Model Performance Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(metric_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)
        
        for bar in bars2:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / "metrics_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_confusion_matrices_comparison(
        self,
        efficientnet_results: Dict[str, Any],
        resnet18_results: Dict[str, Any]
    ) -> None:
        """Plot confusion matrices side by side."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # EfficientNet confusion matrix
        eff_cm = np.array(efficientnet_results["confusion_matrix"])
        eff_cm_norm = eff_cm.astype('float') / eff_cm.sum(axis=1)[:, np.newaxis]
        
        sns.heatmap(eff_cm_norm, annot=True, fmt='.2f', cmap='Blues',
                   xticklabels=self.class_names, yticklabels=self.class_names,
                   ax=ax1)
        ax1.set_title('EfficientNet - Confusion Matrix')
        ax1.set_xlabel('Predicted')
        ax1.set_ylabel('Actual')
        
        # ResNet18 confusion matrix
        res_cm = np.array(resnet18_results["confusion_matrix"])
        res_cm_norm = res_cm.astype('float') / res_cm.sum(axis=1)[:, np.newaxis]
        
        sns.heatmap(res_cm_norm, annot=True, fmt='.2f', cmap='Oranges',
                   xticklabels=self.class_names, yticklabels=self.class_names,
                   ax=ax2)
        ax2.set_title('ResNet18 - Confusion Matrix')
        ax2.set_xlabel('Predicted')
        ax2.set_ylabel('Actual')
        
        plt.tight_layout()
        plt.savefig(self.save_dir / "confusion_matrices_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_class_performance_comparison(self, comparison_results: Dict[str, Any]) -> None:
        """Plot per-class performance comparison."""
        class_comparisons = comparison_results["class_performance_comparisons"]
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        metrics = ["f1_score", "precision", "recall"]
        
        for i, metric in enumerate(metrics):
            class_names = list(class_comparisons.keys())
            eff_values = [class_comparisons[c][metric]["EfficientNet"] for c in class_names]
            res_values = [class_comparisons[c][metric]["ResNet18"] for c in class_names]
            
            x = np.arange(len(class_names))
            width = 0.35
            
            axes[i].bar(x - width/2, eff_values, width, label='EfficientNet', alpha=0.8)
            axes[i].bar(x + width/2, res_values, width, label='ResNet18', alpha=0.8)
            
            axes[i].set_xlabel('Classes')
            axes[i].set_ylabel(metric.replace('_', ' ').title())
            axes[i].set_title(f'{metric.replace("_", " ").title()} Comparison')
            axes[i].set_xticks(x)
            axes[i].set_xticklabels(class_names, rotation=45, ha='right')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / "class_performance_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_performance_comparison(self, comparison_results: Dict[str, Any]) -> None:
        """Plot performance metrics comparison."""
        perf_comparisons = comparison_results["performance_comparisons"]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Inference time comparison
        if "avg_inference_time_ms" in perf_comparisons:
            models = ["EfficientNet", "ResNet18"]
            times = [
                perf_comparisons["avg_inference_time_ms"]["EfficientNet"],
                perf_comparisons["avg_inference_time_ms"]["ResNet18"]
            ]
            
            bars = ax1.bar(models, times, color=['skyblue', 'lightcoral'])
            ax1.set_ylabel('Average Inference Time (ms)')
            ax1.set_title('Inference Time Comparison')
            ax1.grid(True, alpha=0.3)
            
            # Add value labels
            for bar, time in zip(bars, times):
                ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.1,
                        f'{time:.2f}ms', ha='center', va='bottom')
        
        # Throughput comparison
        if "throughput_samples_per_sec" in perf_comparisons:
            throughputs = [
                perf_comparisons["throughput_samples_per_sec"]["EfficientNet"],
                perf_comparisons["throughput_samples_per_sec"]["ResNet18"]
            ]
            
            bars = ax2.bar(models, throughputs, color=['skyblue', 'lightcoral'])
            ax2.set_ylabel('Throughput (samples/sec)')
            ax2.set_title('Throughput Comparison')
            ax2.grid(True, alpha=0.3)
            
            # Add value labels
            for bar, throughput in zip(bars, throughputs):
                ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.1,
                        f'{throughput:.1f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(self.save_dir / "performance_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_model_size_comparison(self, comparison_results: Dict[str, Any]) -> None:
        """Plot model size comparison."""
        size_comparison = comparison_results["model_size_comparison"]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        models = ["EfficientNet", "ResNet18"]
        
        # Parameters comparison
        parameters = [
            size_comparison["EfficientNet"]["parameters"],
            size_comparison["ResNet18"]["parameters"]
        ]
        
        bars1 = ax1.bar(models, parameters, color=['skyblue', 'lightcoral'])
        ax1.set_ylabel('Number of Parameters')
        ax1.set_title('Model Parameters Comparison')
        ax1.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, params in zip(bars1, parameters):
            ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.1,
                    f'{params:,}', ha='center', va='bottom')
        
        # Size comparison
        sizes = [
            size_comparison["EfficientNet"]["size_mb"],
            size_comparison["ResNet18"]["size_mb"]
        ]
        
        bars2 = ax2.bar(models, sizes, color=['skyblue', 'lightcoral'])
        ax2.set_ylabel('Model Size (MB)')
        ax2.set_title('Model Size Comparison')
        ax2.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, size in zip(bars2, sizes):
            ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.1,
                    f'{size:.1f}MB', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(self.save_dir / "model_size_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_comprehensive_report(
        self,
        efficientnet_results: Dict[str, Any],
        resnet18_results: Dict[str, Any],
        comparison_results: Dict[str, Any]
    ) -> None:
        """
        Generate a comprehensive evaluation report.
        
        Args:
            efficientnet_results: EfficientNet evaluation results
            resnet18_results: ResNet18 evaluation results
            comparison_results: Comparison results
        """
        report_path = self.save_dir / "comprehensive_evaluation_report.txt"
        
        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("COMPREHENSIVE MODEL EVALUATION REPORT\n")
            f.write("="*80 + "\n\n")
            
            # Executive Summary
            f.write("EXECUTIVE SUMMARY\n")
            f.write("-"*40 + "\n")
            winner = comparison_results["winner"]["overall"]
            f.write(f"Overall Winner: {winner}\n")
            f.write(f"Key Metrics Winner: {winner} ({comparison_results['winner']['overall']})\n")
            f.write(f"EfficientNet Wins: {comparison_results['winner']['EfficientNet_wins']}\n")
            f.write(f"ResNet18 Wins: {comparison_results['winner']['ResNet18_wins']}\n\n")
            
            # Model Information
            f.write("MODEL INFORMATION\n")
            f.write("-"*40 + "\n")
            
            eff_info = efficientnet_results["model_info"]
            res_info = resnet18_results["model_info"]
            
            f.write(f"EfficientNet:\n")
            f.write(f"  Model: {eff_info['model_name']}\n")
            f.write(f"  Parameters: {eff_info['total_parameters']:,}\n")
            f.write(f"  Size: {eff_info['model_size_mb']:.1f} MB\n")
            f.write(f"  Device: {eff_info['device']}\n\n")
            
            f.write(f"ResNet18:\n")
            f.write(f"  Model: {res_info['model_name']}\n")
            f.write(f"  Parameters: {res_info['total_parameters']:,}\n")
            f.write(f"  Size: {res_info['model_size_mb']:.1f} MB\n")
            f.write(f"  Device: {res_info['device']}\n\n")
            
            # Performance Metrics
            f.write("PERFORMANCE METRICS\n")
            f.write("-"*40 + "\n")
            
            metric_comparisons = comparison_results["metric_comparisons"]
            
            for metric_name, comparison in metric_comparisons.items():
                f.write(f"{metric_name}:\n")
                f.write(f"  EfficientNet: {comparison['EfficientNet']:.4f}\n")
                f.write(f"  ResNet18: {comparison['ResNet18']:.4f}\n")
                f.write(f"  Difference: {comparison['difference']:.4f}\n")
                f.write(f"  Improvement: {comparison['percentage_improvement']:.2f}%\n")
                f.write(f"  Winner: {comparison['winner']}\n\n")
            
            # Performance Comparison
            f.write("INFERENCE PERFORMANCE\n")
            f.write("-"*40 + "\n")
            
            perf_comparisons = comparison_results["performance_comparisons"]
            
            for metric_name, comparison in perf_comparisons.items():
                f.write(f"{metric_name}:\n")
                f.write(f"  EfficientNet: {comparison['EfficientNet']:.4f}\n")
                f.write(f"  ResNet18: {comparison['ResNet18']:.4f}\n")
                f.write(f"  Winner: {comparison['winner']}\n\n")
            
            # Class-specific Performance
            f.write("CLASS-SPECIFIC PERFORMANCE\n")
            f.write("-"*40 + "\n")
            
            class_comparisons = comparison_results["class_performance_comparisons"]
            
            for class_name, metrics in class_comparisons.items():
                f.write(f"{class_name}:\n")
                for metric_name, comparison in metrics.items():
                    f.write(f"  {metric_name}:\n")
                    f.write(f"    EfficientNet: {comparison['EfficientNet']:.4f}\n")
                    f.write(f"    ResNet18: {comparison['ResNet18']:.4f}\n")
                    f.write(f"    Winner: {comparison['winner']}\n")
                f.write("\n")
            
            # Recommendations
            f.write("RECOMMENDATIONS\n")
            f.write("-"*40 + "\n")
            
            if winner == "EfficientNet":
                f.write("✓ EfficientNet shows superior performance across key metrics\n")
                f.write("✓ Recommended for production deployment\n")
                f.write("✓ Consider the trade-offs in model size and complexity\n")
            else:
                f.write("✓ ResNet18 shows competitive performance\n")
                f.write("✓ May be preferred for simpler deployment scenarios\n")
                f.write("✓ Lower complexity and potentially faster inference\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write("End of Report\n")
            f.write("="*80 + "\n")
        
        logger.info(f"Comprehensive report saved to: {report_path}")
        
        # Also save as JSON for programmatic access
        json_report = {
            "efficientnet_results": efficientnet_results,
            "resnet18_results": resnet18_results,
            "comparison_results": comparison_results
        }
        
        with open(self.save_dir / "evaluation_results.json", 'w') as f:
            json.dump(json_report, f, indent=2, default=str)


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate and compare EfficientNet vs ResNet18")
    
    # Required arguments
    parser.add_argument("--efficientnet_model", type=Path, required=True,
                       help="Path to EfficientNet model")
    parser.add_argument("--resnet18_model", type=Path, required=True,
                       help="Path to ResNet18 model")
    parser.add_argument("--test_csv", type=Path, required=True,
                       help="Path to test CSV file")
    parser.add_argument("--output_dir", type=Path, required=True,
                       help="Output directory for results")
    
    # Optional arguments
    parser.add_argument("--efficientnet_config", type=Path, default=None,
                       help="Path to EfficientNet configuration file")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size for evaluation")
    parser.add_argument("--device", type=str, default=None,
                       help="Device to use (cuda, mps, cpu)")
    
    args = parser.parse_args()
    
    # Validate inputs
    for model_path in [args.efficientnet_model, args.resnet18_model]:
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
    
    if not args.test_csv.exists():
        raise FileNotFoundError(f"Test CSV file not found: {args.test_csv}")
    
    # Setup device
    device = None
    if args.device:
        device = torch.device(args.device)
    
    # Initialize evaluator
    evaluator = ModelEvaluator(args.output_dir)
    
    # Evaluate EfficientNet
    logger.info("Starting EfficientNet evaluation...")
    efficientnet_results = evaluator.evaluate_efficientnet(
        model_path=args.efficientnet_model,
        config_path=args.efficientnet_config,
        test_csv=args.test_csv,
        batch_size=args.batch_size,
        device=device
    )
    
    # Evaluate ResNet18
    logger.info("Starting ResNet18 evaluation...")
    resnet18_results = evaluator.evaluate_resnet18(
        model_path=args.resnet18_model,
        test_csv=args.test_csv,
        batch_size=args.batch_size,
        device=device
    )
    
    # Compare models
    logger.info("Comparing models...")
    comparison_results = evaluator.compare_models(
        efficientnet_results, resnet18_results
    )
    
    # Create visualizations
    logger.info("Creating visualizations...")
    evaluator.create_comparison_visualizations(
        efficientnet_results, resnet18_results, comparison_results
    )
    
    # Generate comprehensive report
    logger.info("Generating comprehensive report...")
    evaluator.generate_comprehensive_report(
        efficientnet_results, resnet18_results, comparison_results
    )
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("EVALUATION SUMMARY")
    logger.info("="*60)
    
    winner = comparison_results["winner"]["overall"]
    logger.info(f"Overall Winner: {winner}")
    
    # Key metrics
    key_metrics = ["accuracy", "f1_macro", "f1_weighted"]
    for metric in key_metrics:
        if metric in comparison_results["metric_comparisons"]:
            comp = comparison_results["metric_comparisons"][metric]
            logger.info(f"{metric}: EfficientNet={comp['EfficientNet']:.4f}, "
                       f"ResNet18={comp['ResNet18']:.4f}, "
                       f"Winner={comp['winner']}")
    
    logger.info(f"\nDetailed results saved to: {args.output_dir}")
    logger.info("="*60)


if __name__ == "__main__":
    main()