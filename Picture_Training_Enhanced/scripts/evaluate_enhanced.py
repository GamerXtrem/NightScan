"""
Enhanced Model Evaluation Script for Wildlife Photo Classification

Comprehensive evaluation script with support for multiple metrics,
model comparison, and detailed analysis.
"""

import argparse
import logging
import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from models.photo_config import PhotoConfig, create_model, get_config
from models.data_augmentation import AugmentationManager
from utils.metrics import MetricsCalculator, evaluate_model
from utils.training_utils import load_checkpoint
from scripts.train_enhanced import PhotoDataset

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Comprehensive model evaluation with multiple metrics."""
    
    def __init__(self, class_names: List[str], save_dir: Path):
        self.class_names = class_names
        self.save_dir = save_dir
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.metrics_calc = MetricsCalculator(class_names, save_dir)
        self.results = []
    
    def evaluate_single_model(self, model: nn.Module, test_loader: DataLoader,
                            device: torch.device, model_name: str) -> Dict[str, Any]:
        """Evaluate a single model."""
        logger.info(f"Evaluating model: {model_name}")
        
        model.eval()
        start_time = time.time()
        
        all_predictions = []
        all_probabilities = []
        all_labels = []
        inference_times = []
        
        with torch.no_grad():
            for batch_idx, (inputs, labels) in enumerate(tqdm(test_loader, desc="Evaluating")):
                inputs, labels = inputs.to(device), labels.to(device)
                
                # Measure inference time
                batch_start = time.time()
                outputs = model(inputs)
                batch_time = time.time() - batch_start
                inference_times.append(batch_time / inputs.size(0))  # Per-sample time
                
                probabilities = F.softmax(outputs, dim=1)
                predictions = torch.argmax(outputs, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        total_time = time.time() - start_time
        
        # Convert to numpy arrays
        y_true = np.array(all_labels)
        y_pred = np.array(all_predictions)
        y_proba = np.array(all_probabilities)
        
        # Calculate metrics
        metrics = self.metrics_calc.calculate_metrics(y_true, y_pred, y_proba)
        
        # Add performance metrics
        metrics['inference_time'] = {
            'total_time': total_time,
            'avg_per_sample': np.mean(inference_times),
            'std_per_sample': np.std(inference_times),
            'samples_per_second': len(y_true) / total_time
        }
        
        # Add model info
        metrics['model_name'] = model_name
        metrics['num_parameters'] = sum(p.numel() for p in model.parameters())
        metrics['model_size_mb'] = metrics['num_parameters'] * 4 / (1024 * 1024)
        
        # Generate model-specific visualizations
        model_save_dir = self.save_dir / f"{model_name}_evaluation"
        model_save_dir.mkdir(exist_ok=True)
        
        model_metrics_calc = MetricsCalculator(self.class_names, model_save_dir)
        model_metrics_calc.plot_confusion_matrix(y_true, y_pred, 
                                                title=f"{model_name} - Confusion Matrix")
        model_metrics_calc.plot_confusion_matrix(y_true, y_pred, normalize=True,
                                                title=f"{model_name} - Normalized Confusion Matrix")
        model_metrics_calc.plot_roc_curves(y_true, y_proba, 
                                         title=f"{model_name} - ROC Curves")
        model_metrics_calc.plot_precision_recall_curves(y_true, y_proba,
                                                       title=f"{model_name} - PR Curves")
        model_metrics_calc.plot_per_class_metrics(metrics, 
                                                 title=f"{model_name} - Per-Class Metrics")
        
        # Save metrics
        model_metrics_calc.save_metrics(metrics, model_save_dir / "metrics.json")
        
        # Generate report
        report = model_metrics_calc.generate_report(metrics, model_save_dir / "report.txt")
        
        logger.info(f"{model_name} evaluation complete:")
        logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"  F1-Score: {metrics['f1_macro']:.4f}")
        logger.info(f"  Inference time: {metrics['inference_time']['avg_per_sample']*1000:.2f}ms/sample")
        
        return metrics
    
    def compare_models(self, models_results: List[Dict[str, Any]]):
        """Compare multiple models and generate comparison plots."""
        logger.info("Generating model comparison plots...")
        
        # Create comparison dataframe
        comparison_data = []
        for result in models_results:
            comparison_data.append({
                'Model': result['model_name'],
                'Accuracy': result['accuracy'],
                'F1-Score': result['f1_macro'],
                'Precision': result['precision_macro'],
                'Recall': result['recall_macro'],
                'Inference Time (ms)': result['inference_time']['avg_per_sample'] * 1000,
                'Model Size (MB)': result['model_size_mb'],
                'Parameters': result['num_parameters']
            })
        
        df = pd.DataFrame(comparison_data)
        
        # Save comparison table
        df.to_csv(self.save_dir / "model_comparison.csv", index=False)
        
        # Generate comparison plots
        self._plot_model_comparison(df)
        self._plot_performance_vs_size(df)
        self._plot_performance_vs_inference_time(df)
        
        # Generate detailed comparison report
        self._generate_comparison_report(df, models_results)
    
    def _plot_model_comparison(self, df: pd.DataFrame):
        """Plot model comparison metrics."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        models = df['Model']
        
        # Accuracy comparison
        axes[0, 0].bar(models, df['Accuracy'], color='skyblue', alpha=0.8)
        axes[0, 0].set_title('Model Accuracy Comparison')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].set_ylim(0, 1)
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # F1-Score comparison
        axes[0, 1].bar(models, df['F1-Score'], color='lightcoral', alpha=0.8)
        axes[0, 1].set_title('Model F1-Score Comparison')
        axes[0, 1].set_ylabel('F1-Score')
        axes[0, 1].set_ylim(0, 1)
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Precision vs Recall
        axes[1, 0].scatter(df['Recall'], df['Precision'], s=100, alpha=0.7)
        for i, model in enumerate(models):
            axes[1, 0].annotate(model, (df['Recall'].iloc[i], df['Precision'].iloc[i]),
                               xytext=(5, 5), textcoords='offset points')
        axes[1, 0].set_xlabel('Recall')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].set_title('Precision vs Recall')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Inference time comparison
        axes[1, 1].bar(models, df['Inference Time (ms)'], color='lightgreen', alpha=0.8)
        axes[1, 1].set_title('Inference Time Comparison')
        axes[1, 1].set_ylabel('Inference Time (ms)')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / "model_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_performance_vs_size(self, df: pd.DataFrame):
        """Plot performance vs model size."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Accuracy vs Model Size
        ax1.scatter(df['Model Size (MB)'], df['Accuracy'], s=100, alpha=0.7)
        for i, model in enumerate(df['Model']):
            ax1.annotate(model, (df['Model Size (MB)'].iloc[i], df['Accuracy'].iloc[i]),
                        xytext=(5, 5), textcoords='offset points')
        ax1.set_xlabel('Model Size (MB)')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Accuracy vs Model Size')
        ax1.grid(True, alpha=0.3)
        
        # F1-Score vs Parameters
        ax2.scatter(df['Parameters'] / 1e6, df['F1-Score'], s=100, alpha=0.7)
        for i, model in enumerate(df['Model']):
            ax2.annotate(model, (df['Parameters'].iloc[i] / 1e6, df['F1-Score'].iloc[i]),
                        xytext=(5, 5), textcoords='offset points')
        ax2.set_xlabel('Parameters (Millions)')
        ax2.set_ylabel('F1-Score')
        ax2.set_title('F1-Score vs Parameters')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / "performance_vs_size.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_performance_vs_inference_time(self, df: pd.DataFrame):
        """Plot performance vs inference time."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Accuracy vs Inference Time
        ax1.scatter(df['Inference Time (ms)'], df['Accuracy'], s=100, alpha=0.7)
        for i, model in enumerate(df['Model']):
            ax1.annotate(model, (df['Inference Time (ms)'].iloc[i], df['Accuracy'].iloc[i]),
                        xytext=(5, 5), textcoords='offset points')
        ax1.set_xlabel('Inference Time (ms)')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Accuracy vs Inference Time')
        ax1.grid(True, alpha=0.3)
        
        # Efficiency plot (Accuracy / Inference Time)
        efficiency = df['Accuracy'] / df['Inference Time (ms)']
        ax2.bar(df['Model'], efficiency, alpha=0.7)
        ax2.set_ylabel('Efficiency (Accuracy/ms)')
        ax2.set_title('Model Efficiency')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / "performance_vs_inference_time.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_comparison_report(self, df: pd.DataFrame, detailed_results: List[Dict[str, Any]]):
        """Generate detailed comparison report."""
        report = []
        report.append("=" * 80)
        report.append("MODEL COMPARISON REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Overall comparison table
        report.append("OVERALL PERFORMANCE COMPARISON:")
        report.append(df.to_string(index=False))
        report.append("")
        
        # Best model in each category
        best_accuracy = df.loc[df['Accuracy'].idxmax()]
        best_f1 = df.loc[df['F1-Score'].idxmax()]
        best_speed = df.loc[df['Inference Time (ms)'].idxmin()]
        best_efficiency = df.loc[(df['Accuracy'] / df['Inference Time (ms)']).idxmax()]
        
        report.append("BEST MODELS BY CATEGORY:")
        report.append(f"  Best Accuracy: {best_accuracy['Model']} ({best_accuracy['Accuracy']:.4f})")
        report.append(f"  Best F1-Score: {best_f1['Model']} ({best_f1['F1-Score']:.4f})")
        report.append(f"  Fastest Inference: {best_speed['Model']} ({best_speed['Inference Time (ms)']:.2f}ms)")
        report.append(f"  Best Efficiency: {best_efficiency['Model']} ({best_efficiency['Accuracy'] / best_efficiency['Inference Time (ms)']:.4f})")
        report.append("")
        
        # Per-class performance comparison
        report.append("PER-CLASS PERFORMANCE COMPARISON:")
        for class_name in self.class_names:
            report.append(f"\n{class_name.upper()}:")
            report.append(f"{'Model':<20} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
            report.append("-" * 50)
            
            for result in detailed_results:
                precision = result['per_class']['precision'].get(class_name, 0.0)
                recall = result['per_class']['recall'].get(class_name, 0.0)
                f1 = result['per_class']['f1'].get(class_name, 0.0)
                report.append(f"{result['model_name']:<20} {precision:<10.4f} {recall:<10.4f} {f1:<10.4f}")
        
        report.append("")
        report.append("RECOMMENDATIONS:")
        report.append(f"- For best accuracy: Use {best_accuracy['Model']}")
        report.append(f"- For balanced performance: Use {best_f1['Model']}")
        report.append(f"- For real-time inference: Use {best_speed['Model']}")
        report.append(f"- For best efficiency: Use {best_efficiency['Model']}")
        
        # Save report
        with open(self.save_dir / "comparison_report.txt", 'w') as f:
            f.write("\n".join(report))
        
        logger.info("Comparison report generated successfully")


def load_model_from_checkpoint(checkpoint_path: Path, config: PhotoConfig) -> nn.Module:
    """Load model from checkpoint."""
    model = create_model(config)
    
    if checkpoint_path.suffix == '.pth':
        # State dict only
        model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
    else:
        # Full checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
    
    return model


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Enhanced model evaluation")
    parser.add_argument("--models", type=Path, nargs="+", required=True,
                       help="Paths to model checkpoints")
    parser.add_argument("--model_names", type=str, nargs="+",
                       help="Names for models (optional)")
    parser.add_argument("--configs", type=str, nargs="+",
                       help="Configuration names for models")
    parser.add_argument("--test_csv", type=Path, required=True,
                       help="Test CSV file")
    parser.add_argument("--output_dir", type=Path, required=True,
                       help="Output directory for results")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size for evaluation")
    parser.add_argument("--num_workers", type=int, default=4,
                       help="Number of data loader workers")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use for evaluation")
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else 
                             "mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    logger.info(f"Using device: {device}")
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Validate inputs
    if args.model_names and len(args.model_names) != len(args.models):
        raise ValueError("Number of model names must match number of models")
    
    if args.configs and len(args.configs) != len(args.models):
        raise ValueError("Number of configs must match number of models")
    
    # Default names and configs
    model_names = args.model_names or [f"model_{i}" for i in range(len(args.models))]
    configs = args.configs or ["efficientnet_b1_balanced"] * len(args.models)
    
    # Load first model to get class names
    first_config = get_config(configs[0])
    first_model = load_model_from_checkpoint(args.models[0], first_config)
    
    # Create dummy augmentation manager for dataset
    aug_manager = AugmentationManager(first_config.to_dict())
    
    # Create test dataset
    test_dataset = PhotoDataset(
        args.test_csv,
        aug_manager,
        is_training=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    class_names = test_dataset.class_names
    logger.info(f"Test dataset: {len(test_dataset)} samples, {len(class_names)} classes")
    
    # Initialize evaluator
    evaluator = ModelEvaluator(class_names, args.output_dir)
    
    # Evaluate all models
    all_results = []
    
    for model_path, model_name, config_name in zip(args.models, model_names, configs):
        logger.info(f"Loading model: {model_path}")
        
        # Load configuration and model
        config = get_config(config_name)
        config.num_classes = len(class_names)  # Update for actual dataset
        
        model = load_model_from_checkpoint(model_path, config)
        model.to(device)
        
        # Evaluate model
        results = evaluator.evaluate_single_model(model, test_loader, device, model_name)
        all_results.append(results)
        
        # Clean up
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Generate comparison if multiple models
    if len(all_results) > 1:
        evaluator.compare_models(all_results)
        logger.info("Model comparison complete")
    
    # Save overall results
    with open(args.output_dir / "all_results.json", 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    logger.info(f"Evaluation complete. Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()