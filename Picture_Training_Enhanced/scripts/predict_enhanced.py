"""
Enhanced Prediction Script for Wildlife Photo Classification

Advanced prediction script with support for batch processing, confidence scoring,
and visualization of results.
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
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from models.photo_config import PhotoConfig, create_model, get_config
from models.data_augmentation import AugmentationManager

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PhotoPredictor:
    """Enhanced photo prediction with confidence scoring and visualization."""
    
    def __init__(self, model_path: Path, config_name: str, class_names: List[str],
                 device: torch.device = None):
        """
        Initialize photo predictor.
        
        Args:
            model_path: Path to model checkpoint
            config_name: Configuration name
            class_names: List of class names
            device: Computation device
        """
        self.model_path = model_path
        self.config_name = config_name
        self.class_names = class_names
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load configuration
        self.config = get_config(config_name)
        self.config.num_classes = len(class_names)
        
        # Initialize augmentation manager
        self.aug_manager = AugmentationManager(self.config.to_dict())
        
        # Load model
        self.model = self._load_model()
        
        logger.info(f"Predictor initialized with {len(class_names)} classes")
    
    def _load_model(self) -> nn.Module:
        """Load model from checkpoint."""
        model = create_model(self.config)
        
        if self.model_path.suffix == '.pth':
            # State dict only
            model.load_state_dict(torch.load(self.model_path, map_location='cpu'))
        else:
            # Full checkpoint
            checkpoint = torch.load(self.model_path, map_location='cpu')
            model.load_state_dict(checkpoint['model_state_dict'])
        
        model.to(self.device)
        model.eval()
        
        return model
    
    def predict_single(self, image_path: Path, top_k: int = 3) -> Dict[str, Any]:
        """
        Predict single image.
        
        Args:
            image_path: Path to image
            top_k: Number of top predictions to return
            
        Returns:
            Dictionary with prediction results
        """
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.aug_manager.image_augmentation(image, is_training=False)
            image_tensor = image_tensor.unsqueeze(0).to(self.device)
            
            # Predict
            start_time = time.time()
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = F.softmax(outputs, dim=1)
            
            inference_time = time.time() - start_time
            
            # Get top-k predictions
            top_probs, top_indices = torch.topk(probabilities, top_k, dim=1)
            top_probs = top_probs.cpu().numpy()[0]
            top_indices = top_indices.cpu().numpy()[0]
            
            # Format results
            predictions = []
            for i in range(top_k):
                predictions.append({
                    'class': self.class_names[top_indices[i]],
                    'class_id': int(top_indices[i]),
                    'confidence': float(top_probs[i])
                })
            
            result = {
                'image_path': str(image_path),
                'predictions': predictions,
                'inference_time': inference_time,
                'top_prediction': predictions[0]['class'],
                'top_confidence': predictions[0]['confidence']
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error predicting {image_path}: {e}")
            return {
                'image_path': str(image_path),
                'error': str(e),
                'predictions': [],
                'inference_time': 0.0,
                'top_prediction': None,
                'top_confidence': 0.0
            }
    
    def predict_batch(self, image_paths: List[Path], batch_size: int = 32,
                     top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Predict batch of images.
        
        Args:
            image_paths: List of image paths
            batch_size: Batch size for processing
            top_k: Number of top predictions to return
            
        Returns:
            List of prediction results
        """
        results = []
        
        for i in tqdm(range(0, len(image_paths), batch_size), desc="Predicting"):
            batch_paths = image_paths[i:i + batch_size]
            batch_images = []
            valid_indices = []
            
            # Load and preprocess batch
            for j, path in enumerate(batch_paths):
                try:
                    image = Image.open(path).convert('RGB')
                    image_tensor = self.aug_manager.image_augmentation(image, is_training=False)
                    batch_images.append(image_tensor)
                    valid_indices.append(j)
                except Exception as e:
                    logger.warning(f"Failed to load {path}: {e}")
                    results.append({
                        'image_path': str(path),
                        'error': str(e),
                        'predictions': [],
                        'inference_time': 0.0,
                        'top_prediction': None,
                        'top_confidence': 0.0
                    })
            
            if not batch_images:
                continue
            
            # Stack batch
            batch_tensor = torch.stack(batch_images).to(self.device)
            
            # Predict
            start_time = time.time()
            with torch.no_grad():
                outputs = self.model(batch_tensor)
                probabilities = F.softmax(outputs, dim=1)
            
            inference_time = time.time() - start_time
            
            # Process results
            for j, valid_idx in enumerate(valid_indices):
                path = batch_paths[valid_idx]
                probs = probabilities[j]
                
                # Get top-k predictions
                top_probs, top_indices = torch.topk(probs, top_k)
                top_probs = top_probs.cpu().numpy()
                top_indices = top_indices.cpu().numpy()
                
                # Format results
                predictions = []
                for k in range(top_k):
                    predictions.append({
                        'class': self.class_names[top_indices[k]],
                        'class_id': int(top_indices[k]),
                        'confidence': float(top_probs[k])
                    })
                
                result = {
                    'image_path': str(path),
                    'predictions': predictions,
                    'inference_time': inference_time / len(valid_indices),
                    'top_prediction': predictions[0]['class'],
                    'top_confidence': predictions[0]['confidence']
                }
                
                results.append(result)
        
        return results
    
    def create_prediction_visualization(self, image_path: Path, prediction: Dict[str, Any],
                                      output_path: Path):
        """Create visualization of prediction results."""
        try:
            # Load image
            image = Image.open(image_path).convert('RGB')
            
            # Create figure
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Display original image
            ax1.imshow(image)
            ax1.set_title(f'Original Image: {image_path.name}')
            ax1.axis('off')
            
            # Display predictions
            classes = [p['class'] for p in prediction['predictions']]
            confidences = [p['confidence'] for p in prediction['predictions']]
            
            bars = ax2.barh(classes, confidences)
            ax2.set_xlabel('Confidence')
            ax2.set_title('Top Predictions')
            ax2.set_xlim(0, 1)
            
            # Color bars based on confidence
            for i, (bar, conf) in enumerate(zip(bars, confidences)):
                if i == 0:  # Top prediction
                    bar.set_color('green' if conf > 0.5 else 'orange')
                else:
                    bar.set_color('lightblue')
                
                # Add confidence text
                ax2.text(conf + 0.01, bar.get_y() + bar.get_height()/2,
                        f'{conf:.3f}', va='center')
            
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.error(f"Error creating visualization for {image_path}: {e}")
    
    def analyze_predictions(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze batch prediction results."""
        analysis = {
            'total_predictions': len(results),
            'successful_predictions': 0,
            'failed_predictions': 0,
            'class_distribution': {},
            'confidence_stats': {
                'mean': 0.0,
                'std': 0.0,
                'min': 0.0,
                'max': 0.0
            },
            'inference_time_stats': {
                'mean': 0.0,
                'std': 0.0,
                'min': 0.0,
                'max': 0.0
            }
        }
        
        successful_results = []
        confidences = []
        inference_times = []
        
        for result in results:
            if 'error' in result:
                analysis['failed_predictions'] += 1
            else:
                analysis['successful_predictions'] += 1
                successful_results.append(result)
                confidences.append(result['top_confidence'])
                inference_times.append(result['inference_time'])
                
                # Count class predictions
                top_class = result['top_prediction']
                analysis['class_distribution'][top_class] = \
                    analysis['class_distribution'].get(top_class, 0) + 1
        
        # Calculate statistics
        if confidences:
            analysis['confidence_stats'] = {
                'mean': float(np.mean(confidences)),
                'std': float(np.std(confidences)),
                'min': float(np.min(confidences)),
                'max': float(np.max(confidences))
            }
        
        if inference_times:
            analysis['inference_time_stats'] = {
                'mean': float(np.mean(inference_times)),
                'std': float(np.std(inference_times)),
                'min': float(np.min(inference_times)),
                'max': float(np.max(inference_times))
            }
        
        return analysis


def main():
    """Main prediction function."""
    parser = argparse.ArgumentParser(description="Enhanced photo prediction")
    parser.add_argument("--model", type=Path, required=True,
                       help="Path to model checkpoint")
    parser.add_argument("--config", type=str, default="efficientnet_b1_balanced",
                       help="Configuration name")
    parser.add_argument("--class_names", type=str, nargs="+", required=True,
                       help="List of class names")
    parser.add_argument("--input", type=Path, required=True,
                       help="Input image file or directory")
    parser.add_argument("--output_dir", type=Path, required=True,
                       help="Output directory for results")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size for processing")
    parser.add_argument("--top_k", type=int, default=3,
                       help="Number of top predictions to return")
    parser.add_argument("--visualize", action="store_true",
                       help="Create visualization for predictions")
    parser.add_argument("--confidence_threshold", type=float, default=0.5,
                       help="Confidence threshold for filtering results")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use for prediction")
    
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
    
    # Initialize predictor
    predictor = PhotoPredictor(args.model, args.config, args.class_names, device)
    
    # Get input files
    if args.input.is_file():
        image_paths = [args.input]
    elif args.input.is_dir():
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        image_paths = []
        for ext in image_extensions:
            image_paths.extend(args.input.glob(f'*{ext}'))
            image_paths.extend(args.input.glob(f'*{ext.upper()}'))
    else:
        raise ValueError(f"Input path does not exist: {args.input}")
    
    logger.info(f"Found {len(image_paths)} images to process")
    
    # Make predictions
    if len(image_paths) == 1:
        # Single image prediction
        result = predictor.predict_single(image_paths[0], args.top_k)
        results = [result]
    else:
        # Batch prediction
        results = predictor.predict_batch(image_paths, args.batch_size, args.top_k)
    
    # Filter by confidence threshold
    filtered_results = []
    for result in results:
        if 'error' not in result and result['top_confidence'] >= args.confidence_threshold:
            filtered_results.append(result)
    
    logger.info(f"Predictions complete: {len(filtered_results)}/{len(results)} above threshold")
    
    # Save results
    with open(args.output_dir / "predictions.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    # Create CSV summary
    csv_data = []
    for result in results:
        if 'error' not in result:
            csv_data.append({
                'image_path': result['image_path'],
                'top_prediction': result['top_prediction'],
                'top_confidence': result['top_confidence'],
                'inference_time': result['inference_time']
            })
    
    if csv_data:
        df = pd.DataFrame(csv_data)
        df.to_csv(args.output_dir / "predictions.csv", index=False)
    
    # Generate analysis
    analysis = predictor.analyze_predictions(results)
    with open(args.output_dir / "analysis.json", 'w') as f:
        json.dump(analysis, f, indent=2)
    
    # Create visualizations if requested
    if args.visualize:
        viz_dir = args.output_dir / "visualizations"
        viz_dir.mkdir(exist_ok=True)
        
        # Visualize top predictions
        top_results = sorted(filtered_results, key=lambda x: x['top_confidence'], reverse=True)[:10]
        
        for result in tqdm(top_results, desc="Creating visualizations"):
            image_path = Path(result['image_path'])
            output_path = viz_dir / f"{image_path.stem}_prediction.png"
            predictor.create_prediction_visualization(image_path, result, output_path)
    
    # Print summary
    logger.info("PREDICTION SUMMARY:")
    logger.info(f"  Total images: {analysis['total_predictions']}")
    logger.info(f"  Successful: {analysis['successful_predictions']}")
    logger.info(f"  Failed: {analysis['failed_predictions']}")
    logger.info(f"  Above threshold: {len(filtered_results)}")
    logger.info(f"  Average confidence: {analysis['confidence_stats']['mean']:.3f}")
    logger.info(f"  Average inference time: {analysis['inference_time_stats']['mean']*1000:.1f}ms")
    
    # Print class distribution
    logger.info("CLASS DISTRIBUTION:")
    for class_name, count in analysis['class_distribution'].items():
        logger.info(f"  {class_name}: {count}")
    
    logger.info(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()