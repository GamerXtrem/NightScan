"""
Enhanced Data Preprocessing for Wildlife Photo Classification

Advanced preprocessing script with data validation, augmentation preview,
and comprehensive dataset analysis.
"""

import argparse
import logging
import json
import csv
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import shutil
import multiprocessing as mp
from functools import partial

import numpy as np
import pandas as pd
from PIL import Image, ImageStat
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from tqdm import tqdm
import cv2

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImageValidator:
    """Validate and analyze image quality."""
    
    def __init__(self, min_size: Tuple[int, int] = (32, 32), 
                 max_size: Tuple[int, int] = (4096, 4096)):
        self.min_size = min_size
        self.max_size = max_size
        self.supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    
    def validate_image(self, image_path: Path) -> Dict[str, Any]:
        """Validate a single image and return quality metrics."""
        result = {
            'path': str(image_path),
            'valid': False,
            'error': None,
            'width': 0,
            'height': 0,
            'channels': 0,
            'file_size': 0,
            'format': None,
            'brightness': 0.0,
            'contrast': 0.0,
            'sharpness': 0.0
        }
        
        try:
            # Check file existence and extension
            if not image_path.exists():
                result['error'] = "File not found"
                return result
            
            if image_path.suffix.lower() not in self.supported_formats:
                result['error'] = f"Unsupported format: {image_path.suffix}"
                return result
            
            # Get file size
            result['file_size'] = image_path.stat().st_size
            
            # Open and validate image
            with Image.open(image_path) as img:
                result['width'] = img.width
                result['height'] = img.height
                result['channels'] = len(img.getbands())
                result['format'] = img.format
                
                # Check dimensions
                if (img.width < self.min_size[0] or img.height < self.min_size[1] or
                    img.width > self.max_size[0] or img.height > self.max_size[1]):
                    result['error'] = f"Invalid dimensions: {img.width}x{img.height}"
                    return result
                
                # Convert to RGB for analysis
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Calculate image statistics
                stat = ImageStat.Stat(img)
                result['brightness'] = sum(stat.mean) / 3  # Average brightness
                result['contrast'] = sum(stat.stddev) / 3  # Average standard deviation
                
                # Calculate sharpness (Laplacian variance)
                img_array = np.array(img)
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
                result['sharpness'] = cv2.Laplacian(gray, cv2.CV_64F).var()
                
                result['valid'] = True
                
        except Exception as e:
            result['error'] = str(e)
        
        return result


class DatasetAnalyzer:
    """Analyze dataset characteristics and class distribution."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def analyze_dataset(self, csv_file: Path, validation_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze dataset characteristics."""
        logger.info(f"Analyzing dataset from {csv_file}")
        
        # Read CSV file
        df = pd.read_csv(csv_file)
        
        # Merge with validation results
        validation_df = pd.DataFrame(validation_results)
        df = df.merge(validation_df, left_on='path', right_on='path', how='left')
        
        analysis = {
            'total_samples': len(df),
            'valid_samples': len(df[df['valid'] == True]),
            'invalid_samples': len(df[df['valid'] == False]),
            'class_distribution': {},
            'image_statistics': {},
            'quality_metrics': {}
        }
        
        # Class distribution
        class_counts = df['label'].value_counts()
        analysis['class_distribution'] = class_counts.to_dict()
        
        # Image statistics for valid images
        valid_df = df[df['valid'] == True]
        if len(valid_df) > 0:
            analysis['image_statistics'] = {
                'width': {
                    'mean': valid_df['width'].mean(),
                    'std': valid_df['width'].std(),
                    'min': valid_df['width'].min(),
                    'max': valid_df['width'].max()
                },
                'height': {
                    'mean': valid_df['height'].mean(),
                    'std': valid_df['height'].std(),
                    'min': valid_df['height'].min(),
                    'max': valid_df['height'].max()
                },
                'file_size': {
                    'mean': valid_df['file_size'].mean(),
                    'std': valid_df['file_size'].std(),
                    'min': valid_df['file_size'].min(),
                    'max': valid_df['file_size'].max()
                }
            }
            
            analysis['quality_metrics'] = {
                'brightness': {
                    'mean': valid_df['brightness'].mean(),
                    'std': valid_df['brightness'].std()
                },
                'contrast': {
                    'mean': valid_df['contrast'].mean(),
                    'std': valid_df['contrast'].std()
                },
                'sharpness': {
                    'mean': valid_df['sharpness'].mean(),
                    'std': valid_df['sharpness'].std()
                }
            }
        
        # Save analysis
        with open(self.output_dir / f"{csv_file.stem}_analysis.json", 'w') as f:
            json.dump(analysis, f, indent=2)
        
        # Generate visualizations
        self._plot_class_distribution(class_counts, csv_file.stem)
        self._plot_image_statistics(valid_df, csv_file.stem)
        self._plot_quality_metrics(valid_df, csv_file.stem)
        
        return analysis
    
    def _plot_class_distribution(self, class_counts: pd.Series, prefix: str):
        """Plot class distribution."""
        plt.figure(figsize=(12, 6))
        
        # Bar plot
        plt.subplot(1, 2, 1)
        class_counts.plot(kind='bar')
        plt.title(f'{prefix} - Class Distribution')
        plt.xlabel('Class')
        plt.ylabel('Number of Samples')
        plt.xticks(rotation=45)
        
        # Pie chart
        plt.subplot(1, 2, 2)
        class_counts.plot(kind='pie', autopct='%1.1f%%')
        plt.title(f'{prefix} - Class Distribution')
        plt.ylabel('')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{prefix}_class_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_image_statistics(self, df: pd.DataFrame, prefix: str):
        """Plot image statistics."""
        if len(df) == 0:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Width distribution
        axes[0, 0].hist(df['width'], bins=30, alpha=0.7, color='skyblue')
        axes[0, 0].set_title('Width Distribution')
        axes[0, 0].set_xlabel('Width (pixels)')
        axes[0, 0].set_ylabel('Frequency')
        
        # Height distribution
        axes[0, 1].hist(df['height'], bins=30, alpha=0.7, color='lightcoral')
        axes[0, 1].set_title('Height Distribution')
        axes[0, 1].set_xlabel('Height (pixels)')
        axes[0, 1].set_ylabel('Frequency')
        
        # File size distribution
        axes[1, 0].hist(df['file_size'] / 1024, bins=30, alpha=0.7, color='lightgreen')
        axes[1, 0].set_title('File Size Distribution')
        axes[1, 0].set_xlabel('File Size (KB)')
        axes[1, 0].set_ylabel('Frequency')
        
        # Aspect ratio distribution
        aspect_ratios = df['width'] / df['height']
        axes[1, 1].hist(aspect_ratios, bins=30, alpha=0.7, color='gold')
        axes[1, 1].set_title('Aspect Ratio Distribution')
        axes[1, 1].set_xlabel('Aspect Ratio (W/H)')
        axes[1, 1].set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{prefix}_image_statistics.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_quality_metrics(self, df: pd.DataFrame, prefix: str):
        """Plot quality metrics."""
        if len(df) == 0:
            return
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Brightness distribution
        axes[0].hist(df['brightness'], bins=30, alpha=0.7, color='yellow')
        axes[0].set_title('Brightness Distribution')
        axes[0].set_xlabel('Brightness')
        axes[0].set_ylabel('Frequency')
        
        # Contrast distribution
        axes[1].hist(df['contrast'], bins=30, alpha=0.7, color='purple')
        axes[1].set_title('Contrast Distribution')
        axes[1].set_xlabel('Contrast')
        axes[1].set_ylabel('Frequency')
        
        # Sharpness distribution
        axes[2].hist(df['sharpness'], bins=30, alpha=0.7, color='orange')
        axes[2].set_title('Sharpness Distribution')
        axes[2].set_xlabel('Sharpness')
        axes[2].set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{prefix}_quality_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()


def validate_image_parallel(image_path: Path, validator: ImageValidator) -> Dict[str, Any]:
    """Wrapper for parallel image validation."""
    return validator.validate_image(image_path)


def create_train_val_split(input_csv: Path, output_dir: Path, 
                          val_size: float = 0.2, test_size: float = 0.1,
                          random_state: int = 42) -> Tuple[Path, Path, Path]:
    """Create train/validation/test splits."""
    logger.info(f"Creating train/val/test splits from {input_csv}")
    
    # Read CSV
    df = pd.read_csv(input_csv)
    
    # Stratified split
    X = df['path'].values
    y = df['label'].values
    
    # First split: train+val vs test
    if test_size > 0:
        X_trainval, X_test, y_trainval, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=random_state
        )
        
        # Second split: train vs val
        X_train, X_val, y_train, y_val = train_test_split(
            X_trainval, y_trainval, test_size=val_size/(1-test_size), 
            stratify=y_trainval, random_state=random_state
        )
    else:
        # Only train/val split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=val_size, stratify=y, random_state=random_state
        )
        X_test = y_test = None
    
    # Save splits
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Train set
    train_df = pd.DataFrame({'path': X_train, 'label': y_train})
    train_path = output_dir / 'train.csv'
    train_df.to_csv(train_path, index=False)
    
    # Validation set
    val_df = pd.DataFrame({'path': X_val, 'label': y_val})
    val_path = output_dir / 'val.csv'
    val_df.to_csv(val_path, index=False)
    
    # Test set (if created)
    test_path = None
    if X_test is not None:
        test_df = pd.DataFrame({'path': X_test, 'label': y_test})
        test_path = output_dir / 'test.csv'
        test_df.to_csv(test_path, index=False)
    
    logger.info(f"Train samples: {len(train_df)}")
    logger.info(f"Validation samples: {len(val_df)}")
    if test_path:
        logger.info(f"Test samples: {len(test_df)}")
    
    return train_path, val_path, test_path


def create_cross_validation_splits(input_csv: Path, output_dir: Path,
                                 n_splits: int = 5, random_state: int = 42):
    """Create k-fold cross-validation splits."""
    logger.info(f"Creating {n_splits}-fold CV splits from {input_csv}")
    
    # Read CSV
    df = pd.read_csv(input_csv)
    
    # Stratified k-fold
    X = df['path'].values
    y = df['label'].values
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    cv_dir = output_dir / 'cross_validation'
    cv_dir.mkdir(parents=True, exist_ok=True)
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        fold_dir = cv_dir / f'fold_{fold}'
        fold_dir.mkdir(parents=True, exist_ok=True)
        
        # Train set
        train_df = pd.DataFrame({
            'path': X[train_idx],
            'label': y[train_idx]
        })
        train_df.to_csv(fold_dir / 'train.csv', index=False)
        
        # Validation set
        val_df = pd.DataFrame({
            'path': X[val_idx],
            'label': y[val_idx]
        })
        val_df.to_csv(fold_dir / 'val.csv', index=False)
        
        logger.info(f"Fold {fold}: Train={len(train_df)}, Val={len(val_df)}")


def main():
    """Main preprocessing function."""
    parser = argparse.ArgumentParser(description="Enhanced data preprocessing for wildlife photos")
    parser.add_argument("--input_csv", type=Path, required=True,
                       help="Input CSV file with image paths and labels")
    parser.add_argument("--output_dir", type=Path, required=True,
                       help="Output directory for processed data")
    parser.add_argument("--validate_images", action="store_true",
                       help="Validate all images in the dataset")
    parser.add_argument("--create_splits", action="store_true",
                       help="Create train/val/test splits")
    parser.add_argument("--create_cv", action="store_true",
                       help="Create cross-validation splits")
    parser.add_argument("--val_size", type=float, default=0.2,
                       help="Validation set size (default: 0.2)")
    parser.add_argument("--test_size", type=float, default=0.1,
                       help="Test set size (default: 0.1)")
    parser.add_argument("--cv_folds", type=int, default=5,
                       help="Number of CV folds (default: 5)")
    parser.add_argument("--num_workers", type=int, default=4,
                       help="Number of worker processes for validation")
    parser.add_argument("--min_size", type=int, nargs=2, default=[32, 32],
                       help="Minimum image size (width height)")
    parser.add_argument("--max_size", type=int, nargs=2, default=[4096, 4096],
                       help="Maximum image size (width height)")
    
    args = parser.parse_args()
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize components
    validator = ImageValidator(
        min_size=tuple(args.min_size),
        max_size=tuple(args.max_size)
    )
    analyzer = DatasetAnalyzer(args.output_dir / 'analysis')
    
    # Validate images if requested
    validation_results = []
    if args.validate_images:
        logger.info("Validating images...")
        
        # Read image paths from CSV
        df = pd.read_csv(args.input_csv)
        image_paths = [Path(p) for p in df['path']]
        
        # Parallel validation
        with mp.Pool(args.num_workers) as pool:
            validate_func = partial(validate_image_parallel, validator=validator)
            validation_results = list(tqdm(
                pool.imap(validate_func, image_paths),
                total=len(image_paths),
                desc="Validating images"
            ))
        
        # Save validation results
        validation_df = pd.DataFrame(validation_results)
        validation_df.to_csv(args.output_dir / 'validation_results.csv', index=False)
        
        # Log validation summary
        valid_count = sum(1 for r in validation_results if r['valid'])
        invalid_count = len(validation_results) - valid_count
        logger.info(f"Validation complete: {valid_count} valid, {invalid_count} invalid")
        
        # Create clean dataset (valid images only)
        clean_df = df.merge(validation_df[validation_df['valid']], 
                           left_on='path', right_on='path', how='inner')
        clean_df = clean_df[['path', 'label']]  # Keep only original columns
        clean_df.to_csv(args.output_dir / 'clean_dataset.csv', index=False)
        logger.info(f"Clean dataset saved with {len(clean_df)} valid images")
    
    # Analyze dataset
    analysis = analyzer.analyze_dataset(args.input_csv, validation_results)
    logger.info(f"Dataset analysis complete. Total samples: {analysis['total_samples']}")
    
    # Create train/val/test splits
    if args.create_splits:
        input_csv = args.output_dir / 'clean_dataset.csv' if args.validate_images else args.input_csv
        train_path, val_path, test_path = create_train_val_split(
            input_csv, args.output_dir / 'splits',
            val_size=args.val_size, test_size=args.test_size
        )
        logger.info("Train/val/test splits created")
    
    # Create cross-validation splits
    if args.create_cv:
        input_csv = args.output_dir / 'clean_dataset.csv' if args.validate_images else args.input_csv
        create_cross_validation_splits(
            input_csv, args.output_dir,
            n_splits=args.cv_folds
        )
        logger.info(f"{args.cv_folds}-fold CV splits created")
    
    logger.info(f"Preprocessing complete. Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()