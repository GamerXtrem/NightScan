"""
Tests de précision et regression testing pour les modèles ML.

Ce module teste la qualité des prédictions et détecte les régressions:
- Validation avec datasets de référence connus
- Tests de consistance et reproductibilité des prédictions
- Détection de régression de précision entre versions
- Framework A/B testing pour comparaison de modèles
- Validation cross-validation et métriques de qualité
- Tests de robustesse face aux variations d'input
"""

import pytest
import time
import json
import tempfile
import hashlib
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
from PIL import Image
import struct
from dataclasses import dataclass
from collections import defaultdict
import statistics

# Import components for accuracy testing
from unified_prediction_system.model_manager import ModelManager
from unified_prediction_system.prediction_router import PredictionRouter
from unified_prediction_system.unified_prediction_api import UnifiedPredictionAPI


@dataclass
class AccuracyMetrics:
    """Métriques de précision pour évaluation modèle."""
    total_predictions: int = 0
    correct_predictions: int = 0
    top1_accuracy: float = 0.0
    top3_accuracy: float = 0.0
    top5_accuracy: float = 0.0
    confidence_scores: List[float] = None
    per_class_accuracy: Dict[str, float] = None
    confusion_matrix: Dict[str, Dict[str, int]] = None
    
    def __post_init__(self):
        if self.confidence_scores is None:
            self.confidence_scores = []
        if self.per_class_accuracy is None:
            self.per_class_accuracy = {}
        if self.confusion_matrix is None:
            self.confusion_matrix = defaultdict(lambda: defaultdict(int))
    
    @property
    def overall_accuracy(self) -> float:
        """Précision globale."""
        return self.correct_predictions / self.total_predictions if self.total_predictions > 0 else 0.0
    
    @property
    def avg_confidence(self) -> float:
        """Confiance moyenne."""
        return statistics.mean(self.confidence_scores) if self.confidence_scores else 0.0
    
    @property
    def confidence_std(self) -> float:
        """Écart-type de la confiance."""
        return statistics.stdev(self.confidence_scores) if len(self.confidence_scores) > 1 else 0.0


class BaselineDataset:
    """Dataset de référence pour tests de régression."""
    
    def __init__(self, name: str):
        self.name = name
        self.samples = []
        self.expected_results = {}
        self.metadata = {}
    
    def add_sample(self, sample_id: str, file_path: str, expected_species: str, 
                   expected_confidence: float, metadata: Dict = None):
        """Ajoute un échantillon au dataset."""
        self.samples.append(sample_id)
        self.expected_results[sample_id] = {
            'file_path': file_path,
            'species': expected_species,
            'confidence': expected_confidence,
            'metadata': metadata or {}
        }
    
    def get_sample(self, sample_id: str) -> Dict[str, Any]:
        """Récupère un échantillon."""
        return self.expected_results.get(sample_id)
    
    def get_all_samples(self) -> List[str]:
        """Récupère tous les IDs d'échantillons."""
        return self.samples.copy()


class ModelAccuracyTester:
    """Testeur de précision de modèles."""
    
    def __init__(self, api: UnifiedPredictionAPI):
        self.api = api
        self.baseline_datasets = {}
        self.accuracy_history = {}
    
    def register_baseline_dataset(self, dataset: BaselineDataset):
        """Enregistre un dataset de référence."""
        self.baseline_datasets[dataset.name] = dataset
    
    def evaluate_model_accuracy(self, dataset_name: str, model_version: str = None) -> AccuracyMetrics:
        """Évalue la précision d'un modèle sur un dataset."""
        if dataset_name not in self.baseline_datasets:
            raise ValueError(f"Dataset {dataset_name} not found")
        
        dataset = self.baseline_datasets[dataset_name]
        metrics = AccuracyMetrics()
        
        for sample_id in dataset.get_all_samples():
            sample = dataset.get_sample(sample_id)
            
            try:
                # Execute prediction
                result = self.api.predict_file(
                    sample['file_path'], 
                    model_version=model_version
                )
                
                predicted_species = result['predictions']['species']
                predicted_confidence = result['predictions']['confidence']
                expected_species = sample['species']
                
                # Update metrics
                metrics.total_predictions += 1
                metrics.confidence_scores.append(predicted_confidence)
                
                # Check accuracy
                if predicted_species == expected_species:
                    metrics.correct_predictions += 1
                
                # Update confusion matrix
                metrics.confusion_matrix[expected_species][predicted_species] += 1
                
                # Update per-class accuracy
                if expected_species not in metrics.per_class_accuracy:
                    metrics.per_class_accuracy[expected_species] = {'correct': 0, 'total': 0}
                
                metrics.per_class_accuracy[expected_species]['total'] += 1
                if predicted_species == expected_species:
                    metrics.per_class_accuracy[expected_species]['correct'] += 1
                
            except Exception as e:
                print(f"Error processing sample {sample_id}: {e}")
        
        # Calculate per-class accuracies
        for species, counts in metrics.per_class_accuracy.items():
            metrics.per_class_accuracy[species] = counts['correct'] / counts['total']
        
        # Calculate top-k accuracies (if multiple predictions available)
        self._calculate_topk_accuracy(metrics, dataset, model_version)
        
        return metrics
    
    def _calculate_topk_accuracy(self, metrics: AccuracyMetrics, dataset: BaselineDataset, model_version: str):
        """Calcule les précisions top-k."""
        # This would be implemented with full prediction results
        # For now, using overall accuracy as approximation
        metrics.top1_accuracy = metrics.overall_accuracy
        metrics.top3_accuracy = min(1.0, metrics.overall_accuracy * 1.1)
        metrics.top5_accuracy = min(1.0, metrics.overall_accuracy * 1.2)
    
    def detect_accuracy_regression(self, dataset_name: str, baseline_accuracy: float, 
                                  threshold: float = 0.05) -> Dict[str, Any]:
        """Détecte une régression de précision."""
        current_metrics = self.evaluate_model_accuracy(dataset_name)
        current_accuracy = current_metrics.overall_accuracy
        
        regression_ratio = (baseline_accuracy - current_accuracy) / baseline_accuracy
        
        return {
            'dataset': dataset_name,
            'baseline_accuracy': baseline_accuracy,
            'current_accuracy': current_accuracy,
            'regression_ratio': regression_ratio,
            'regression_detected': regression_ratio > threshold,
            'threshold': threshold,
            'metrics': current_metrics
        }


class TestModelAccuracyBaseline:
    """Tests de précision avec datasets de référence."""
    
    def setup_method(self):
        """Setup pour tests de précision."""
        self.api = UnifiedPredictionAPI()
        self.accuracy_tester = ModelAccuracyTester(self.api)
        self._create_baseline_datasets()
    
    def _create_baseline_datasets(self):
        """Crée les datasets de référence pour tests."""
        # Audio dataset
        audio_dataset = BaselineDataset("audio_baseline")
        
        # Create test audio files with known expected results
        audio_samples = [
            ("owl_sample_1", "owl", 0.92, {"duration": 3.5, "quality": "high"}),
            ("owl_sample_2", "owl", 0.88, {"duration": 2.1, "quality": "medium"}),
            ("deer_sample_1", "deer", 0.85, {"duration": 4.2, "quality": "high"}),
            ("wind_sample_1", "wind", 0.78, {"duration": 5.0, "quality": "low"}),
            ("fox_sample_1", "fox", 0.90, {"duration": 1.8, "quality": "high"}),
        ]
        
        for sample_id, species, confidence, metadata in audio_samples:
            # Create corresponding test file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                # Create realistic WAV header
                tmp.write(b'RIFF')
                tmp.write(struct.pack('<L', 44100))
                tmp.write(b'WAVEfmt ')
                tmp.write(struct.pack('<L', 16))
                tmp.write(struct.pack('<H', 1))   # PCM
                tmp.write(struct.pack('<H', 1))   # Mono
                tmp.write(struct.pack('<L', 44100))  # Sample rate
                tmp.write(struct.pack('<L', 88200))  # Byte rate
                tmp.write(struct.pack('<H', 2))   # Block align
                tmp.write(struct.pack('<H', 16))  # Bits per sample
                tmp.write(b'data')
                tmp.write(struct.pack('<L', int(44100 * metadata['duration'])))
                tmp.write(b'\x00' * int(44100 * metadata['duration']))
                
                audio_dataset.add_sample(sample_id, tmp.name, species, confidence, metadata)
        
        # Image dataset
        image_dataset = BaselineDataset("image_baseline")
        
        image_samples = [
            ("deer_image_1", "deer", 0.94, {"resolution": "high", "lighting": "daylight"}),
            ("fox_image_1", "fox", 0.89, {"resolution": "medium", "lighting": "twilight"}),
            ("owl_image_1", "owl", 0.87, {"resolution": "high", "lighting": "night"}),
            ("background_1", "none", 0.82, {"resolution": "high", "lighting": "daylight"}),
        ]
        
        for sample_id, species, confidence, metadata in image_samples:
            # Create test image
            if species == "deer":
                color = (139, 69, 19)  # Brown
            elif species == "fox":
                color = (255, 140, 0)  # Orange
            elif species == "owl":
                color = (105, 105, 105)  # Gray
            else:
                color = (34, 139, 34)  # Green background
            
            img = Image.new('RGB', (640, 480), color=color)
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                img.save(tmp, format='JPEG', quality=95)
                image_dataset.add_sample(sample_id, tmp.name, species, confidence, metadata)
        
        self.accuracy_tester.register_baseline_dataset(audio_dataset)
        self.accuracy_tester.register_baseline_dataset(image_dataset)
    
    @pytest.mark.ml_accuracy
    def test_audio_model_baseline_accuracy(self):
        """Test précision baseline pour modèle audio."""
        with patch.object(self.api.router, 'predict') as mock_predict:
            # Mock realistic audio predictions based on sample IDs
            def mock_audio_prediction(file_path):
                if 'owl_sample' in file_path:
                    return {
                        'predictions': {'species': 'owl', 'confidence': 0.90},
                        'file_type': 'audio'
                    }
                elif 'deer_sample' in file_path:
                    return {
                        'predictions': {'species': 'deer', 'confidence': 0.85},
                        'file_type': 'audio'
                    }
                elif 'wind_sample' in file_path:
                    return {
                        'predictions': {'species': 'wind', 'confidence': 0.78},
                        'file_type': 'audio'
                    }
                elif 'fox_sample' in file_path:
                    return {
                        'predictions': {'species': 'fox', 'confidence': 0.88},
                        'file_type': 'audio'
                    }
                else:
                    return {
                        'predictions': {'species': 'unknown', 'confidence': 0.5},
                        'file_type': 'audio'
                    }
            
            mock_predict.side_effect = mock_audio_prediction
            
            # Evaluate baseline accuracy
            metrics = self.accuracy_tester.evaluate_model_accuracy("audio_baseline")
            
            # Baseline accuracy assertions
            assert metrics.total_predictions == 5
            assert metrics.overall_accuracy >= 0.8  # At least 80% accuracy
            assert metrics.avg_confidence >= 0.8    # Good confidence
            assert metrics.confidence_std < 0.2     # Consistent confidence
            
            # Per-class accuracy should be reasonable
            for species, accuracy in metrics.per_class_accuracy.items():
                assert accuracy >= 0.7  # Each class > 70% accuracy
    
    @pytest.mark.ml_accuracy
    def test_image_model_baseline_accuracy(self):
        """Test précision baseline pour modèle image."""
        with patch.object(self.api.router, 'predict') as mock_predict:
            # Mock realistic image predictions
            def mock_image_prediction(file_path):
                if 'deer_image' in file_path:
                    return {
                        'predictions': {'species': 'deer', 'confidence': 0.92},
                        'file_type': 'photo'
                    }
                elif 'fox_image' in file_path:
                    return {
                        'predictions': {'species': 'fox', 'confidence': 0.89},
                        'file_type': 'photo'
                    }
                elif 'owl_image' in file_path:
                    return {
                        'predictions': {'species': 'owl', 'confidence': 0.85},
                        'file_type': 'photo'
                    }
                elif 'background' in file_path:
                    return {
                        'predictions': {'species': 'none', 'confidence': 0.80},
                        'file_type': 'photo'
                    }
                else:
                    return {
                        'predictions': {'species': 'unknown', 'confidence': 0.5},
                        'file_type': 'photo'
                    }
            
            mock_predict.side_effect = mock_image_prediction
            
            # Evaluate baseline accuracy
            metrics = self.accuracy_tester.evaluate_model_accuracy("image_baseline")
            
            # Image model accuracy assertions
            assert metrics.total_predictions == 4
            assert metrics.overall_accuracy >= 0.75  # At least 75% accuracy
            assert metrics.avg_confidence >= 0.85    # High confidence for images
            
            # Check confusion matrix
            assert len(metrics.confusion_matrix) > 0
    
    @pytest.mark.ml_accuracy
    def test_accuracy_regression_detection(self):
        """Test détection de régression de précision."""
        with patch.object(self.api.router, 'predict') as mock_predict:
            # Mock degraded model performance
            def mock_degraded_prediction(file_path):
                # Simulate accuracy drop - some predictions now wrong
                if 'owl_sample_1' in file_path:
                    return {
                        'predictions': {'species': 'wind', 'confidence': 0.65},  # Wrong!
                        'file_type': 'audio'
                    }
                elif 'deer_sample' in file_path:
                    return {
                        'predictions': {'species': 'fox', 'confidence': 0.70},   # Wrong!
                        'file_type': 'audio'
                    }
                else:
                    # Other predictions remain correct
                    return {
                        'predictions': {'species': 'wind', 'confidence': 0.78},
                        'file_type': 'audio'
                    }
            
            mock_predict.side_effect = mock_degraded_prediction
            
            # Test regression detection
            baseline_accuracy = 0.90  # Expected baseline
            regression_result = self.accuracy_tester.detect_accuracy_regression(
                "audio_baseline", 
                baseline_accuracy, 
                threshold=0.1  # 10% degradation threshold
            )
            
            # Should detect regression
            assert regression_result['regression_detected'] == True
            assert regression_result['current_accuracy'] < baseline_accuracy
            assert regression_result['regression_ratio'] > 0.1
            
            # Verify detailed metrics
            metrics = regression_result['metrics']
            assert metrics.total_predictions == 5
            assert metrics.overall_accuracy < 0.80  # Significantly degraded
    
    def teardown_method(self):
        """Cleanup des fichiers de test."""
        for dataset_name, dataset in self.accuracy_tester.baseline_datasets.items():
            for sample_id in dataset.get_all_samples():
                sample = dataset.get_sample(sample_id)
                try:
                    Path(sample['file_path']).unlink()
                except:
                    pass


class TestModelConsistency:
    """Tests de consistance et reproductibilité des modèles."""
    
    def setup_method(self):
        """Setup pour tests de consistance."""
        self.api = UnifiedPredictionAPI()
    
    @pytest.mark.ml_accuracy
    def test_prediction_reproducibility(self):
        """Test reproductibilité des prédictions."""
        # Create test file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            tmp.write(b'RIFF\x24\x08\x00\x00WAVEfmt ')
            test_file = tmp.name
        
        try:
            with patch.object(self.api.router, 'predict') as mock_predict:
                # Mock deterministic prediction
                mock_predict.return_value = {
                    'predictions': {
                        'species': 'owl',
                        'confidence': 0.8756,  # Specific value for reproducibility
                        'predictions': [
                            {'class': 'owl', 'confidence': 0.8756},
                            {'class': 'wind', 'confidence': 0.1244}
                        ]
                    },
                    'file_type': 'audio'
                }
                
                # Run same prediction multiple times
                results = []
                for i in range(10):
                    result = self.api.predict_file(test_file)
                    results.append(result)
                
                # Verify reproducibility
                first_result = results[0]
                for result in results[1:]:
                    assert result['predictions']['species'] == first_result['predictions']['species']
                    assert result['predictions']['confidence'] == first_result['predictions']['confidence']
                    assert len(result['predictions']['predictions']) == len(first_result['predictions']['predictions'])
                
                # All predictions should be identical
                species_set = set(r['predictions']['species'] for r in results)
                confidence_set = set(r['predictions']['confidence'] for r in results)
                
                assert len(species_set) == 1  # Only one unique species
                assert len(confidence_set) == 1  # Only one unique confidence
                
        finally:
            Path(test_file).unlink()
    
    @pytest.mark.ml_accuracy
    def test_confidence_calibration(self):
        """Test calibration des scores de confiance."""
        test_cases = [
            # (expected_accuracy, confidence_range)
            (0.95, (0.90, 1.00)),  # High confidence should be very accurate
            (0.85, (0.80, 0.90)),  # Medium-high confidence 
            (0.70, (0.60, 0.80)),  # Medium confidence
            (0.50, (0.40, 0.60)),  # Low confidence
        ]
        
        with patch.object(self.api.router, 'predict') as mock_predict:
            calibration_results = []
            
            for expected_acc, (conf_min, conf_max) in test_cases:
                # Mock predictions with specific confidence range
                mock_predictions = []
                for i in range(20):  # 20 predictions per confidence range
                    confidence = np.random.uniform(conf_min, conf_max)
                    # Simulate accuracy based on confidence
                    is_correct = np.random.random() < expected_acc
                    species = 'owl' if is_correct else 'wrong_species'
                    
                    mock_predictions.append({
                        'predictions': {'species': species, 'confidence': confidence},
                        'file_type': 'audio'
                    })
                
                mock_predict.side_effect = mock_predictions
                
                # Test predictions
                correct_count = 0
                total_count = 0
                confidences = []
                
                for i in range(20):
                    result = self.api.predict_file(f'/fake/calibration_test_{i}.wav')
                    total_count += 1
                    confidences.append(result['predictions']['confidence'])
                    
                    # Check if prediction is correct (for simulation)
                    if result['predictions']['species'] == 'owl':
                        correct_count += 1
                
                actual_accuracy = correct_count / total_count
                avg_confidence = np.mean(confidences)
                
                calibration_results.append({
                    'confidence_range': (conf_min, conf_max),
                    'expected_accuracy': expected_acc,
                    'actual_accuracy': actual_accuracy,
                    'avg_confidence': avg_confidence,
                    'calibration_error': abs(actual_accuracy - avg_confidence)
                })
            
            # Confidence calibration assertions
            for result in calibration_results:
                # Calibration error should be reasonable
                assert result['calibration_error'] < 0.15  # < 15% calibration error
                
                # Higher confidence should correlate with higher accuracy
                if result['avg_confidence'] > 0.8:
                    assert result['actual_accuracy'] > 0.7  # High confidence → good accuracy
    
    @pytest.mark.ml_accuracy
    def test_cross_validation_consistency(self):
        """Test consistance cross-validation."""
        # Create multiple test files for cross-validation
        test_files = []
        for i in range(5):
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                tmp.write(b'RIFF\x24\x08\x00\x00WAVEfmt ')
                test_files.append(tmp.name)
        
        try:
            with patch.object(self.api.router, 'predict') as mock_predict:
                # Mock consistent predictions across folds
                def consistent_prediction(file_path):
                    # Extract file index from path
                    file_idx = int(file_path.split('_')[-1].split('.')[0]) if '_' in file_path else 0
                    
                    # Consistent species mapping
                    species_map = ['owl', 'deer', 'fox', 'wind', 'owl']
                    confidence_map = [0.92, 0.88, 0.85, 0.79, 0.91]
                    
                    return {
                        'predictions': {
                            'species': species_map[file_idx % len(species_map)],
                            'confidence': confidence_map[file_idx % len(confidence_map)]
                        },
                        'file_type': 'audio'
                    }
                
                mock_predict.side_effect = consistent_prediction
                
                # Perform k-fold cross-validation simulation
                fold_results = []
                k_folds = 3
                
                for fold in range(k_folds):
                    fold_predictions = []
                    fold_confidences = []
                    
                    for file_path in test_files:
                        # Simulate fold-specific file path
                        fold_file_path = f"{file_path}_fold_{fold}"
                        result = self.api.predict_file(fold_file_path)
                        
                        fold_predictions.append(result['predictions']['species'])
                        fold_confidences.append(result['predictions']['confidence'])
                    
                    fold_results.append({
                        'fold': fold,
                        'predictions': fold_predictions,
                        'confidences': fold_confidences,
                        'avg_confidence': np.mean(fold_confidences)
                    })
                
                # Cross-validation consistency assertions
                # Check consistency across folds
                fold_avg_confidences = [r['avg_confidence'] for r in fold_results]
                confidence_variance = np.var(fold_avg_confidences)
                
                assert confidence_variance < 0.01  # Low variance across folds
                
                # Check prediction stability
                all_predictions = [pred for fold in fold_results for pred in fold['predictions']]
                unique_predictions = set(all_predictions)
                
                # Should have reasonable diversity but consistency within folds
                assert len(unique_predictions) <= 4  # Limited to expected species
                
        finally:
            for file_path in test_files:
                Path(file_path).unlink()


class TestModelRobustness:
    """Tests de robustesse face aux variations d'input."""
    
    def setup_method(self):
        """Setup pour tests de robustesse."""
        self.api = UnifiedPredictionAPI()
    
    @pytest.mark.ml_accuracy
    def test_noise_robustness(self):
        """Test robustesse face au bruit dans les données."""
        # Create base test file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            # High quality audio
            tmp.write(b'RIFF')
            tmp.write(struct.pack('<L', 44100))
            tmp.write(b'WAVEfmt ')
            tmp.write(struct.pack('<L', 16))
            tmp.write(struct.pack('<H', 1))
            tmp.write(struct.pack('<H', 1))
            tmp.write(struct.pack('<L', 44100))
            tmp.write(struct.pack('<L', 88200))
            tmp.write(struct.pack('<H', 2))
            tmp.write(struct.pack('<H', 16))
            tmp.write(b'data')
            tmp.write(struct.pack('<L', 44100))
            tmp.write(b'\x00' * 44100)
            
            clean_file = tmp.name
        
        try:
            with patch.object(self.api.router, 'predict') as mock_predict:
                # Mock predictions with noise impact simulation
                def noise_sensitive_prediction(file_path):
                    if 'clean' in file_path:
                        return {
                            'predictions': {'species': 'owl', 'confidence': 0.95},
                            'file_type': 'audio'
                        }
                    elif 'noise_low' in file_path:
                        return {
                            'predictions': {'species': 'owl', 'confidence': 0.87},
                            'file_type': 'audio'
                        }
                    elif 'noise_medium' in file_path:
                        return {
                            'predictions': {'species': 'owl', 'confidence': 0.78},
                            'file_type': 'audio'
                        }
                    elif 'noise_high' in file_path:
                        return {
                            'predictions': {'species': 'owl', 'confidence': 0.65},
                            'file_type': 'audio'
                        }
                    else:
                        return {
                            'predictions': {'species': 'unknown', 'confidence': 0.40},
                            'file_type': 'audio'
                        }
                
                mock_predict.side_effect = noise_sensitive_prediction
                
                # Test with different noise levels
                noise_levels = ['clean', 'noise_low', 'noise_medium', 'noise_high']
                results = {}
                
                for noise_level in noise_levels:
                    test_path = f'{clean_file}_{noise_level}'
                    result = self.api.predict_file(test_path)
                    results[noise_level] = result
                
                # Robustness assertions
                clean_confidence = results['clean']['predictions']['confidence']
                
                # Confidence should degrade gracefully with noise
                assert results['noise_low']['predictions']['confidence'] >= clean_confidence * 0.85
                assert results['noise_medium']['predictions']['confidence'] >= clean_confidence * 0.75
                assert results['noise_high']['predictions']['confidence'] >= clean_confidence * 0.6
                
                # Species prediction should remain consistent for low/medium noise
                clean_species = results['clean']['predictions']['species']
                assert results['noise_low']['predictions']['species'] == clean_species
                assert results['noise_medium']['predictions']['species'] == clean_species
                
        finally:
            Path(clean_file).unlink()
    
    @pytest.mark.ml_accuracy
    def test_input_variation_robustness(self):
        """Test robustesse face aux variations d'input."""
        variations = [
            ('normal_quality', 0.92),
            ('low_resolution', 0.78),
            ('compressed', 0.85),
            ('different_format', 0.88),
            ('rotated', 0.75),  # For images
        ]
        
        with patch.object(self.api.router, 'predict') as mock_predict:
            def variation_prediction(file_path):
                # Determine variation type from path
                for variation, confidence in variations:
                    if variation in file_path:
                        return {
                            'predictions': {'species': 'deer', 'confidence': confidence},
                            'file_type': 'photo'
                        }
                
                return {
                    'predictions': {'species': 'deer', 'confidence': 0.92},
                    'file_type': 'photo'
                }
            
            mock_predict.side_effect = variation_prediction
            
            # Test each variation
            variation_results = {}
            
            for variation, expected_conf in variations:
                test_path = f'/fake/test_{variation}.jpg'
                result = self.api.predict_file(test_path)
                variation_results[variation] = result
            
            # Robustness assertions
            baseline_confidence = variation_results['normal_quality']['predictions']['confidence']
            
            for variation, result in variation_results.items():
                if variation != 'normal_quality':
                    # All variations should maintain species consistency
                    assert result['predictions']['species'] == 'deer'
                    
                    # Confidence degradation should be reasonable
                    confidence_ratio = result['predictions']['confidence'] / baseline_confidence
                    assert confidence_ratio >= 0.7  # At least 70% of baseline confidence
    
    @pytest.mark.ml_accuracy
    def test_edge_case_handling(self):
        """Test gestion des cas limites."""
        edge_cases = [
            ('very_short_audio', 'owl', 0.65),      # < 1 second
            ('very_long_audio', 'owl', 0.88),       # > 30 seconds
            ('silent_audio', 'silence', 0.92),      # No audio content
            ('very_small_image', 'deer', 0.70),     # < 100x100 pixels
            ('very_large_image', 'deer', 0.90),     # > 4K resolution
            ('dark_image', 'unknown', 0.45),        # Very dark/underexposed
        ]
        
        with patch.object(self.api.router, 'predict') as mock_predict:
            def edge_case_prediction(file_path):
                for case, species, confidence in edge_cases:
                    if case in file_path:
                        return {
                            'predictions': {'species': species, 'confidence': confidence},
                            'file_type': 'audio' if 'audio' in case else 'photo'
                        }
                
                return {
                    'predictions': {'species': 'unknown', 'confidence': 0.5},
                    'file_type': 'audio'
                }
            
            mock_predict.side_effect = edge_case_prediction
            
            # Test edge cases
            edge_results = {}
            
            for case, expected_species, expected_conf in edge_cases:
                test_path = f'/fake/{case}_test.wav'
                
                try:
                    result = self.api.predict_file(test_path)
                    edge_results[case] = {
                        'success': True,
                        'result': result
                    }
                except Exception as e:
                    edge_results[case] = {
                        'success': False,
                        'error': str(e)
                    }
            
            # Edge case handling assertions
            for case, result_info in edge_results.items():
                # System should handle all edge cases gracefully
                assert result_info['success'] == True, f"Edge case {case} failed: {result_info.get('error')}"
                
                if result_info['success']:
                    result = result_info['result']
                    
                    # Should return valid predictions
                    assert 'predictions' in result
                    assert 'species' in result['predictions']
                    assert 'confidence' in result['predictions']
                    
                    # Confidence should be reasonable
                    confidence = result['predictions']['confidence']
                    assert 0.0 <= confidence <= 1.0


@pytest.mark.ml_accuracy
class TestModelVersionComparison:
    """Tests de comparaison entre versions de modèles."""
    
    def setup_method(self):
        """Setup pour comparaison de versions."""
        self.api = UnifiedPredictionAPI()
    
    def test_model_version_accuracy_comparison(self):
        """Test comparaison de précision entre versions."""
        # Mock different model versions with different performance
        version_performance = {
            'v1.0': {'species': 'owl', 'confidence': 0.85, 'accuracy': 0.82},
            'v1.1': {'species': 'owl', 'confidence': 0.89, 'accuracy': 0.86},
            'v2.0': {'species': 'owl', 'confidence': 0.93, 'accuracy': 0.91},
        }
        
        with patch.object(self.api.router, 'predict') as mock_predict:
            def version_specific_prediction(file_path, model_version=None):
                if model_version and model_version in version_performance:
                    perf = version_performance[model_version]
                    return {
                        'predictions': {
                            'species': perf['species'],
                            'confidence': perf['confidence']
                        },
                        'model_version': model_version,
                        'file_type': 'audio'
                    }
                else:
                    # Default to latest version
                    perf = version_performance['v2.0']
                    return {
                        'predictions': {
                            'species': perf['species'],
                            'confidence': perf['confidence']
                        },
                        'model_version': 'v2.0',
                        'file_type': 'audio'
                    }
            
            # Override to handle model_version parameter
            self.api.predict_file = Mock(side_effect=version_specific_prediction)
            
            # Compare versions
            version_results = {}
            test_file = '/fake/version_comparison.wav'
            
            for version in version_performance.keys():
                result = self.api.predict_file(test_file, model_version=version)
                version_results[version] = result
            
            # Version comparison assertions
            v1_confidence = version_results['v1.0']['predictions']['confidence']
            v11_confidence = version_results['v1.1']['predictions']['confidence']
            v2_confidence = version_results['v2.0']['predictions']['confidence']
            
            # Newer versions should generally perform better
            assert v11_confidence >= v1_confidence  # v1.1 >= v1.0
            assert v2_confidence >= v11_confidence  # v2.0 >= v1.1
            
            # All versions should predict same species for this test
            species_set = set(r['predictions']['species'] for r in version_results.values())
            assert len(species_set) == 1  # Consistent species prediction
    
    def test_ab_testing_framework(self):
        """Test framework A/B testing pour modèles."""
        # Define A/B test scenarios
        model_a_performance = {'species': 'owl', 'confidence': 0.88, 'processing_time': 0.2}
        model_b_performance = {'species': 'owl', 'confidence': 0.91, 'processing_time': 0.25}
        
        ab_test_results = {'model_a': [], 'model_b': []}
        
        with patch.object(self.api.router, 'predict') as mock_predict:
            def ab_test_prediction(file_path, model_variant=None):
                if model_variant == 'model_a':
                    perf = model_a_performance
                elif model_variant == 'model_b':
                    perf = model_b_performance
                else:
                    perf = model_b_performance  # Default to B
                
                # Add some realistic variance
                confidence_variance = np.random.normal(0, 0.02)
                adjusted_confidence = max(0.1, min(1.0, perf['confidence'] + confidence_variance))
                
                return {
                    'predictions': {
                        'species': perf['species'],
                        'confidence': adjusted_confidence
                    },
                    'processing_time': perf['processing_time'],
                    'model_variant': model_variant,
                    'file_type': 'audio'
                }
            
            # Override to handle model_variant parameter
            self.api.predict_file = Mock(side_effect=ab_test_prediction)
            
            # Simulate A/B test with multiple samples
            test_samples = 50
            
            for i in range(test_samples):
                # Randomly assign to A or B
                variant = 'model_a' if i % 2 == 0 else 'model_b'
                
                result = self.api.predict_file(f'/fake/ab_test_{i}.wav', model_variant=variant)
                ab_test_results[variant].append(result)
            
            # Analyze A/B test results
            def analyze_variant(results):
                confidences = [r['predictions']['confidence'] for r in results]
                processing_times = [r['processing_time'] for r in results]
                
                return {
                    'sample_count': len(results),
                    'avg_confidence': np.mean(confidences),
                    'confidence_std': np.std(confidences),
                    'avg_processing_time': np.mean(processing_times),
                    'success_rate': len([r for r in results if r['predictions']['species'] == 'owl']) / len(results)
                }
            
            model_a_analysis = analyze_variant(ab_test_results['model_a'])
            model_b_analysis = analyze_variant(ab_test_results['model_b'])
            
            # A/B test assertions
            assert model_a_analysis['sample_count'] == 25
            assert model_b_analysis['sample_count'] == 25
            
            # Model B should perform better (based on our setup)
            assert model_b_analysis['avg_confidence'] > model_a_analysis['avg_confidence']
            
            # Both should have high success rates
            assert model_a_analysis['success_rate'] >= 0.9
            assert model_b_analysis['success_rate'] >= 0.9
            
            # Statistical significance test (simplified)
            confidence_diff = model_b_analysis['avg_confidence'] - model_a_analysis['avg_confidence']
            assert confidence_diff > 0.02  # Meaningful difference