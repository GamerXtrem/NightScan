"""
Tests pour le routeur de prédiction unifié.

Ce module teste tous les aspects du PredictionRouter:
- Détection automatique du type de fichier et routing
- Workflows de prédiction audio et photo  
- Traitement batch et concurrent
- Validation de fichiers et gestion d'erreurs
- Statistiques et tracking de performance
- Conversion WAV vers spectrogramme
"""

import pytest
import threading
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, mock_open
from typing import Dict, Any, List
import tempfile
import numpy as np
from io import BytesIO

# Import the components to test
from unified_prediction_system.prediction_router import (
    PredictionRouter,
    PredictionError
)
from unified_prediction_system.file_type_detector import FileType, NightScanFile


class TestPredictionRouter:
    """Tests pour le routeur principal de prédiction."""
    
    @patch('unified_prediction_system.prediction_router.UnifiedModelManager')
    @patch('unified_prediction_system.prediction_router.FileTypeDetector')
    def test_router_initialization(self, mock_detector, mock_manager):
        """Test initialisation du PredictionRouter."""
        router = PredictionRouter()
        
        # Verify components initialized
        mock_detector.assert_called_once()
        mock_manager.assert_called_once_with(None)
        
        # Check initial stats
        assert router.stats['total_predictions'] == 0
        assert router.stats['successful_predictions'] == 0
        assert router.stats['failed_predictions'] == 0
        assert router.stats['by_type']['audio'] == 0
        assert router.stats['by_type']['photo'] == 0
        assert router.stats['by_type']['unknown'] == 0
        assert router.stats['processing_times'] == []
    
    @patch('unified_prediction_system.prediction_router.UnifiedModelManager')
    @patch('unified_prediction_system.prediction_router.FileTypeDetector')
    def test_router_with_config_path(self, mock_detector, mock_manager):
        """Test initialisation avec fichier de configuration."""
        config_path = Path('/fake/config.json')
        router = PredictionRouter(config_path=config_path)
        
        mock_manager.assert_called_once_with(config_path)
    
    def test_audio_file_routing(self):
        """Test routing automatique pour fichier audio."""
        router = PredictionRouter()
        
        # Mock file detection
        mock_file = NightScanFile(
            path=Path('/test/audio.wav'),
            file_type=FileType.AUDIO,
            metadata={'duration': 10.5, 'sample_rate': 44100}
        )
        router.file_detector.detect_file_type = Mock(return_value=mock_file)
        
        # Mock audio prediction
        mock_audio_result = {
            'species': 'owl',
            'confidence': 0.85,
            'predictions': [{'class': 'owl', 'confidence': 0.85}]
        }
        router.model_manager.predict_audio = Mock(return_value=mock_audio_result)
        
        # Test prediction
        result = router.predict('/test/audio.wav')
        
        # Verify routing
        router.file_detector.detect_file_type.assert_called_once()
        router.model_manager.predict_audio.assert_called_once()
        
        # Verify result structure
        assert result['file_type'] == 'audio'
        assert result['predictions'] == mock_audio_result
        assert result['metadata']['duration'] == 10.5
        assert 'processing_time' in result
    
    def test_photo_file_routing(self):
        """Test routing automatique pour fichier photo."""
        router = PredictionRouter()
        
        # Mock file detection
        mock_file = NightScanFile(
            path=Path('/test/photo.jpg'),
            file_type=FileType.PHOTO,
            metadata={'width': 1920, 'height': 1080, 'format': 'JPEG'}
        )
        router.file_detector.detect_file_type = Mock(return_value=mock_file)
        
        # Mock photo prediction
        mock_photo_result = {
            'species': 'deer',
            'confidence': 0.92,
            'bounding_boxes': [{'x': 100, 'y': 100, 'w': 200, 'h': 150}]
        }
        router.model_manager.predict_photo = Mock(return_value=mock_photo_result)
        
        # Test prediction
        result = router.predict('/test/photo.jpg')
        
        # Verify routing
        router.file_detector.detect_file_type.assert_called_once()
        router.model_manager.predict_photo.assert_called_once()
        
        # Verify result structure
        assert result['file_type'] == 'photo'
        assert result['predictions'] == mock_photo_result
        assert result['metadata']['width'] == 1920
        assert 'processing_time' in result
    
    def test_unknown_file_type_handling(self):
        """Test gestion des fichiers de type inconnu."""
        router = PredictionRouter()
        
        # Mock unknown file type
        mock_file = NightScanFile(
            path=Path('/test/unknown.txt'),
            file_type=FileType.UNKNOWN,
            metadata={}
        )
        router.file_detector.detect_file_type = Mock(return_value=mock_file)
        
        # Test prediction should raise error
        with pytest.raises(PredictionError) as exc_info:
            router.predict('/test/unknown.txt')
        
        assert "Type de fichier non supporté" in str(exc_info.value)
        
        # Verify stats updated for failed prediction
        assert router.stats['failed_predictions'] == 1
        assert router.stats['by_type']['unknown'] == 1
    
    def test_batch_prediction_mixed_types(self):
        """Test prédiction batch avec types de fichiers mixtes."""
        router = PredictionRouter()
        
        # Mock files of different types
        audio_file = NightScanFile(
            path=Path('/test/audio.wav'),
            file_type=FileType.AUDIO,
            metadata={'duration': 5.0}
        )
        photo_file = NightScanFile(
            path=Path('/test/photo.jpg'), 
            file_type=FileType.PHOTO,
            metadata={'width': 800, 'height': 600}
        )
        
        def mock_detect_file_type(file_path):
            if 'audio' in str(file_path):
                return audio_file
            return photo_file
        
        router.file_detector.detect_file_type.side_effect = mock_detect_file_type
        
        # Mock predictions
        router.model_manager.predict_audio = Mock(return_value={'species': 'owl'})
        router.model_manager.predict_photo = Mock(return_value={'species': 'deer'})
        
        # Test batch prediction
        file_paths = ['/test/audio.wav', '/test/photo.jpg']
        results = router.predict_batch(file_paths)
        
        # Verify results
        assert len(results) == 2
        assert results[0]['file_type'] == 'audio'
        assert results[1]['file_type'] == 'photo'
        assert router.model_manager.predict_audio.call_count == 1
        assert router.model_manager.predict_photo.call_count == 1
    
    def test_concurrent_predictions(self):
        """Test prédictions concurrentes thread-safe."""
        router = PredictionRouter()
        results = []
        errors = []
        
        # Mock successful prediction
        mock_file = NightScanFile(
            path=Path('/test/audio.wav'),
            file_type=FileType.AUDIO,
            metadata={'duration': 1.0}
        )
        router.file_detector.detect_file_type = Mock(return_value=mock_file)
        router.model_manager.predict_audio = Mock(return_value={'species': 'test'})
        
        def predict_thread(thread_id):
            try:
                result = router.predict(f'/test/audio_{thread_id}.wav')
                results.append((thread_id, result))
            except Exception as e:
                errors.append((thread_id, e))
        
        # Start multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=predict_thread, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Verify results
        assert len(errors) == 0
        assert len(results) == 5
        assert router.stats['successful_predictions'] == 5
        assert router.stats['total_predictions'] == 5
    
    def test_statistics_tracking(self):
        """Test tracking des statistiques de performance."""
        router = PredictionRouter()
        
        # Mock successful audio prediction
        audio_file = NightScanFile(
            path=Path('/test/audio.wav'),
            file_type=FileType.AUDIO,
            metadata={}
        )
        router.file_detector.detect_file_type = Mock(return_value=audio_file)
        router.model_manager.predict_audio = Mock(return_value={'species': 'owl'})
        
        # Make prediction
        result = router.predict('/test/audio.wav')
        
        # Verify stats updated
        assert router.stats['total_predictions'] == 1
        assert router.stats['successful_predictions'] == 1
        assert router.stats['failed_predictions'] == 0
        assert router.stats['by_type']['audio'] == 1
        assert len(router.stats['processing_times']) == 1
        assert router.stats['processing_times'][0] > 0
    
    def test_error_handling_model_failure(self):
        """Test gestion d'erreur lors d'échec du modèle."""
        router = PredictionRouter()
        
        # Mock file detection success
        mock_file = NightScanFile(
            path=Path('/test/audio.wav'),
            file_type=FileType.AUDIO,
            metadata={}
        )
        router.file_detector.detect_file_type = Mock(return_value=mock_file)
        
        # Mock model prediction failure
        router.model_manager.predict_audio = Mock(
            side_effect=PredictionError("Modèle en erreur")
        )
        
        # Test prediction should propagate error
        with pytest.raises(PredictionError):
            router.predict('/test/audio.wav')
        
        # Verify stats updated for failure
        assert router.stats['failed_predictions'] == 1
        assert router.stats['total_predictions'] == 1
        assert router.stats['successful_predictions'] == 0
    
    def test_wav_to_spectrogram_conversion(self):
        """Test conversion WAV vers spectrogramme."""
        router = PredictionRouter()
        
        # Mock audio file with WAV format
        mock_file = NightScanFile(
            path=Path('/test/audio.wav'),
            file_type=FileType.AUDIO,
            metadata={'format': 'WAV', 'duration': 10.0}
        )
        router.file_detector.detect_file_type = Mock(return_value=mock_file)
        
        # Mock conversion process
        mock_spectrogram = np.random.rand(128, 128)
        router.model_manager.wav_to_spectrogram = Mock(return_value=mock_spectrogram)
        router.model_manager.predict_spectrogram = Mock(return_value={'species': 'owl'})
        
        # Test spectrogram prediction
        result = router.predict_spectrogram('/test/audio.wav')
        
        # Verify conversion called
        router.model_manager.wav_to_spectrogram.assert_called_once()
        router.model_manager.predict_spectrogram.assert_called_once_with(mock_spectrogram)
        
        assert result['file_type'] == 'audio'
        assert result['conversion_type'] == 'wav_to_spectrogram'
    
    def test_get_prediction_stats(self):
        """Test récupération des statistiques détaillées."""
        router = PredictionRouter()
        
        # Simulate some predictions
        router.stats['total_predictions'] = 10
        router.stats['successful_predictions'] = 8
        router.stats['failed_predictions'] = 2
        router.stats['by_type']['audio'] = 6
        router.stats['by_type']['photo'] = 4
        router.stats['processing_times'] = [0.5, 1.2, 0.8, 0.9, 1.1]
        
        stats = router.get_prediction_stats()
        
        # Verify comprehensive stats
        assert stats['total_predictions'] == 10
        assert stats['success_rate'] == 0.8
        assert stats['failure_rate'] == 0.2
        assert stats['average_processing_time'] == 0.9  # Mean of processing times
        assert stats['by_type']['audio'] == 6
        assert stats['by_type']['photo'] == 4
        assert 'model_stats' in stats
    
    def test_reset_stats(self):
        """Test remise à zéro des statistiques."""
        router = PredictionRouter()
        
        # Set some stats
        router.stats['total_predictions'] = 5
        router.stats['successful_predictions'] = 4
        router.stats['processing_times'] = [1.0, 2.0]
        
        # Reset stats
        router.reset_stats()
        
        # Verify reset
        assert router.stats['total_predictions'] == 0
        assert router.stats['successful_predictions'] == 0
        assert router.stats['failed_predictions'] == 0
        assert router.stats['processing_times'] == []
        assert router.stats['by_type']['audio'] == 0


class TestPredictionRouterIntegration:
    """Tests d'intégration pour le routeur de prédiction."""
    
    @pytest.mark.ml_integration
    def test_end_to_end_audio_prediction(self):
        """Test complet de prédiction audio E2E."""
        router = PredictionRouter()
        
        # Create temporary WAV-like file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            # Write minimal WAV header
            tmp.write(b'RIFF\x24\x08\x00\x00WAVEfmt ')
            wav_path = tmp.name
        
        try:
            # Mock the entire prediction pipeline
            mock_file = NightScanFile(
                path=Path(wav_path),
                file_type=FileType.AUDIO,
                metadata={'duration': 5.0, 'sample_rate': 44100}
            )
            router.file_detector.detect_file_type = Mock(return_value=mock_file)
            
            mock_prediction = {
                'species': 'owl',
                'confidence': 0.87,
                'predictions': [
                    {'class': 'owl', 'confidence': 0.87},
                    {'class': 'wind', 'confidence': 0.13}
                ]
            }
            router.model_manager.predict_audio = Mock(return_value=mock_prediction)
            
            # Execute prediction
            result = router.predict(wav_path)
            
            # Verify complete result structure
            assert result['file_path'] == wav_path
            assert result['file_type'] == 'audio'
            assert result['predictions']['species'] == 'owl'
            assert result['metadata']['duration'] == 5.0
            assert result['processing_time'] > 0
            assert 'timestamp' in result
            
            # Verify stats updated
            assert router.stats['total_predictions'] == 1
            assert router.stats['successful_predictions'] == 1
            assert router.stats['by_type']['audio'] == 1
            
        finally:
            Path(wav_path).unlink()
    
    @pytest.mark.ml_integration
    def test_batch_processing_performance(self):
        """Test performance du traitement batch."""
        router = PredictionRouter()
        
        # Create multiple temporary files
        file_paths = []
        for i in range(10):
            with tempfile.NamedTemporaryFile(suffix=f'_{i}.wav', delete=False) as tmp:
                tmp.write(b'RIFF\x24\x08\x00\x00WAVEfmt ')
                file_paths.append(tmp.name)
        
        try:
            # Mock predictions for all files
            mock_file = NightScanFile(
                path=Path('dummy'),
                file_type=FileType.AUDIO,
                metadata={'duration': 2.0}
            )
            router.file_detector.detect_file_type = Mock(return_value=mock_file)
            router.model_manager.predict_audio = Mock(
                return_value={'species': 'test', 'confidence': 0.8}
            )
            
            # Execute batch prediction
            start_time = time.time()
            results = router.predict_batch(file_paths, max_workers=4)
            processing_time = time.time() - start_time
            
            # Verify results
            assert len(results) == 10
            assert all(r['file_type'] == 'audio' for r in results)
            assert processing_time < 10.0  # Should be reasonably fast
            
            # Verify stats
            assert router.stats['total_predictions'] == 10
            assert router.stats['successful_predictions'] == 10
            
        finally:
            for path in file_paths:
                Path(path).unlink()


@pytest.mark.ml_unit
class TestPredictionRouterEdgeCases:
    """Tests pour les cas limites et erreurs."""
    
    def test_nonexistent_file_handling(self):
        """Test gestion des fichiers inexistants."""
        router = PredictionRouter()
        
        with pytest.raises(FileNotFoundError):
            router.predict('/nonexistent/file.wav')
    
    def test_corrupted_file_handling(self):
        """Test gestion des fichiers corrompus."""
        router = PredictionRouter()
        
        # Mock corrupted file detection
        router.file_detector.detect_file_type = Mock(
            side_effect=Exception("Fichier corrompu")
        )
        
        with pytest.raises(PredictionError) as exc_info:
            router.predict('/test/corrupted.wav')
        
        assert "Erreur lors de la détection" in str(exc_info.value)
    
    def test_empty_batch_prediction(self):
        """Test prédiction batch avec liste vide."""
        router = PredictionRouter()
        
        results = router.predict_batch([])
        
        assert results == []
        assert router.stats['total_predictions'] == 0
    
    def test_large_file_memory_handling(self):
        """Test gestion mémoire pour gros fichiers."""
        router = PredictionRouter()
        
        # Mock large file
        large_file = NightScanFile(
            path=Path('/test/large.wav'),
            file_type=FileType.AUDIO,
            metadata={'duration': 3600, 'file_size': 1000000000}  # 1GB file
        )
        router.file_detector.detect_file_type = Mock(return_value=large_file)
        router.model_manager.predict_audio = Mock(return_value={'species': 'test'})
        
        # Should handle large files gracefully
        result = router.predict('/test/large.wav')
        
        assert result['file_type'] == 'audio'
        assert result['metadata']['file_size'] == 1000000000
    
    def test_prediction_timeout_handling(self):
        """Test gestion des timeouts de prédiction."""
        router = PredictionRouter()
        
        # Mock slow prediction
        mock_file = NightScanFile(
            path=Path('/test/slow.wav'),
            file_type=FileType.AUDIO,
            metadata={}
        )
        router.file_detector.detect_file_type = Mock(return_value=mock_file)
        
        def slow_predict(*args, **kwargs):
            time.sleep(0.1)  # Simulate slow processing
            return {'species': 'test'}
        
        router.model_manager.predict_audio = Mock(side_effect=slow_predict)
        
        # Should complete within reasonable time
        start_time = time.time()
        result = router.predict('/test/slow.wav')
        processing_time = time.time() - start_time
        
        assert result['file_type'] == 'audio'
        assert processing_time < 1.0  # Should be under 1 second for mock


@pytest.mark.performance_critical
class TestPredictionRouterPerformance:
    """Tests de performance pour le routeur."""
    
    def test_prediction_latency_benchmark(self):
        """Test benchmark de latence de prédiction."""
        router = PredictionRouter()
        
        # Mock fast prediction
        mock_file = NightScanFile(
            path=Path('/test/audio.wav'),
            file_type=FileType.AUDIO,
            metadata={}
        )
        router.file_detector.detect_file_type = Mock(return_value=mock_file)
        router.model_manager.predict_audio = Mock(
            return_value={'species': 'test', 'confidence': 0.9}
        )
        
        # Measure prediction latency
        latencies = []
        for _ in range(100):
            start_time = time.time()
            router.predict('/test/audio.wav')
            latency = time.time() - start_time
            latencies.append(latency)
        
        # Performance assertions
        avg_latency = sum(latencies) / len(latencies)
        p95_latency = sorted(latencies)[94]  # 95th percentile
        
        assert avg_latency < 0.01  # Average < 10ms
        assert p95_latency < 0.05  # P95 < 50ms
        assert max(latencies) < 0.1  # Max < 100ms
    
    def test_memory_usage_stability(self):
        """Test stabilité de l'usage mémoire."""
        router = PredictionRouter()
        
        # Mock prediction
        mock_file = NightScanFile(
            path=Path('/test/audio.wav'),
            file_type=FileType.AUDIO,
            metadata={}
        )
        router.file_detector.detect_file_type = Mock(return_value=mock_file)
        router.model_manager.predict_audio = Mock(return_value={'species': 'test'})
        
        # Run many predictions to check for memory leaks
        for i in range(1000):
            router.predict(f'/test/audio_{i}.wav')
        
        # Memory should be stable (no obvious leaks in mock scenario)
        assert router.stats['total_predictions'] == 1000
        assert router.stats['successful_predictions'] == 1000
        assert len(router.stats['processing_times']) == 1000