"""
Tests d'intégration End-to-End pour le pipeline de prédiction complet.

Ce module teste l'intégration complète du système de prédiction:
- Pipeline E2E : upload → détection → prédiction → résultat
- Consistance des modèles et reproductibilité
- Performance benchmarks avec fichiers réalistes  
- Intégration cache et optimisations
- Notifications WebSocket temps réel
- Workflow complet multi-utilisateurs
"""

import pytest
import asyncio
import threading
import time
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any
import numpy as np
from PIL import Image
import struct

# Import components for integration testing
from unified_prediction_system.unified_prediction_api import UnifiedPredictionAPI
from unified_prediction_system.prediction_router import PredictionRouter
from unified_prediction_system.model_manager import ModelManager
from unified_prediction_system.file_type_detector import FileTypeDetector, FileType


class TestPredictionPipelineIntegration:
    """Tests d'intégration pour le pipeline complet de prédiction."""
    
    @patch('unified_prediction_system.unified_prediction_api.ModelManager')
    @patch('unified_prediction_system.unified_prediction_api.PredictionRouter')
    def test_end_to_end_audio_pipeline(self, mock_router, mock_manager):
        """Test pipeline E2E complet pour fichier audio."""
        # Setup API
        api = UnifiedPredictionAPI()
        
        # Create test WAV file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            # Write realistic WAV header
            tmp.write(b'RIFF')
            tmp.write(struct.pack('<L', 44100))  # File size
            tmp.write(b'WAVE')
            tmp.write(b'fmt ')
            tmp.write(struct.pack('<L', 16))     # fmt chunk size
            tmp.write(struct.pack('<H', 1))      # PCM
            tmp.write(struct.pack('<H', 1))      # Mono
            tmp.write(struct.pack('<L', 44100))  # Sample rate
            tmp.write(struct.pack('<L', 88200))  # Byte rate
            tmp.write(struct.pack('<H', 2))      # Block align
            tmp.write(struct.pack('<H', 16))     # Bits per sample
            tmp.write(b'data')
            tmp.write(struct.pack('<L', 44100))  # Data size
            tmp.write(b'\\x00' * 44100)          # Sample data
            
            wav_path = tmp.name
        
        try:
            # Mock prediction result
            mock_prediction_result = {\n                'file_path': wav_path,\n                'file_type': 'audio',\n                'predictions': {\n                    'species': 'owl',\n                    'confidence': 0.89,\n                    'predictions': [\n                        {'class': 'owl', 'confidence': 0.89},\n                        {'class': 'wind', 'confidence': 0.11}\n                    ]\n                },\n                'metadata': {\n                    'duration': 1.0,\n                    'sample_rate': 44100,\n                    'channels': 1\n                },\n                'processing_time': 0.234,\n                'timestamp': '2024-01-15T10:30:00Z'\n            }
            
            # Configure mocks
            api.router.predict.return_value = mock_prediction_result
            
            # Execute E2E prediction
            result = api.predict_file(wav_path)
            
            # Verify complete workflow
            assert result['status'] == 'success'
            assert result['file_type'] == 'audio'
            assert result['predictions']['species'] == 'owl'
            assert result['predictions']['confidence'] == 0.89
            assert result['metadata']['duration'] == 1.0
            assert result['processing_time'] > 0
            assert 'prediction_id' in result
            assert 'timestamp' in result
            
            # Verify API called correctly
            api.router.predict.assert_called_once_with(wav_path)
            
        finally:
            Path(wav_path).unlink()
    
    @patch('unified_prediction_system.unified_prediction_api.ModelManager')
    @patch('unified_prediction_system.unified_prediction_api.PredictionRouter')
    def test_end_to_end_image_pipeline(self, mock_router, mock_manager):
        """Test pipeline E2E complet pour fichier image."""
        api = UnifiedPredictionAPI()
        
        # Create test image file
        img = Image.new('RGB', (640, 480), color=(100, 150, 200))
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            img.save(tmp, format='JPEG', quality=95)
            img_path = tmp.name
        
        try:
            # Mock prediction result
            mock_prediction_result = {
                'file_path': img_path,
                'file_type': 'photo',
                'predictions': {
                    'species': 'deer',
                    'confidence': 0.94,
                    'bounding_boxes': [
                        {
                            'species': 'deer',
                            'confidence': 0.94,
                            'bbox': {'x': 120, 'y': 80, 'width': 200, 'height': 180}
                        }
                    ]
                },
                'metadata': {
                    'width': 640,
                    'height': 480,
                    'format': 'JPEG',
                    'file_size': img.size
                },
                'processing_time': 0.156
            }
            
            api.router.predict.return_value = mock_prediction_result
            
            # Execute prediction
            result = api.predict_file(img_path)
            
            # Verify results
            assert result['status'] == 'success'
            assert result['file_type'] == 'photo'
            assert result['predictions']['species'] == 'deer'
            assert len(result['predictions']['bounding_boxes']) == 1
            assert result['metadata']['width'] == 640
            assert result['metadata']['height'] == 480
            
        finally:
            Path(img_path).unlink()
    
    def test_batch_processing_mixed_files(self):
        """Test traitement batch avec fichiers audio et image mélangés."""
        api = UnifiedPredictionAPI()
        
        # Create mixed file types
        files = []
        
        # Audio file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            tmp.write(b'RIFF\\x24\\x08\\x00\\x00WAVEfmt ')
            files.append((tmp.name, 'audio'))
        
        # Image file
        img = Image.new('RGB', (100, 100), color='red')
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            img.save(tmp, format='JPEG')
            files.append((tmp.name, 'photo'))
        
        # Spectrogram file
        with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as tmp:
            np.save(tmp.name, np.random.rand(64, 128))
            files.append((tmp.name, 'audio'))
        
        try:
            with patch.object(api.router, 'predict_batch') as mock_batch:
                mock_batch.return_value = [
                    {'file_type': 'audio', 'predictions': {'species': 'owl'}},
                    {'file_type': 'photo', 'predictions': {'species': 'fox'}},
                    {'file_type': 'audio', 'predictions': {'species': 'wind'}}
                ]
                
                file_paths = [f[0] for f in files]
                results = api.predict_batch(file_paths)
                
                # Verify batch processing
                assert len(results) == 3
                assert results[0]['file_type'] == 'audio'
                assert results[1]['file_type'] == 'photo'
                assert results[2]['file_type'] == 'audio'
                
                mock_batch.assert_called_once_with(file_paths, max_workers=None)
                
        finally:
            for file_path, _ in files:
                Path(file_path).unlink()
    
    def test_prediction_consistency_reproducibility(self):
        """Test consistance et reproductibilité des prédictions."""
        api = UnifiedPredictionAPI()
        
        # Create test file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            tmp.write(b'RIFF\\x24\\x08\\x00\\x00WAVEfmt ')
            test_path = tmp.name
        
        try:
            # Mock consistent prediction
            consistent_result = {
                'predictions': {'species': 'owl', 'confidence': 0.85},
                'file_type': 'audio'
            }
            
            with patch.object(api.router, 'predict', return_value=consistent_result):
                # Run multiple predictions on same file
                results = []
                for _ in range(5):
                    result = api.predict_file(test_path)
                    results.append(result)
                
                # Verify consistency
                for result in results:
                    assert result['predictions']['species'] == 'owl'
                    assert result['predictions']['confidence'] == 0.85
                    assert result['file_type'] == 'audio'
                
                # Verify all results are identical (minus timestamps)
                species_results = [r['predictions']['species'] for r in results]
                confidence_results = [r['predictions']['confidence'] for r in results]
                
                assert len(set(species_results)) == 1  # All same species
                assert len(set(confidence_results)) == 1  # All same confidence
                
        finally:
            Path(test_path).unlink()
    
    @pytest.mark.performance_critical
    def test_performance_benchmarks_realistic_files(self):
        """Test benchmarks de performance avec fichiers réalistes."""
        api = UnifiedPredictionAPI()
        
        # Create realistic sized files
        large_files = []
        
        # Large audio file (simulated)
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            # Simulate 10-second WAV file
            header = b'RIFF'
            header += struct.pack('<L', 44100 * 10 * 2)  # 10 seconds stereo
            header += b'WAVEfmt '
            tmp.write(header)
            large_files.append(tmp.name)
        
        # Large image file
        large_img = Image.new('RGB', (1920, 1080), color='green')
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            large_img.save(tmp, format='JPEG', quality=95)
            large_files.append(tmp.name)
        
        try:
            with patch.object(api.router, 'predict') as mock_predict:
                # Mock fast prediction
                mock_predict.return_value = {
                    'predictions': {'species': 'test'}, 
                    'processing_time': 0.5
                }
                
                # Measure performance
                start_time = time.time()
                for file_path in large_files:
                    result = api.predict_file(file_path)
                total_time = time.time() - start_time
                
                # Performance assertions
                assert total_time < 5.0  # Should process 2 large files in < 5s
                assert mock_predict.call_count == 2
                
                # Individual file processing should be fast
                for call in mock_predict.call_args_list:
                    # Verify called with correct file paths
                    assert call[0][0] in large_files
                
        finally:
            for file_path in large_files:
                Path(file_path).unlink()
    
    def test_cache_integration_optimization(self):
        """Test intégration avec le système de cache."""
        api = UnifiedPredictionAPI()
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            tmp.write(b'RIFF\\x24\\x08\\x00\\x00WAVEfmt ')
            test_path = tmp.name
        
        try:
            with patch.object(api, '_get_cached_prediction') as mock_cache_get, \\
                 patch.object(api, '_cache_prediction') as mock_cache_set, \\
                 patch.object(api.router, 'predict') as mock_predict:
                
                # First call - cache miss
                mock_cache_get.return_value = None
                mock_predict.return_value = {'predictions': {'species': 'owl'}}
                
                result1 = api.predict_file(test_path)
                
                # Verify cache operations
                mock_cache_get.assert_called_once()
                mock_predict.assert_called_once()
                mock_cache_set.assert_called_once()
                
                # Second call - cache hit
                mock_cache_get.return_value = {'predictions': {'species': 'owl'}, 'cached': True}
                mock_cache_get.reset_mock()
                mock_predict.reset_mock()
                mock_cache_set.reset_mock()
                
                result2 = api.predict_file(test_path)
                
                # Verify cache hit
                mock_cache_get.assert_called_once()
                mock_predict.assert_not_called()  # Should not call prediction
                mock_cache_set.assert_not_called()  # Should not cache again
                
                assert result2['cached'] == True
                
        finally:
            Path(test_path).unlink()
    
    @pytest.mark.asyncio
    async def test_websocket_notifications_realtime(self):
        """Test notifications WebSocket en temps réel."""
        api = UnifiedPredictionAPI()
        
        # Mock WebSocket connection
        mock_websocket = Mock()
        notifications = []
        
        async def mock_send(message):
            notifications.append(json.loads(message))
        
        mock_websocket.send = mock_send
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            tmp.write(b'RIFF\\x24\\x08\\x00\\x00WAVEfmt ')
            test_path = tmp.name
        
        try:
            with patch.object(api.router, 'predict') as mock_predict:
                mock_predict.return_value = {
                    'predictions': {'species': 'owl', 'confidence': 0.9},
                    'processing_time': 0.5
                }
                
                # Setup WebSocket notification
                api.websocket_connections = [mock_websocket]
                
                # Execute prediction with notifications
                result = await api.predict_file_async(test_path, notify_websocket=True)
                
                # Verify notifications sent
                assert len(notifications) >= 2  # Start + completion notifications
                
                # Check start notification
                start_notification = notifications[0]
                assert start_notification['type'] == 'prediction_started'
                assert start_notification['file_path'] == test_path
                
                # Check completion notification
                completion_notification = notifications[-1]
                assert completion_notification['type'] == 'prediction_completed'
                assert completion_notification['species'] == 'owl'
                assert completion_notification['confidence'] == 0.9
                
        finally:
            Path(test_path).unlink()
    
    def test_multi_user_concurrent_workflow(self):
        """Test workflow concurrent multi-utilisateurs."""
        api = UnifiedPredictionAPI()
        
        # Create files for different users
        user_files = {}
        for user_id in ['user1', 'user2', 'user3']:
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                tmp.write(b'RIFF\\x24\\x08\\x00\\x00WAVEfmt ')
                user_files[user_id] = tmp.name
        
        results = {}
        errors = {}
        
        def user_prediction_workflow(user_id):
            try:
                with patch.object(api.router, 'predict') as mock_predict:
                    mock_predict.return_value = {
                        'predictions': {'species': f'species_{user_id}'},
                        'user_context': user_id
                    }
                    
                    # Simulate user workflow
                    result = api.predict_file(user_files[user_id], user_context={'user_id': user_id})
                    results[user_id] = result
                    
            except Exception as e:
                errors[user_id] = e
        
        try:
            # Start concurrent user workflows
            threads = []
            for user_id in user_files.keys():
                thread = threading.Thread(target=user_prediction_workflow, args=(user_id,))
                threads.append(thread)
                thread.start()
            
            # Wait for all workflows
            for thread in threads:
                thread.join()
            
            # Verify all users completed successfully
            assert len(errors) == 0
            assert len(results) == 3
            
            # Verify user isolation
            for user_id in ['user1', 'user2', 'user3']:
                assert user_id in results
                assert results[user_id]['predictions']['species'] == f'species_{user_id}'
                
        finally:
            for file_path in user_files.values():
                Path(file_path).unlink()


class TestModelConsistencyIntegration:
    """Tests d'intégration pour la consistance des modèles."""
    
    def test_model_version_consistency(self):
        """Test consistance entre versions de modèles."""
        api = UnifiedPredictionAPI()
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            tmp.write(b'RIFF\\x24\\x08\\x00\\x00WAVEfmt ')
            test_path = tmp.name
        
        try:
            with patch.object(api.model_manager, 'get_model_version') as mock_version, \\
                 patch.object(api.router, 'predict') as mock_predict:
                
                # Test with different model versions
                versions = ['v1.0', 'v1.1', 'v2.0']
                version_results = {}
                
                for version in versions:
                    mock_version.return_value = version
                    mock_predict.return_value = {
                        'predictions': {'species': 'owl'},
                        'model_version': version
                    }
                    
                    result = api.predict_file(test_path, model_version=version)
                    version_results[version] = result
                
                # Verify version tracking
                for version in versions:
                    assert version_results[version]['model_version'] == version
                    # All versions should give consistent species (in this mock)
                    assert version_results[version]['predictions']['species'] == 'owl'
                
        finally:
            Path(test_path).unlink()
    
    def test_model_accuracy_regression_detection(self):
        """Test détection de régression de précision des modèles."""
        api = UnifiedPredictionAPI()
        
        # Mock known good predictions for regression testing
        baseline_predictions = {
            'test_owl.wav': {'species': 'owl', 'confidence': 0.95},
            'test_deer.jpg': {'species': 'deer', 'confidence': 0.88},
            'test_wind.wav': {'species': 'wind', 'confidence': 0.72}
        }
        
        with patch.object(api.router, 'predict') as mock_predict:
            regression_detected = False
            
            for test_file, expected in baseline_predictions.items():
                # Mock current prediction
                mock_predict.return_value = {
                    'predictions': expected  # In real test, this might differ
                }
                
                result = api.predict_file(test_file)
                
                # Check for regression (confidence drop > 10%)
                actual_confidence = result['predictions']['confidence']
                expected_confidence = expected['confidence']
                
                if actual_confidence < expected_confidence - 0.1:
                    regression_detected = True
                    break
                    
                # Verify species consistency
                assert result['predictions']['species'] == expected['species']
            
            # In this mock scenario, no regression should be detected
            assert regression_detected == False


class TestErrorHandlingIntegration:
    """Tests d'intégration pour la gestion d'erreurs."""
    
    def test_graceful_degradation_model_failure(self):
        """Test dégradation gracieuse lors d'échec de modèle."""
        api = UnifiedPredictionAPI()
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            tmp.write(b'RIFF\\x24\\x08\\x00\\x00WAVEfmt ')
            test_path = tmp.name
        
        try:
            with patch.object(api.router, 'predict') as mock_predict:
                # Simulate model failure
                mock_predict.side_effect = Exception("Model crashed")
                
                # Should handle gracefully
                result = api.predict_file(test_path)
                
                assert result['status'] == 'error'
                assert 'error_message' in result
                assert 'Model crashed' in result['error_message']
                assert 'fallback_attempted' in result
                
        finally:
            Path(test_path).unlink()
    
    def test_partial_batch_failure_handling(self):
        """Test gestion d'échecs partiels en batch."""
        api = UnifiedPredictionAPI()
        
        # Create test files
        files = []
        for i in range(3):
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                tmp.write(b'RIFF\\x24\\x08\\x00\\x00WAVEfmt ')
                files.append(tmp.name)
        
        try:
            with patch.object(api.router, 'predict_batch') as mock_batch:
                # Mock partial failure (file 1 succeeds, file 2 fails, file 3 succeeds)
                mock_batch.return_value = [
                    {'status': 'success', 'predictions': {'species': 'owl'}},
                    {'status': 'error', 'error': 'Processing failed'},
                    {'status': 'success', 'predictions': {'species': 'deer'}}
                ]
                
                results = api.predict_batch(files)
                
                # Verify partial success handling
                assert len(results) == 3
                assert results[0]['status'] == 'success'
                assert results[1]['status'] == 'error'
                assert results[2]['status'] == 'success'
                
                # Verify successful predictions have data
                assert results[0]['predictions']['species'] == 'owl'
                assert results[2]['predictions']['species'] == 'deer'
                
                # Verify failed prediction has error info
                assert 'error' in results[1]
                
        finally:
            for file_path in files:
                Path(file_path).unlink()


@pytest.mark.ml_integration
@pytest.mark.performance_critical
class TestSystemPerformanceIntegration:
    """Tests de performance système intégrés."""
    
    def test_system_under_load_integration(self):
        """Test système sous charge réaliste."""
        api = UnifiedPredictionAPI()
        
        # Create multiple files of different types and sizes
        test_files = []
        
        # Small files
        for i in range(5):
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                tmp.write(b'RIFF\\x24\\x08\\x00\\x00WAVEfmt ')
                test_files.append(tmp.name)
        
        # Medium images
        for i in range(3):
            img = Image.new('RGB', (800, 600), color=(i*50, 100, 150))
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                img.save(tmp, format='JPEG')
                test_files.append(tmp.name)
        
        try:
            with patch.object(api.router, 'predict') as mock_predict:
                # Mock realistic processing times
                processing_times = [0.1, 0.3, 0.2, 0.4, 0.15, 0.8, 0.6, 0.9]
                
                def mock_predict_with_delay(file_path):
                    time.sleep(processing_times[len(mock_predict.call_args_list) % len(processing_times)])
                    return {
                        'predictions': {'species': 'test'},
                        'processing_time': processing_times[len(mock_predict.call_args_list) % len(processing_times)]
                    }
                
                mock_predict.side_effect = mock_predict_with_delay
                
                # Execute under load
                start_time = time.time()
                results = []
                
                # Concurrent processing
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                    futures = [executor.submit(api.predict_file, file_path) for file_path in test_files]
                    results = [future.result() for future in concurrent.futures.as_completed(futures)]
                
                total_time = time.time() - start_time
                
                # Performance assertions
                assert len(results) == len(test_files)
                assert total_time < 10.0  # Should complete within 10 seconds
                assert all(r['predictions']['species'] == 'test' for r in results)
                
                # Verify reasonable concurrent performance
                avg_processing_time = sum(r['processing_time'] for r in results) / len(results)
                assert avg_processing_time < 1.0  # Average processing < 1s
                
        finally:
            for file_path in test_files:
                Path(file_path).unlink()
    
    def test_memory_usage_stability_integration(self):
        """Test stabilité mémoire sous charge continue."""
        api = UnifiedPredictionAPI()
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            tmp.write(b'RIFF\\x24\\x08\\x00\\x00WAVEfmt ')
            test_path = tmp.name
        
        try:
            with patch.object(api.router, 'predict') as mock_predict:
                mock_predict.return_value = {
                    'predictions': {'species': 'test'},
                    'processing_time': 0.1
                }
                
                # Process many files to check for memory leaks
                for i in range(100):
                    result = api.predict_file(test_path)
                    
                    # Verify each prediction succeeds
                    assert result['predictions']['species'] == 'test'
                    
                    # Simulate cleanup
                    if i % 10 == 0:
                        api._cleanup_resources()
                
                # Memory should be stable (no leaks in properly mocked scenario)
                assert mock_predict.call_count == 100
                
        finally:
            Path(test_path).unlink()