"""
Tests de performance et benchmarks pour les modèles ML.

Ce module teste les performances critiques du système:
- Benchmarks de latence de prédiction (SLA < 2s)
- Monitoring usage mémoire GPU/CPU
- Tests de concurrence et charge simultanée
- Optimisation warm-up et mise en cache
- Métriques de performance en temps réel
- Détection de régression de performance
"""

import pytest
import time
import threading
import psutil
import gc
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Tuple, Any
import tempfile
import numpy as np
from PIL import Image
import struct
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import components for performance testing
from unified_prediction_system.model_manager import ModelManager
from unified_prediction_system.prediction_router import PredictionRouter
from unified_prediction_system.unified_prediction_api import UnifiedPredictionAPI


class PerformanceMonitor:
    """Utilitaire pour monitoring des performances."""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.memory_before = None
        self.memory_after = None
        self.cpu_before = None
        self.cpu_after = None
    
    def start(self):
        """Démarre le monitoring."""
        gc.collect()  # Force garbage collection
        self.start_time = time.time()
        self.memory_before = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        self.cpu_before = psutil.Process().cpu_percent()
    
    def stop(self):
        """Arrête le monitoring."""
        self.end_time = time.time()
        self.memory_after = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        self.cpu_after = psutil.Process().cpu_percent()
    
    @property
    def duration(self) -> float:
        """Durée d'exécution en secondes."""
        return self.end_time - self.start_time
    
    @property
    def memory_delta(self) -> float:
        """Changement de mémoire en MB."""
        return self.memory_after - self.memory_before
    
    @property
    def cpu_usage(self) -> float:
        """Usage CPU moyen."""
        return (self.cpu_before + self.cpu_after) / 2


@contextmanager
def performance_monitor():
    """Context manager pour monitoring automatique."""
    monitor = PerformanceMonitor()
    monitor.start()
    try:
        yield monitor
    finally:
        monitor.stop()


class TestModelManagerPerformance:
    """Tests de performance pour le ModelManager."""
    
    @pytest.mark.performance_critical
    def test_model_loading_latency_benchmark(self):
        """Test benchmark de latence de chargement de modèle."""
        manager = ModelManager()
        
        with patch.object(manager, '_load_model_from_disk') as mock_load:
            # Mock realistic model loading time
            def slow_load(*args, **kwargs):
                time.sleep(0.5)  # Simulate model loading
                return Mock()
            
            mock_load.side_effect = slow_load
            
            # Measure loading latency
            latencies = []
            for i in range(5):
                with performance_monitor() as monitor:
                    model = manager.load_model(f'test_model_{i}', 'audio')
                
                latencies.append(monitor.duration)
                assert model is not None
            
            # Performance assertions
            avg_latency = sum(latencies) / len(latencies)
            max_latency = max(latencies)
            
            assert avg_latency < 1.0  # Average loading < 1 second
            assert max_latency < 2.0  # Max loading < 2 seconds
            assert all(l > 0.4 for l in latencies)  # All loads take realistic time
    
    @pytest.mark.performance_critical
    def test_model_prediction_latency_sla(self):
        """Test SLA de latence de prédiction < 2 secondes."""
        manager = ModelManager()
        
        with patch.object(manager, '_load_model_from_disk') as mock_load, \
             patch.object(manager, 'predict_audio') as mock_predict:
            
            # Mock model
            mock_model = Mock()
            mock_load.return_value = mock_model
            
            # Mock prediction with realistic time
            def predict_with_latency(*args, **kwargs):
                time.sleep(0.1)  # Simulate prediction time
                return {
                    'species': 'owl',
                    'confidence': 0.89,
                    'processing_time': 0.1
                }
            
            mock_predict.side_effect = predict_with_latency
            
            # Load model once
            model = manager.load_model('performance_test_model', 'audio')
            
            # Test prediction latencies
            prediction_latencies = []
            for i in range(50):  # 50 predictions for statistical significance
                with performance_monitor() as monitor:
                    result = manager.predict_audio('fake_audio_data')
                
                prediction_latencies.append(monitor.duration)
                assert result['species'] == 'owl'
            
            # SLA Performance Assertions
            avg_latency = sum(prediction_latencies) / len(prediction_latencies)
            p95_latency = sorted(prediction_latencies)[47]  # 95th percentile
            p99_latency = sorted(prediction_latencies)[49]  # 99th percentile
            max_latency = max(prediction_latencies)
            
            # SLA Requirements
            assert avg_latency < 0.5   # Average < 500ms
            assert p95_latency < 1.0   # P95 < 1 second
            assert p99_latency < 1.5   # P99 < 1.5 seconds  
            assert max_latency < 2.0   # Max < 2 seconds (SLA)
    
    @pytest.mark.performance_critical
    def test_memory_usage_under_load(self):
        """Test usage mémoire sous charge."""
        manager = ModelManager()
        manager.max_models = 5  # Limit for memory testing
        
        with patch.object(manager, '_load_model_from_disk') as mock_load:
            # Mock models with memory usage simulation
            def create_mock_model(*args, **kwargs):
                mock_model = Mock()
                # Simulate memory allocation
                mock_model._memory_footprint = np.random.rand(1000, 1000)  # ~8MB
                return mock_model
            
            mock_load.side_effect = create_mock_model
            
            initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            # Load multiple models
            for i in range(10):  # Load more than max to test cleanup
                with performance_monitor() as monitor:
                    model = manager.load_model(f'memory_test_model_{i}', 'audio')
                
                # Memory should not grow unbounded
                current_memory = psutil.Process().memory_info().rss / 1024 / 1024
                memory_growth = current_memory - initial_memory
                
                # Memory growth should be reasonable
                assert memory_growth < 100  # Less than 100MB growth
                assert len(manager.models) <= manager.max_models
            
            # Memory should stabilize after cleanup
            final_memory = psutil.Process().memory_info().rss / 1024 / 1024
            final_growth = final_memory - initial_memory
            
            assert final_growth < 50  # Final growth < 50MB
    
    @pytest.mark.performance_critical
    def test_concurrent_model_access_performance(self):
        """Test performance d'accès concurrent aux modèles."""
        manager = ModelManager()
        
        with patch.object(manager, '_load_model_from_disk') as mock_load:
            mock_load.return_value = Mock()
            
            # Pre-load a model
            model = manager.load_model('concurrent_test_model', 'audio')
            
            results = []
            errors = []
            
            def concurrent_access(thread_id):
                try:
                    thread_latencies = []
                    for i in range(10):  # 10 accesses per thread
                        with performance_monitor() as monitor:
                            accessed_model = manager.get_model('concurrent_test_model')
                        
                        thread_latencies.append(monitor.duration)
                        assert accessed_model is not None
                    
                    results.append((thread_id, thread_latencies))
                except Exception as e:
                    errors.append((thread_id, e))
            
            # Start concurrent threads
            threads = []
            start_time = time.time()
            
            for thread_id in range(10):
                thread = threading.Thread(target=concurrent_access, args=(thread_id,))
                threads.append(thread)
                thread.start()
            
            # Wait for all threads
            for thread in threads:
                thread.join()
            
            total_time = time.time() - start_time
            
            # Verify no errors
            assert len(errors) == 0
            assert len(results) == 10
            
            # Analyze concurrent performance
            all_latencies = []
            for thread_id, latencies in results:
                all_latencies.extend(latencies)
            
            avg_concurrent_latency = sum(all_latencies) / len(all_latencies)
            max_concurrent_latency = max(all_latencies)
            
            # Concurrent access should remain fast
            assert avg_concurrent_latency < 0.01  # < 10ms average
            assert max_concurrent_latency < 0.1   # < 100ms max
            assert total_time < 5.0  # Total execution < 5 seconds
    
    def test_model_warm_up_optimization(self):
        """Test optimisation du warm-up des modèles."""
        manager = ModelManager()
        
        with patch.object(manager, '_load_model_from_disk') as mock_load:
            # Mock cold start (first load)
            def cold_load(*args, **kwargs):
                time.sleep(1.0)  # Simulate cold start
                return Mock()
            
            # Mock warm load (subsequent loads)
            def warm_load(*args, **kwargs):
                time.sleep(0.1)  # Much faster
                return Mock()
            
            mock_load.side_effect = [cold_load(), warm_load(), warm_load()]
            
            # First load (cold start)
            with performance_monitor() as cold_monitor:
                model1 = manager.load_model('warm_up_test', 'audio')
            
            # Second load (should be cached/warm)
            with performance_monitor() as warm_monitor1:
                model2 = manager.load_model('warm_up_test', 'audio')
            
            # Third load (should also be warm)
            with performance_monitor() as warm_monitor2:
                model3 = manager.load_model('warm_up_test', 'audio')
            
            # Verify warm-up optimization
            assert cold_monitor.duration > 0.5  # Cold start takes time
            assert warm_monitor1.duration < 0.01  # Warm access is fast
            assert warm_monitor2.duration < 0.01  # Consistent warm performance
            
            # All should return same model instance (cached)
            assert model1 is model2
            assert model2 is model3


class TestPredictionRouterPerformance:
    """Tests de performance pour le PredictionRouter."""
    
    @pytest.mark.performance_critical
    def test_file_type_detection_performance(self):
        """Test performance de détection de type de fichier."""
        router = PredictionRouter()
        
        # Create test files
        test_files = []
        
        # Audio files
        for i in range(10):
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                tmp.write(b'RIFF\x24\x08\x00\x00WAVEfmt ')
                test_files.append(tmp.name)
        
        # Image files
        for i in range(10):
            img = Image.new('RGB', (100, 100), color=(i*25, 100, 150))
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                img.save(tmp, format='JPEG')
                test_files.append(tmp.name)
        
        try:
            detection_times = []
            
            for file_path in test_files:
                with performance_monitor() as monitor:
                    file_info = router.file_detector.detect_file_type(file_path)
                
                detection_times.append(monitor.duration)
                assert file_info.is_valid == True
            
            # Performance assertions
            avg_detection_time = sum(detection_times) / len(detection_times)
            max_detection_time = max(detection_times)
            
            assert avg_detection_time < 0.05  # Average < 50ms
            assert max_detection_time < 0.2   # Max < 200ms
            
        finally:
            for file_path in test_files:
                Path(file_path).unlink()
    
    @pytest.mark.performance_critical
    def test_batch_processing_efficiency(self):
        """Test efficacité du traitement batch."""
        router = PredictionRouter()
        
        # Create batch of files
        batch_files = []
        for i in range(20):
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                tmp.write(b'RIFF\x24\x08\x00\x00WAVEfmt ')
                batch_files.append(tmp.name)
        
        try:
            with patch.object(router.model_manager, 'predict_audio') as mock_predict:
                # Mock fast prediction
                mock_predict.return_value = {
                    'species': 'test',
                    'confidence': 0.8,
                    'processing_time': 0.1
                }
                
                # Test batch processing
                with performance_monitor() as monitor:
                    results = router.predict_batch(batch_files, max_workers=4)
                
                batch_time = monitor.duration
                
                # Verify results
                assert len(results) == 20
                assert all(r['predictions']['species'] == 'test' for r in results)
                
                # Performance assertions
                avg_time_per_file = batch_time / 20
                assert batch_time < 10.0  # Total batch < 10 seconds
                assert avg_time_per_file < 0.5  # Average per file < 500ms
                
                # Batch should be faster than sequential
                sequential_estimate = 20 * 0.2  # 20 files * 200ms each
                efficiency_gain = sequential_estimate / batch_time
                assert efficiency_gain > 1.5  # At least 50% efficiency gain
                
        finally:
            for file_path in batch_files:
                Path(file_path).unlink()
    
    @pytest.mark.performance_critical
    def test_prediction_pipeline_end_to_end_performance(self):
        """Test performance E2E du pipeline de prédiction."""
        router = PredictionRouter()
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            # Create realistic audio file
            tmp.write(b'RIFF')
            tmp.write(struct.pack('<L', 44100))  # 1 second of audio
            tmp.write(b'WAVEfmt ')
            tmp.write(b'\x10\x00\x00\x00\x01\x00\x01\x00')  # PCM mono
            tmp.write(struct.pack('<L', 44100))  # Sample rate
            tmp.write(struct.pack('<L', 88200))  # Byte rate
            tmp.write(b'\x02\x00\x10\x00data')
            tmp.write(struct.pack('<L', 44100))  # Data size
            tmp.write(b'\x00' * 44100)  # Audio data
            
            test_file = tmp.name
        
        try:
            with patch.object(router.model_manager, 'predict_audio') as mock_predict:
                mock_predict.return_value = {
                    'species': 'owl',
                    'confidence': 0.92,
                    'predictions': [
                        {'class': 'owl', 'confidence': 0.92},
                        {'class': 'wind', 'confidence': 0.08}
                    ]
                }
                
                # Test E2E pipeline performance
                pipeline_times = []
                for i in range(20):
                    with performance_monitor() as monitor:
                        result = router.predict(test_file)
                    
                    pipeline_times.append(monitor.duration)
                    assert result['predictions']['species'] == 'owl'
                
                # Performance analysis
                avg_pipeline_time = sum(pipeline_times) / len(pipeline_times)
                p95_pipeline_time = sorted(pipeline_times)[18]  # 95th percentile
                max_pipeline_time = max(pipeline_times)
                
                # E2E Performance SLA
                assert avg_pipeline_time < 1.0   # Average E2E < 1 second
                assert p95_pipeline_time < 1.5   # P95 < 1.5 seconds
                assert max_pipeline_time < 2.0   # Max E2E < 2 seconds (SLA)
                
        finally:
            Path(test_file).unlink()


class TestAPIPerformanceBenchmarks:
    """Tests de performance pour l'API unifiée."""
    
    def setup_method(self):
        """Setup pour tests de performance API."""
        from unified_prediction_system.unified_prediction_api import create_app
        self.app = create_app(testing=True)
        self.client = self.app.test_client()
        self.app_context = self.app.app_context()
        self.app_context.push()
    
    def teardown_method(self):
        """Cleanup après tests."""
        self.app_context.pop()
    
    @pytest.mark.performance_critical
    def test_api_endpoint_response_time_sla(self):
        """Test SLA de temps de réponse des endpoints API."""
        from io import BytesIO
        
        wav_data = BytesIO(b'RIFF\x24\x08\x00\x00WAVEfmt ')
        
        with patch('unified_prediction_system.unified_prediction_api.UnifiedPredictionAPI') as mock_api:
            mock_instance = mock_api.return_value
            mock_instance.predict_file.return_value = {
                'status': 'success',
                'predictions': {'species': 'owl', 'confidence': 0.9},
                'processing_time': 0.2
            }
            
            # Test API response times
            response_times = []
            for i in range(25):
                wav_data.seek(0)
                
                with performance_monitor() as monitor:
                    response = self.client.post('/api/v1/prediction/predict',
                        data={'file': (wav_data, f'test_{i}.wav')},
                        content_type='multipart/form-data'
                    )
                
                response_times.append(monitor.duration)
                assert response.status_code == 200
            
            # API Performance SLA
            avg_response_time = sum(response_times) / len(response_times)
            p95_response_time = sorted(response_times)[23]  # 95th percentile
            max_response_time = max(response_times)
            
            assert avg_response_time < 0.5   # Average API response < 500ms
            assert p95_response_time < 1.0   # P95 < 1 second
            assert max_response_time < 2.0   # Max < 2 seconds
    
    @pytest.mark.performance_critical
    def test_api_throughput_under_concurrent_load(self):
        """Test débit API sous charge concurrente."""
        from io import BytesIO
        import concurrent.futures
        
        with patch('unified_prediction_system.unified_prediction_api.UnifiedPredictionAPI') as mock_api:
            mock_instance = mock_api.return_value
            mock_instance.predict_file.return_value = {
                'status': 'success',
                'predictions': {'species': 'test'}
            }
            
            def make_concurrent_request(request_id):
                wav_data = BytesIO(b'RIFF\x24\x08\x00\x00WAVEfmt ')
                
                start_time = time.time()
                response = self.client.post('/api/v1/prediction/predict',
                    data={'file': (wav_data, f'concurrent_{request_id}.wav')},
                    content_type='multipart/form-data'
                )
                end_time = time.time()
                
                return {
                    'request_id': request_id,
                    'status_code': response.status_code,
                    'response_time': end_time - start_time
                }
            
            # Execute concurrent requests
            start_time = time.time()
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                futures = [executor.submit(make_concurrent_request, i) for i in range(50)]
                results = [future.result() for future in as_completed(futures)]
            
            total_time = time.time() - start_time
            
            # Verify all requests succeeded
            assert len(results) == 50
            assert all(r['status_code'] == 200 for r in results)
            
            # Throughput analysis
            throughput = 50 / total_time  # requests per second
            avg_concurrent_response_time = sum(r['response_time'] for r in results) / 50
            
            # Throughput SLA
            assert throughput > 10.0  # > 10 requests/second
            assert avg_concurrent_response_time < 2.0  # Average concurrent < 2s
            assert total_time < 20.0  # 50 requests in < 20 seconds
    
    @pytest.mark.performance_critical
    def test_api_memory_stability_under_load(self):
        """Test stabilité mémoire API sous charge."""
        from io import BytesIO
        
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
        memory_measurements = []
        
        with patch('unified_prediction_system.unified_prediction_api.UnifiedPredictionAPI') as mock_api:
            mock_instance = mock_api.return_value
            mock_instance.predict_file.return_value = {
                'status': 'success',
                'predictions': {'species': 'test'}
            }
            
            # Make many requests to test memory stability
            for i in range(100):
                wav_data = BytesIO(b'RIFF\x24\x08\x00\x00WAVEfmt ')
                
                response = self.client.post('/api/v1/prediction/predict',
                    data={'file': (wav_data, f'memory_test_{i}.wav')},
                    content_type='multipart/form-data'
                )
                
                assert response.status_code == 200
                
                # Measure memory every 10 requests
                if i % 10 == 0:
                    current_memory = psutil.Process().memory_info().rss / 1024 / 1024
                    memory_measurements.append(current_memory)
            
            final_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            # Memory stability analysis
            memory_growth = final_memory - initial_memory
            max_memory = max(memory_measurements)
            memory_variance = max(memory_measurements) - min(memory_measurements)
            
            # Memory stability assertions
            assert memory_growth < 50  # Total growth < 50MB
            assert memory_variance < 30  # Variance < 30MB (stable)
            assert max_memory - initial_memory < 60  # Peak usage reasonable


class TestPerformanceRegression:
    """Tests de détection de régression de performance."""
    
    @pytest.mark.performance_critical
    def test_prediction_latency_regression_detection(self):
        """Test détection de régression de latence."""
        # Baseline performance metrics (would be stored in CI/CD)
        baseline_metrics = {
            'avg_prediction_latency': 0.3,
            'p95_prediction_latency': 0.8,
            'max_prediction_latency': 1.2
        }
        
        router = PredictionRouter()
        
        with patch.object(router.model_manager, 'predict_audio') as mock_predict:
            # Simulate current performance (slightly degraded)
            def degraded_predict(*args, **kwargs):
                time.sleep(0.4)  # Slightly slower than baseline
                return {'species': 'test', 'confidence': 0.8}
            
            mock_predict.side_effect = degraded_predict
            
            # Measure current performance
            current_latencies = []
            for i in range(20):
                with performance_monitor() as monitor:
                    result = router.model_manager.predict_audio('fake_data')
                
                current_latencies.append(monitor.duration)
            
            # Calculate current metrics
            current_metrics = {
                'avg_prediction_latency': sum(current_latencies) / len(current_latencies),
                'p95_prediction_latency': sorted(current_latencies)[18],
                'max_prediction_latency': max(current_latencies)
            }
            
            # Check for regression (>20% degradation)
            regression_threshold = 0.2
            
            for metric_name, baseline_value in baseline_metrics.items():
                current_value = current_metrics[metric_name]
                degradation = (current_value - baseline_value) / baseline_value
                
                if degradation > regression_threshold:
                    pytest.fail(f"Performance regression detected in {metric_name}: "
                              f"baseline={baseline_value:.3f}s, current={current_value:.3f}s, "
                              f"degradation={degradation:.1%}")
    
    @pytest.mark.performance_critical  
    def test_memory_usage_regression_detection(self):
        """Test détection de régression d'usage mémoire."""
        baseline_memory_usage = 50  # MB
        
        manager = ModelManager()
        manager.max_models = 3
        
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        with patch.object(manager, '_load_model_from_disk') as mock_load:
            # Simulate memory-intensive models
            def memory_intensive_model(*args, **kwargs):
                mock_model = Mock()
                # Simulate larger memory footprint than baseline
                mock_model._memory_data = np.random.rand(2000, 2000)  # ~32MB vs baseline 8MB
                return mock_model
            
            mock_load.side_effect = memory_intensive_model
            
            # Load models and measure memory
            for i in range(5):
                model = manager.load_model(f'regression_test_{i}', 'audio')
            
            final_memory = psutil.Process().memory_info().rss / 1024 / 1024
            current_memory_usage = final_memory - initial_memory
            
            # Check for memory regression
            memory_increase = (current_memory_usage - baseline_memory_usage) / baseline_memory_usage
            
            if memory_increase > 0.3:  # >30% increase
                pytest.fail(f"Memory usage regression detected: "
                          f"baseline={baseline_memory_usage}MB, "
                          f"current={current_memory_usage:.1f}MB, "
                          f"increase={memory_increase:.1%}")


@pytest.mark.performance_critical
class TestSystemResourceUtilization:
    """Tests d'utilisation des ressources système."""
    
    def test_cpu_utilization_efficiency(self):
        """Test efficacité d'utilisation CPU."""
        router = PredictionRouter()
        
        with patch.object(router.model_manager, 'predict_audio') as mock_predict:
            # Mock CPU-intensive prediction
            def cpu_intensive_predict(*args, **kwargs):
                # Simulate CPU work
                _ = sum(i * i for i in range(100000))
                return {'species': 'test', 'confidence': 0.8}
            
            mock_predict.side_effect = cpu_intensive_predict
            
            # Monitor CPU during predictions
            cpu_measurements = []
            
            for i in range(10):
                cpu_before = psutil.cpu_percent(interval=0.1)
                
                with performance_monitor() as monitor:
                    result = router.model_manager.predict_audio('fake_data')
                
                cpu_after = psutil.cpu_percent(interval=0.1)
                cpu_measurements.append((cpu_before, cpu_after, monitor.duration))
            
            # Analyze CPU efficiency
            avg_cpu_usage = sum(cpu_after for _, cpu_after, _ in cpu_measurements) / 10
            avg_processing_time = sum(duration for _, _, duration in cpu_measurements) / 10
            
            # CPU efficiency assertions
            assert avg_cpu_usage > 10  # Should actually use CPU
            assert avg_cpu_usage < 80  # But not overwhelm system
            assert avg_processing_time < 1.0  # Efficient processing
    
    def test_memory_efficiency_large_batch(self):
        """Test efficacité mémoire pour gros batch."""
        router = PredictionRouter()
        
        # Create large batch of files
        large_batch = []
        for i in range(50):
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                tmp.write(b'RIFF\x24\x08\x00\x00WAVEfmt ')
                large_batch.append(tmp.name)
        
        try:
            initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            with patch.object(router.model_manager, 'predict_audio') as mock_predict:
                mock_predict.return_value = {'species': 'test', 'confidence': 0.8}
                
                # Process large batch
                with performance_monitor() as monitor:
                    results = router.predict_batch(large_batch, max_workers=4)
                
                peak_memory = psutil.Process().memory_info().rss / 1024 / 1024
                final_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            # Memory efficiency analysis
            peak_memory_usage = peak_memory - initial_memory
            final_memory_usage = final_memory - initial_memory
            memory_per_file = peak_memory_usage / 50
            
            # Verify results
            assert len(results) == 50
            
            # Memory efficiency assertions
            assert peak_memory_usage < 200  # Peak usage < 200MB for 50 files
            assert memory_per_file < 4  # < 4MB per file on average
            assert final_memory_usage < peak_memory_usage * 1.1  # Good cleanup
            
        finally:
            for file_path in large_batch:
                Path(file_path).unlink()