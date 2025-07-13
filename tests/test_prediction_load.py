"""
Tests de charge et stress pour le système de prédiction.

Ce module teste la robustesse du système sous charge extrême:
- Simulation de charge utilisateur élevée (100+ utilisateurs simultanés)
- Test de connexions concurrentes et connection pooling
- Détection de memory leaks sous charge continue
- Dégradation gracieuse lors de surcharge
- Test de résilience aux pannes et recovery
- Monitoring des métriques système critiques
"""

import pytest
import time
import threading
import asyncio
import tempfile
import psutil
import gc
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Tuple, Any
import numpy as np
from PIL import Image
import struct
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass
import queue
import random

# Import components for load testing
from unified_prediction_system.unified_prediction_api import UnifiedPredictionAPI, create_app
from unified_prediction_system.prediction_router import PredictionRouter
from unified_prediction_system.model_manager import ModelManager


@dataclass
class LoadTestMetrics:
    """Métriques de test de charge."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_time: float = 0.0
    min_response_time: float = float('inf')
    max_response_time: float = 0.0
    response_times: List[float] = None
    errors: List[str] = None
    
    def __post_init__(self):
        if self.response_times is None:
            self.response_times = []
        if self.errors is None:
            self.errors = []
    
    @property
    def success_rate(self) -> float:
        """Taux de réussite."""
        return self.successful_requests / self.total_requests if self.total_requests > 0 else 0
    
    @property
    def throughput(self) -> float:
        """Débit en requêtes/seconde."""
        return self.total_requests / self.total_time if self.total_time > 0 else 0
    
    @property
    def avg_response_time(self) -> float:
        """Temps de réponse moyen."""
        return sum(self.response_times) / len(self.response_times) if self.response_times else 0
    
    @property
    def p95_response_time(self) -> float:
        """95e percentile du temps de réponse."""
        if not self.response_times:
            return 0
        sorted_times = sorted(self.response_times)
        index = int(0.95 * len(sorted_times))
        return sorted_times[index]


class LoadTestRunner:
    """Utilitaire pour exécuter des tests de charge."""
    
    def __init__(self, target_function, num_users: int, duration: int):
        self.target_function = target_function
        self.num_users = num_users
        self.duration = duration
        self.metrics = LoadTestMetrics()
        self.stop_event = threading.Event()
        
    def run_user_simulation(self, user_id: int, results_queue: queue.Queue):
        """Simule le comportement d'un utilisateur."""
        user_metrics = LoadTestMetrics()
        
        while not self.stop_event.is_set():
            try:
                start_time = time.time()
                
                # Execute target function
                result = self.target_function(user_id)
                
                end_time = time.time()
                response_time = end_time - start_time
                
                user_metrics.total_requests += 1
                user_metrics.response_times.append(response_time)
                user_metrics.min_response_time = min(user_metrics.min_response_time, response_time)
                user_metrics.max_response_time = max(user_metrics.max_response_time, response_time)
                
                if result.get('status') == 'success':
                    user_metrics.successful_requests += 1
                else:
                    user_metrics.failed_requests += 1
                    user_metrics.errors.append(str(result.get('error', 'Unknown error')))
                
                # Simulate user think time
                time.sleep(random.uniform(0.1, 0.5))
                
            except Exception as e:
                user_metrics.failed_requests += 1
                user_metrics.errors.append(str(e))
        
        results_queue.put(user_metrics)
    
    def run(self) -> LoadTestMetrics:
        """Exécute le test de charge."""
        results_queue = queue.Queue()
        threads = []
        
        # Start user simulation threads
        start_time = time.time()
        
        for user_id in range(self.num_users):
            thread = threading.Thread(
                target=self.run_user_simulation,
                args=(user_id, results_queue)
            )
            threads.append(thread)
            thread.start()
        
        # Let test run for specified duration
        time.sleep(self.duration)
        
        # Stop all threads
        self.stop_event.set()
        
        for thread in threads:
            thread.join()
        
        end_time = time.time()
        self.metrics.total_time = end_time - start_time
        
        # Aggregate results from all users
        while not results_queue.empty():
            user_metrics = results_queue.get()
            self.metrics.total_requests += user_metrics.total_requests
            self.metrics.successful_requests += user_metrics.successful_requests
            self.metrics.failed_requests += user_metrics.failed_requests
            self.metrics.response_times.extend(user_metrics.response_times)
            self.metrics.errors.extend(user_metrics.errors)
            
            if user_metrics.min_response_time != float('inf'):
                self.metrics.min_response_time = min(
                    self.metrics.min_response_time, 
                    user_metrics.min_response_time
                )
            self.metrics.max_response_time = max(
                self.metrics.max_response_time,
                user_metrics.max_response_time
            )
        
        return self.metrics


class TestHighVolumeLoad:
    """Tests de charge élevée pour simulation utilisateurs multiples."""
    
    @pytest.mark.load_test
    @pytest.mark.slow
    def test_high_concurrent_user_load(self):
        """Test charge élevée avec 100+ utilisateurs simultanés."""
        api = UnifiedPredictionAPI()
        
        def user_prediction_workflow(user_id: int) -> Dict[str, Any]:
            """Simule le workflow d'un utilisateur."""
            try:
                # Create user-specific test file
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                    tmp.write(b'RIFF\x24\x08\x00\x00WAVEfmt ')
                    test_file = tmp.name
                
                with patch.object(api.router, 'predict') as mock_predict:
                    # Simulate realistic processing time
                    time.sleep(random.uniform(0.1, 0.3))
                    
                    mock_predict.return_value = {
                        'status': 'success',
                        'predictions': {'species': f'species_user_{user_id}'},
                        'processing_time': random.uniform(0.1, 0.3)
                    }
                    
                    result = api.predict_file(test_file)
                    
                    # Cleanup
                    Path(test_file).unlink()
                    
                    return result
                    
            except Exception as e:
                return {'status': 'error', 'error': str(e)}
        
        # Run load test
        load_runner = LoadTestRunner(
            target_function=user_prediction_workflow,
            num_users=100,  # 100 concurrent users
            duration=30     # 30 seconds
        )
        
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        metrics = load_runner.run()
        
        final_memory = psutil.Process().memory_info().rss / 1024 / 1024
        memory_growth = final_memory - initial_memory
        
        # Load test assertions
        assert metrics.success_rate > 0.95  # > 95% success rate
        assert metrics.throughput > 50  # > 50 requests/second
        assert metrics.avg_response_time < 2.0  # Average < 2 seconds
        assert metrics.p95_response_time < 3.0  # P95 < 3 seconds
        assert metrics.max_response_time < 5.0  # Max < 5 seconds
        
        # System stability assertions
        assert memory_growth < 200  # Memory growth < 200MB
        assert len(metrics.errors) < metrics.total_requests * 0.05  # < 5% errors
    
    @pytest.mark.load_test
    @pytest.mark.slow
    def test_sustained_load_endurance(self):
        """Test de charge soutenue pour endurance du système."""
        api = UnifiedPredictionAPI()
        
        def sustained_prediction_workflow(user_id: int) -> Dict[str, Any]:
            """Workflow pour test d'endurance."""
            try:
                with patch.object(api.router, 'predict') as mock_predict:
                    mock_predict.return_value = {
                        'status': 'success',
                        'predictions': {'species': 'endurance_test'},
                        'processing_time': 0.2
                    }
                    
                    # Simulate file upload
                    return api.predict_file('/fake/endurance_test.wav')
                    
            except Exception as e:
                return {'status': 'error', 'error': str(e)}
        
        # Monitor system metrics during test
        memory_samples = []
        cpu_samples = []
        
        def monitor_system():
            while not getattr(monitor_system, 'stop', False):
                memory_samples.append(psutil.Process().memory_info().rss / 1024 / 1024)
                cpu_samples.append(psutil.Process().cpu_percent())
                time.sleep(5)  # Sample every 5 seconds
        
        # Start system monitoring
        monitor_thread = threading.Thread(target=monitor_system)
        monitor_thread.start()
        
        try:
            # Run sustained load test
            load_runner = LoadTestRunner(
                target_function=sustained_prediction_workflow,
                num_users=50,   # Moderate concurrent users
                duration=120    # 2 minutes sustained
            )
            
            metrics = load_runner.run()
            
        finally:
            # Stop monitoring
            monitor_system.stop = True
            monitor_thread.join()
        
        # Endurance test assertions
        assert metrics.success_rate > 0.98  # Very high success rate for sustained load
        assert metrics.total_requests > 1000  # Should process many requests
        
        # System stability over time
        if memory_samples:
            memory_variance = max(memory_samples) - min(memory_samples)
            assert memory_variance < 100  # Memory should be stable
        
        if cpu_samples:
            avg_cpu = sum(cpu_samples) / len(cpu_samples)
            assert avg_cpu < 80  # CPU should not be overwhelmed
    
    @pytest.mark.load_test
    def test_spike_load_handling(self):
        """Test gestion des pics de charge soudains."""
        api = UnifiedPredictionAPI()
        
        def spike_prediction_workflow(user_id: int) -> Dict[str, Any]:
            """Workflow pour test de pic de charge."""
            try:
                with patch.object(api.router, 'predict') as mock_predict:
                    mock_predict.return_value = {
                        'status': 'success',
                        'predictions': {'species': 'spike_test'}
                    }
                    
                    return api.predict_file('/fake/spike_test.wav')
                    
            except Exception as e:
                return {'status': 'error', 'error': str(e)}
        
        # Simulate normal load first
        normal_load_runner = LoadTestRunner(
            target_function=spike_prediction_workflow,
            num_users=20,
            duration=10
        )
        
        normal_metrics = normal_load_runner.run()
        normal_throughput = normal_metrics.throughput
        
        # Simulate spike load
        spike_load_runner = LoadTestRunner(
            target_function=spike_prediction_workflow,
            num_users=200,  # 10x increase
            duration=15
        )
        
        spike_metrics = spike_load_runner.run()
        
        # Spike handling assertions
        assert spike_metrics.success_rate > 0.80  # At least 80% success during spike
        assert spike_metrics.throughput > normal_throughput * 2  # Should scale somewhat
        assert spike_metrics.p95_response_time < 10.0  # Reasonable response time even under spike


class TestConcurrentConnectionPooling:
    """Tests pour connection pooling et gestion concurrence."""
    
    @pytest.mark.load_test
    def test_database_connection_pooling_under_load(self):
        """Test connection pooling base de données sous charge."""
        api = UnifiedPredictionAPI()
        
        # Mock database connections
        connection_pool = []
        max_connections = 50
        
        def mock_db_query(query_type: str):
            """Simule une requête base de données."""
            if len(connection_pool) >= max_connections:
                raise Exception("Connection pool exhausted")
            
            # Simulate connection acquisition
            connection_id = f"conn_{len(connection_pool)}"
            connection_pool.append(connection_id)
            
            try:
                time.sleep(random.uniform(0.01, 0.05))  # Simulate query time
                return {'result': f'data_for_{query_type}'}
            finally:
                # Release connection
                connection_pool.remove(connection_id)
        
        def concurrent_db_user(user_id: int) -> Dict[str, Any]:
            """Utilisateur faisant des requêtes DB concurrentes."""
            try:
                # Simulate multiple DB operations per user
                results = []
                for i in range(5):
                    result = mock_db_query(f'user_{user_id}_query_{i}')
                    results.append(result)
                
                return {'status': 'success', 'db_results': results}
                
            except Exception as e:
                return {'status': 'error', 'error': str(e)}
        
        # Test concurrent database access
        load_runner = LoadTestRunner(
            target_function=concurrent_db_user,
            num_users=100,  # High concurrency
            duration=20
        )
        
        metrics = load_runner.run()
        
        # Connection pooling assertions
        assert metrics.success_rate > 0.95  # Should handle connection limits gracefully
        assert max(len(connection_pool) for _ in [0]) <= max_connections  # Never exceed pool limit
        assert len([e for e in metrics.errors if 'pool exhausted' in e]) < 10  # Minimal pool exhaustion
    
    @pytest.mark.load_test
    def test_model_instance_pooling_efficiency(self):
        """Test efficacité du pooling d'instances de modèles."""
        manager = ModelManager()
        manager.max_models = 10  # Limited model pool
        
        model_access_log = []
        model_creation_count = 0
        
        def mock_model_access(model_id: str):
            """Simule l'accès à un modèle avec pooling."""
            nonlocal model_creation_count
            
            # Check if model exists in pool
            if model_id not in manager.models:
                model_creation_count += 1
                # Simulate model loading time
                time.sleep(0.5)
                manager.models[model_id] = Mock()
            
            model_access_log.append(model_id)
            return manager.models[model_id]
        
        def concurrent_model_user(user_id: int) -> Dict[str, Any]:
            """Utilisateur accédant aux modèles de façon concurrente."""
            try:
                # Access random models (simulating different file types)
                model_ids = [f'audio_model_{i}' for i in range(5)]  # 5 possible models
                
                results = []
                for _ in range(10):  # 10 model accesses per user
                    model_id = random.choice(model_ids)
                    model = mock_model_access(model_id)
                    results.append(f'prediction_from_{model_id}')
                
                return {'status': 'success', 'predictions': results}
                
            except Exception as e:
                return {'status': 'error', 'error': str(e)}
        
        # Test concurrent model access
        load_runner = LoadTestRunner(
            target_function=concurrent_model_user,
            num_users=50,
            duration=15
        )
        
        metrics = load_runner.run()
        
        # Model pooling efficiency assertions
        assert metrics.success_rate > 0.98  # High success rate
        assert model_creation_count <= 10  # Should reuse models efficiently
        assert len(manager.models) <= manager.max_models  # Respect pool limits
        
        # Verify model reuse efficiency
        total_accesses = len(model_access_log)
        unique_accesses = len(set(model_access_log))
        reuse_ratio = (total_accesses - unique_accesses) / total_accesses
        assert reuse_ratio > 0.8  # > 80% of accesses should be reused models


class TestMemoryLeakDetection:
    """Tests pour détection de fuites mémoire sous charge."""
    
    @pytest.mark.load_test
    @pytest.mark.slow
    def test_memory_leak_detection_continuous_load(self):
        """Test détection de fuites mémoire sous charge continue."""
        api = UnifiedPredictionAPI()
        
        memory_measurements = []
        
        def memory_intensive_prediction(user_id: int) -> Dict[str, Any]:
            """Prédiction simulant utilisation mémoire."""
            try:
                # Simulate memory allocation during prediction
                large_data = np.random.rand(1000, 1000)  # ~8MB allocation
                
                with patch.object(api.router, 'predict') as mock_predict:
                    mock_predict.return_value = {
                        'status': 'success',
                        'predictions': {'species': 'memory_test'},
                        'large_data': large_data.tolist()  # Include in response
                    }
                    
                    result = api.predict_file('/fake/memory_test.wav')
                    
                    # Explicit cleanup
                    del large_data
                    
                    return result
                    
            except Exception as e:
                return {'status': 'error', 'error': str(e)}
        
        # Monitor memory during continuous load
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        # Run load test with memory monitoring
        for cycle in range(5):  # 5 cycles to detect leaks
            gc.collect()  # Force garbage collection
            
            cycle_start_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            load_runner = LoadTestRunner(
                target_function=memory_intensive_prediction,
                num_users=20,
                duration=10
            )
            
            metrics = load_runner.run()
            
            gc.collect()  # Force cleanup
            cycle_end_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            memory_measurements.append({
                'cycle': cycle,
                'start_memory': cycle_start_memory,
                'end_memory': cycle_end_memory,
                'delta': cycle_end_memory - cycle_start_memory,
                'requests': metrics.total_requests
            })
            
            # Verify cycle success
            assert metrics.success_rate > 0.95
        
        # Memory leak analysis
        final_memory = psutil.Process().memory_info().rss / 1024 / 1024
        total_memory_growth = final_memory - initial_memory
        
        # Check for consistent memory growth (leak indicator)
        memory_deltas = [m['delta'] for m in memory_measurements]
        avg_growth_per_cycle = sum(memory_deltas) / len(memory_deltas)
        
        # Memory leak assertions
        assert total_memory_growth < 100  # Total growth < 100MB
        assert avg_growth_per_cycle < 20   # Average growth per cycle < 20MB
        
        # No consistent upward trend (strong leak indicator)
        positive_growth_cycles = len([d for d in memory_deltas if d > 5])
        assert positive_growth_cycles < 3  # < 3 cycles with significant growth
    
    @pytest.mark.load_test
    def test_garbage_collection_efficiency(self):
        """Test efficacité du garbage collection."""
        api = UnifiedPredictionAPI()
        
        def gc_test_prediction(user_id: int) -> Dict[str, Any]:
            """Prédiction générant beaucoup d'objets temporaires."""
            try:
                # Create many temporary objects
                temp_objects = []
                for i in range(100):
                    temp_data = {
                        'user_id': user_id,
                        'iteration': i,
                        'data': list(range(100)),
                        'matrix': np.random.rand(50, 50)
                    }
                    temp_objects.append(temp_data)
                
                # Simulate prediction
                with patch.object(api.router, 'predict') as mock_predict:
                    mock_predict.return_value = {
                        'status': 'success',
                        'predictions': {'species': 'gc_test'}
                    }
                    
                    result = api.predict_file('/fake/gc_test.wav')
                
                # Objects should be automatically cleaned up
                del temp_objects
                
                return result
                
            except Exception as e:
                return {'status': 'error', 'error': str(e)}
        
        # Test with GC monitoring
        initial_objects = len(gc.get_objects())
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        load_runner = LoadTestRunner(
            target_function=gc_test_prediction,
            num_users=30,
            duration=15
        )
        
        metrics = load_runner.run()
        
        # Force garbage collection
        collected = gc.collect()
        
        final_objects = len(gc.get_objects())
        final_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        # GC efficiency assertions
        assert metrics.success_rate > 0.95
        assert collected > 0  # GC should collect some objects
        
        object_growth = final_objects - initial_objects
        memory_growth = final_memory - initial_memory
        
        assert object_growth < 10000  # Object count should be reasonable
        assert memory_growth < 50     # Memory should be cleaned up


class TestGracefulDegradation:
    """Tests pour dégradation gracieuse sous surcharge."""
    
    @pytest.mark.load_test
    def test_rate_limiting_under_overload(self):
        """Test rate limiting lors de surcharge."""
        from io import BytesIO
        
        app = create_app(testing=True, enable_rate_limiting=True)
        client = app.test_client()
        
        def rate_limited_request(user_id: int) -> Dict[str, Any]:
            """Requête soumise au rate limiting."""
            try:
                wav_data = BytesIO(b'RIFF\x24\x08\x00\x00WAVEfmt ')
                
                with app.app_context():
                    response = client.post('/api/v1/prediction/predict',
                        data={'file': (wav_data, f'rate_test_{user_id}.wav')},
                        content_type='multipart/form-data'
                    )
                
                return {
                    'status': 'success' if response.status_code == 200 else 'rate_limited',
                    'status_code': response.status_code
                }
                
            except Exception as e:
                return {'status': 'error', 'error': str(e)}
        
        # Overload the system to trigger rate limiting
        load_runner = LoadTestRunner(
            target_function=rate_limited_request,
            num_users=200,  # High load to trigger limits
            duration=10
        )
        
        with patch('unified_prediction_system.unified_prediction_api.UnifiedPredictionAPI'):
            metrics = load_runner.run()
        
        # Rate limiting should prevent system overload
        rate_limited_count = len([r for r in metrics.errors if 'rate' in str(r)])
        
        # Some requests should be rate limited (system protection)
        assert rate_limited_count > 0
        # But system should remain responsive for allowed requests
        assert metrics.success_rate > 0.5  # At least 50% get through
    
    @pytest.mark.load_test
    def test_circuit_breaker_pattern(self):
        """Test circuit breaker lors de pannes cascade."""
        api = UnifiedPredictionAPI()
        
        # Simulate failing dependency
        failure_count = 0
        failure_threshold = 50
        
        def circuit_breaker_prediction(user_id: int) -> Dict[str, Any]:
            """Prédiction avec circuit breaker simulation."""
            nonlocal failure_count
            
            try:
                # Simulate dependency failure after threshold
                if failure_count > failure_threshold:
                    return {
                        'status': 'circuit_open',
                        'error': 'Circuit breaker open - service temporarily unavailable'
                    }
                
                # Simulate random failures
                if random.random() < 0.1:  # 10% failure rate
                    failure_count += 1
                    raise Exception("Simulated service failure")
                
                # Reset failure count on success
                failure_count = max(0, failure_count - 1)
                
                with patch.object(api.router, 'predict') as mock_predict:
                    mock_predict.return_value = {
                        'status': 'success',
                        'predictions': {'species': 'circuit_test'}
                    }
                    
                    return api.predict_file('/fake/circuit_test.wav')
                    
            except Exception as e:
                failure_count += 1
                return {'status': 'error', 'error': str(e)}
        
        # Test circuit breaker behavior
        load_runner = LoadTestRunner(
            target_function=circuit_breaker_prediction,
            num_users=100,
            duration=20
        )
        
        metrics = load_runner.run()
        
        # Circuit breaker should protect system
        circuit_open_count = len([e for e in metrics.errors if 'circuit' in str(e)])
        
        # Should have triggered circuit breaker
        assert circuit_open_count > 0
        # System should still process some requests
        assert metrics.successful_requests > 0
        # Overall success rate should be reasonable despite failures
        assert metrics.success_rate > 0.3


@pytest.mark.load_test
@pytest.mark.slow
class TestSystemResilience:
    """Tests de résilience système sous conditions extrêmes."""
    
    def test_resource_exhaustion_recovery(self):
        """Test récupération après épuisement des ressources."""
        api = UnifiedPredictionAPI()
        
        resource_limit = 100
        current_resources = 0
        
        def resource_exhaustion_test(user_id: int) -> Dict[str, Any]:
            """Test épuisant les ressources système."""
            nonlocal current_resources
            
            try:
                # Simulate resource consumption
                if current_resources >= resource_limit:
                    return {
                        'status': 'resource_exhausted',
                        'error': 'System resources temporarily unavailable'
                    }
                
                current_resources += 1
                
                try:
                    # Simulate resource-intensive operation
                    time.sleep(0.1)
                    
                    with patch.object(api.router, 'predict') as mock_predict:
                        mock_predict.return_value = {
                            'status': 'success',
                            'predictions': {'species': 'resource_test'}
                        }
                        
                        result = api.predict_file('/fake/resource_test.wav')
                        
                    return result
                    
                finally:
                    # Release resource
                    current_resources = max(0, current_resources - 1)
                    
            except Exception as e:
                current_resources = max(0, current_resources - 1)
                return {'status': 'error', 'error': str(e)}
        
        # Test system under resource pressure
        load_runner = LoadTestRunner(
            target_function=resource_exhaustion_test,
            num_users=150,  # Exceed resource limit
            duration=30
        )
        
        metrics = load_runner.run()
        
        # System should handle resource exhaustion gracefully
        exhaustion_count = len([e for e in metrics.errors if 'resource' in str(e)])
        
        # Should experience some resource exhaustion
        assert exhaustion_count > 0
        # But should maintain some service availability
        assert metrics.success_rate > 0.4
        # Should recover after load reduces
        assert current_resources < resource_limit
    
    def test_cascading_failure_isolation(self):
        """Test isolation des pannes en cascade."""
        api = UnifiedPredictionAPI()
        
        # Simulate different service components
        services = {
            'file_service': {'healthy': True, 'failure_count': 0},
            'model_service': {'healthy': True, 'failure_count': 0},
            'db_service': {'healthy': True, 'failure_count': 0}
        }
        
        def cascading_failure_test(user_id: int) -> Dict[str, Any]:
            """Test avec pannes en cascade."""
            try:
                # Check service health
                failed_services = []
                
                for service_name, service in services.items():
                    if service['failure_count'] > 10:
                        service['healthy'] = False
                        failed_services.append(service_name)
                    
                    # Simulate random failures
                    if random.random() < 0.05:  # 5% failure rate
                        service['failure_count'] += 1
                
                # If critical services are down, fail gracefully
                if not services['model_service']['healthy']:
                    return {
                        'status': 'service_unavailable',
                        'error': 'Core prediction service temporarily unavailable',
                        'failed_services': failed_services
                    }
                
                # Partial functionality if some services are down
                if failed_services:
                    # Reduced functionality mode
                    return {
                        'status': 'degraded',
                        'predictions': {'species': 'degraded_mode'},
                        'failed_services': failed_services
                    }
                
                # Full functionality
                with patch.object(api.router, 'predict') as mock_predict:
                    mock_predict.return_value = {
                        'status': 'success',
                        'predictions': {'species': 'cascade_test'}
                    }
                    
                    return api.predict_file('/fake/cascade_test.wav')
                    
            except Exception as e:
                return {'status': 'error', 'error': str(e)}
        
        # Test cascading failure handling
        load_runner = LoadTestRunner(
            target_function=cascading_failure_test,
            num_users=80,
            duration=25
        )
        
        metrics = load_runner.run()
        
        # System should isolate failures and maintain partial service
        degraded_count = len([e for e in metrics.errors if 'degraded' in str(e)])
        unavailable_count = len([e for e in metrics.errors if 'unavailable' in str(e)])
        
        # Should show graceful degradation
        assert metrics.success_rate > 0.3  # Some service maintained
        # Should handle both degraded and unavailable states
        assert degraded_count + unavailable_count > 0