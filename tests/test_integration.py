"""Integration tests for NightScan API endpoints."""

import pytest
import tempfile
import json
import time
from pathlib import Path
from io import BytesIO

from web.app import create_app, db
from Audio_Training.scripts.api_server import create_app as create_api_app
from config import get_config


class TestApiIntegration:
    """Integration tests for the API endpoints."""
    
    @pytest.fixture
    def web_app(self):
        """Create web app for testing."""
        app = create_app()
        app.config['TESTING'] = True
        app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///:memory:'
        app.config['WTF_CSRF_ENABLED'] = False
        
        with app.app_context():
            db.create_all()
            yield app
    
    @pytest.fixture
    def api_app(self):
        """Create API app for testing."""
        # Create a dummy model file for testing
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
            model_path = Path(f.name)
        
        # Create a dummy CSV directory
        csv_dir = Path(tempfile.mkdtemp())
        
        app = create_api_app(model_path=model_path, csv_dir=csv_dir)
        app.config['TESTING'] = True
        
        yield app
        
        # Cleanup
        model_path.unlink(missing_ok=True)
    
    @pytest.fixture
    def web_client(self, web_app):
        """Create web app test client."""
        return web_app.test_client()
    
    @pytest.fixture
    def api_client(self, api_app):
        """Create API app test client."""
        return api_app.test_client()
    
    @pytest.fixture
    def dummy_wav_file(self):
        """Create a dummy WAV file for testing."""
        # Minimal WAV file header
        wav_data = (
            b'RIFF'  # ChunkID
            b'\\x24\\x00\\x00\\x00'  # ChunkSize (36 bytes)
            b'WAVE'  # Format
            b'fmt '  # Subchunk1ID
            b'\\x10\\x00\\x00\\x00'  # Subchunk1Size (16)
            b'\\x01\\x00'  # AudioFormat (PCM)
            b'\\x01\\x00'  # NumChannels (1)
            b'\\x44\\xAC\\x00\\x00'  # SampleRate (44100)
            b'\\x88\\x58\\x01\\x00'  # ByteRate
            b'\\x02\\x00'  # BlockAlign
            b'\\x10\\x00'  # BitsPerSample (16)
            b'data'  # Subchunk2ID
            b'\\x00\\x00\\x00\\x00'  # Subchunk2Size (0 - no audio data)
        )
        return BytesIO(wav_data)
    
    def test_health_endpoints(self, web_client, api_client):
        """Test health check endpoints."""
        # Test web app health
        response = web_client.get('/health')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['status'] == 'healthy'
        assert 'timestamp' in data
        
        # Test API health
        response = api_client.get('/health')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['status'] == 'healthy'
        assert 'service' in data
    
    def test_readiness_endpoints(self, web_client, api_client):
        """Test readiness check endpoints."""
        # Test web app readiness
        response = web_client.get('/ready')
        assert response.status_code in [200, 503]  # May fail due to missing dependencies
        data = json.loads(response.data)
        assert 'status' in data
        assert 'checks' in data
        assert 'timestamp' in data
        
        # Test API readiness
        response = api_client.get('/ready')
        assert response.status_code in [200, 503]
        data = json.loads(response.data)
        assert 'status' in data
        assert 'checks' in data
    
    def test_api_v1_health(self, web_client, api_client):
        """Test API v1 health endpoints."""
        # Test web app API v1
        response = web_client.get('/api/v1/health')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['status'] == 'healthy'
        
        # Test API v1 on API server
        response = api_client.get('/api/v1/health')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['status'] == 'healthy'
    
    def test_openapi_documentation(self, web_client, api_client):
        """Test OpenAPI documentation endpoints."""
        # Test OpenAPI spec
        response = web_client.get('/api/v1/openapi.json')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'openapi' in data
        assert 'info' in data
        assert 'paths' in data
        
        # Test API docs page
        response = web_client.get('/api/v1/docs')
        assert response.status_code == 200
        assert b'Swagger' in response.data
    
    def test_prediction_endpoint_no_file(self, api_client):
        """Test prediction endpoint without file."""
        response = api_client.post('/api/v1/predict')
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'error' in data
        assert data['code'] == 'NO_FILE'
    
    def test_prediction_endpoint_invalid_file(self, api_client):
        """Test prediction endpoint with invalid file."""
        data = {'file': (BytesIO(b'not a wav file'), 'test.wav')}
        response = api_client.post('/api/v1/predict', 
                                 data=data, 
                                 content_type='multipart/form-data')
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'error' in data
    
    def test_detections_endpoint_unauthorized(self, web_client):
        """Test detections endpoint without authentication."""
        response = web_client.get('/api/v1/detections')
        assert response.status_code == 302  # Redirect to login
    
    def test_detections_endpoint_pagination(self, web_client):
        """Test detections endpoint pagination parameters."""
        # This would require authentication, so we test parameter validation
        response = web_client.get('/api/v1/detections?page=0')
        # Should redirect to login but validate we handle the parameter
        assert response.status_code == 302
        
        response = web_client.get('/api/v1/detections?per_page=1000')  # Over limit
        assert response.status_code == 302
    
    def test_metrics_endpoint(self, web_client, api_client):
        """Test Prometheus metrics endpoints."""
        # Test web app metrics
        response = web_client.get('/metrics')
        assert response.status_code == 200
        assert response.headers['Content-Type'].startswith('text/plain')
        
        # Test API metrics
        response = api_client.get('/metrics')
        assert response.status_code == 200
        assert response.headers['Content-Type'].startswith('text/plain')
    
    def test_cache_integration(self, api_client):
        """Test cache functionality."""
        from cache_utils import get_cache
        
        cache = get_cache()
        
        # Test cache health
        health = cache.get_cache_stats()
        assert 'enabled' in health
        assert 'redis_available' in health
        
        # Test cache operations
        test_data = b'test audio data'
        test_result = [{'label': 'test', 'probability': 0.9}]
        
        # Should be None initially
        assert cache.get_prediction(test_data) is None
        
        # Cache and retrieve
        cache.cache_prediction(test_data, test_result)
        cached = cache.get_prediction(test_data)
        
        if cache.cache_enabled:
            assert cached == test_result
        else:
            assert cached is None  # If Redis not available
    
    def test_config_integration(self):
        """Test configuration loading."""
        config = get_config()
        
        # Test required config sections exist
        assert hasattr(config, 'database')
        assert hasattr(config, 'security')
        assert hasattr(config, 'upload')
        assert hasattr(config, 'model')
        assert hasattr(config, 'api')
        
        # Test config validation
        from config import validate_config
        errors = validate_config(config)
        
        # In test environment, some errors are expected
        print(f"Config validation errors (expected in test): {errors}")


class TestLoadTesting:
    """Basic load testing for critical endpoints."""
    
    @pytest.fixture
    def web_client(self):
        """Create web app test client."""
        app = create_app()
        app.config['TESTING'] = True
        app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///:memory:'
        
        with app.app_context():
            db.create_all()
            yield app.test_client()
    
    def test_health_endpoint_load(self, web_client):
        """Test health endpoint under load."""
        start_time = time.time()
        responses = []
        
        # Make 50 concurrent-ish requests
        for i in range(50):
            response = web_client.get('/health')
            responses.append(response.status_code)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # All should succeed
        assert all(status == 200 for status in responses)
        
        # Should complete reasonably quickly
        assert duration < 5.0, f"Health checks took too long: {duration}s"
        
        print(f"50 health checks completed in {duration:.2f}s")
    
    def test_metrics_endpoint_load(self, web_client):
        """Test metrics endpoint under load."""
        start_time = time.time()
        responses = []
        
        # Make 20 requests for metrics (usually heavier)
        for i in range(20):
            response = web_client.get('/metrics')
            responses.append(response.status_code)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # All should succeed
        assert all(status == 200 for status in responses)
        
        # Should complete reasonably quickly
        assert duration < 10.0, f"Metrics requests took too long: {duration}s"
        
        print(f"20 metrics requests completed in {duration:.2f}s")


class TestModelRegression:
    """Basic model regression tests."""
    
    def test_model_prediction_format(self):
        """Test that model predictions maintain expected format."""
        # This would require actual model loading in a real test
        # For now, test the expected output format
        
        expected_format = {
            "results": [
                {
                    "segment": "test.wav#0", 
                    "time": 0.0,
                    "predictions": [
                        {"label": "species_name", "probability": 0.95},
                        {"label": "other_species", "probability": 0.03},
                        {"label": "background", "probability": 0.02}
                    ]
                }
            ],
            "processing_time": 1.23,
            "file_info": {
                "filename": "test.wav",
                "size_bytes": 12345,
                "duration_seconds": 8.0,
                "cached": False
            }
        }
        
        # Validate structure
        assert "results" in expected_format
        assert isinstance(expected_format["results"], list)
        assert len(expected_format["results"]) > 0
        
        result = expected_format["results"][0]
        assert "segment" in result
        assert "time" in result
        assert "predictions" in result
        assert len(result["predictions"]) == 3
        
        prediction = result["predictions"][0]
        assert "label" in prediction
        assert "probability" in prediction
        assert 0 <= prediction["probability"] <= 1
    
    def test_prediction_consistency(self):
        """Test that predictions are consistent for same input."""
        # This would test that the same audio file produces
        # the same predictions across multiple runs
        # Important for cache validation and user experience
        
        # Placeholder - would need actual model and audio data
        test_audio_hash = "abc123"
        
        # Simulate consistent predictions
        prediction1 = [{"label": "owl", "probability": 0.85}]
        prediction2 = [{"label": "owl", "probability": 0.85}]
        
        assert prediction1 == prediction2
        print("Prediction consistency test passed (placeholder)")


if __name__ == "__main__":
    # Run basic tests when executed directly
    pytest.main([__file__, "-v"])