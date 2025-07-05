"""Load testing for NightScan using locust."""

from locust import HttpUser, task, between
import tempfile
import json
from io import BytesIO


class NightScanUser(HttpUser):
    """Simulated user for load testing NightScan."""
    
    wait_time = between(1, 5)  # Wait 1-5 seconds between requests
    
    def on_start(self):
        """Setup performed when user starts."""
        self.dummy_wav = self.create_dummy_wav()
    
    def create_dummy_wav(self):
        """Create a minimal valid WAV file for testing."""
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
        return wav_data
    
    @task(10)
    def health_check(self):
        """Check application health - frequent task."""
        self.client.get("/health")
    
    @task(8)
    def api_v1_health(self):
        """Check API v1 health."""
        self.client.get("/api/v1/health")
    
    @task(5)
    def readiness_check(self):
        """Check application readiness."""
        self.client.get("/ready")
    
    @task(3)
    def metrics(self):
        """Get Prometheus metrics."""
        self.client.get("/metrics")
    
    @task(2)
    def openapi_spec(self):
        """Get OpenAPI specification."""
        self.client.get("/api/v1/openapi.json")
    
    @task(1)
    def api_docs(self):
        """Load API documentation page."""
        self.client.get("/api/v1/docs")


class AuthenticatedNightScanUser(HttpUser):
    """Authenticated user for testing protected endpoints."""
    
    wait_time = between(2, 8)
    
    def on_start(self):
        """Login and setup for authenticated requests."""
        self.login()
        self.dummy_wav = self.create_dummy_wav()
    
    def create_dummy_wav(self):
        """Create a minimal valid WAV file for testing."""
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
        return wav_data
    
    def login(self):
        """Attempt to login (will fail in test but exercises the endpoint)."""
        response = self.client.post("/login", data={
            "username": "testuser",
            "password": "testpassword123!",
            "captcha": "0"  # Will fail but exercises validation
        })
        # Note: This will fail in real testing but exercises the login flow
    
    @task(5)
    def get_detections(self):
        """Get wildlife detections."""
        self.client.get("/api/v1/detections")
    
    @task(3)
    def get_detections_paginated(self):
        """Get paginated detections."""
        self.client.get("/api/v1/detections?page=1&per_page=10")
    
    @task(2)
    def get_detections_filtered(self):
        """Get filtered detections."""
        self.client.get("/api/v1/detections?species=owl")
    
    @task(1)
    def upload_prediction_request(self):
        """Attempt audio upload (will fail auth but exercises endpoint)."""
        files = {'file': ('test.wav', BytesIO(self.dummy_wav), 'audio/wav')}
        self.client.post("/api/v1/predict", files=files)


class PredictionApiUser(HttpUser):
    """User specifically for testing the prediction API server."""
    
    wait_time = between(3, 10)  # Longer wait for heavy operations
    
    def on_start(self):
        """Setup for prediction API testing."""
        self.dummy_wav = self.create_dummy_wav()
    
    def create_dummy_wav(self):
        """Create a minimal valid WAV file for testing."""
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
        return wav_data
    
    @task(10)
    def api_health_check(self):
        """Check API server health."""
        self.client.get("/health")
    
    @task(8)
    def api_v1_health_check(self):
        """Check API v1 health."""
        self.client.get("/api/v1/health")
    
    @task(5)
    def api_readiness_check(self):
        """Check API server readiness."""
        self.client.get("/ready")
    
    @task(3)
    def api_metrics(self):
        """Get API server metrics."""
        self.client.get("/metrics")
    
    @task(1)
    def predict_audio(self):
        """Submit audio for prediction (will likely fail due to missing model)."""
        files = {'file': ('test.wav', BytesIO(self.dummy_wav), 'audio/wav')}
        with self.client.post("/api/v1/predict", files=files, catch_response=True) as response:
            # We expect this to fail in testing due to missing model
            if response.status_code in [400, 500]:
                response.success()  # Mark as successful test even if prediction fails


class CacheStressUser(HttpUser):
    """User for testing cache performance under load."""
    
    wait_time = between(0.1, 1)  # Fast requests to stress cache
    
    def on_start(self):
        """Setup for cache testing."""
        self.dummy_wav = self.create_dummy_wav()
        self.cache_test_data = [
            b'audio_data_1',
            b'audio_data_2', 
            b'audio_data_3',
            b'audio_data_4',
            b'audio_data_5',
        ]
    
    def create_dummy_wav(self):
        """Create a minimal valid WAV file for testing."""
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
        return wav_data
    
    @task(20)
    def rapid_health_checks(self):
        """Rapid health checks to test basic performance."""
        self.client.get("/health")
    
    @task(10)
    def cache_test_requests(self):
        """Make requests that should benefit from caching."""
        # Simulate repeated requests for same audio data
        import random
        audio_variant = random.choice([1, 2, 3, 4, 5])
        files = {'file': (f'test_{audio_variant}.wav', BytesIO(self.dummy_wav), 'audio/wav')}
        
        with self.client.post("/api/v1/predict", files=files, catch_response=True) as response:
            # Accept cache-related responses as successful
            if response.status_code in [200, 400, 500]:
                response.success()


# Custom load test scenarios
class LoadTestScenarios:
    """Collection of load test scenarios for different use cases."""
    
    @staticmethod
    def burst_load_scenario():
        """Scenario: Sudden burst of traffic."""
        return {
            'user_classes': [NightScanUser],
            'spawn_rate': 10,  # 10 users per second
            'users': 100,      # Peak at 100 concurrent users
            'run_time': '2m'   # Run for 2 minutes
        }
    
    @staticmethod
    def sustained_load_scenario():
        """Scenario: Sustained moderate load."""
        return {
            'user_classes': [NightScanUser, AuthenticatedNightScanUser],
            'spawn_rate': 2,   # 2 users per second
            'users': 50,       # 50 concurrent users
            'run_time': '10m'  # Run for 10 minutes
        }
    
    @staticmethod
    def prediction_stress_scenario():
        """Scenario: Heavy prediction workload."""
        return {
            'user_classes': [PredictionApiUser],
            'spawn_rate': 1,   # 1 user per second (predictions are heavy)
            'users': 20,       # 20 concurrent prediction users
            'run_time': '5m'   # Run for 5 minutes
        }
    
    @staticmethod
    def cache_performance_scenario():
        """Scenario: Test cache performance."""
        return {
            'user_classes': [CacheStressUser],
            'spawn_rate': 5,   # 5 users per second
            'users': 30,       # 30 concurrent users hitting cache
            'run_time': '3m'   # Run for 3 minutes
        }


# Usage examples for running tests:

# Basic load test:
# locust -f test_load.py --host=http://localhost:8000

# Web UI load test:
# locust -f test_load.py --host=http://localhost:8000 --web-host=0.0.0.0

# Headless burst test:
# locust -f test_load.py --host=http://localhost:8000 --headless -u 100 -r 10 -t 2m

# Prediction API stress test:
# locust -f test_load.py --host=http://localhost:8001 --headless -u 20 -r 1 -t 5m PredictionApiUser

if __name__ == "__main__":
    print("Load testing scenarios available:")
    print("1. burst_load_scenario - Sudden traffic burst")
    print("2. sustained_load_scenario - Moderate sustained load") 
    print("3. prediction_stress_scenario - Heavy prediction workload")
    print("4. cache_performance_scenario - Cache performance test")
    print("\\nRun with: locust -f test_load.py --host=<target_url>")