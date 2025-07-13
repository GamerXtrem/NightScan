"""
Tests pour l'API unifiée de prédiction.

Ce module teste tous les endpoints et fonctionnalités de l'API:
- Endpoints REST pour upload et prédiction
- Validation des requêtes et paramètres
- Formats de réponse et codes de statut
- Authentification et autorisation
- Rate limiting et quotas utilisateurs
- Gestion d'erreurs et responses d'erreur
- Performance des endpoints sous charge
"""

import pytest
import json
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any
import numpy as np
from PIL import Image
import struct
from io import BytesIO

# Import Flask testing utilities
from flask import Flask
from werkzeug.test import Client
from werkzeug.datastructures import FileStorage

# Import the API components to test
from unified_prediction_system.unified_prediction_api import UnifiedPredictionAPI, create_app


class TestUnifiedPredictionAPIEndpoints:
    """Tests pour les endpoints de l'API unifiée."""
    
    def setup_method(self):
        """Setup pour chaque test."""
        self.app = create_app(testing=True)
        self.client = self.app.test_client()
        self.app_context = self.app.app_context()
        self.app_context.push()
    
    def teardown_method(self):
        """Cleanup après chaque test."""
        self.app_context.pop()
    
    def test_health_check_endpoint(self):
        """Test endpoint de vérification de santé."""
        response = self.client.get('/api/v1/prediction/health')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['status'] == 'healthy'
        assert 'version' in data
        assert 'models_loaded' in data
        assert 'uptime' in data
    
    def test_predict_endpoint_audio_file(self):
        """Test endpoint de prédiction avec fichier audio."""
        # Create test WAV file
        wav_data = BytesIO()
        wav_data.write(b'RIFF')
        wav_data.write(struct.pack('<L', 44))
        wav_data.write(b'WAVEfmt ')
        wav_data.write(struct.pack('<L', 16))
        wav_data.write(struct.pack('<H', 1))   # PCM
        wav_data.write(struct.pack('<H', 1))   # Mono
        wav_data.write(struct.pack('<L', 44100))  # Sample rate
        wav_data.write(struct.pack('<L', 88200))  # Byte rate
        wav_data.write(struct.pack('<H', 2))   # Block align
        wav_data.write(struct.pack('<H', 16))  # Bits per sample
        wav_data.write(b'data')
        wav_data.write(struct.pack('<L', 0))   # Data size
        wav_data.seek(0)
        
        with patch('unified_prediction_system.unified_prediction_api.UnifiedPredictionAPI') as mock_api:
            # Mock prediction result
            mock_instance = mock_api.return_value
            mock_instance.predict_file.return_value = {
                'status': 'success',
                'prediction_id': 'pred_123',
                'file_type': 'audio',
                'predictions': {
                    'species': 'owl',
                    'confidence': 0.89,
                    'predictions': [
                        {'class': 'owl', 'confidence': 0.89},
                        {'class': 'wind', 'confidence': 0.11}
                    ]
                },
                'metadata': {
                    'duration': 1.0,
                    'sample_rate': 44100
                },
                'processing_time': 0.234
            }
            
            # Make request
            response = self.client.post('/api/v1/prediction/predict', 
                data={
                    'file': (wav_data, 'test.wav'),
                    'model_type': 'auto',
                    'confidence_threshold': '0.5'
                },
                content_type='multipart/form-data'
            )
            
            assert response.status_code == 200
            data = json.loads(response.data)
            assert data['status'] == 'success'
            assert data['file_type'] == 'audio'
            assert data['predictions']['species'] == 'owl'
            assert data['predictions']['confidence'] == 0.89
            assert 'prediction_id' in data
            assert 'processing_time' in data
    
    def test_predict_endpoint_image_file(self):
        """Test endpoint de prédiction avec fichier image."""
        # Create test image
        img = Image.new('RGB', (640, 480), color='red')
        img_data = BytesIO()
        img.save(img_data, format='JPEG')
        img_data.seek(0)
        
        with patch('unified_prediction_system.unified_prediction_api.UnifiedPredictionAPI') as mock_api:
            mock_instance = mock_api.return_value
            mock_instance.predict_file.return_value = {
                'status': 'success',
                'prediction_id': 'pred_456',
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
                    'format': 'JPEG'
                },
                'processing_time': 0.156
            }
            
            response = self.client.post('/api/v1/prediction/predict',
                data={
                    'file': (img_data, 'test.jpg'),
                    'model_type': 'photo'
                },
                content_type='multipart/form-data'
            )
            
            assert response.status_code == 200
            data = json.loads(response.data)
            assert data['status'] == 'success'
            assert data['file_type'] == 'photo'
            assert data['predictions']['species'] == 'deer'
            assert len(data['predictions']['bounding_boxes']) == 1
    
    def test_predict_endpoint_no_file(self):
        """Test endpoint de prédiction sans fichier."""
        response = self.client.post('/api/v1/prediction/predict',
            data={'model_type': 'auto'},
            content_type='multipart/form-data'
        )
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert data['status'] == 'error'
        assert 'No file provided' in data['error']
    
    def test_predict_endpoint_invalid_file_type(self):
        """Test endpoint avec type de fichier non supporté."""
        text_data = BytesIO(b'This is not an audio or image file')
        text_data.seek(0)
        
        with patch('unified_prediction_system.unified_prediction_api.UnifiedPredictionAPI') as mock_api:
            mock_instance = mock_api.return_value
            mock_instance.predict_file.side_effect = ValueError("Type de fichier non supporté")
            
            response = self.client.post('/api/v1/prediction/predict',
                data={'file': (text_data, 'test.txt')},
                content_type='multipart/form-data'
            )
            
            assert response.status_code == 400
            data = json.loads(response.data)
            assert data['status'] == 'error'
            assert 'Type de fichier non supporté' in data['error']
    
    def test_batch_predict_endpoint(self):
        """Test endpoint de prédiction batch."""
        # Create multiple test files
        files_data = []
        
        # Audio file
        wav_data = BytesIO()
        wav_data.write(b'RIFF\x24\x08\x00\x00WAVEfmt ')
        wav_data.seek(0)
        files_data.append(('files', (wav_data, 'audio1.wav')))
        
        # Image file
        img = Image.new('RGB', (100, 100), color='blue')
        img_data = BytesIO()
        img.save(img_data, format='JPEG')
        img_data.seek(0)
        files_data.append(('files', (img_data, 'image1.jpg')))
        
        with patch('unified_prediction_system.unified_prediction_api.UnifiedPredictionAPI') as mock_api:
            mock_instance = mock_api.return_value
            mock_instance.predict_batch.return_value = [
                {
                    'status': 'success',
                    'file_type': 'audio',
                    'predictions': {'species': 'owl', 'confidence': 0.8}
                },
                {
                    'status': 'success', 
                    'file_type': 'photo',
                    'predictions': {'species': 'fox', 'confidence': 0.9}
                }
            ]
            
            response = self.client.post('/api/v1/prediction/predict/batch',
                data=files_data,
                content_type='multipart/form-data'
            )
            
            assert response.status_code == 200
            data = json.loads(response.data)
            assert data['status'] == 'success'
            assert len(data['results']) == 2
            assert data['results'][0]['file_type'] == 'audio'
            assert data['results'][1]['file_type'] == 'photo'
    
    def test_get_prediction_status_endpoint(self):
        """Test endpoint de statut de prédiction."""
        prediction_id = 'pred_123'
        
        with patch('unified_prediction_system.unified_prediction_api.UnifiedPredictionAPI') as mock_api:
            mock_instance = mock_api.return_value
            mock_instance.get_prediction_status.return_value = {
                'prediction_id': prediction_id,
                'status': 'processing',
                'progress': 65,
                'stage': 'feature_extraction',
                'estimated_completion': '2024-01-15T10:35:00Z'
            }
            
            response = self.client.get(f'/api/v1/prediction/status/{prediction_id}')
            
            assert response.status_code == 200
            data = json.loads(response.data)
            assert data['prediction_id'] == prediction_id
            assert data['status'] == 'processing'
            assert data['progress'] == 65
    
    def test_get_prediction_result_endpoint(self):
        """Test endpoint de récupération de résultat."""
        prediction_id = 'pred_456'
        
        with patch('unified_prediction_system.unified_prediction_api.UnifiedPredictionAPI') as mock_api:
            mock_instance = mock_api.return_value
            mock_instance.get_prediction_result.return_value = {
                'prediction_id': prediction_id,
                'status': 'completed',
                'file_type': 'audio',
                'predictions': {
                    'species': 'owl',
                    'confidence': 0.92
                },
                'metadata': {
                    'duration': 5.2,
                    'sample_rate': 44100
                },
                'completed_at': '2024-01-15T10:30:15Z'
            }
            
            response = self.client.get(f'/api/v1/prediction/result/{prediction_id}')
            
            assert response.status_code == 200
            data = json.loads(response.data)
            assert data['prediction_id'] == prediction_id
            assert data['status'] == 'completed'
            assert data['predictions']['species'] == 'owl'
    
    def test_get_prediction_result_not_found(self):
        """Test endpoint avec prédiction inexistante."""
        prediction_id = 'nonexistent_pred'
        
        with patch('unified_prediction_system.unified_prediction_api.UnifiedPredictionAPI') as mock_api:
            mock_instance = mock_api.return_value
            mock_instance.get_prediction_result.side_effect = KeyError("Prediction not found")
            
            response = self.client.get(f'/api/v1/prediction/result/{prediction_id}')
            
            assert response.status_code == 404
            data = json.loads(response.data)
            assert data['status'] == 'error'
            assert 'not found' in data['error'].lower()
    
    def test_list_predictions_endpoint(self):
        """Test endpoint de listing des prédictions."""
        with patch('unified_prediction_system.unified_prediction_api.UnifiedPredictionAPI') as mock_api:
            mock_instance = mock_api.return_value
            mock_instance.list_predictions.return_value = {
                'predictions': [
                    {
                        'prediction_id': 'pred_001',
                        'status': 'completed',
                        'file_type': 'audio',
                        'species': 'owl',
                        'created_at': '2024-01-15T10:30:00Z'
                    },
                    {
                        'prediction_id': 'pred_002',
                        'status': 'processing',
                        'file_type': 'photo',
                        'created_at': '2024-01-15T10:25:00Z'
                    }
                ],
                'pagination': {
                    'page': 1,
                    'per_page': 20,
                    'total': 2,
                    'has_next': False
                }
            }
            
            response = self.client.get('/api/v1/prediction/predictions?page=1&per_page=20')
            
            assert response.status_code == 200
            data = json.loads(response.data)
            assert len(data['predictions']) == 2
            assert data['pagination']['total'] == 2
    
    def test_delete_prediction_endpoint(self):
        """Test endpoint de suppression de prédiction."""
        prediction_id = 'pred_to_delete'
        
        with patch('unified_prediction_system.unified_prediction_api.UnifiedPredictionAPI') as mock_api:
            mock_instance = mock_api.return_value
            mock_instance.delete_prediction.return_value = {
                'prediction_id': prediction_id,
                'status': 'deleted',
                'deleted_at': '2024-01-15T10:40:00Z'
            }
            
            response = self.client.delete(f'/api/v1/prediction/predictions/{prediction_id}')
            
            assert response.status_code == 200
            data = json.loads(response.data)
            assert data['prediction_id'] == prediction_id
            assert data['status'] == 'deleted'


class TestAPIAuthentication:
    """Tests pour l'authentification et autorisation."""
    
    def setup_method(self):
        """Setup avec authentification activée."""
        self.app = create_app(testing=True, require_auth=True)
        self.client = self.app.test_client()
        self.app_context = self.app.app_context()
        self.app_context.push()
    
    def teardown_method(self):
        """Cleanup après chaque test."""
        self.app_context.pop()
    
    def test_prediction_requires_authentication(self):
        """Test que la prédiction nécessite une authentification."""
        wav_data = BytesIO(b'RIFF\x24\x08\x00\x00WAVEfmt ')
        
        response = self.client.post('/api/v1/prediction/predict',
            data={'file': (wav_data, 'test.wav')},
            content_type='multipart/form-data'
        )
        
        assert response.status_code == 401
        data = json.loads(response.data)
        assert data['status'] == 'error'
        assert 'authentication' in data['error'].lower()
    
    def test_prediction_with_valid_token(self):
        """Test prédiction avec token valide."""
        wav_data = BytesIO(b'RIFF\x24\x08\x00\x00WAVEfmt ')
        
        with patch('unified_prediction_system.unified_prediction_api.verify_token') as mock_verify:
            mock_verify.return_value = {'user_id': 'user123', 'valid': True}
            
            with patch('unified_prediction_system.unified_prediction_api.UnifiedPredictionAPI') as mock_api:
                mock_instance = mock_api.return_value
                mock_instance.predict_file.return_value = {
                    'status': 'success',
                    'predictions': {'species': 'owl'}
                }
                
                response = self.client.post('/api/v1/prediction/predict',
                    data={'file': (wav_data, 'test.wav')},
                    content_type='multipart/form-data',
                    headers={'Authorization': 'Bearer valid_token_123'}
                )
                
                assert response.status_code == 200
                data = json.loads(response.data)
                assert data['status'] == 'success'
    
    def test_prediction_with_invalid_token(self):
        """Test prédiction avec token invalide."""
        wav_data = BytesIO(b'RIFF\x24\x08\x00\x00WAVEfmt ')
        
        with patch('unified_prediction_system.unified_prediction_api.verify_token') as mock_verify:
            mock_verify.return_value = {'valid': False, 'error': 'Token expired'}
            
            response = self.client.post('/api/v1/prediction/predict',
                data={'file': (wav_data, 'test.wav')},
                content_type='multipart/form-data',
                headers={'Authorization': 'Bearer invalid_token'}
            )
            
            assert response.status_code == 401
            data = json.loads(response.data)
            assert data['status'] == 'error'
            assert 'token' in data['error'].lower()


class TestAPIRateLimiting:
    """Tests pour le rate limiting et quotas."""
    
    def setup_method(self):
        """Setup avec rate limiting activé."""
        self.app = create_app(testing=True, enable_rate_limiting=True)
        self.client = self.app.test_client()
        self.app_context = self.app.app_context()
        self.app_context.push()
    
    def teardown_method(self):
        """Cleanup après chaque test."""
        self.app_context.pop()
    
    def test_rate_limiting_exceeded(self):
        """Test dépassement de limite de taux."""
        wav_data = BytesIO(b'RIFF\x24\x08\x00\x00WAVEfmt ')
        
        with patch('unified_prediction_system.unified_prediction_api.check_rate_limit') as mock_rate_limit:
            # First few requests succeed
            mock_rate_limit.side_effect = [True, True, True, False]  # 4th request fails
            
            with patch('unified_prediction_system.unified_prediction_api.UnifiedPredictionAPI') as mock_api:
                mock_instance = mock_api.return_value
                mock_instance.predict_file.return_value = {'status': 'success'}
                
                # Make requests until rate limit hit
                responses = []
                for i in range(4):
                    wav_data.seek(0)
                    response = self.client.post('/api/v1/prediction/predict',
                        data={'file': (wav_data, f'test_{i}.wav')},
                        content_type='multipart/form-data'
                    )
                    responses.append(response)
                
                # First 3 should succeed
                for i in range(3):
                    assert responses[i].status_code == 200
                
                # 4th should be rate limited
                assert responses[3].status_code == 429
                data = json.loads(responses[3].data)
                assert data['status'] == 'error'
                assert 'rate limit' in data['error'].lower()
    
    def test_user_quota_exceeded(self):
        """Test dépassement de quota utilisateur."""
        wav_data = BytesIO(b'RIFF\x24\x08\x00\x00WAVEfmt ')
        
        with patch('unified_prediction_system.unified_prediction_api.check_user_quota') as mock_quota:
            mock_quota.return_value = {
                'allowed': False,
                'quota_exceeded': True,
                'remaining': 0,
                'reset_time': '2024-01-16T00:00:00Z'
            }
            
            response = self.client.post('/api/v1/prediction/predict',
                data={'file': (wav_data, 'test.wav')},
                content_type='multipart/form-data',
                headers={'Authorization': 'Bearer user_token'}
            )
            
            assert response.status_code == 429
            data = json.loads(response.data)
            assert data['status'] == 'error'
            assert 'quota' in data['error'].lower()
            assert 'reset_time' in data
    
    def test_quota_headers_in_response(self):
        """Test présence des headers de quota dans les réponses."""
        wav_data = BytesIO(b'RIFF\x24\x08\x00\x00WAVEfmt ')
        
        with patch('unified_prediction_system.unified_prediction_api.check_user_quota') as mock_quota:
            mock_quota.return_value = {
                'allowed': True,
                'remaining': 95,
                'limit': 100,
                'reset_time': '2024-01-16T00:00:00Z'
            }
            
            with patch('unified_prediction_system.unified_prediction_api.UnifiedPredictionAPI') as mock_api:
                mock_instance = mock_api.return_value
                mock_instance.predict_file.return_value = {'status': 'success'}
                
                response = self.client.post('/api/v1/prediction/predict',
                    data={'file': (wav_data, 'test.wav')},
                    content_type='multipart/form-data'
                )
                
                assert response.status_code == 200
                assert 'X-RateLimit-Remaining' in response.headers
                assert 'X-RateLimit-Limit' in response.headers
                assert 'X-RateLimit-Reset' in response.headers
                assert response.headers['X-RateLimit-Remaining'] == '95'


class TestAPIValidation:
    """Tests pour la validation des paramètres d'API."""
    
    def setup_method(self):
        """Setup pour tests de validation."""
        self.app = create_app(testing=True)
        self.client = self.app.test_client()
        self.app_context = self.app.app_context()
        self.app_context.push()
    
    def teardown_method(self):
        """Cleanup après chaque test."""
        self.app_context.pop()
    
    def test_invalid_confidence_threshold(self):
        """Test validation du seuil de confiance."""
        wav_data = BytesIO(b'RIFF\x24\x08\x00\x00WAVEfmt ')
        
        response = self.client.post('/api/v1/prediction/predict',
            data={
                'file': (wav_data, 'test.wav'),
                'confidence_threshold': '1.5'  # > 1.0, invalid
            },
            content_type='multipart/form-data'
        )
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert data['status'] == 'error'
        assert 'confidence' in data['error'].lower()
    
    def test_invalid_model_type(self):
        """Test validation du type de modèle."""
        wav_data = BytesIO(b'RIFF\x24\x08\x00\x00WAVEfmt ')
        
        response = self.client.post('/api/v1/prediction/predict',
            data={
                'file': (wav_data, 'test.wav'),
                'model_type': 'invalid_model'
            },
            content_type='multipart/form-data'
        )
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert data['status'] == 'error'
        assert 'model_type' in data['error'].lower()
    
    def test_file_size_validation(self):
        """Test validation de la taille de fichier."""
        # Create oversized file (simulate)
        large_data = BytesIO(b'x' * (100 * 1024 * 1024))  # 100MB
        
        response = self.client.post('/api/v1/prediction/predict',
            data={'file': (large_data, 'large_file.wav')},
            content_type='multipart/form-data'
        )
        
        assert response.status_code == 413
        data = json.loads(response.data)
        assert data['status'] == 'error'
        assert 'size' in data['error'].lower()
    
    def test_pagination_validation(self):
        """Test validation des paramètres de pagination."""
        # Invalid page number
        response = self.client.get('/api/v1/prediction/predictions?page=-1')
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'page' in data['error'].lower()
        
        # Invalid per_page
        response = self.client.get('/api/v1/prediction/predictions?per_page=1000')
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'per_page' in data['error'].lower()


class TestAPIErrorHandling:
    """Tests pour la gestion d'erreurs de l'API."""
    
    def setup_method(self):
        """Setup pour tests d'erreurs."""
        self.app = create_app(testing=True)
        self.client = self.app.test_client()
        self.app_context = self.app.app_context()
        self.app_context.push()
    
    def teardown_method(self):
        """Cleanup après chaque test."""
        self.app_context.pop()
    
    def test_internal_server_error_handling(self):
        """Test gestion d'erreur interne du serveur."""
        wav_data = BytesIO(b'RIFF\x24\x08\x00\x00WAVEfmt ')
        
        with patch('unified_prediction_system.unified_prediction_api.UnifiedPredictionAPI') as mock_api:
            mock_instance = mock_api.return_value
            mock_instance.predict_file.side_effect = Exception("Internal error")
            
            response = self.client.post('/api/v1/prediction/predict',
                data={'file': (wav_data, 'test.wav')},
                content_type='multipart/form-data'
            )
            
            assert response.status_code == 500
            data = json.loads(response.data)
            assert data['status'] == 'error'
            assert 'error_id' in data  # Error tracking ID
            assert 'timestamp' in data
    
    def test_model_unavailable_error(self):
        """Test erreur de modèle indisponible."""
        wav_data = BytesIO(b'RIFF\x24\x08\x00\x00WAVEfmt ')
        
        with patch('unified_prediction_system.unified_prediction_api.UnifiedPredictionAPI') as mock_api:
            mock_instance = mock_api.return_value
            mock_instance.predict_file.side_effect = RuntimeError("Model not available")
            
            response = self.client.post('/api/v1/prediction/predict',
                data={'file': (wav_data, 'test.wav')},
                content_type='multipart/form-data'
            )
            
            assert response.status_code == 503
            data = json.loads(response.data)
            assert data['status'] == 'error'
            assert 'model' in data['error'].lower()
    
    def test_malformed_request_handling(self):
        """Test gestion des requêtes mal formées."""
        # Send malformed JSON
        response = self.client.post('/api/v1/prediction/predict',
            data='{"invalid": json}',
            content_type='application/json'
        )
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert data['status'] == 'error'
    
    def test_timeout_error_handling(self):
        """Test gestion des timeouts."""
        wav_data = BytesIO(b'RIFF\x24\x08\x00\x00WAVEfmt ')
        
        with patch('unified_prediction_system.unified_prediction_api.UnifiedPredictionAPI') as mock_api:
            mock_instance = mock_api.return_value
            
            def slow_predict(*args, **kwargs):
                time.sleep(10)  # Simulate very slow processing
                return {'status': 'success'}
            
            mock_instance.predict_file.side_effect = slow_predict
            
            response = self.client.post('/api/v1/prediction/predict',
                data={'file': (wav_data, 'test.wav')},
                content_type='multipart/form-data'
            )
            
            # Should timeout before 10 seconds
            assert response.status_code == 504
            data = json.loads(response.data)
            assert data['status'] == 'error'
            assert 'timeout' in data['error'].lower()


@pytest.mark.ml_integration
class TestAPIPerformance:
    """Tests de performance pour l'API."""
    
    def setup_method(self):
        """Setup pour tests de performance."""
        self.app = create_app(testing=True)
        self.client = self.app.test_client()
        self.app_context = self.app.app_context()
        self.app_context.push()
    
    def teardown_method(self):
        """Cleanup après chaque test."""
        self.app_context.pop()
    
    @pytest.mark.performance_critical
    def test_prediction_endpoint_latency(self):
        """Test latence de l'endpoint de prédiction."""
        wav_data = BytesIO(b'RIFF\x24\x08\x00\x00WAVEfmt ')
        
        with patch('unified_prediction_system.unified_prediction_api.UnifiedPredictionAPI') as mock_api:
            mock_instance = mock_api.return_value
            mock_instance.predict_file.return_value = {
                'status': 'success',
                'predictions': {'species': 'test'},
                'processing_time': 0.1
            }
            
            # Measure response times
            latencies = []
            for _ in range(10):
                wav_data.seek(0)
                start_time = time.time()
                
                response = self.client.post('/api/v1/prediction/predict',
                    data={'file': (wav_data, 'test.wav')},
                    content_type='multipart/form-data'
                )
                
                latency = time.time() - start_time
                latencies.append(latency)
                
                assert response.status_code == 200
            
            # Performance assertions
            avg_latency = sum(latencies) / len(latencies)
            p95_latency = sorted(latencies)[8]  # 95th percentile for 10 samples
            
            assert avg_latency < 0.5  # Average API response < 500ms
            assert p95_latency < 1.0   # P95 < 1 second
            assert max(latencies) < 2.0  # Max < 2 seconds
    
    @pytest.mark.performance_critical  
    def test_concurrent_requests_handling(self):
        """Test gestion des requêtes concurrentes."""
        wav_data = BytesIO(b'RIFF\x24\x08\x00\x00WAVEfmt ')
        
        with patch('unified_prediction_system.unified_prediction_api.UnifiedPredictionAPI') as mock_api:
            mock_instance = mock_api.return_value
            mock_instance.predict_file.return_value = {
                'status': 'success',
                'predictions': {'species': 'test'}
            }
            
            import concurrent.futures
            import threading
            
            results = []
            errors = []
            
            def make_request(request_id):
                try:
                    wav_data.seek(0)
                    response = self.client.post('/api/v1/prediction/predict',
                        data={'file': (wav_data, f'test_{request_id}.wav')},
                        content_type='multipart/form-data'
                    )
                    results.append((request_id, response.status_code))
                except Exception as e:
                    errors.append((request_id, e))
            
            # Execute concurrent requests
            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                futures = [executor.submit(make_request, i) for i in range(20)]
                concurrent.futures.wait(futures)
            
            # Verify all requests succeeded
            assert len(errors) == 0
            assert len(results) == 20
            assert all(status_code == 200 for _, status_code in results)
    
    @pytest.mark.performance_critical
    def test_batch_endpoint_performance(self):
        """Test performance de l'endpoint batch."""
        # Create multiple small files
        files_data = []
        for i in range(5):
            wav_data = BytesIO(b'RIFF\x24\x08\x00\x00WAVEfmt ')
            files_data.append(('files', (wav_data, f'test_{i}.wav')))
        
        with patch('unified_prediction_system.unified_prediction_api.UnifiedPredictionAPI') as mock_api:
            mock_instance = mock_api.return_value
            mock_instance.predict_batch.return_value = [
                {'status': 'success', 'predictions': {'species': f'species_{i}'}}
                for i in range(5)
            ]
            
            start_time = time.time()
            
            response = self.client.post('/api/v1/prediction/predict/batch',
                data=files_data,
                content_type='multipart/form-data'
            )
            
            processing_time = time.time() - start_time
            
            assert response.status_code == 200
            assert processing_time < 5.0  # Batch of 5 files < 5 seconds
            
            data = json.loads(response.data)
            assert len(data['results']) == 5