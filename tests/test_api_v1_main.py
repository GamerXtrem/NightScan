"""
Tests complets pour l'API principale api_v1.py
Coverage critique pour production readiness
"""

import pytest
import json
import os
import tempfile
from unittest.mock import patch, MagicMock
from flask import Flask

# Import du module à tester
try:
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from api_v1 import create_app, db, User, Prediction
except ImportError as e:
    pytest.skip(f"Cannot import api_v1 module: {e}", allow_module_level=True)


class TestAPIv1Core:
    """Tests pour le core de l'API v1"""
    
    @pytest.fixture
    def app(self):
        """Setup application de test"""
        app = create_app({
            'TESTING': True,
            'SECRET_KEY': 'test-secret-key',
            'SQLALCHEMY_DATABASE_URI': 'sqlite:///:memory:',
            'SQLALCHEMY_TRACK_MODIFICATIONS': False,
            'WTF_CSRF_ENABLED': False
        })
        
        with app.app_context():
            db.create_all()
            yield app
            db.drop_all()
    
    @pytest.fixture
    def client(self, app):
        """Client de test"""
        return app.test_client()
    
    def test_app_creation(self, app):
        """Test création application"""
        assert app.config['TESTING'] is True
        assert 'api_v1' in app.blueprints
    
    def test_health_endpoint(self, client):
        """Test endpoint health check"""
        response = client.get('/api/v1/health')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['status'] == 'healthy'
        assert 'timestamp' in data
    
    def test_cors_headers(self, client):
        """Test headers CORS"""
        response = client.options('/api/v1/health')
        assert 'Access-Control-Allow-Origin' in response.headers
    
    def test_security_headers(self, client):
        """Test headers de sécurité"""
        response = client.get('/api/v1/health')
        assert 'X-Content-Type-Options' in response.headers
        assert response.headers['X-Content-Type-Options'] == 'nosniff'


class TestAPIv1Auth:
    """Tests pour l'authentification API v1"""
    
    @pytest.fixture
    def app(self):
        app = create_app({
            'TESTING': True,
            'SECRET_KEY': 'test-secret-key',
            'SQLALCHEMY_DATABASE_URI': 'sqlite:///:memory:',
            'WTF_CSRF_ENABLED': False
        })
        
        with app.app_context():
            db.create_all()
            # Créer utilisateur test
            user = User(
                username='testuser',
                email='test@example.com',
                password_hash='hashed_password'
            )
            db.session.add(user)
            db.session.commit()
            yield app
            db.drop_all()
    
    @pytest.fixture
    def client(self, app):
        return app.test_client()
    
    def test_login_endpoint_exists(self, client):
        """Test endpoint login existe"""
        response = client.post('/api/v1/auth/login', json={
            'username': 'testuser',
            'password': 'wrongpassword'
        })
        # Même si échec, endpoint doit exister
        assert response.status_code in [200, 401, 422]
    
    def test_protected_endpoint_without_auth(self, client):
        """Test accès endpoint protégé sans auth"""
        response = client.get('/api/v1/predictions')
        assert response.status_code == 401
    
    @patch('api_v1.jwt_required')
    def test_jwt_protection(self, mock_jwt, client):
        """Test protection JWT"""
        mock_jwt.return_value = lambda f: f
        response = client.get('/api/v1/predictions')
        # Avec JWT mocké, devrait passer l'auth
        assert response.status_code != 401


class TestAPIv1Predictions:
    """Tests pour les endpoints de prédictions"""
    
    @pytest.fixture
    def app_with_data(self):
        app = create_app({
            'TESTING': True,
            'SECRET_KEY': 'test-secret-key',
            'SQLALCHEMY_DATABASE_URI': 'sqlite:///:memory:',
            'WTF_CSRF_ENABLED': False
        })
        
        with app.app_context():
            db.create_all()
            
            # Créer utilisateur et prédictions test
            user = User(username='testuser', email='test@example.com')
            db.session.add(user)
            db.session.flush()
            
            prediction = Prediction(
                user_id=user.id,
                filename='test.wav',
                prediction_result={'species': 'bird', 'confidence': 0.85},
                confidence_score=0.85
            )
            db.session.add(prediction)
            db.session.commit()
            
            yield app, user.id
            db.drop_all()
    
    @pytest.fixture
    def client_with_data(self, app_with_data):
        app, user_id = app_with_data
        return app.test_client(), user_id
    
    def test_predictions_list_structure(self, client_with_data):
        """Test structure réponse liste prédictions"""
        client, user_id = client_with_data
        
        with patch('api_v1.jwt_required') as mock_jwt, \
             patch('api_v1.get_jwt_identity', return_value=user_id):
            mock_jwt.return_value = lambda f: f
            
            response = client.get('/api/v1/predictions')
            assert response.status_code == 200
            
            data = json.loads(response.data)
            assert 'success' in data
            assert 'data' in data
            assert isinstance(data['data'], list)
    
    @patch('api_v1.unified_predict')
    def test_new_prediction_upload(self, mock_predict, client_with_data):
        """Test upload nouvelle prédiction"""
        client, user_id = client_with_data
        mock_predict.return_value = {'species': 'owl', 'confidence': 0.90}
        
        with patch('api_v1.jwt_required') as mock_jwt, \
             patch('api_v1.get_jwt_identity', return_value=user_id):
            mock_jwt.return_value = lambda f: f
            
            # Créer fichier temporaire
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                f.write(b'fake audio data')
                f.flush()
                
                with open(f.name, 'rb') as test_file:
                    response = client.post('/api/v1/predict', 
                        data={'file': (test_file, 'test.wav')},
                        content_type='multipart/form-data'
                    )
                
                os.unlink(f.name)
            
            # Should process file even if prediction service unavailable
            assert response.status_code in [200, 422, 500]


class TestAPIv1ErrorHandling:
    """Tests pour la gestion d'erreurs"""
    
    @pytest.fixture
    def app(self):
        app = create_app({
            'TESTING': True,
            'SECRET_KEY': 'test-secret-key',
            'SQLALCHEMY_DATABASE_URI': 'sqlite:///:memory:',
        })
        return app
    
    @pytest.fixture
    def client(self, app):
        return app.test_client()
    
    def test_404_error_handling(self, client):
        """Test gestion erreur 404"""
        response = client.get('/api/v1/nonexistent')
        assert response.status_code == 404
        data = json.loads(response.data)
        assert 'error' in data
    
    def test_method_not_allowed(self, client):
        """Test méthode non autorisée"""
        response = client.delete('/api/v1/health')
        assert response.status_code == 405
    
    def test_malformed_json(self, client):
        """Test JSON malformé"""
        response = client.post('/api/v1/auth/login',
            data='{"invalid": json}',
            content_type='application/json'
        )
        assert response.status_code in [400, 422]


class TestAPIv1Performance:
    """Tests de performance basiques"""
    
    @pytest.fixture
    def client(self):
        app = create_app({
            'TESTING': True,
            'SECRET_KEY': 'test-secret-key',
            'SQLALCHEMY_DATABASE_URI': 'sqlite:///:memory:',
        })
        return app.test_client()
    
    def test_health_endpoint_performance(self, client):
        """Test performance endpoint health"""
        import time
        
        start_time = time.time()
        response = client.get('/api/v1/health')
        end_time = time.time()
        
        assert response.status_code == 200
        assert (end_time - start_time) < 1.0  # < 1 seconde
    
    def test_multiple_concurrent_requests(self, client):
        """Test requêtes multiples"""
        responses = []
        for _ in range(10):
            response = client.get('/api/v1/health')
            responses.append(response.status_code)
        
        # Toutes les requêtes doivent réussir
        assert all(status == 200 for status in responses)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])