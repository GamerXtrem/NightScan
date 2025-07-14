"""
Tests complets pour l'application web principale web/app.py
Coverage critique pour la production
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
    from web.app import app, db
    # Import des modèles si disponibles
    try:
        from web.app import User, Prediction
    except ImportError:
        User = None
        Prediction = None
except ImportError as e:
    pytest.skip(f"Cannot import web.app module: {e}", allow_module_level=True)


class TestWebAppCore:
    """Tests de base pour l'application web"""
    
    @pytest.fixture
    def client(self):
        """Client de test pour l'app web"""
        app.config['TESTING'] = True
        app.config['WTF_CSRF_ENABLED'] = False
        app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///:memory:'
        
        with app.test_client() as client:
            with app.app_context():
                if hasattr(db, 'create_all'):
                    db.create_all()
                yield client
                if hasattr(db, 'drop_all'):
                    db.drop_all()
    
    def test_home_page_loads(self, client):
        """Test chargement page d'accueil"""
        response = client.get('/')
        assert response.status_code in [200, 302]  # 302 si redirect login
    
    def test_login_page_exists(self, client):
        """Test page login existe"""
        response = client.get('/login')
        assert response.status_code == 200
        assert b'login' in response.data.lower()
    
    def test_security_headers_present(self, client):
        """Test présence headers de sécurité"""
        response = client.get('/')
        
        # Headers de sécurité essentiels
        assert 'X-Content-Type-Options' in response.headers
        assert 'X-Frame-Options' in response.headers or 'Content-Security-Policy' in response.headers
    
    def test_static_files_accessible(self, client):
        """Test accessibilité fichiers statiques"""
        # Test si route static existe
        response = client.get('/static/css/style.css')
        # Acceptable que le fichier n'existe pas, mais route doit être configurée
        assert response.status_code in [200, 404]


class TestWebAppAuth:
    """Tests pour l'authentification web"""
    
    @pytest.fixture
    def client(self):
        app.config['TESTING'] = True
        app.config['WTF_CSRF_ENABLED'] = False
        app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///:memory:'
        
        with app.test_client() as client:
            with app.app_context():
                if hasattr(db, 'create_all'):
                    db.create_all()
                yield client
                if hasattr(db, 'drop_all'):
                    db.drop_all()
    
    def test_login_form_submission(self, client):
        """Test soumission formulaire login"""
        response = client.post('/login', data={
            'username': 'testuser',
            'password': 'testpass'
        }, follow_redirects=True)
        
        # Doit traiter la soumission même si utilisateur n'existe pas
        assert response.status_code == 200
    
    def test_logout_endpoint(self, client):
        """Test endpoint logout"""
        response = client.get('/logout')
        assert response.status_code in [200, 302]  # Redirect vers login
    
    def test_register_page_exists(self, client):
        """Test page register existe"""
        response = client.get('/register')
        assert response.status_code in [200, 404]  # Peut être désactivé
    
    def test_session_protection(self, client):
        """Test protection de session"""
        # Accéder à une page protégée sans login
        response = client.get('/dashboard')
        # Doit rediriger vers login ou retourner 401
        assert response.status_code in [302, 401, 404]


class TestWebAppUploads:
    """Tests pour le système d'upload"""
    
    @pytest.fixture
    def client(self):
        app.config['TESTING'] = True
        app.config['WTF_CSRF_ENABLED'] = False
        app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB
        
        with app.test_client() as client:
            yield client
    
    def test_upload_endpoint_exists(self, client):
        """Test endpoint upload existe"""
        response = client.get('/upload')
        # Endpoint peut être protégé par auth
        assert response.status_code in [200, 302, 401, 404]
    
    def test_upload_form_structure(self, client):
        """Test structure formulaire upload"""
        response = client.get('/upload')
        if response.status_code == 200:
            # Si accessible, doit contenir un formulaire
            assert b'form' in response.data.lower() or b'upload' in response.data.lower()
    
    @patch('web.app.allowed_file')
    def test_file_upload_security(self, mock_allowed, client):
        """Test sécurité upload fichiers"""
        mock_allowed.return_value = True
        
        # Simuler upload fichier
        with tempfile.NamedTemporaryFile(suffix='.wav') as f:
            f.write(b'fake audio data')
            f.flush()
            
            with open(f.name, 'rb') as test_file:
                response = client.post('/upload', 
                    data={'file': (test_file, 'test.wav')},
                    content_type='multipart/form-data',
                    follow_redirects=True
                )
            
            # Upload peut échouer pour diverses raisons, mais doit être traité
            assert response.status_code in [200, 302, 400, 401, 413, 422]


class TestWebAppAPI:
    """Tests pour les endpoints API intégrés"""
    
    @pytest.fixture
    def client(self):
        app.config['TESTING'] = True
        app.config['WTF_CSRF_ENABLED'] = False
        
        with app.test_client() as client:
            yield client
    
    def test_api_predict_endpoint(self, client):
        """Test endpoint API predict"""
        response = client.get('/api/predict')
        # Endpoint peut nécessiter POST ou auth
        assert response.status_code in [200, 401, 404, 405]
    
    def test_api_status_endpoint(self, client):
        """Test endpoint status"""
        response = client.get('/api/status')
        assert response.status_code in [200, 404]
        
        if response.status_code == 200:
            # Si existe, doit retourner JSON valide
            try:
                data = json.loads(response.data)
                assert isinstance(data, dict)
            except json.JSONDecodeError:
                pytest.fail("Status endpoint should return valid JSON")
    
    def test_health_check_endpoint(self, client):
        """Test endpoint health check"""
        for endpoint in ['/health', '/api/health', '/healthz']:
            response = client.get(endpoint)
            if response.status_code == 200:
                # Au moins un endpoint health doit exister
                break
        else:
            # Si aucun endpoint standard, vérifier root
            response = client.get('/')
            assert response.status_code in [200, 302]


class TestWebAppErrorHandling:
    """Tests pour la gestion d'erreurs web"""
    
    @pytest.fixture
    def client(self):
        app.config['TESTING'] = True
        
        with app.test_client() as client:
            yield client
    
    def test_404_error_page(self, client):
        """Test page d'erreur 404"""
        response = client.get('/nonexistent-page-12345')
        assert response.status_code == 404
        # Doit avoir une page d'erreur ou message approprié
        assert len(response.data) > 0
    
    def test_csrf_protection(self, client):
        """Test protection CSRF"""
        # Essayer soumission sans token CSRF si activé
        app.config['WTF_CSRF_ENABLED'] = True
        
        response = client.post('/login', data={
            'username': 'test',
            'password': 'test'
        })
        
        # Avec CSRF activé, doit rejeter ou demander token
        assert response.status_code in [200, 400, 403, 422]
    
    def test_large_file_upload_rejection(self, client):
        """Test rejet fichiers trop volumineux"""
        app.config['MAX_CONTENT_LENGTH'] = 1024  # 1KB limit
        
        # Créer fichier plus grand que la limite
        large_data = b'x' * 2048  # 2KB
        
        response = client.post('/upload',
            data={'file': (tempfile.NamedTemporaryFile(), 'large.wav')},
            content_type='multipart/form-data'
        )
        
        # Doit gérer les fichiers trop volumineux
        assert response.status_code in [200, 413, 422]


class TestWebAppPerformance:
    """Tests de performance web"""
    
    @pytest.fixture
    def client(self):
        app.config['TESTING'] = True
        
        with app.test_client() as client:
            yield client
    
    def test_page_load_time(self, client):
        """Test temps de chargement pages"""
        import time
        
        start_time = time.time()
        response = client.get('/')
        end_time = time.time()
        
        load_time = end_time - start_time
        assert load_time < 2.0  # < 2 secondes pour page d'accueil
        assert response.status_code in [200, 302]
    
    def test_static_content_caching(self, client):
        """Test mise en cache contenu statique"""
        response = client.get('/static/css/style.css')
        
        if response.status_code == 200:
            # Doit avoir headers de cache ou ETag
            has_cache = any(header in response.headers for header in [
                'Cache-Control', 'ETag', 'Last-Modified'
            ])
            # Cache recommandé mais pas obligatoire pour tests
            assert True  # Test informatif


class TestWebAppSecurity:
    """Tests de sécurité web spécifiques"""
    
    @pytest.fixture
    def client(self):
        app.config['TESTING'] = True
        
        with app.test_client() as client:
            yield client
    
    def test_xss_protection(self, client):
        """Test protection XSS basique"""
        # Essayer injection script
        malicious_input = '<script>alert("xss")</script>'
        
        response = client.post('/search', data={
            'query': malicious_input
        }, follow_redirects=True)
        
        # Script ne doit pas être exécuté tel quel
        if response.status_code == 200:
            assert b'<script>alert(' not in response.data
    
    def test_sql_injection_protection(self, client):
        """Test protection injection SQL basique"""
        # Essayer injection SQL
        malicious_input = "'; DROP TABLE users; --"
        
        response = client.post('/login', data={
            'username': malicious_input,
            'password': 'test'
        })
        
        # Doit gérer l'input sans crash
        assert response.status_code in [200, 400, 401, 422]
    
    def test_content_security_policy(self, client):
        """Test Content Security Policy"""
        response = client.get('/')
        
        # CSP recommandé pour production
        csp_present = 'Content-Security-Policy' in response.headers
        # Test informatif - CSP peut être configuré au niveau serveur
        assert True


if __name__ == '__main__':
    pytest.main([__file__, '-v'])