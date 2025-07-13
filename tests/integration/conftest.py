"""
Fixtures spécialisées pour les tests d'intégration.

Ces fixtures configurent des environnements de test réalistes
avec de vrais services et données.
"""

import pytest
import tempfile
import shutil
import os
import time
import redis
import psycopg2
from pathlib import Path
from unittest.mock import Mock, patch
import subprocess
import signal
from typing import Dict, Any, Generator
import uuid
import hashlib
import struct
import numpy as np
from PIL import Image

from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash

from web.app import create_app, db as main_db
from config import get_config


@pytest.fixture(scope="session")
def integration_config():
    """Configuration pour tests d'intégration."""
    config = get_config()
    
    # Override avec valeurs de test
    config.database.database_url = os.environ.get(
        'TEST_DATABASE_URL', 
        'postgresql://test:test@localhost:5432/nightscan_test'
    )
    config.redis.redis_url = os.environ.get(
        'TEST_REDIS_URL',
        'redis://localhost:6379/1'  # Use DB 1 for tests
    )
    config.security.secret_key = 'test-secret-key-integration'
    config.security.csrf_secret_key = 'test-csrf-secret-key'
    
    return config


@pytest.fixture(scope="session")
def test_database_url(integration_config):
    """URL de base de données de test."""
    return integration_config.database.database_url


@pytest.fixture(scope="session")
def test_redis_url(integration_config):
    """URL Redis de test."""
    return integration_config.redis.redis_url


@pytest.fixture(scope="session")
def redis_client(test_redis_url):
    """Client Redis pour tests d'intégration."""
    try:
        client = redis.from_url(test_redis_url)
        # Test connection
        client.ping()
        
        # Clear test DB
        client.flushdb()
        
        yield client
        
        # Cleanup
        client.flushdb()
        client.close()
        
    except redis.ConnectionError:
        pytest.skip("Redis not available for integration tests")


@pytest.fixture(scope="session")
def database_connection(test_database_url):
    """Connexion directe à la base de données de test."""
    try:
        conn = psycopg2.connect(test_database_url)
        conn.autocommit = True
        
        yield conn
        
        conn.close()
        
    except psycopg2.OperationalError:
        pytest.skip("PostgreSQL not available for integration tests")


@pytest.fixture(scope="function")
def integration_app(integration_config, test_database_url, test_redis_url):
    """Application Flask configurée pour tests d'intégration."""
    
    # Create app with test config
    app = create_app()
    app.config.update({
        'TESTING': True,
        'WTF_CSRF_ENABLED': False,  # Disable CSRF for tests
        'SQLALCHEMY_DATABASE_URI': test_database_url,
        'REDIS_URL': test_redis_url,
        'SECRET_KEY': integration_config.security.secret_key,
        'LOGIN_DISABLED': False,  # Enable login for auth tests
        'RATELIMIT_ENABLED': False,  # Disable rate limiting for tests
    })
    
    with app.app_context():
        # Ensure all tables are created
        main_db.create_all()
        
        # Clear any existing data
        main_db.session.execute('TRUNCATE TABLE "user" CASCADE')
        main_db.session.execute('TRUNCATE TABLE prediction CASCADE') 
        main_db.session.execute('TRUNCATE TABLE detection CASCADE')
        main_db.session.commit()
        
        yield app
        
        # Cleanup
        main_db.session.rollback()
        main_db.drop_all()


@pytest.fixture(scope="function")
def integration_client(integration_app):
    """Client de test pour l'application d'intégration."""
    return integration_app.test_client()


@pytest.fixture(scope="function")
def integration_db(integration_app):
    """Base de données configurée pour tests d'intégration."""
    return main_db


@pytest.fixture(scope="function")
def test_user_factory(integration_db):
    """Factory pour créer des utilisateurs de test."""
    from web.app import User
    
    created_users = []
    
    def create_user(username="testuser", password="testpass123", **kwargs):
        user = User(
            username=username,
            **kwargs
        )
        user.set_password(password)
        integration_db.session.add(user)
        integration_db.session.commit()
        
        created_users.append(user)
        return user
    
    yield create_user
    
    # Cleanup
    for user in created_users:
        try:
            integration_db.session.delete(user)
            integration_db.session.commit()
        except:
            integration_db.session.rollback()


@pytest.fixture(scope="function")
def authenticated_user(integration_client, test_user_factory):
    """Utilisateur authentifié pour tests."""
    user = test_user_factory(username="authuser", password="authpass123")
    
    # Login via web interface
    response = integration_client.post('/login', data={
        'username': 'authuser',
        'password': 'authpass123'
    }, follow_redirects=True)
    
    assert response.status_code == 200
    
    return user


@pytest.fixture(scope="function")
def test_jwt_token(integration_app, test_user_factory):
    """Token JWT valide pour tests API."""
    user = test_user_factory(username="apiuser", password="apipass123")
    
    with integration_app.test_client() as client:
        # Get JWT token via API
        response = client.post('/api/auth/login', 
            json={
                'username': 'apiuser',
                'password': 'apipass123'
            },
            headers={'Content-Type': 'application/json'}
        )
        
        if response.status_code == 200:
            data = response.get_json()
            return data.get('access_token')
        else:
            # Fallback: create mock token for tests
            from auth.jwt_manager import get_jwt_manager
            jwt_manager = get_jwt_manager()
            return jwt_manager.create_access_token(user.id)


@pytest.fixture(scope="session")
def test_files_dir():
    """Répertoire temporaire pour fichiers de test."""
    temp_dir = tempfile.mkdtemp(prefix="nightscan_integration_")
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture(scope="function")
def test_audio_file(test_files_dir):
    """Créer un fichier audio WAV valide pour tests."""
    audio_file = test_files_dir / "test_audio.wav"
    
    # Create valid WAV file
    sample_rate = 44100
    duration = 2.0  # 2 seconds
    samples = int(sample_rate * duration)
    
    # Generate sine wave
    t = np.linspace(0, duration, samples)
    frequency = 440  # A note
    audio_data = (np.sin(2 * np.pi * frequency * t) * 32767).astype(np.int16)
    
    with open(audio_file, 'wb') as f:
        # WAV header
        f.write(b'RIFF')
        f.write(struct.pack('<L', samples * 2 + 36))  # File size
        f.write(b'WAVE')
        
        # fmt chunk
        f.write(b'fmt ')
        f.write(struct.pack('<L', 16))  # Chunk size
        f.write(struct.pack('<H', 1))   # Audio format (PCM)
        f.write(struct.pack('<H', 1))   # Channels (mono)
        f.write(struct.pack('<L', sample_rate))  # Sample rate
        f.write(struct.pack('<L', sample_rate * 2))  # Byte rate
        f.write(struct.pack('<H', 2))   # Block align
        f.write(struct.pack('<H', 16))  # Bits per sample
        
        # data chunk
        f.write(b'data')
        f.write(struct.pack('<L', samples * 2))  # Data size
        f.write(audio_data.tobytes())
    
    return audio_file


@pytest.fixture(scope="function")
def test_image_file(test_files_dir):
    """Créer un fichier image valide pour tests."""
    image_file = test_files_dir / "test_image.jpg"
    
    # Create test image
    width, height = 640, 480
    img = Image.new('RGB', (width, height), color=(100, 150, 200))
    
    # Add some pattern for realism
    pixels = img.load()
    for x in range(0, width, 20):
        for y in range(0, height, 20):
            if (x + y) % 40 == 0:
                pixels[x, y] = (255, 255, 255)
    
    img.save(image_file, format='JPEG', quality=85)
    return image_file


@pytest.fixture(scope="function")
def test_large_audio_file(test_files_dir):
    """Créer un gros fichier audio pour tests de performance."""
    large_file = test_files_dir / "large_audio.wav"
    
    # Create 50MB WAV file
    sample_rate = 44100
    duration = 300.0  # 5 minutes 
    samples = int(sample_rate * duration)
    
    with open(large_file, 'wb') as f:
        # WAV header
        f.write(b'RIFF')
        f.write(struct.pack('<L', samples * 2 + 36))
        f.write(b'WAVE')
        f.write(b'fmt ')
        f.write(struct.pack('<L', 16))
        f.write(struct.pack('<H', 1))   # PCM
        f.write(struct.pack('<H', 1))   # Mono
        f.write(struct.pack('<L', sample_rate))
        f.write(struct.pack('<L', sample_rate * 2))
        f.write(struct.pack('<H', 2))
        f.write(struct.pack('<H', 16))
        f.write(b'data')
        f.write(struct.pack('<L', samples * 2))
        
        # Write audio data in chunks
        chunk_size = 44100  # 1 second chunks
        for i in range(0, samples, chunk_size):
            chunk_samples = min(chunk_size, samples - i)
            t = np.linspace(i/sample_rate, (i+chunk_samples)/sample_rate, chunk_samples)
            frequency = 440 + (i % 1000)  # Varying frequency
            audio_chunk = (np.sin(2 * np.pi * frequency * t) * 32767).astype(np.int16)
            f.write(audio_chunk.tobytes())
    
    return large_file


@pytest.fixture(scope="function")
def test_invalid_file(test_files_dir):
    """Créer un fichier invalide pour tests d'erreur."""
    invalid_file = test_files_dir / "invalid.wav"
    
    # Create file with .wav extension but invalid content
    with open(invalid_file, 'wb') as f:
        f.write(b'This is not a valid WAV file content')
    
    return invalid_file


@pytest.fixture(scope="function")
def mock_prediction_service():
    """Mock du service de prédiction pour tests."""
    with patch('web.tasks.run_prediction') as mock_task:
        mock_task.delay.return_value.id = 'test-task-id'
        mock_task.delay.return_value.status = 'SUCCESS'
        mock_task.delay.return_value.result = {
            'species': 'owl',
            'confidence': 0.89,
            'predictions': [
                {'class': 'owl', 'confidence': 0.89},
                {'class': 'wind', 'confidence': 0.11}
            ]
        }
        yield mock_task


@pytest.fixture(scope="function")
def mock_celery_worker():
    """Mock Celery worker pour tests asynchrones."""
    from celery import Celery
    
    # Create minimal Celery app for testing
    celery_app = Celery('test')
    celery_app.conf.update(
        task_always_eager=True,  # Execute tasks synchronously
        task_eager_propagates=True,
        broker_url='memory://',
        result_backend='cache+memory://'
    )
    
    return celery_app


@pytest.fixture(scope="function")
def test_session_data():
    """Données de session pour tests."""
    return {
        'user_id': 1,
        'username': 'testuser',
        'csrf_token': 'test-csrf-token',
        'login_time': time.time()
    }


@pytest.fixture(scope="function")
def test_metrics_data():
    """Données de métriques pour tests."""
    return {
        'request_count': 0,
        'failed_logins': 0,
        'upload_count': 0,
        'prediction_count': 0
    }


class IntegrationTestHelpers:
    """Utilitaires pour tests d'intégration."""
    
    @staticmethod
    def create_test_prediction(user, filename="test.wav", result=None):
        """Créer une prédiction de test."""
        from web.app import Prediction
        
        if result is None:
            result = '{"species": "owl", "confidence": 0.85}'
        
        prediction = Prediction(
            user_id=user.id,
            filename=filename,
            result=result,
            file_size=44100 * 2,  # 1 second of 16-bit mono at 44.1kHz
        )
        
        main_db.session.add(prediction)
        main_db.session.commit()
        return prediction
    
    @staticmethod
    def assert_user_logged_in(client, username):
        """Vérifier qu'un utilisateur est connecté."""
        response = client.get('/api/user/profile')
        assert response.status_code == 200
        data = response.get_json()
        assert data.get('username') == username
    
    @staticmethod
    def assert_user_logged_out(client):
        """Vérifier qu'aucun utilisateur n'est connecté."""
        response = client.get('/api/user/profile')
        assert response.status_code == 401
    
    @staticmethod
    def calculate_file_hash(file_path):
        """Calculer le hash d'un fichier."""
        hash_sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    
    @staticmethod
    def wait_for_task_completion(task_id, timeout=30):
        """Attendre qu'une tâche Celery soit terminée."""
        from celery.result import AsyncResult
        
        result = AsyncResult(task_id)
        start_time = time.time()
        
        while not result.ready() and time.time() - start_time < timeout:
            time.sleep(0.1)
        
        return result.ready(), result.result if result.ready() else None


@pytest.fixture(scope="function")
def integration_helpers():
    """Helper utilities pour tests d'intégration."""
    return IntegrationTestHelpers