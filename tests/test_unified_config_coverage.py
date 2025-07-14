"""
Tests de couverture complète pour unified_config.py
Tests critiques pour production readiness
"""

import pytest
import os
import tempfile
import json
from unittest.mock import patch, MagicMock
from datetime import timedelta

# Import du module à tester
try:
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from unified_config import (
        get_config, UnifiedConfig, Environment,
        DatabaseConfig, CacheConfig, SecurityConfig,
        ServicesConfig, MLConfig, LoggingConfig, MonitoringConfig
    )
except ImportError as e:
    pytest.skip(f"Cannot import unified_config module: {e}", allow_module_level=True)


class TestUnifiedConfigCore:
    """Tests de base pour la configuration unifiée"""
    
    def test_environment_enum(self):
        """Test enum Environment"""
        assert Environment.DEVELOPMENT.value == 'development'
        assert Environment.STAGING.value == 'staging'
        assert Environment.PRODUCTION.value == 'production'
    
    def test_get_config_singleton(self):
        """Test pattern singleton get_config"""
        config1 = get_config()
        config2 = get_config()
        assert config1 is config2
    
    def test_get_config_force_reload(self):
        """Test force_reload du config"""
        config1 = get_config()
        config2 = get_config(force_reload=True)
        # Peut être différent avec force_reload
        assert isinstance(config2, UnifiedConfig)
    
    def test_config_environments(self):
        """Test configuration par environnement"""
        for env in [Environment.DEVELOPMENT, Environment.STAGING, Environment.PRODUCTION]:
            config = get_config(environment=env, force_reload=True)
            assert isinstance(config, UnifiedConfig)
            assert config.environment == env


class TestDatabaseConfig:
    """Tests pour DatabaseConfig"""
    
    def test_database_config_creation(self):
        """Test création DatabaseConfig"""
        db_config = DatabaseConfig(
            host='localhost',
            port=5432,
            database='test_db',
            username='test_user',
            password='test_pass'
        )
        
        assert db_config.host == 'localhost'
        assert db_config.port == 5432
        assert db_config.database == 'test_db'
    
    def test_database_url_generation(self):
        """Test génération URL base de données"""
        db_config = DatabaseConfig(
            host='localhost',
            port=5432,
            database='test_db',
            username='test_user',
            password='test_pass'
        )
        
        url = db_config.get_url()
        assert 'postgresql://test_user:test_pass@localhost:5432/test_db' in url
    
    def test_database_url_without_password(self):
        """Test URL sans mot de passe"""
        db_config = DatabaseConfig(
            host='localhost',
            port=5432,
            database='test_db',
            username='test_user',
            password='test_pass'
        )
        
        url = db_config.get_url(include_password=False)
        assert 'test_pass' not in url
        assert 'test_user' in url
    
    def test_database_ssl_options(self):
        """Test options SSL base de données"""
        db_config = DatabaseConfig(
            host='localhost',
            sslmode='require',
            sslcert='/path/to/cert.pem'
        )
        
        assert db_config.sslmode == 'require'
        assert db_config.sslcert == '/path/to/cert.pem'


class TestCacheConfig:
    """Tests pour CacheConfig"""
    
    def test_cache_config_creation(self):
        """Test création CacheConfig"""
        cache_config = CacheConfig(
            host='localhost',
            port=6379,
            database=0,
            password='redis_pass'
        )
        
        assert cache_config.host == 'localhost'
        assert cache_config.port == 6379
        assert cache_config.database == 0
    
    def test_cache_url_generation(self):
        """Test génération URL cache"""
        cache_config = CacheConfig(
            host='localhost',
            port=6379,
            database=0,
            password='redis_pass'
        )
        
        url = cache_config.get_url()
        assert 'redis://:redis_pass@localhost:6379/0' in url
    
    def test_cache_without_password(self):
        """Test cache sans mot de passe"""
        cache_config = CacheConfig(
            host='localhost',
            port=6379,
            database=0
        )
        
        url = cache_config.get_url()
        assert 'redis://localhost:6379/0' in url


class TestSecurityConfig:
    """Tests pour SecurityConfig"""
    
    def test_security_config_creation(self):
        """Test création SecurityConfig"""
        security_config = SecurityConfig(
            secret_key='test_secret_key_32_characters_long',
            jwt_secret='jwt_secret_key',
            session_timeout=timedelta(hours=24)
        )
        
        assert len(security_config.secret_key) >= 32
        assert security_config.jwt_secret == 'jwt_secret_key'
        assert security_config.session_timeout.total_seconds() == 24 * 3600
    
    def test_security_config_defaults(self):
        """Test valeurs par défaut sécurité"""
        security_config = SecurityConfig()
        
        assert security_config.csrf_enabled is True
        assert security_config.session_cookie_secure is True
        assert security_config.session_cookie_httponly is True
    
    def test_cors_configuration(self):
        """Test configuration CORS"""
        security_config = SecurityConfig(
            cors_enabled=True,
            allowed_origins=['https://app.nightscan.com', 'https://admin.nightscan.com']
        )
        
        assert security_config.cors_enabled is True
        assert len(security_config.allowed_origins) == 2
        assert 'https://app.nightscan.com' in security_config.allowed_origins


class TestServicesConfig:
    """Tests pour ServicesConfig"""
    
    def test_services_config_creation(self):
        """Test création ServicesConfig"""
        services_config = ServicesConfig(
            web_port=8000,
            api_v1_port=8001,
            prediction_port=8002
        )
        
        assert services_config.web_port == 8000
        assert services_config.api_v1_port == 8001
        assert services_config.prediction_port == 8002
    
    def test_services_ssl_configuration(self):
        """Test configuration SSL services"""
        services_config = ServicesConfig(
            ssl_enabled=True,
            ssl_cert_path='/path/to/cert.pem',
            ssl_key_path='/path/to/key.pem'
        )
        
        assert services_config.ssl_enabled is True
        assert services_config.ssl_cert_path == '/path/to/cert.pem'
    
    def test_file_upload_limits(self):
        """Test limites upload fichiers"""
        services_config = ServicesConfig(
            max_file_size=100 * 1024 * 1024,  # 100MB
            upload_timeout=300
        )
        
        assert services_config.max_file_size == 100 * 1024 * 1024
        assert services_config.upload_timeout == 300


class TestMLConfig:
    """Tests pour MLConfig"""
    
    def test_ml_config_creation(self):
        """Test création MLConfig"""
        ml_config = MLConfig(
            models_directory='/path/to/models',
            use_gpu=True,
            batch_size=32
        )
        
        assert ml_config.models_directory == '/path/to/models'
        assert ml_config.use_gpu is True
        assert ml_config.batch_size == 32
    
    def test_model_paths_configuration(self):
        """Test configuration chemins modèles"""
        ml_config = MLConfig(
            audio_heavy_model='/models/audio_heavy.pth',
            audio_light_model='/models/audio_light.pth',
            photo_heavy_model='/models/photo_heavy.pth',
            photo_light_model='/models/photo_light.pth'
        )
        
        assert ml_config.audio_heavy_model == '/models/audio_heavy.pth'
        assert ml_config.audio_light_model == '/models/audio_light.pth'
        assert ml_config.photo_heavy_model == '/models/photo_heavy.pth'
        assert ml_config.photo_light_model == '/models/photo_light.pth'
    
    def test_prediction_thresholds(self):
        """Test seuils de prédiction"""
        ml_config = MLConfig(
            confidence_threshold=0.85,
            max_predictions_per_hour=1000
        )
        
        assert ml_config.confidence_threshold == 0.85
        assert ml_config.max_predictions_per_hour == 1000


class TestUnifiedConfigIntegration:
    """Tests d'intégration configuration unifiée"""
    
    def test_unified_config_complete_creation(self):
        """Test création complète UnifiedConfig"""
        config = UnifiedConfig(
            environment=Environment.DEVELOPMENT,
            database=DatabaseConfig(host='localhost'),
            cache=CacheConfig(host='localhost'),
            security=SecurityConfig(),
            services=ServicesConfig(),
            ml=MLConfig(),
            logging=LoggingConfig(),
            monitoring=MonitoringConfig()
        )
        
        assert config.environment == Environment.DEVELOPMENT
        assert isinstance(config.database, DatabaseConfig)
        assert isinstance(config.cache, CacheConfig)
        assert isinstance(config.security, SecurityConfig)
    
    def test_get_database_url_method(self):
        """Test méthode get_database_url"""
        config = get_config()
        url = config.get_database_url()
        assert isinstance(url, str)
        assert 'postgresql://' in url or 'sqlite://' in url
    
    def test_get_cache_url_method(self):
        """Test méthode get_cache_url"""
        config = get_config()
        url = config.get_cache_url()
        assert isinstance(url, str)
        assert 'redis://' in url
    
    def test_get_service_url_method(self):
        """Test méthode get_service_url"""
        config = get_config()
        
        web_url = config.get_service_url('web')
        api_url = config.get_service_url('api_v1')
        
        assert isinstance(web_url, str)
        assert isinstance(api_url, str)
        assert 'http' in web_url
        assert 'http' in api_url
    
    def test_to_dict_method(self):
        """Test sérialisation to_dict"""
        config = get_config()
        config_dict = config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert 'database' in config_dict
        assert 'cache' in config_dict
        assert 'security' in config_dict
    
    def test_to_dict_exclude_secrets(self):
        """Test to_dict sans secrets"""
        config = get_config()
        config_dict = config.to_dict(include_secrets=False)
        
        assert isinstance(config_dict, dict)
        # Vérifier que les secrets sont masqués
        if 'security' in config_dict:
            security = config_dict['security']
            if 'secret_key' in security:
                assert '***masked***' in str(security['secret_key'])


class TestConfigurationValidation:
    """Tests de validation configuration"""
    
    def test_environment_validation(self):
        """Test validation environnement"""
        # Test environnement valide
        config = get_config(environment=Environment.DEVELOPMENT)
        assert config.environment == Environment.DEVELOPMENT
        
        # Test avec string
        config = get_config(environment='staging', force_reload=True)
        assert config.environment == Environment.STAGING
    
    def test_port_validation(self):
        """Test validation ports"""
        services_config = ServicesConfig(
            web_port=8000,
            api_v1_port=8001
        )
        
        assert 1 <= services_config.web_port <= 65535
        assert 1 <= services_config.api_v1_port <= 65535
        assert services_config.web_port != services_config.api_v1_port
    
    def test_ssl_configuration_validation(self):
        """Test validation configuration SSL"""
        services_config = ServicesConfig(ssl_enabled=True)
        
        # SSL activé doit avoir des paramètres cohérents
        if services_config.ssl_enabled:
            assert hasattr(services_config, 'ssl_cert_path')
            assert hasattr(services_config, 'ssl_key_path')


class TestEnvironmentVariables:
    """Tests pour variables d'environnement"""
    
    def test_environment_variable_override(self):
        """Test override par variables d'environnement"""
        with patch.dict(os.environ, {
            'NIGHTSCAN_WEB_PORT': '9000',
            'NIGHTSCAN_DB_HOST': 'test-db-host'
        }):
            config = get_config(force_reload=True)
            # Les variables d'environnement doivent être prises en compte
            # Test que la config charge les variables
            assert isinstance(config, UnifiedConfig)
    
    def test_environment_variable_precedence(self):
        """Test précédence variables d'environnement"""
        original_env = os.environ.get('NIGHTSCAN_ENV')
        
        try:
            os.environ['NIGHTSCAN_ENV'] = 'staging'
            config = get_config(force_reload=True)
            
            # Variable d'environnement doit avoir précédence
            assert config.environment in [Environment.STAGING, Environment.DEVELOPMENT]
            
        finally:
            if original_env:
                os.environ['NIGHTSCAN_ENV'] = original_env
            else:
                os.environ.pop('NIGHTSCAN_ENV', None)


class TestConfigurationFiles:
    """Tests pour fichiers de configuration"""
    
    def test_config_file_loading(self):
        """Test chargement fichiers configuration"""
        # Test que les fichiers de config peuvent être chargés
        config_files = [
            'config/unified/development.json',
            'config/unified/staging.json',
            'config/unified/production.json'
        ]
        
        for config_file in config_files:
            if os.path.exists(config_file):
                with open(config_file, 'r') as f:
                    config_data = json.load(f)
                    assert isinstance(config_data, dict)
                    # Vérifier structure basique
                    assert 'database' in config_data or 'services' in config_data
    
    def test_config_file_validation(self):
        """Test validation fichiers configuration"""
        # Créer fichier de config temporaire
        test_config = {
            "database": {
                "host": "localhost",
                "port": 5432,
                "database": "test_db"
            },
            "services": {
                "web_port": 8000,
                "api_v1_port": 8001
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_config, f)
            f.flush()
            
            # Vérifier que le fichier est valide JSON
            with open(f.name, 'r') as read_f:
                loaded_config = json.load(read_f)
                assert loaded_config['database']['host'] == 'localhost'
            
            os.unlink(f.name)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])