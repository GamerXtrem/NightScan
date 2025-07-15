import pytest
import os
import tempfile
import json
from pathlib import Path
from unittest.mock import patch, mock_open
import sys

# Add the parent directory to Python path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from config import (
    SecurityConfig,
    DatabaseConfig,
    RateLimitConfig,
    FileUploadConfig,
    get_config
)


class TestSecurityConfig:
    """Test security configuration."""
    
    def test_security_config_initialization(self):
        """Test security config with default values."""
        config = SecurityConfig()
        
        assert config.secret_key is not None
        assert len(config.secret_key) >= 32
        assert config.csrf_secret_key is None  # Should be None by default
        assert config.password_min_length == 10
        assert config.lockout_threshold == 5
        assert config.lockout_window == 900
        assert config.lockout_file == "failed_logins.json"
        
    def test_security_config_with_custom_values(self):
        """Test security config with custom values."""
        config = SecurityConfig(
            password_min_length=12,
            lockout_threshold=3,
            lockout_window=1200
        )
        
        assert config.password_min_length == 12
        assert config.lockout_threshold == 3
        assert config.lockout_window == 1200
        
    def test_security_config_secret_key_generation(self):
        """Test that secret key is properly generated."""
        config1 = SecurityConfig()
        config2 = SecurityConfig()
        
        # Each instance should have a different secret key
        assert config1.secret_key != config2.secret_key
        assert len(config1.secret_key) >= 32
        assert len(config2.secret_key) >= 32


class TestDatabaseConfig:
    """Test database configuration."""
    
    def test_database_config_default(self):
        """Test database config with default values."""
        config = DatabaseConfig()
        
        assert config.uri == "sqlite:///nightscan.db"
        assert config.pool_size == 10
        assert config.pool_timeout == 30
        assert config.pool_recycle == 3600
        assert config.echo == False
        
    def test_database_config_from_env(self):
        """Test database config from environment variables."""
        with patch.dict(os.environ, {
            'SQLALCHEMY_DATABASE_URI': 'postgresql://user:pass@localhost/db',
            'DB_POOL_SIZE': '20',
            'DB_POOL_TIMEOUT': '60',
            'DB_ECHO': 'true'
        }):
            config = DatabaseConfig()
            
            assert config.uri == 'postgresql://user:pass@localhost/db'
            assert config.pool_size == 20
            assert config.pool_timeout == 60
            assert config.echo == True
            
    def test_database_config_custom_values(self):
        """Test database config with custom values."""
        config = DatabaseConfig(
            uri="mysql://user:pass@localhost/test",
            pool_size=15,
            echo=True
        )
        
        assert config.uri == "mysql://user:pass@localhost/test"
        assert config.pool_size == 15
        assert config.echo == True


class TestRateLimitConfig:
    """Test rate limiting configuration."""
    
    def test_rate_limit_config_default(self):
        """Test rate limit config with default values."""
        config = RateLimitConfig()
        
        assert config.enabled == True
        assert config.default_limit == "100 per hour"
        assert config.login_limit == "10 per minute"
        assert config.upload_limit == "5 per minute"
        
    def test_rate_limit_config_disabled(self):
        """Test rate limit config when disabled."""
        with patch.dict(os.environ, {'RATE_LIMIT_ENABLED': 'false'}):
            config = RateLimitConfig()
            assert config.enabled == False
            
    def test_rate_limit_config_custom_limits(self):
        """Test rate limit config with custom limits."""
        config = RateLimitConfig(
            default_limit="200 per hour",
            login_limit="20 per minute",
            upload_limit="10 per minute"
        )
        
        assert config.default_limit == "200 per hour"
        assert config.login_limit == "20 per minute"
        assert config.upload_limit == "10 per minute"


class TestUploadConfig:
    """Test upload configuration."""
    
    def test_upload_config_default(self):
        """Test upload config with default values."""
        config = FileUploadConfig()
        
        assert config.max_file_size == 104857600  # 100MB
        assert config.max_total_size == 10737418240  # 10GB
        assert config.allowed_extensions == ['.wav']
        
    def test_upload_config_from_env(self):
        """Test upload config from environment variables."""
        with patch.dict(os.environ, {
            'MAX_FILE_SIZE': str(50 * 1024 * 1024),  # 50MB
            'MAX_TOTAL_SIZE': str(5 * 1024 * 1024 * 1024)  # 5GB
        }):
            config = FileUploadConfig()
            
            assert config.max_file_size == 50 * 1024 * 1024
            assert config.max_total_size == 5 * 1024 * 1024 * 1024
            
    def test_upload_config_custom_extensions(self):
        """Test upload config with custom extensions."""
        config = FileUploadConfig(allowed_extensions=['.wav', '.mp3', '.flac'])
        
        assert '.wav' in config.allowed_extensions
        assert '.mp3' in config.allowed_extensions
        assert '.flac' in config.allowed_extensions


class TestEmailConfig:
    """Test email configuration."""
    
    def test_email_config_default(self):
        """Test email config with default values."""
        config = EmailConfig()
        
        assert config.smtp_server == 'localhost'
        assert config.smtp_port == 587
        assert config.smtp_username is None
        assert config.smtp_password is None
        assert config.from_email == 'noreply@nightscan.com'
        assert config.use_tls == True
        
    def test_email_config_from_env(self):
        """Test email config from environment variables."""
        with patch.dict(os.environ, {
            'SMTP_SERVER': 'smtp.gmail.com',
            'SMTP_PORT': '465',
            'SMTP_USERNAME': 'user@gmail.com',
            'SMTP_PASSWORD': 'secret',
            'FROM_EMAIL': 'nightscan@gmail.com',
            'SMTP_USE_TLS': 'false'
        }):
            config = EmailConfig()
            
            assert config.smtp_server == 'smtp.gmail.com'
            assert config.smtp_port == 465
            assert config.smtp_username == 'user@gmail.com'
            assert config.smtp_password == 'secret'
            assert config.from_email == 'nightscan@gmail.com'
            assert config.use_tls == False
            
    def test_email_config_custom_values(self):
        """Test email config with custom values."""
        config = EmailConfig(
            smtp_server='smtp.mailgun.org',
            smtp_port=587,
            smtp_username='api',
            smtp_password='key-123',
            from_email='alerts@nightscan.org'
        )
        
        assert config.smtp_server == 'smtp.mailgun.org'
        assert config.smtp_username == 'api'
        assert config.from_email == 'alerts@nightscan.org'


class TestNotificationConfig:
    """Test notification configuration."""
    
    def test_notification_config_default(self):
        """Test notification config with default values."""
        config = NotificationConfig()
        
        assert config.enable_email == True
        assert config.enable_push == True
        assert config.enable_websocket == True
        assert config.enable_slack == False
        assert config.enable_discord == False
        assert config.expo_access_token is None
        
    def test_notification_config_from_env(self):
        """Test notification config from environment variables."""
        with patch.dict(os.environ, {
            'ENABLE_EMAIL_NOTIFICATIONS': 'false',
            'ENABLE_PUSH_NOTIFICATIONS': 'true',
            'ENABLE_SLACK_NOTIFICATIONS': 'true',
            'EXPO_ACCESS_TOKEN': 'expo_token_123'
        }):
            config = NotificationConfig()
            
            assert config.enable_email == False
            assert config.enable_push == True
            assert config.enable_slack == True
            assert config.expo_access_token == 'expo_token_123'
            
    def test_notification_config_all_enabled(self):
        """Test notification config with all notifications enabled."""
        config = NotificationConfig(
            enable_email=True,
            enable_push=True,
            enable_websocket=True,
            enable_slack=True,
            enable_discord=True
        )
        
        assert config.enable_email == True
        assert config.enable_push == True
        assert config.enable_websocket == True
        assert config.enable_slack == True
        assert config.enable_discord == True


class TestMainConfig:
    """Test main configuration class."""
    
    def test_config_initialization(self):
        """Test main config initialization."""
        config = Config()
        
        assert isinstance(config.security, SecurityConfig)
        assert isinstance(config.database, DatabaseConfig)
        assert isinstance(config.rate_limit, RateLimitConfig)
        assert isinstance(config.upload, UploadConfig)
        assert isinstance(config.email, EmailConfig)
        assert isinstance(config.notifications, NotificationConfig)
        
    def test_config_with_custom_components(self):
        """Test config with custom component configurations."""
        custom_security = SecurityConfig(password_min_length=15)
        custom_database = DatabaseConfig(pool_size=25)
        
        config = Config(
            security=custom_security,
            database=custom_database
        )
        
        assert config.security.password_min_length == 15
        assert config.database.pool_size == 25
        
    def test_config_from_file(self):
        """Test loading config from JSON file."""
        config_data = {
            'security': {
                'password_min_length': 12,
                'lockout_threshold': 3
            },
            'database': {
                'pool_size': 20,
                'echo': True
            },
            'upload': {
                'max_file_size': 50 * 1024 * 1024
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            config_file = f.name
            
        try:
            config = Config.from_file(config_file)
            
            assert config.security.password_min_length == 12
            assert config.security.lockout_threshold == 3
            assert config.database.pool_size == 20
            assert config.database.echo == True
            assert config.upload.max_file_size == 50 * 1024 * 1024
        finally:
            os.unlink(config_file)
            
    def test_config_from_nonexistent_file(self):
        """Test loading config from nonexistent file."""
        with pytest.raises(FileNotFoundError):
            Config.from_file('/nonexistent/config.json')
            
    def test_config_from_invalid_json(self):
        """Test loading config from invalid JSON file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write('invalid json content')
            config_file = f.name
            
        try:
            with pytest.raises(json.JSONDecodeError):
                Config.from_file(config_file)
        finally:
            os.unlink(config_file)


class TestConfigSingleton:
    """Test configuration singleton functionality."""
    
    def test_get_config_singleton(self):
        """Test that get_config returns the same instance."""
        config1 = get_config()
        config2 = get_config()
        
        assert config1 is config2
        assert isinstance(config1, Config)
        
    def test_get_config_with_file(self):
        """Test get_config with config file."""
        config_data = {
            'security': {'password_min_length': 8},
            'database': {'pool_size': 15}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            config_file = f.name
            
        try:
            # Clear any existing singleton
            if hasattr(get_config, '_instance'):
                delattr(get_config, '_instance')
                
            config = get_config(config_file)
            
            assert config.security.password_min_length == 8
            assert config.database.pool_size == 15
            
            # Second call should return same instance
            config2 = get_config()
            assert config is config2
        finally:
            os.unlink(config_file)


class TestConfigValidation:
    """Test configuration validation."""
    
    def test_security_config_validation(self):
        """Test security config parameter validation."""
        # Test invalid password length
        with pytest.raises(ValueError):
            SecurityConfig(password_min_length=0)
            
        with pytest.raises(ValueError):
            SecurityConfig(password_min_length=-5)
            
        # Test invalid lockout threshold
        with pytest.raises(ValueError):
            SecurityConfig(lockout_threshold=0)
            
    def test_database_config_validation(self):
        """Test database config parameter validation."""
        # Test invalid pool size
        with pytest.raises(ValueError):
            DatabaseConfig(pool_size=0)
            
        with pytest.raises(ValueError):
            DatabaseConfig(pool_size=-1)
            
        # Test invalid pool timeout
        with pytest.raises(ValueError):
            DatabaseConfig(pool_timeout=0)
            
    def test_upload_config_validation(self):
        """Test upload config parameter validation."""
        # Test invalid file size
        with pytest.raises(ValueError):
            UploadConfig(max_file_size=0)
            
        with pytest.raises(ValueError):
            UploadConfig(max_total_size=-1)
            
    def test_email_config_validation(self):
        """Test email config parameter validation."""
        # Test invalid SMTP port
        with pytest.raises(ValueError):
            EmailConfig(smtp_port=0)
            
        with pytest.raises(ValueError):
            EmailConfig(smtp_port=70000)  # Port too high


class TestConfigEnvironmentOverrides:
    """Test environment variable overrides."""
    
    def test_environment_override_precedence(self):
        """Test that environment variables override config file values."""
        config_data = {
            'security': {'password_min_length': 8},
            'database': {'pool_size': 10}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            config_file = f.name
            
        try:
            with patch.dict(os.environ, {
                'PASSWORD_MIN_LENGTH': '12',
                'DB_POOL_SIZE': '20'
            }):
                # Clear singleton
                if hasattr(get_config, '_instance'):
                    delattr(get_config, '_instance')
                    
                config = get_config(config_file)
                
                # Environment variables should override file values
                assert config.security.password_min_length == 12
                assert config.database.pool_size == 20
        finally:
            os.unlink(config_file)
            
    def test_partial_environment_override(self):
        """Test partial override of configuration."""
        with patch.dict(os.environ, {
            'PASSWORD_MIN_LENGTH': '15',
            # Don't set other security config
        }):
            # Clear singleton
            if hasattr(get_config, '_instance'):
                delattr(get_config, '_instance')
                
            config = get_config()
            
            # Only the specified env var should be overridden
            assert config.security.password_min_length == 15
            assert config.security.lockout_threshold == 5  # Default value


class TestConfigSerialization:
    """Test configuration serialization."""
    
    def test_config_to_dict(self):
        """Test converting config to dictionary."""
        config = Config()
        config_dict = config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert 'security' in config_dict
        assert 'database' in config_dict
        assert 'rate_limit' in config_dict
        assert 'upload' in config_dict
        assert 'email' in config_dict
        assert 'notifications' in config_dict
        
        # Check nested structure
        assert isinstance(config_dict['security'], dict)
        assert 'password_min_length' in config_dict['security']
        
    def test_config_to_json(self):
        """Test converting config to JSON."""
        config = Config()
        json_str = config.to_json()
        
        assert isinstance(json_str, str)
        
        # Should be valid JSON
        parsed = json.loads(json_str)
        assert isinstance(parsed, dict)
        assert 'security' in parsed


if __name__ == '__main__':
    pytest.main([__file__])