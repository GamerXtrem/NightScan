"""
Environment-specific configuration loaders for NightScan.

This module provides pre-configured settings for different deployment environments
(development, testing, staging, production) with appropriate defaults and overrides.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional

# Import the configuration
sys.path.append(str(Path(__file__).parent.parent))
from config import NightScanConfig, get_config, load_config


def get_development_config() -> Dict[str, Any]:
    """Get development environment configuration."""
    return {
        "environment": "development",
        "debug": True,
        "database": {
            "uri": "sqlite:///nightscan_dev.db",
            "echo": True,  # Enable SQL logging
            "pool_size": 5,
            "max_overflow": 0,  # No overflow needed for development
            "pool_pre_ping": False,  # Not needed for SQLite
            "pool_recycle": 3600,
            "echo_pool": True  # Debug pool events in development
        },
        "redis": {
            "url": "redis://localhost:6379/0",
            "enabled": True,
            "max_connections": 5
        },
        "security": {
            "force_https": False,
            "password_min_length": 8  # Relaxed for development
        },
        "rate_limit": {
            "enabled": False  # Disable rate limiting in dev
        },
        "logging": {
            "level": "DEBUG",
            "format": "text",  # Human-readable logs
            "file_path": "logs/dev.log"
        },
        "upload": {
            "upload_dir": "uploads/dev",
            "max_file_size": 52428800  # 50MB for testing
        },
        "model": {
            "batch_size": 8,  # Smaller batch for development
            "device": "cpu"  # Use CPU in development
        },
        "api": {
            "enable_docs": True,
            "cors_origins": ["http://localhost:3000", "http://localhost:8080"]
        }
    }


def get_testing_config() -> Dict[str, Any]:
    """Get testing environment configuration."""
    return {
        "environment": "testing",
        "debug": False,
        "database": {
            "uri": "sqlite:///:memory:",  # In-memory for tests
            "pool_size": 1,
            "max_overflow": 0,  # No overflow for tests
            "pool_pre_ping": False,  # Not needed for in-memory
            "pool_recycle": 3600,
            "echo": False
        },
        "redis": {
            "url": "redis://localhost:6379/15",  # Separate Redis DB
            "enabled": True,
            "max_connections": 2
        },
        "security": {
            "secret_key": "test-secret-key-do-not-use-in-production",
            "force_https": False
        },
        "rate_limit": {
            "enabled": False
        },
        "logging": {
            "level": "WARNING",  # Less verbose for tests
            "format": "json",
            "file": None  # No file logging during tests
        },
        "upload": {
            "upload_dir": "/tmp/nightscan_test_uploads",
            "max_file_size": 10485760  # 10MB
        },
        "model": {
            "audio_model_path": "tests/fixtures/test_audio_model.pth",
            "audio_csv_dir": "tests/fixtures/audio_csv",
            "photo_model_path": "tests/fixtures/test_photo_model.pth",
            "photo_data_dir": "tests/fixtures/photo_data",
            "device": "cpu"
        },
        "ports": {
            "web_app": 9000,
            "api_v1": 9001,
            "prediction_api": 9002
        }
    }


def get_staging_config() -> Dict[str, Any]:
    """Get staging environment configuration."""
    return {
        "environment": "staging",
        "debug": False,
        "database": {
            "uri": os.environ.get("STAGING_DATABASE_URL", 
                                "postgresql://nightscan:password@localhost/nightscan_staging"),
            "pool_size": 15,
            "max_overflow": 10,  # Allow temporary connections for spikes
            "pool_pre_ping": True,  # Ensure connections are alive
            "pool_recycle": 1800  # 30 minutes
        },
        "redis": {
            "url": os.environ.get("STAGING_REDIS_URL", "redis://localhost:6379/1"),
            "enabled": True,
            "max_connections": 10
        },
        "security": {
            "secret_key": os.environ.get("STAGING_SECRET_KEY"),
            "force_https": True
        },
        "rate_limit": {
            "enabled": True,
            "default_limit": "2000 per day"  # Higher limits for staging
        },
        "logging": {
            "level": "INFO",
            "format": "json",
            "file": "/var/log/nightscan/staging.log"
        },
        "upload": {
            "upload_dir": "/data/nightscan/staging/uploads",
            "max_file_size": 104857600  # 100MB
        },
        "model": {
            "audio_model_path": "/models/staging/audio/best_model.pth",
            "photo_model_path": "/models/staging/photo/best_model.pth",
            "batch_size": 16,
            "device": "cuda" if os.environ.get("CUDA_AVAILABLE") else "cpu"
        },
        "api": {
            "base_url": "https://api-staging.nightscan.com",
            "enable_docs": True,  # Keep docs in staging
            "cors_origins": [os.environ.get("STAGING_FRONTEND_URL", "https://staging.nightscan.com")]
        }
    }


def get_production_config() -> Dict[str, Any]:
    """Get production environment configuration."""
    return {
        "environment": "production",
        "debug": False,
        "database": {
            "uri": os.environ.get("DATABASE_URL"),  # Required in production
            "pool_size": 50,  # Increased for production load
            "max_overflow": 25,  # Handle traffic spikes
            "pool_pre_ping": True,  # Critical for production stability
            "pool_recycle": 1800,  # 30 minutes to avoid stale connections
            "pool_timeout": 60,  # Longer timeout for production
            "echo": False,
            "echo_pool": False  # Disable pool logging in production
        },
        "redis": {
            "url": os.environ.get("REDIS_URL"),  # Required in production
            "enabled": True,
            "max_connections": 20
        },
        "security": {
            "secret_key": os.environ.get("SECRET_KEY"),  # Required
            "csrf_secret_key": os.environ.get("CSRF_SECRET_KEY"),  # Required
            "force_https": True,
            "password_min_length": 12,
            "lockout_threshold": 3
        },
        "rate_limit": {
            "enabled": True,
            "default_limit": "1000 per day",
            "login_limit": "3 per minute",
            "prediction_limit": "10 per minute"
        },
        "logging": {
            "level": "WARNING",
            "format": "json",
            "file": "/var/log/nightscan/app.log"
        },
        "upload": {
            "upload_dir": "/data/nightscan/uploads",
            "max_file_size": 104857600  # 100MB
        },
        "model": {
            "audio_model_path": "/models/production/audio/best_model.pth",
            "audio_csv_dir": "/data/nightscan/audio_csv",
            "photo_model_path": "/models/production/photo/best_model.pth",
            "photo_data_dir": "/data/nightscan/photo_data",
            "batch_size": 32,
            "device": "cuda"
        },
        "api": {
            "base_url": "https://api.nightscan.com",
            "enable_docs": False,  # Disable docs in production
            "cors_origins": ["https://app.nightscan.com", "https://nightscan.com"]
        },
        "optimized_serving": {
            "enabled": True,
            "max_batch_size": 16,
            "worker_threads": 8
        }
    }


def get_raspberry_pi_config() -> Dict[str, Any]:
    """Get Raspberry Pi specific configuration."""
    return {
        "environment": "raspberry_pi",
        "debug": False,
        "database": {
            "uri": "sqlite:///home/pi/nightscan/nightscan.db",
            "pool_size": 2,  # Limited resources
            "max_overflow": 1,  # Minimal overflow for Pi
            "pool_pre_ping": False,  # Save resources on Pi
            "pool_recycle": 3600
        },
        "redis": {
            "enabled": False  # No Redis on Pi
        },
        "security": {
            "force_https": False
        },
        "rate_limit": {
            "enabled": False
        },
        "logging": {
            "level": "INFO",
            "format": "text",
            "file": "/home/pi/nightscan/logs/nightscan.log",
            "max_file_size": 5242880  # 5MB
        },
        "raspberry_pi": {
            "data_dir": "/home/pi/nightscan/data",
            "capture_duration": 10,  # Longer capture for Pi
            "base_sleep": 0.2,  # Slightly longer sleep
            "max_sleep": 5.0
        },
        "model": {
            "device": "cpu",
            "batch_size": 1  # Process one at a time
        }
    }


def get_vps_lite_config() -> Dict[str, Any]:
    """Get VPS Lite configuration (optimized for limited resources)."""
    return {
        "environment": "vps_lite",
        "debug": False,
        "database": {
            "uri": os.environ.get("DATABASE_URL"),  # Required
            "pool_size": 20,  # Balanced for VPS resources
            "max_overflow": 10,  # Some flexibility
            "pool_pre_ping": True,  # Important for VPS
            "pool_recycle": 1800,  # 30 minutes
            "pool_timeout": 30
        },
        "redis": {
            "url": os.environ.get("REDIS_URL"),
            "enabled": True,
            "max_connections": 15
        },
        "security": {
            "secret_key": os.environ.get("SECRET_KEY"),  # Required
            "csrf_secret_key": os.environ.get("CSRF_SECRET_KEY"),  # Required
            "force_https": True,
            "password_min_length": 10
        },
        "rate_limit": {
            "enabled": True,
            "default_limit": "500 per day",
            "login_limit": "5 per minute"
        },
        "logging": {
            "level": "WARNING",  # Less verbose to save disk
            "format": "json"
        },
        "upload": {
            "max_file_size": 104857600,  # 100MB
            "max_total_size": 5368709120  # 5GB total
        },
        "model": {
            "batch_size": 16,  # Optimized for VPS
            "max_workers": 2,  # Limited CPU
            "cache_enabled": True
        }
    }


# Environment configuration mapping
ENVIRONMENT_CONFIGS = {
    "development": get_development_config,
    "testing": get_testing_config,
    "staging": get_staging_config,
    "production": get_production_config,
    "vps_lite": get_vps_lite_config,
    "raspberry_pi": get_raspberry_pi_config
}


def load_environment_config(environment: Optional[str] = None) -> NightScanConfig:
    """
    Load configuration for a specific environment.
    
    Args:
        environment: Environment name (development, testing, staging, production, raspberry_pi)
                    If not provided, uses NIGHTSCAN_ENV environment variable
    
    Returns:
        NightScanConfig instance configured for the environment
    """
    if environment is None:
        environment = os.environ.get("NIGHTSCAN_ENV", "development")
    
    if environment not in ENVIRONMENT_CONFIGS:
        raise ValueError(f"Unknown environment: {environment}. "
                        f"Valid options: {list(ENVIRONMENT_CONFIGS.keys())}")
    
    # Get environment-specific configuration
    env_config = ENVIRONMENT_CONFIGS[environment]()
    
    # Check for environment-specific config file
    config_file = f"config/{environment}.json"
    if Path(config_file).exists():
        # Load from file if it exists
        return load_config(config_file)
    
    # Otherwise create from environment config dict
    os.environ["NIGHTSCAN_ENV"] = environment
    
    # If pydantic is available, use it to create config from dict
    try:
        return NightScanConfig(**env_config)
    except:
        # Otherwise just load with environment set
        return load_config()


def validate_environment_config(environment: str) -> bool:
    """
    Validate that all required settings are present for an environment.
    
    Args:
        environment: Environment name to validate
    
    Returns:
        True if valid, raises ValueError if not
    """
    config = load_environment_config(environment)
    
    # Production-specific validation
    if environment == "production":
        required_vars = [
            ("DATABASE_URL", "Database URL"),
            ("REDIS_URL", "Redis URL"),
            ("SECRET_KEY", "Flask secret key"),
            ("CSRF_SECRET_KEY", "CSRF secret key")
        ]
        
        missing = []
        for var, desc in required_vars:
            if not os.environ.get(var):
                missing.append(f"{desc} ({var})")
        
        if missing:
            raise ValueError(f"Missing required production settings: {', '.join(missing)}")
    
    # Validate model paths exist
    if hasattr(config, 'model'):
        if environment != "testing":  # Skip for testing environment
            model_paths = []
            if hasattr(config.model, 'audio_model_path'):
                model_paths.append(("Audio", config.model.audio_model_path))
            if hasattr(config.model, 'photo_model_path'):
                model_paths.append(("Photo", config.model.photo_model_path))
            
            for model_type, path in model_paths:
                if not Path(path).exists() and environment in ["staging", "production"]:
                    raise ValueError(f"{model_type} model not found at {path}")
    
    return True


def get_environment_info() -> Dict[str, Any]:
    """Get information about the current environment configuration."""
    env = os.environ.get("NIGHTSCAN_ENV", "development")
    config = load_environment_config(env)
    
    info = {
        "environment": env,
        "debug": getattr(config, 'debug', False),
        "database_type": "postgresql" if "postgresql" in getattr(config.database, 'uri', '') else "sqlite",
        "redis_enabled": getattr(config.redis, 'enabled', False) if hasattr(config, 'redis') else False,
        "https_enforced": getattr(config.security, 'force_https', False) if hasattr(config, 'security') else False,
        "rate_limiting": getattr(config.rate_limit, 'enabled', False) if hasattr(config, 'rate_limit') else False,
        "api_docs_enabled": getattr(config.api, 'enable_docs', True) if hasattr(config, 'api') else True
    }
    
    # Add device info for ML models
    if hasattr(config, 'model'):
        info["ml_device"] = getattr(config.model, 'device', 'cpu')
    
    return info


if __name__ == "__main__":
    import sys
    
    # Test environment configuration loading
    env_to_test = sys.argv[1] if len(sys.argv) > 1 else "development"
    
    print(f"Loading configuration for environment: {env_to_test}")
    
    try:
        config = load_environment_config(env_to_test)
        print(f"✓ Configuration loaded successfully")
        
        # Validate configuration
        validate_environment_config(env_to_test)
        print(f"✓ Configuration validated successfully")
        
        # Show environment info
        info = get_environment_info()
        print(f"\nEnvironment Information:")
        for key, value in info.items():
            print(f"  {key}: {value}")
        
        # Show some key settings
        print(f"\nKey Settings:")
        print(f"  Database: {config.database.uri if hasattr(config, 'database') else 'N/A'}")
        # Debug status logged to application logs only
        
        if hasattr(config, 'ports'):
            print(f"  Web Port: {config.ports.web_app}")
            print(f"  API Port: {config.ports.api_v1}")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        sys.exit(1)