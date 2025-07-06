"""Centralized configuration management for NightScan."""

import os
import logging
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from pathlib import Path
import json

try:
    from pydantic import BaseSettings, validator, Field
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    # Fallback for environments without pydantic
    class BaseSettings:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)


@dataclass
class DatabaseConfig:
    """Database configuration."""
    uri: str = "sqlite:///nightscan.db"
    pool_size: int = 10
    pool_timeout: int = 30
    pool_recycle: int = 3600
    echo: bool = False


@dataclass 
class RedisConfig:
    """Redis caching configuration."""
    url: str = "redis://localhost:6379/0"
    enabled: bool = True
    default_ttl: int = 3600  # 1 hour
    max_connections: int = 10


@dataclass
class SecurityConfig:
    """Security configuration."""
    secret_key: Optional[str] = None
    csrf_secret_key: Optional[str] = None
    password_min_length: int = 10
    lockout_threshold: int = 5
    lockout_window: int = 1800  # 30 minutes
    lockout_file: str = "failed_logins.json"
    force_https: bool = True


@dataclass
class RateLimitConfig:
    """Rate limiting configuration."""
    enabled: bool = True
    default_limit: str = "1000 per day"
    login_limit: str = "5 per minute"
    prediction_limit: str = "10 per minute"
    upload_limit: str = "10 per minute"


@dataclass
class ModelConfig:
    """ML model configuration."""
    model_path: str = "models/best_model.pth"
    csv_dir: str = "data/processed/csv"
    batch_size: int = 1
    max_audio_duration: int = 600  # 10 minutes
    supported_formats: List[str] = field(default_factory=lambda: ["wav"])


@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = "INFO"
    format: str = "json"  # json or text
    file: Optional[str] = None
    max_file_size: int = 10485760  # 10MB
    backup_count: int = 5


@dataclass
class FileUploadConfig:
    """File upload configuration."""
    max_file_size: int = 104857600  # 100MB
    max_total_size: int = 10737418240  # 10GB
    allowed_extensions: List[str] = field(default_factory=lambda: [".wav"])
    upload_dir: str = "uploads"


@dataclass
class ApiConfig:
    """API configuration."""
    version: str = "1.0.0"
    cors_origins: List[str] = field(default_factory=list)
    rate_limit: str = "60 per minute"
    enable_docs: bool = True
    base_url: str = "http://localhost:8001"


@dataclass
class OptimizedServingConfig:
    """Optimized ML serving configuration."""
    enabled: bool = True
    
    # Connection pooling
    db_min_connections: int = 5
    db_max_connections: int = 20
    db_connection_timeout: float = 10.0
    
    redis_min_connections: int = 3
    redis_max_connections: int = 15
    redis_connection_timeout: float = 5.0
    
    http_max_connections: int = 100
    http_connection_timeout: float = 10.0
    
    # Model instance pooling
    model_pool_size: int = 3
    model_warmup_requests: int = 5
    
    # Batch processing
    batch_timeout_ms: float = 100.0
    max_batch_size: int = 8
    request_queue_size: int = 1000
    worker_threads: int = 4
    
    # Performance settings
    enable_async_inference: bool = True
    enable_batch_processing: bool = True
    enable_model_pooling: bool = True
    fallback_to_original: bool = True


@dataclass
class RaspberryPiConfig:
    """Raspberry Pi specific configuration."""
    data_dir: str = "data"
    audio_dir: str = "data/audio"
    image_dir: str = "data/images"
    spectrogram_dir: str = "data/spectrograms"
    start_hour: int = 18
    stop_hour: int = 10
    capture_duration: int = 8  # seconds
    base_sleep: float = 0.1
    max_sleep: float = 5.0


if PYDANTIC_AVAILABLE:
    class NightScanConfig(BaseSettings):
        """Main configuration class with validation."""
        
        # Environment
        environment: str = Field(default="development", env="NIGHTSCAN_ENV")
        debug: bool = Field(default=False, env="DEBUG")
        
        # Database
        database: DatabaseConfig = DatabaseConfig()
        
        # Redis
        redis: RedisConfig = RedisConfig()
        
        # Security
        security: SecurityConfig = SecurityConfig()
        
        # Rate limiting
        rate_limit: RateLimitConfig = RateLimitConfig()
        
        # ML Model
        model: ModelConfig = ModelConfig()
        
        # Logging
        logging: LoggingConfig = LoggingConfig()
        
        # File uploads
        upload: FileUploadConfig = FileUploadConfig()
        
        # API
        api: ApiConfig = ApiConfig()
        
        # Raspberry Pi
        raspberry_pi: RaspberryPiConfig = RaspberryPiConfig()
        
        # Optimized serving
        optimized_serving: OptimizedServingConfig = OptimizedServingConfig()
        
        class Config:
            env_prefix = "NIGHTSCAN_"
            case_sensitive = False
            
        @validator('database')
        def validate_database_uri(cls, v):
            if not v.uri:
                raise ValueError("Database URI is required")
            return v
        
        @validator('security')
        def validate_security(cls, v):
            if not v.secret_key:
                v.secret_key = os.urandom(32).hex()
                logging.warning("Generated random SECRET_KEY. Set NIGHTSCAN_SECURITY__SECRET_KEY for production.")
            return v
        
        @validator('model')
        def validate_model_paths(cls, v):
            model_path = Path(v.model_path)
            csv_dir = Path(v.csv_dir)
            
            if not model_path.exists() and os.getenv("NIGHTSCAN_ENV") == "production":
                raise ValueError(f"Model file not found: {model_path}")
            
            if not csv_dir.exists() and os.getenv("NIGHTSCAN_ENV") == "production":
                raise ValueError(f"CSV directory not found: {csv_dir}")
            
            return v
        
        @validator('upload')
        def validate_upload_config(cls, v):
            if v.max_file_size > v.max_total_size:
                raise ValueError("max_file_size cannot be larger than max_total_size")
            
            # Create upload directory if it doesn't exist
            Path(v.upload_dir).mkdir(parents=True, exist_ok=True)
            
            return v
        
        @validator('raspberry_pi')
        def validate_pi_hours(cls, v):
            if not (0 <= v.start_hour <= 23) or not (0 <= v.stop_hour <= 23):
                raise ValueError("start_hour and stop_hour must be between 0 and 23")
            
            if v.base_sleep > v.max_sleep:
                raise ValueError("base_sleep cannot be greater than max_sleep")
            
            return v

else:
    # Fallback configuration without validation
    class NightScanConfig:
        def __init__(self):
            self.environment = os.getenv("NIGHTSCAN_ENV", "development")
            self.debug = os.getenv("DEBUG", "false").lower() == "true"
            
            self.database = DatabaseConfig()
            self.redis = RedisConfig()
            self.security = SecurityConfig()
            self.rate_limit = RateLimitConfig()
            self.model = ModelConfig()
            self.logging = LoggingConfig()
            self.upload = FileUploadConfig()
            self.api = ApiConfig()
            self.raspberry_pi = RaspberryPiConfig()
            self.optimized_serving = OptimizedServingConfig()
            
            # Load from environment variables
            self._load_from_env()
        
        def _load_from_env(self):
            """Load configuration from environment variables."""
            # Database
            if os.getenv("SQLALCHEMY_DATABASE_URI"):
                self.database.uri = os.getenv("SQLALCHEMY_DATABASE_URI")
            
            # Redis
            if os.getenv("REDIS_URL"):
                self.redis.url = os.getenv("REDIS_URL")
            
            # Security
            if os.getenv("SECRET_KEY"):
                self.security.secret_key = os.getenv("SECRET_KEY")
            
            # Model
            if os.getenv("MODEL_PATH"):
                self.model.model_path = os.getenv("MODEL_PATH")
            if os.getenv("CSV_DIR"):
                self.model.csv_dir = os.getenv("CSV_DIR")


# Global configuration instance
_config: Optional[NightScanConfig] = None


def get_config() -> NightScanConfig:
    """Get the global configuration instance."""
    global _config
    
    if _config is None:
        _config = load_config()
    
    return _config


def load_config(config_file: Optional[str] = None) -> NightScanConfig:
    """Load configuration from file and environment variables."""
    
    # Load from JSON file if provided
    config_data = {}
    if config_file and Path(config_file).exists():
        try:
            with open(config_file, 'r') as f:
                config_data = json.load(f)
        except Exception as e:
            logging.warning(f"Failed to load config file {config_file}: {e}")
    
    if PYDANTIC_AVAILABLE:
        return NightScanConfig(**config_data)
    else:
        config = NightScanConfig()
        # Apply config file data
        for section, values in config_data.items():
            if hasattr(config, section):
                section_obj = getattr(config, section)
                for key, value in values.items():
                    if hasattr(section_obj, key):
                        setattr(section_obj, key, value)
        return config


def load_config_for_environment(env: str) -> NightScanConfig:
    """Load configuration for a specific environment."""
    config_file = f"config/{env}.json"
    
    if Path(config_file).exists():
        return load_config(config_file)
    else:
        # Fallback to environment variables
        os.environ["NIGHTSCAN_ENV"] = env
        return load_config()


def validate_config(config: NightScanConfig) -> List[str]:
    """Validate configuration and return list of errors."""
    errors = []
    
    # Check required paths exist in production
    if config.environment == "production":
        model_path = Path(config.model.model_path)
        if not model_path.exists():
            errors.append(f"Model file not found: {model_path}")
        
        csv_dir = Path(config.model.csv_dir)
        if not csv_dir.exists():
            errors.append(f"CSV directory not found: {csv_dir}")
    
    # Check database URI
    if not config.database.uri:
        errors.append("Database URI is required")
    
    # Check secret key in production
    if config.environment == "production" and not config.security.secret_key:
        errors.append("SECRET_KEY is required in production")
    
    # Check file size limits
    if config.upload.max_file_size > config.upload.max_total_size:
        errors.append("max_file_size cannot be larger than max_total_size")
    
    return errors


def create_example_config(output_file: str = "config/example.json"):
    """Create an example configuration file."""
    config = NightScanConfig() if PYDANTIC_AVAILABLE else NightScanConfig()
    
    config_dict = {
        "environment": "production",
        "debug": False,
        "database": {
            "uri": "postgresql://user:password@localhost/nightscan",
            "pool_size": 10,
            "echo": False
        },
        "redis": {
            "url": "redis://localhost:6379/0",
            "enabled": True,
            "default_ttl": 3600
        },
        "security": {
            "secret_key": os.environ.get("SECRET_KEY"),
            "force_https": True,
            "lockout_threshold": 5
        },
        "model": {
            "model_path": "/path/to/best_model.pth",
            "csv_dir": "/path/to/csv/files",
            "batch_size": 4
        },
        "api": {
            "cors_origins": ["https://your-frontend.com"],
            "rate_limit": "60 per minute",
            "base_url": "https://api.nightscan.com"
        },
        "logging": {
            "level": "INFO",
            "format": "json",
            "file": "/var/log/nightscan/app.log"
        }
    }
    
    # Create config directory
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    print(f"Example configuration created at {output_file}")


if __name__ == "__main__":
    # Create example config when run directly
    create_example_config()
    
    # Test configuration loading
    config = get_config()
    errors = validate_config(config)
    
    if errors:
        print("Configuration errors:")
        for error in errors:
            print(f"  - {error}")
    else:
        print("Configuration is valid!")
    
    print(f"Environment: {config.environment}")
    print(f"Database: {config.database.uri}")
    print(f"Redis enabled: {config.redis.enabled}")
    print(f"Model path: {config.model.model_path}")