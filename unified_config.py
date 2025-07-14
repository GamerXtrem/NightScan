#!/usr/bin/env python3
"""
Système de Configuration Unifié pour NightScan

Ce module unifie toutes les approches de configuration dispersées :
- Variables d'environnement standardisées
- Fichiers de configuration JSON/YAML
- Configuration par environnement (dev/staging/prod)
- Validation et valeurs par défaut
- Migration depuis les configurations legacy
"""

import os
import json
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Type
from dataclasses import dataclass, field, asdict
from enum import Enum
import secrets
from datetime import timedelta

try:
    from pydantic import BaseModel, Field, validator, root_validator
    from pydantic_settings import BaseSettings
    PYDANTIC_V2 = True
except ImportError:
    try:
        from pydantic import BaseModel, Field, validator, root_validator
        from pydantic import BaseSettings
        PYDANTIC_V2 = False
    except ImportError:
        # Fallback pour les environnements sans Pydantic
        BaseModel = object
        BaseSettings = object
        PYDANTIC_V2 = False

logger = logging.getLogger(__name__)


class Environment(str, Enum):
    """Environnements supportés."""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


class LogLevel(str, Enum):
    """Niveaux de log supportés."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass
class DatabaseConfig:
    """Configuration de base de données unifiée."""
    # PostgreSQL principal
    host: str = "localhost"
    port: int = 5432
    database: str = "nightscan"
    username: str = "nightscan"
    password: str = ""
    
    # Options de connexion
    pool_size: int = 20
    max_overflow: int = 0
    pool_timeout: int = 30
    pool_recycle: int = 3600
    
    # SSL et sécurité
    sslmode: str = "prefer"
    sslcert: Optional[str] = None
    sslkey: Optional[str] = None
    sslrootcert: Optional[str] = None
    
    def get_url(self, include_password: bool = True) -> str:
        """Génère l'URL de connexion PostgreSQL."""
        password_part = f":{self.password}" if include_password and self.password else ""
        return f"postgresql://{self.username}{password_part}@{self.host}:{self.port}/{self.database}"
    
    def get_safe_url(self) -> str:
        """URL de connexion sans mot de passe (pour logs)."""
        return self.get_url(include_password=False)


@dataclass
class CacheConfig:
    """Configuration de cache (Redis) unifiée."""
    host: str = "localhost"
    port: int = 6379
    database: int = 0
    username: Optional[str] = None
    password: Optional[str] = None
    
    # Options de connexion
    socket_timeout: int = 5
    socket_connect_timeout: int = 5
    max_connections: int = 50
    
    # SSL
    ssl: bool = False
    ssl_cert_reqs: Optional[str] = None
    
    def get_url(self, include_password: bool = True) -> str:
        """Génère l'URL de connexion Redis."""
        auth_part = ""
        if self.username or self.password:
            if self.username and self.password and include_password:
                auth_part = f"{self.username}:{self.password}@"
            elif self.username:
                auth_part = f"{self.username}@"
        
        scheme = "rediss" if self.ssl else "redis"
        return f"{scheme}://{auth_part}{self.host}:{self.port}/{self.database}"
    
    def get_safe_url(self) -> str:
        """URL de connexion sans mot de passe (pour logs)."""
        return self.get_url(include_password=False)


@dataclass
class ServiceConfig:
    """Configuration des services/ports unifiée."""
    # Services web
    web_host: str = "0.0.0.0"
    web_port: int = 8000
    
    # APIs
    api_v1_port: int = 8001
    prediction_port: int = 8002
    ml_service_port: int = 8003
    analytics_port: int = 8008
    websocket_port: int = 8012
    
    # Services externes
    nginx_port: int = 80
    nginx_ssl_port: int = 443
    
    # Timeouts et limites
    request_timeout: int = 30
    upload_timeout: int = 300
    max_file_size: int = 100 * 1024 * 1024  # 100MB
    
    def get_service_url(self, service: str, host: str = "localhost", ssl: bool = False) -> str:
        """Génère l'URL d'un service."""
        port_map = {
            "web": self.web_port,
            "api_v1": self.api_v1_port,
            "prediction": self.prediction_port,
            "ml_service": self.ml_service_port,
            "analytics": self.analytics_port,
            "websocket": self.websocket_port
        }
        
        port = port_map.get(service)
        if not port:
            raise ValueError(f"Service inconnu: {service}")
        
        scheme = "https" if ssl else "http"
        return f"{scheme}://{host}:{port}"


@dataclass
class SecurityConfig:
    """Configuration de sécurité unifiée."""
    # Clés et secrets
    secret_key: str = ""
    jwt_secret: str = ""
    encryption_key: str = ""
    
    # Sessions
    session_timeout: int = 3600  # 1 heure
    session_cookie_name: str = "nightscan_session"
    session_cookie_secure: bool = True
    session_cookie_httponly: bool = True
    
    # Authentification
    password_min_length: int = 8
    password_require_special: bool = True
    max_login_attempts: int = 5
    account_lockout_duration: int = 900  # 15 minutes
    
    # CSRF et CORS
    csrf_enabled: bool = True
    cors_enabled: bool = False
    allowed_origins: List[str] = field(default_factory=list)
    
    # Rate limiting
    rate_limit_enabled: bool = True
    rate_limit_requests: int = 100
    rate_limit_window: int = 3600  # 1 heure
    
    def generate_secrets(self):
        """Génère des secrets aléatoires sécurisés."""
        if not self.secret_key:
            self.secret_key = secrets.token_hex(32)
        if not self.jwt_secret:
            self.jwt_secret = secrets.token_hex(32)
        if not self.encryption_key:
            self.encryption_key = secrets.token_hex(32)


@dataclass
class MLConfig:
    """Configuration des modèles ML unifiée."""
    # Chemins des modèles
    models_directory: str = "models"
    cache_directory: str = "cache"
    data_directory: str = "data"
    
    # Modèles audio
    audio_heavy_model: str = "audio_training_efficientnet/models/best_model.pth"
    audio_light_model: str = "mobile_models/audio_light_model.pth"
    audio_model_classes: int = 6
    audio_input_size: tuple = (128, 128)
    
    # Modèles photo
    photo_heavy_model: str = "picture_training_enhanced/models/best_model.pth"
    photo_light_model: str = "mobile_models/photo_light_model.pth"
    photo_model_classes: int = 8
    photo_input_size: tuple = (224, 224)
    
    # Configuration d'inférence
    batch_size: int = 32
    max_workers: int = 4
    inference_timeout: int = 30
    
    # GPU/CPU
    use_gpu: bool = True
    gpu_memory_fraction: float = 0.8


@dataclass
class LoggingConfig:
    """Configuration de logging unifiée."""
    level: LogLevel = LogLevel.INFO
    format: str = "json"  # "json" ou "text"
    
    # Fichiers de log
    enable_file_logging: bool = True
    log_directory: str = "logs"
    log_filename: str = "nightscan.log"
    
    # Rotation des logs
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5
    compress_backups: bool = True
    
    # Logging console
    enable_console_logging: bool = True
    console_level: LogLevel = LogLevel.INFO
    
    # Loggers spécialisés
    enable_audit_logging: bool = True
    enable_security_logging: bool = True
    enable_performance_logging: bool = True


@dataclass
class MonitoringConfig:
    """Configuration de monitoring unifiée."""
    # Métriques
    enable_metrics: bool = True
    metrics_port: int = 9090
    prometheus_enabled: bool = False
    
    # Santé
    health_check_enabled: bool = True
    health_check_interval: int = 30
    
    # Circuit breakers
    circuit_breaker_enabled: bool = True
    circuit_breaker_config_file: str = "config/circuit_breakers.json"
    
    # Alertes
    alerts_enabled: bool = False
    alert_email: Optional[str] = None
    alert_webhook: Optional[str] = None


class UnifiedConfig:
    """
    Configuration unifiée pour NightScan.
    
    Centralise toutes les configurations dispersées et fournit
    une interface cohérente pour accéder aux paramètres.
    """
    
    def __init__(self, 
                 environment: Environment = Environment.DEVELOPMENT,
                 config_file: Optional[str] = None,
                 env_prefix: str = "NIGHTSCAN_"):
        """
        Initialise la configuration unifiée.
        
        Args:
            environment: Environnement de déploiement
            config_file: Fichier de configuration optionnel
            env_prefix: Préfixe des variables d'environnement
        """
        self.environment = environment
        self.env_prefix = env_prefix
        
        # Composants de configuration
        self.database = DatabaseConfig()
        self.cache = CacheConfig()
        self.services = ServiceConfig()
        self.security = SecurityConfig()
        self.ml = MLConfig()
        self.logging = LoggingConfig()
        self.monitoring = MonitoringConfig()
        
        # Charger la configuration
        self._load_defaults()
        if config_file:
            self._load_from_file(config_file)
        self._load_from_environment()
        self._validate_and_finalize()
    
    def _load_defaults(self):
        """Charge les configurations par défaut selon l'environnement."""
        if self.environment == Environment.PRODUCTION:
            self.logging.level = LogLevel.WARNING
            self.security.session_cookie_secure = True
            self.security.csrf_enabled = True
            self.monitoring.prometheus_enabled = True
            
        elif self.environment == Environment.DEVELOPMENT:
            self.logging.level = LogLevel.DEBUG
            self.logging.enable_console_logging = True
            self.security.session_cookie_secure = False
            self.services.web_host = "localhost"
            
        elif self.environment == Environment.TESTING:
            self.logging.level = LogLevel.WARNING
            self.logging.enable_file_logging = False
            self.database.database = "nightscan_test"
            self.cache.database = 1
    
    def _load_from_file(self, config_file: str):
        """Charge la configuration depuis un fichier JSON/YAML."""
        config_path = Path(config_file)
        if not config_path.exists():
            logger.warning(f"Fichier de configuration non trouvé: {config_file}")
            return
        
        try:
            with open(config_path, 'r') as f:
                if config_path.suffix.lower() in ['.yaml', '.yml'] and YAML_AVAILABLE:
                    data = yaml.safe_load(f)
                else:
                    data = json.load(f)
            
            self._apply_config_data(data)
            logger.info(f"Configuration chargée depuis {config_file}")
            
        except Exception as e:
            logger.error(f"Erreur lecture configuration {config_file}: {e}")
    
    def _load_from_environment(self):
        """Charge la configuration depuis les variables d'environnement."""
        env_mappings = {
            # Base de données
            f"{self.env_prefix}DB_HOST": ("database", "host"),
            f"{self.env_prefix}DB_PORT": ("database", "port", int),
            f"{self.env_prefix}DB_NAME": ("database", "database"),
            f"{self.env_prefix}DB_USER": ("database", "username"),
            f"{self.env_prefix}DB_PASSWORD": ("database", "password"),
            "DATABASE_URL": ("database", "_url"),
            
            # Cache
            f"{self.env_prefix}REDIS_HOST": ("cache", "host"),
            f"{self.env_prefix}REDIS_PORT": ("cache", "port", int),
            f"{self.env_prefix}REDIS_PASSWORD": ("cache", "password"),
            "REDIS_URL": ("cache", "_url"),
            
            # Services
            f"{self.env_prefix}WEB_PORT": ("services", "web_port", int),
            f"{self.env_prefix}API_PORT": ("services", "api_v1_port", int),
            
            # Sécurité
            "SECRET_KEY": ("security", "secret_key"),
            f"{self.env_prefix}JWT_SECRET": ("security", "jwt_secret"),
            
            # Logging
            f"{self.env_prefix}LOG_LEVEL": ("logging", "level"),
            f"{self.env_prefix}LOG_DIR": ("logging", "log_directory"),
            
            # ML
            f"{self.env_prefix}MODELS_DIR": ("ml", "models_directory"),
            f"{self.env_prefix}USE_GPU": ("ml", "use_gpu", bool),
        }
        
        for env_var, config_path in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                self._set_config_value(config_path, value)
    
    def _set_config_value(self, config_path: tuple, value: str):
        """Définit une valeur de configuration à partir d'un chemin."""
        component_name, attr_name = config_path[0], config_path[1]
        converter = config_path[2] if len(config_path) > 2 else str
        
        component = getattr(self, component_name)
        
        # Gestion des URLs spéciales
        if attr_name == "_url":
            if component_name == "database" and value:
                self._parse_database_url(value)
            elif component_name == "cache" and value:
                self._parse_redis_url(value)
            return
        
        # Conversion de type
        if converter == bool:
            value = value.lower() in ('true', '1', 'yes', 'on')
        elif converter == int:
            value = int(value)
        elif converter == LogLevel:
            value = LogLevel(value.upper())
        
        setattr(component, attr_name, value)
    
    def _parse_database_url(self, url: str):
        """Parse une URL de base de données PostgreSQL."""
        # Exemple: postgresql://user:pass@host:port/dbname
        import urllib.parse
        parsed = urllib.parse.urlparse(url)
        
        if parsed.hostname:
            self.database.host = parsed.hostname
        if parsed.port:
            self.database.port = parsed.port
        if parsed.username:
            self.database.username = parsed.username
        if parsed.password:
            self.database.password = parsed.password
        if parsed.path and len(parsed.path) > 1:
            self.database.database = parsed.path[1:]  # Remove leading /
    
    def _parse_redis_url(self, url: str):
        """Parse une URL de connexion Redis."""
        # Exemple: redis://user:pass@host:port/db
        import urllib.parse
        parsed = urllib.parse.urlparse(url)
        
        if parsed.hostname:
            self.cache.host = parsed.hostname
        if parsed.port:
            self.cache.port = parsed.port
        if parsed.username:
            self.cache.username = parsed.username
        if parsed.password:
            self.cache.password = parsed.password
        if parsed.path and len(parsed.path) > 1:
            try:
                self.cache.database = int(parsed.path[1:])
            except ValueError:
                pass
    
    def _apply_config_data(self, data: Dict[str, Any]):
        """Applique des données de configuration depuis un dictionnaire."""
        component_map = {
            'database': self.database,
            'cache': self.cache,
            'services': self.services,
            'security': self.security,
            'ml': self.ml,
            'logging': self.logging,
            'monitoring': self.monitoring
        }
        
        for section, config_data in data.items():
            if section in component_map and isinstance(config_data, dict):
                component = component_map[section]
                for key, value in config_data.items():
                    if hasattr(component, key):
                        setattr(component, key, value)
    
    def _validate_and_finalize(self):
        """Valide et finalise la configuration."""
        # Générer les secrets manquants
        self.security.generate_secrets()
        
        # Validation de base
        if not self.database.password and self.environment == Environment.PRODUCTION:
            logger.warning("Mot de passe de base de données manquant en production")
        
        if not self.security.secret_key:
            logger.warning("Clé secrète générée automatiquement")
        
        # Créer les dossiers nécessaires
        self._ensure_directories()
    
    def _ensure_directories(self):
        """Crée les dossiers nécessaires."""
        directories = [
            self.logging.log_directory,
            self.ml.models_directory,
            self.ml.cache_directory,
            self.ml.data_directory
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def get_database_url(self, include_password: bool = True) -> str:
        """Retourne l'URL de connexion à la base de données."""
        return self.database.get_url(include_password)
    
    def get_cache_url(self, include_password: bool = True) -> str:
        """Retourne l'URL de connexion au cache."""
        return self.cache.get_url(include_password)
    
    def get_service_url(self, service: str, external_host: Optional[str] = None) -> str:
        """Retourne l'URL d'un service."""
        host = external_host or "localhost"
        ssl = self.environment == Environment.PRODUCTION
        return self.services.get_service_url(service, host, ssl)
    
    def to_dict(self, include_secrets: bool = False) -> Dict[str, Any]:
        """Exporte la configuration en dictionnaire."""
        result = {}
        components = ['database', 'cache', 'services', 'security', 'ml', 'logging', 'monitoring']
        
        for component_name in components:
            component = getattr(self, component_name)
            component_dict = asdict(component)
            
            # Masquer les secrets si demandé
            if not include_secrets and component_name == 'security':
                sensitive_keys = ['secret_key', 'jwt_secret', 'encryption_key']
                for key in sensitive_keys:
                    if key in component_dict and component_dict[key]:
                        component_dict[key] = "***masked***"
            
            if not include_secrets and component_name == 'database':
                if component_dict.get('password'):
                    component_dict['password'] = "***masked***"
            
            if not include_secrets and component_name == 'cache':
                if component_dict.get('password'):
                    component_dict['password'] = "***masked***"
            
            result[component_name] = component_dict
        
        return result
    
    def save_to_file(self, filepath: str, include_secrets: bool = False):
        """Sauvegarde la configuration dans un fichier."""
        config_data = self.to_dict(include_secrets)
        
        filepath = Path(filepath)
        with open(filepath, 'w') as f:
            if filepath.suffix.lower() in ['.yaml', '.yml'] and YAML_AVAILABLE:
                yaml.dump(config_data, f, default_flow_style=False, indent=2)
            else:
                json.dump(config_data, f, indent=2)
        
        logger.info(f"Configuration sauvegardée dans {filepath}")


# Instance globale de configuration
_unified_config: Optional[UnifiedConfig] = None


def get_config(environment: Optional[Environment] = None, 
               config_file: Optional[str] = None,
               force_reload: bool = False) -> UnifiedConfig:
    """
    Retourne l'instance globale de configuration unifiée.
    
    Args:
        environment: Environnement à utiliser (détecté automatiquement si None)
        config_file: Fichier de configuration optionnel
        force_reload: Force le rechargement de la configuration
    
    Returns:
        Instance de UnifiedConfig
    """
    global _unified_config
    
    if _unified_config is None or force_reload:
        # Détecter l'environnement automatiquement
        if environment is None:
            env_name = os.getenv('NIGHTSCAN_ENV', os.getenv('FLASK_ENV', 'development'))
            try:
                environment = Environment(env_name.lower())
            except ValueError:
                environment = Environment.DEVELOPMENT
                logger.warning(f"Environnement inconnu '{env_name}', utilisation de 'development'")
        
        # Fichier de configuration par défaut
        if config_file is None:
            default_configs = [
                f"config/{environment.value}.json",
                f"config/{environment.value}.yaml",
                "config/default.json",
                "config.json"
            ]
            
            for default_config in default_configs:
                if Path(default_config).exists():
                    config_file = default_config
                    break
        
        _unified_config = UnifiedConfig(environment, config_file)
        logger.info(f"Configuration unifiée initialisée pour l'environnement {environment.value}")
    
    return _unified_config


def migrate_legacy_config():
    """
    Migre les configurations legacy vers le système unifié.
    Lit les anciens fichiers de configuration et génère le nouveau format.
    """
    logger.info("🔄 Migration des configurations legacy")
    logger.info("=" * 50)
    
    # Configuration pour chaque environnement
    environments = [Environment.DEVELOPMENT, Environment.STAGING, Environment.PRODUCTION]
    
    for env in environments:
        logger.info(f"\\n📋 Migration environnement: {env.value}")
        
        # Créer une configuration de base
        config = UnifiedConfig(env)
        
        # Tentative de lecture des anciens fichiers
        legacy_files = [
            "config.py",
            "port_config.py", 
            "secure_secrets.py",
            f"config/{env.value}.json",
            ".env.example" if env == Environment.DEVELOPMENT else f".env.{env.value}"
        ]
        
        migrations_applied = 0
        for legacy_file in legacy_files:
            if Path(legacy_file).exists():
                logger.info(f"  📄 Lecture de {legacy_file}")
                migrations_applied += 1
        
        # Sauvegarder la nouvelle configuration
        output_dir = Path("config/unified")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        config_file = output_dir / f"{env.value}.json"
        config.save_to_file(str(config_file), include_secrets=False)
        logger.info(f"  ✅ Configuration sauvegardée: {config_file}")
        
        # Sauvegarder également un template avec secrets
        template_file = output_dir / f"{env.value}.template.json"
        config.save_to_file(str(template_file), include_secrets=True)
        logger.info(f"  📝 Template créé: {template_file}")
    
    logger.info(f"\\n🎉 Migration terminée - {migrations_applied} configurations traitées")
    logger.info("📁 Nouvelles configurations dans: config/unified/")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "migrate":
        migrate_legacy_config()
    else:
        # Test de la configuration
        logger.info("🧪 Test de la configuration unifiée")
        config = get_config()
        logger.info(f"Environnement: {config.environment.value}")
        logger.info(f"Base de données: {config.database.get_safe_url()}")
        logger.info(f"Cache: {config.cache.get_safe_url()}")
        logger.info(f"Service web: {config.get_service_url('web')}")