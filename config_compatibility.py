#!/usr/bin/env python3
"""
Wrappers de Compatibilité pour la Migration des Configurations

Ce module fournit des interfaces de compatibilité pour les systèmes existants
qui utilisent les anciennes approches de configuration, permettant une migration
progressive vers le système unifié.
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from unified_config import get_config, UnifiedConfig

logger = logging.getLogger(__name__)


class LegacyConfigWrapper:
    """
    Wrapper qui émule les anciennes interfaces de configuration.
    
    Ce wrapper permet aux anciens systèmes de continuer à fonctionner
    pendant la migration vers le système unifié.
    """
    
    def __init__(self):
        self._unified_config = get_config()
        
        # Cache des valeurs legacy
        self._legacy_cache = {}
        
        # Mapping des anciennes variables vers le nouveau système
        self._legacy_mappings = self._build_legacy_mappings()
        
        logger.info("Wrapper de compatibilité configuré")
    
    def _build_legacy_mappings(self) -> Dict[str, str]:
        """Construit les mappings entre anciennes et nouvelles configurations."""
        return {
            # Variables legacy → unified_config path
            "WEB_PORT": "services.web_port",
            "API_PORT": "services.api_v1_port",
            "PREDICTION_PORT": "services.prediction_port",
            "DB_HOST": "database.host",
            "DB_PORT": "database.port",
            "DB_NAME": "database.database",
            "DB_USER": "database.username",
            "DB_PASSWORD": "database.password",
            "REDIS_HOST": "cache.host",
            "REDIS_PORT": "cache.port",
            "SECRET_KEY": "security.secret_key",
            "LOG_LEVEL": "logging.level",
            "MODELS_DIR": "ml.models_directory",
            "USE_GPU": "ml.use_gpu",
        }
    
    def get_legacy_value(self, key: str, default: Any = None) -> Any:
        """Récupère une valeur en utilisant les anciennes clés."""
        # Vérifier le cache d'abord
        if key in self._legacy_cache:
            return self._legacy_cache[key]
        
        # Essayer la variable d'environnement d'abord (pour compatibilité)
        env_value = os.getenv(key)
        if env_value is not None:
            self._legacy_cache[key] = env_value
            return env_value
        
        # Mapper vers le système unifié
        if key in self._legacy_mappings:
            unified_path = self._legacy_mappings[key]
            try:
                value = self._get_nested_value(self._unified_config, unified_path)
                self._legacy_cache[key] = value
                return value
            except (AttributeError, KeyError):
                logger.warning(f"Impossible de mapper la clé legacy '{key}'")
        
        return default
    
    def _get_nested_value(self, obj: Any, path: str) -> Any:
        """Récupère une valeur imbriquée via un chemin pointé."""
        parts = path.split('.')
        current = obj
        
        for part in parts:
            if hasattr(current, part):
                current = getattr(current, part)
            else:
                raise AttributeError(f"Attribut '{part}' non trouvé")
        
        return current


class FlaskConfigAdapter:
    """
    Adaptateur pour Flask app.config.
    
    Permet d'utiliser la configuration unifiée avec Flask
    tout en maintenant la compatibilité avec app.config.
    """
    
    def __init__(self, app=None):
        self.app = app
        self._unified_config = get_config()
        
        if app is not None:
            self.init_app(app)
    
    def init_app(self, app):
        """Initialise l'adaptateur avec une app Flask."""
        self.app = app
        
        # Mapper la configuration unifiée vers Flask
        self._apply_flask_config()
        
        logger.info("Configuration Flask adaptée depuis le système unifié")
    
    def _apply_flask_config(self):
        """Applique la configuration unifiée à Flask."""
        if not self.app:
            return
        
        config = self._unified_config
        
        # Configuration Flask standard
        self.app.config.update({
            'SECRET_KEY': config.security.secret_key,
            'SQLALCHEMY_DATABASE_URI': config.get_database_url(),
            'SQLALCHEMY_TRACK_MODIFICATIONS': False,
            'SQLALCHEMY_ENGINE_OPTIONS': {
                'pool_size': config.database.pool_size,
                'pool_timeout': config.database.pool_timeout,
                'pool_recycle': config.database.pool_recycle,
                'max_overflow': config.database.max_overflow,
            },
            
            # Session configuration
            'SESSION_COOKIE_NAME': config.security.session_cookie_name,
            'SESSION_COOKIE_SECURE': config.security.session_cookie_secure,
            'SESSION_COOKIE_HTTPONLY': config.security.session_cookie_httponly,
            'PERMANENT_SESSION_LIFETIME': config.security.session_timeout,
            
            # Upload configuration
            'MAX_CONTENT_LENGTH': config.services.max_file_size,
            
            # Custom NightScan config
            'NIGHTSCAN_CONFIG': config,
            'REDIS_URL': config.get_cache_url(),
            'PREDICT_API_URL': config.get_service_url('prediction'),
        })
        
        # Configuration WTF-CSRF
        if config.security.csrf_enabled:
            self.app.config['WTF_CSRF_ENABLED'] = True
            self.app.config['WTF_CSRF_TIME_LIMIT'] = config.security.session_timeout
        
        # Configuration CORS
        if hasattr(self.app, 'config') and config.security.cors_enabled:
            self.app.config['CORS_ORIGINS'] = config.security.allowed_origins


class DockerComposeGenerator:
    """
    Générateur de configurations Docker Compose depuis le système unifié.
    """
    
    def __init__(self, environment: str = "development"):
        self.environment = environment
        self.config = get_config()
    
    def generate_compose_config(self) -> Dict[str, Any]:
        """Génère une configuration docker-compose.yml."""
        return {
            'version': '3.8',
            'services': {
                'web': {
                    'build': '.',
                    'ports': [f"{self.config.services.web_port}:8000"],
                    'environment': self._get_web_environment(),
                    'depends_on': ['db', 'redis'],
                    'volumes': [
                        './logs:/app/logs',
                        './models:/app/models',
                        './data:/app/data'
                    ]
                },
                'db': {
                    'image': 'postgres:15',
                    'environment': {
                        'POSTGRES_DB': self.config.database.database,
                        'POSTGRES_USER': self.config.database.username,
                        'POSTGRES_PASSWORD': self.config.database.password
                    },
                    'ports': [f"{self.config.database.port}:5432"],
                    'volumes': ['postgres_data:/var/lib/postgresql/data']
                },
                'redis': {
                    'image': 'redis:7-alpine',
                    'ports': [f"{self.config.cache.port}:6379"],
                    'command': 'redis-server --appendonly yes',
                    'volumes': ['redis_data:/data']
                },
                'prediction-api': {
                    'build': '.',
                    'command': 'python -m unified_prediction_system.unified_prediction_api',
                    'ports': [f"{self.config.services.prediction_port}:8002"],
                    'environment': self._get_prediction_environment(),
                    'depends_on': ['db', 'redis'],
                    'volumes': [
                        './models:/app/models',
                        './cache:/app/cache'
                    ]
                }
            },
            'volumes': {
                'postgres_data': {},
                'redis_data': {}
            }
        }
    
    def _get_web_environment(self) -> Dict[str, str]:
        """Récupère les variables d'environnement pour le service web."""
        return {
            'NIGHTSCAN_ENV': self.environment,
            'DATABASE_URL': self.config.get_database_url(),
            'REDIS_URL': self.config.get_cache_url(),
            'SECRET_KEY': self.config.security.secret_key,
            'NIGHTSCAN_LOG_LEVEL': self.config.logging.level.value,
        }
    
    def _get_prediction_environment(self) -> Dict[str, str]:
        """Récupère les variables d'environnement pour l'API de prédiction."""
        return {
            'NIGHTSCAN_ENV': self.environment,
            'NIGHTSCAN_MODELS_DIR': self.config.ml.models_directory,
            'NIGHTSCAN_USE_GPU': str(self.config.ml.use_gpu).lower(),
            'NIGHTSCAN_ML_BATCH_SIZE': str(self.config.ml.batch_size),
        }


class KubernetesConfigGenerator:
    """
    Générateur de configurations Kubernetes depuis le système unifié.
    """
    
    def __init__(self, environment: str = "production"):
        self.environment = environment
        self.config = get_config()
    
    def generate_configmap(self) -> Dict[str, Any]:
        """Génère un ConfigMap Kubernetes."""
        return {
            'apiVersion': 'v1',
            'kind': 'ConfigMap',
            'metadata': {
                'name': f'nightscan-config-{self.environment}',
                'namespace': 'nightscan'
            },
            'data': {
                'NIGHTSCAN_ENV': self.environment,
                'NIGHTSCAN_LOG_LEVEL': self.config.logging.level.value,
                'NIGHTSCAN_WEB_PORT': str(self.config.services.web_port),
                'NIGHTSCAN_API_PORT': str(self.config.services.api_v1_port),
                'NIGHTSCAN_PREDICTION_PORT': str(self.config.services.prediction_port),
                'NIGHTSCAN_MODELS_DIR': self.config.ml.models_directory,
                'NIGHTSCAN_USE_GPU': str(self.config.ml.use_gpu).lower(),
                'NIGHTSCAN_ENABLE_METRICS': str(self.config.monitoring.enable_metrics).lower(),
                'config.json': self._get_config_json()
            }
        }
    
    def generate_secret(self) -> Dict[str, Any]:
        """Génère un Secret Kubernetes."""
        import base64
        
        def encode_secret(value: str) -> str:
            return base64.b64encode(value.encode()).decode()
        
        return {
            'apiVersion': 'v1',
            'kind': 'Secret',
            'metadata': {
                'name': f'nightscan-secrets-{self.environment}',
                'namespace': 'nightscan'
            },
            'type': 'Opaque',
            'data': {
                'SECRET_KEY': encode_secret(self.config.security.secret_key),
                'JWT_SECRET': encode_secret(self.config.security.jwt_secret),
                'DATABASE_URL': encode_secret(self.config.get_database_url()),
                'REDIS_URL': encode_secret(self.config.get_cache_url()),
            }
        }
    
    def _get_config_json(self) -> str:
        """Récupère la configuration complète en JSON."""
        import json
        return json.dumps(self.config.to_dict(include_secrets=False), indent=2)


# Instances globales pour compatibilité
_legacy_wrapper = None
_flask_adapter = None


def get_legacy_config() -> LegacyConfigWrapper:
    """Retourne l'instance globale du wrapper legacy."""
    global _legacy_wrapper
    if _legacy_wrapper is None:
        _legacy_wrapper = LegacyConfigWrapper()
    return _legacy_wrapper


def get_flask_adapter():
    """Retourne l'adaptateur Flask global."""
    global _flask_adapter
    if _flask_adapter is None:
        _flask_adapter = FlaskConfigAdapter()
    return _flask_adapter


def get_legacy_value(key: str, default: Any = None) -> Any:
    """
    Fonction utilitaire pour récupérer une valeur legacy.
    
    Args:
        key: Clé legacy (ex: 'WEB_PORT', 'DB_HOST')
        default: Valeur par défaut
    
    Returns:
        Valeur mappée depuis le système unifié
    """
    return get_legacy_config().get_legacy_value(key, default)


def configure_flask_app(app):
    """
    Configure une app Flask avec le système unifié.
    
    Args:
        app: Instance Flask
    """
    adapter = FlaskConfigAdapter(app)
    return adapter


def main():
    """Fonctions de test et génération de configurations."""
    import sys
    import json
    try:
        import yaml
        YAML_AVAILABLE = True
    except ImportError:
        YAML_AVAILABLE = False
    
    if len(sys.argv) < 2:
        print("Usage: python config_compatibility.py <command>")
        print("Commands: test-legacy, generate-docker, generate-k8s")
        return
    
    command = sys.argv[1]
    
    if command == "test-legacy":
        print("🧪 Test du wrapper de compatibilité")
        wrapper = get_legacy_config()
        
        test_keys = ["WEB_PORT", "DB_HOST", "SECRET_KEY", "LOG_LEVEL"]
        for key in test_keys:
            value = wrapper.get_legacy_value(key, "NOT_FOUND")
            print(f"  {key}: {value}")
    
    elif command == "generate-docker":
        env = sys.argv[2] if len(sys.argv) > 2 else "development"
        print(f"🐳 Génération docker-compose pour {env}")
        
        generator = DockerComposeGenerator(env)
        config = generator.generate_compose_config()
        
        output_file = f"docker-compose.{env}.json"
        with open(output_file, 'w') as f:
            if YAML_AVAILABLE:
                output_file = f"docker-compose.{env}.yml"
                yaml.dump(config, f, default_flow_style=False, indent=2)
            else:
                json.dump(config, f, indent=2)
        
        print(f"✅ Configuration générée: {output_file}")
    
    elif command == "generate-k8s":
        env = sys.argv[2] if len(sys.argv) > 2 else "production"
        print(f"☸️ Génération Kubernetes pour {env}")
        
        generator = KubernetesConfigGenerator(env)
        
        # ConfigMap
        configmap = generator.generate_configmap()
        configmap_file = f"k8s-configmap-{env}.json"
        with open(configmap_file, 'w') as f:
            if YAML_AVAILABLE:
                configmap_file = f"k8s-configmap-{env}.yaml"
                yaml.dump(configmap, f, default_flow_style=False, indent=2)
            else:
                json.dump(configmap, f, indent=2)
        
        # Secret
        secret = generator.generate_secret()
        secret_file = f"k8s-secret-{env}.json"
        with open(secret_file, 'w') as f:
            if YAML_AVAILABLE:
                secret_file = f"k8s-secret-{env}.yaml"
                yaml.dump(secret, f, default_flow_style=False, indent=2)
            else:
                json.dump(secret, f, indent=2)
        
        print(f"✅ Configurations K8s générées pour {env}")
    
    else:
        print(f"Commande inconnue: {command}")


if __name__ == "__main__":
    main()