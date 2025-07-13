"""
Centralized Circuit Breaker Configuration for NightScan

Provides unified configuration management for all circuit breakers
across the application with environment-based overrides and
monitoring integration.

Features:
- Environment-based configuration
- Service-specific settings
- Health check endpoints
- Metrics aggregation
- Circuit breaker registry
"""

import os
import json
import logging
from typing import Dict, Any, List, Optional, Type
from dataclasses import dataclass, field, asdict
from datetime import datetime

from circuit_breaker import (
    CircuitBreaker, CircuitBreakerConfig, get_all_circuit_breakers,
    get_circuit_breaker_metrics, reset_all_circuit_breakers,
    cleanup_circuit_breakers
)
from database_circuit_breaker import DatabaseCircuitBreakerConfig, DatabaseCircuitBreaker
from cache_circuit_breaker import CacheCircuitBreakerConfig, CacheCircuitBreaker
from http_circuit_breaker import HTTPCircuitBreakerConfig, HTTPCircuitBreaker
from ml_circuit_breaker import MLCircuitBreakerConfig, MLCircuitBreaker

logger = logging.getLogger(__name__)


@dataclass
class CircuitBreakerSettings:
    """Global circuit breaker settings."""
    enabled: bool = True
    monitoring_enabled: bool = True
    metrics_port: int = 9090
    health_check_port: int = 8080
    config_file: Optional[str] = None
    auto_recovery: bool = True
    cleanup_on_shutdown: bool = True
    
    # Global defaults
    default_failure_threshold: int = 5
    default_timeout: float = 60.0
    default_success_threshold: int = 3
    
    # Environment overrides
    environment: str = field(default_factory=lambda: os.environ.get('ENVIRONMENT', 'development'))


@dataclass
class ServiceCircuitBreakerConfig:
    """Configuration for a specific service's circuit breakers."""
    service_name: str
    enabled: bool = True
    
    # Database circuit breaker
    database: Optional[Dict[str, Any]] = None
    
    # Cache circuit breaker
    cache: Optional[Dict[str, Any]] = None
    
    # HTTP circuit breaker
    http: Optional[Dict[str, Any]] = None
    
    # ML circuit breaker
    ml: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Set default configurations if not provided."""
        if self.database is None:
            self.database = {
                "read_timeout": 3.0,
                "write_timeout": 10.0,
                "failure_threshold": 3,
                "timeout": 120.0
            }
            
        if self.cache is None:
            self.cache = {
                "redis_host": os.environ.get("REDIS_HOST", "localhost"),
                "redis_port": int(os.environ.get("REDIS_PORT", "6379")),
                "failure_threshold": 3,
                "timeout": 60.0,
                "enable_memory_fallback": True,
                "enable_disk_fallback": True
            }
            
        if self.http is None:
            self.http = {
                "connect_timeout": 3.0,
                "read_timeout": 10.0,
                "total_timeout": 30.0,
                "failure_threshold": 3,
                "timeout": 60.0,
                "max_retries": 3
            }
            
        if self.ml is None:
            self.ml = {
                "model_path": os.environ.get("MODEL_PATH"),
                "failure_threshold": 2,
                "timeout": 300.0,  # 5 minutes
                "inference_timeout": 30.0,
                "enable_lightweight_fallback": True,
                "enable_cached_fallback": True
            }


class CircuitBreakerManager:
    """Centralized manager for all circuit breakers in NightScan."""
    
    def __init__(self, config_file: Optional[str] = None):
        self.settings = CircuitBreakerSettings()
        self.service_configs: Dict[str, ServiceCircuitBreakerConfig] = {}
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.config_file = config_file or os.environ.get('CIRCUIT_BREAKER_CONFIG')
        
        # Load configuration
        self._load_configuration()
        
        # Initialize default service configurations
        self._setup_default_services()
        
        logger.info(f"Circuit breaker manager initialized with {len(self.service_configs)} services")
    
    def _load_configuration(self):
        """Load configuration from file if available."""
        if self.config_file and os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    config_data = json.load(f)
                
                # Update settings
                if 'settings' in config_data:
                    for key, value in config_data['settings'].items():
                        if hasattr(self.settings, key):
                            setattr(self.settings, key, value)
                
                # Load service configurations
                if 'services' in config_data:
                    for service_name, service_config in config_data['services'].items():
                        self.service_configs[service_name] = ServiceCircuitBreakerConfig(
                            service_name=service_name,
                            **service_config
                        )
                
                logger.info(f"Configuration loaded from {self.config_file}")
                
            except Exception as e:
                logger.error(f"Failed to load configuration from {self.config_file}: {e}")
        else:
            logger.info("No configuration file found, using defaults")
    
    def _setup_default_services(self):
        """Setup default service configurations."""
        default_services = [
            'web_app',
            'api_v1', 
            'celery_workers',
            'ml_prediction',
            'notification_service'
        ]
        
        for service_name in default_services:
            if service_name not in self.service_configs:
                self.service_configs[service_name] = ServiceCircuitBreakerConfig(
                    service_name=service_name
                )
    
    def get_database_circuit_breaker(self, service_name: str, db_session=None) -> DatabaseCircuitBreaker:
        """Get or create database circuit breaker for a service."""
        circuit_name = f"{service_name}_database"
        
        if circuit_name in self.circuit_breakers:
            return self.circuit_breakers[circuit_name]
        
        if not self.settings.enabled:
            logger.info(f"Circuit breakers disabled, returning pass-through for {circuit_name}")
            return None
        
        service_config = self.service_configs.get(service_name)
        if not service_config or not service_config.enabled:
            logger.info(f"Service {service_name} circuit breakers disabled")
            return None
        
        # Create configuration
        db_config = service_config.database.copy()
        db_config.update({
            'name': circuit_name,
            'failure_threshold': db_config.get('failure_threshold', self.settings.default_failure_threshold),
            'timeout': db_config.get('timeout', self.settings.default_timeout),
            'success_threshold': db_config.get('success_threshold', self.settings.default_success_threshold)
        })
        
        config = DatabaseCircuitBreakerConfig(**db_config)
        circuit_breaker = DatabaseCircuitBreaker(config, db_session)
        
        self.circuit_breakers[circuit_name] = circuit_breaker
        logger.info(f"Created database circuit breaker: {circuit_name}")
        
        return circuit_breaker
    
    def get_cache_circuit_breaker(self, service_name: str) -> CacheCircuitBreaker:
        """Get or create cache circuit breaker for a service."""
        circuit_name = f"{service_name}_cache"
        
        if circuit_name in self.circuit_breakers:
            return self.circuit_breakers[circuit_name]
        
        if not self.settings.enabled:
            return None
        
        service_config = self.service_configs.get(service_name)
        if not service_config or not service_config.enabled:
            return None
        
        # Create configuration
        cache_config = service_config.cache.copy()
        cache_config.update({
            'name': circuit_name,
            'failure_threshold': cache_config.get('failure_threshold', self.settings.default_failure_threshold),
            'timeout': cache_config.get('timeout', self.settings.default_timeout),
            'success_threshold': cache_config.get('success_threshold', self.settings.default_success_threshold)
        })
        
        config = CacheCircuitBreakerConfig(**cache_config)
        circuit_breaker = CacheCircuitBreaker(config)
        
        self.circuit_breakers[circuit_name] = circuit_breaker
        logger.info(f"Created cache circuit breaker: {circuit_name}")
        
        return circuit_breaker
    
    def get_http_circuit_breaker(self, service_name: str, base_url: str = None, 
                                service_type: str = None) -> HTTPCircuitBreaker:
        """Get or create HTTP circuit breaker for a service."""
        circuit_name = f"{service_name}_http"
        if service_type:
            circuit_name = f"{service_name}_{service_type}_http"
        
        if circuit_name in self.circuit_breakers:
            return self.circuit_breakers[circuit_name]
        
        if not self.settings.enabled:
            return None
        
        service_config = self.service_configs.get(service_name)
        if not service_config or not service_config.enabled:
            return None
        
        # Create configuration
        http_config = service_config.http.copy()
        http_config.update({
            'name': circuit_name,
            'service_name': f"{service_name} {service_type or 'HTTP'} Service",
            'base_url': base_url,
            'failure_threshold': http_config.get('failure_threshold', self.settings.default_failure_threshold),
            'timeout': http_config.get('timeout', self.settings.default_timeout),
            'success_threshold': http_config.get('success_threshold', self.settings.default_success_threshold)
        })
        
        config = HTTPCircuitBreakerConfig(**http_config)
        circuit_breaker = HTTPCircuitBreaker(config)
        
        self.circuit_breakers[circuit_name] = circuit_breaker
        logger.info(f"Created HTTP circuit breaker: {circuit_name}")
        
        return circuit_breaker
    
    def get_ml_circuit_breaker(self, service_name: str, model_path: str = None) -> MLCircuitBreaker:
        """Get or create ML circuit breaker for a service."""
        circuit_name = f"{service_name}_ml"
        
        if circuit_name in self.circuit_breakers:
            return self.circuit_breakers[circuit_name]
        
        if not self.settings.enabled:
            return None
        
        service_config = self.service_configs.get(service_name)
        if not service_config or not service_config.enabled:
            return None
        
        # Create configuration
        ml_config = service_config.ml.copy()
        ml_config.update({
            'name': circuit_name,
            'model_path': model_path or ml_config.get('model_path'),
            'failure_threshold': ml_config.get('failure_threshold', self.settings.default_failure_threshold),
            'timeout': ml_config.get('timeout', self.settings.default_timeout),
            'success_threshold': ml_config.get('success_threshold', self.settings.default_success_threshold)
        })
        
        config = MLCircuitBreakerConfig(**ml_config)
        circuit_breaker = MLCircuitBreaker(config)
        
        self.circuit_breakers[circuit_name] = circuit_breaker
        logger.info(f"Created ML circuit breaker: {circuit_name}")
        
        return circuit_breaker
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """Get metrics for all circuit breakers."""
        if not self.settings.monitoring_enabled:
            return {}
        
        all_metrics = {}
        
        # Include global circuit breakers
        global_metrics = get_circuit_breaker_metrics()
        all_metrics.update(global_metrics)
        
        # Include managed circuit breakers
        for name, circuit_breaker in self.circuit_breakers.items():
            try:
                if hasattr(circuit_breaker, 'get_database_metrics'):
                    all_metrics[name] = circuit_breaker.get_database_metrics()
                elif hasattr(circuit_breaker, 'get_cache_stats'):
                    all_metrics[name] = circuit_breaker.get_cache_stats()
                elif hasattr(circuit_breaker, 'get_http_metrics'):
                    all_metrics[name] = circuit_breaker.get_http_metrics()
                elif hasattr(circuit_breaker, 'get_ml_metrics'):
                    all_metrics[name] = circuit_breaker.get_ml_metrics()
                else:
                    all_metrics[name] = circuit_breaker.get_metrics()
            except Exception as e:
                logger.error(f"Failed to get metrics for {name}: {e}")
                all_metrics[name] = {'error': str(e)}
        
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'circuit_breakers': all_metrics,
            'manager_info': {
                'total_circuits': len(all_metrics),
                'services_configured': len(self.service_configs),
                'environment': self.settings.environment,
                'monitoring_enabled': self.settings.monitoring_enabled
            }
        }
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of all circuit breakers."""
        health_status = {
            'overall_status': 'healthy',
            'timestamp': datetime.utcnow().isoformat(),
            'circuit_breakers': {},
            'summary': {
                'total': 0,
                'healthy': 0,
                'degraded': 0,
                'unhealthy': 0
            }
        }
        
        for name, circuit_breaker in self.circuit_breakers.items():
            try:
                if hasattr(circuit_breaker, 'health_check'):
                    cb_health = circuit_breaker.health_check()
                else:
                    cb_health = {
                        'circuit_state': circuit_breaker.get_state().value,
                        'available': circuit_breaker.is_available()
                    }
                
                # Determine health status
                if cb_health.get('available', True) and cb_health.get('circuit_state') == 'closed':
                    status = 'healthy'
                    health_status['summary']['healthy'] += 1
                elif cb_health.get('circuit_state') == 'half_open':
                    status = 'degraded'
                    health_status['summary']['degraded'] += 1
                else:
                    status = 'unhealthy'
                    health_status['summary']['unhealthy'] += 1
                    health_status['overall_status'] = 'degraded'
                
                health_status['circuit_breakers'][name] = {
                    'status': status,
                    'details': cb_health
                }
                health_status['summary']['total'] += 1
                
            except Exception as e:
                logger.error(f"Health check failed for {name}: {e}")
                health_status['circuit_breakers'][name] = {
                    'status': 'unhealthy',
                    'error': str(e)
                }
                health_status['summary']['total'] += 1
                health_status['summary']['unhealthy'] += 1
                health_status['overall_status'] = 'degraded'
        
        # Set overall status
        if health_status['summary']['unhealthy'] > 0:
            health_status['overall_status'] = 'unhealthy'
        elif health_status['summary']['degraded'] > 0:
            health_status['overall_status'] = 'degraded'
        
        return health_status
    
    def reset_all_circuits(self):
        """Reset all circuit breakers to closed state."""
        logger.info("Resetting all circuit breakers")
        
        # Reset global circuit breakers
        reset_all_circuit_breakers()
        
        # Reset managed circuit breakers
        for name, circuit_breaker in self.circuit_breakers.items():
            try:
                circuit_breaker.reset()
                logger.info(f"Reset circuit breaker: {name}")
            except Exception as e:
                logger.error(f"Failed to reset circuit breaker {name}: {e}")
    
    def cleanup(self):
        """Clean up all circuit breakers."""
        if self.settings.cleanup_on_shutdown:
            logger.info("Cleaning up circuit breakers")
            
            # Cleanup managed circuit breakers
            for name, circuit_breaker in self.circuit_breakers.items():
                try:
                    if hasattr(circuit_breaker, 'cleanup'):
                        circuit_breaker.cleanup()
                    logger.info(f"Cleaned up circuit breaker: {name}")
                except Exception as e:
                    logger.error(f"Failed to cleanup circuit breaker {name}: {e}")
            
            # Cleanup global circuit breakers
            cleanup_circuit_breakers()
            
            self.circuit_breakers.clear()
    
    def save_configuration(self, file_path: str = None):
        """Save current configuration to file."""
        file_path = file_path or self.config_file or 'circuit_breaker_config.json'
        
        config_data = {
            'settings': asdict(self.settings),
            'services': {}
        }
        
        for service_name, service_config in self.service_configs.items():
            config_data['services'][service_name] = asdict(service_config)
        
        try:
            with open(file_path, 'w') as f:
                json.dump(config_data, f, indent=2)
            logger.info(f"Configuration saved to {file_path}")
        except Exception as e:
            logger.error(f"Failed to save configuration to {file_path}: {e}")


# Global circuit breaker manager instance
_circuit_breaker_manager: Optional[CircuitBreakerManager] = None


def get_circuit_breaker_manager() -> CircuitBreakerManager:
    """Get the global circuit breaker manager instance."""
    global _circuit_breaker_manager
    if _circuit_breaker_manager is None:
        _circuit_breaker_manager = CircuitBreakerManager()
    return _circuit_breaker_manager


def initialize_circuit_breakers(config_file: str = None) -> CircuitBreakerManager:
    """Initialize the circuit breaker manager with optional config file."""
    global _circuit_breaker_manager
    _circuit_breaker_manager = CircuitBreakerManager(config_file)
    return _circuit_breaker_manager


# Convenience functions for getting circuit breakers

def get_database_circuit_breaker(service_name: str, db_session=None) -> Optional[DatabaseCircuitBreaker]:
    """Get database circuit breaker for a service."""
    manager = get_circuit_breaker_manager()
    return manager.get_database_circuit_breaker(service_name, db_session)


def get_cache_circuit_breaker(service_name: str) -> Optional[CacheCircuitBreaker]:
    """Get cache circuit breaker for a service."""
    manager = get_circuit_breaker_manager()
    return manager.get_cache_circuit_breaker(service_name)


def get_http_circuit_breaker(service_name: str, base_url: str = None, 
                           service_type: str = None) -> Optional[HTTPCircuitBreaker]:
    """Get HTTP circuit breaker for a service."""
    manager = get_circuit_breaker_manager()
    return manager.get_http_circuit_breaker(service_name, base_url, service_type)


def get_ml_circuit_breaker(service_name: str, model_path: str = None) -> Optional[MLCircuitBreaker]:
    """Get ML circuit breaker for a service."""
    manager = get_circuit_breaker_manager()
    return manager.get_ml_circuit_breaker(service_name, model_path)


# Circuit breaker monitoring endpoints

def circuit_breaker_metrics_endpoint():
    """Flask endpoint for circuit breaker metrics."""
    manager = get_circuit_breaker_manager()
    return manager.get_all_metrics()


def circuit_breaker_health_endpoint():
    """Flask endpoint for circuit breaker health."""
    manager = get_circuit_breaker_manager()
    return manager.get_health_status()