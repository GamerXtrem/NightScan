"""
NightScan Port Configuration - Wrapper for centralized config.

This module now uses the centralized configuration system.
It's kept for backward compatibility.
"""

from config import get_config

# Get centralized config
_config = get_config()

# Default port assignments (from centralized config)
DEFAULT_PORTS = {
    'web_app': _config.ports.web_app,
    'prediction_api': _config.ports.prediction_api,
    'ml_service': _config.ports.ml_service,
    'websocket': _config.ports.websocket,
    'metrics': _config.ports.metrics,
    'health_check': 8080,  # Not in main config
    'redis': _config.ports.redis,
    'postgres': _config.ports.postgres,
    'monitoring': 3000  # Not in main config
}

# Keep the same API for backward compatibility
def get_port(service: str) -> int:
    """Get port for a specific service."""
    if hasattr(_config.ports, service):
        return getattr(_config.ports, service)
    return DEFAULT_PORTS.get(service, 8000)

def get_all_ports() -> dict:
    """Get all configured ports."""
    return DEFAULT_PORTS.copy()

def check_port_conflicts() -> dict:
    """Check for port conflicts."""
    # Use the centralized config's port checking
    ports = get_all_ports()
    conflicts = {}
    
    port_usage = {}
    for service, port in ports.items():
        if port in port_usage:
            if port not in conflicts:
                conflicts[port] = [port_usage[port]]
            conflicts[port].append(service)
        else:
            port_usage[port] = service
            
    return conflicts

def get_service_url(service: str, host: str = 'localhost', protocol: str = 'http') -> str:
    """Get full URL for a service."""
    port = get_port(service)
    return f"{protocol}://{host}:{port}"

# For backward compatibility
PORT_ENV_VARS = {
    'web_app': 'WEB_PORT',
    'prediction_api': 'PREDICTION_PORT',
    'ml_service': 'ML_SERVICE_PORT',
    'websocket': 'WEBSOCKET_PORT',
    'metrics': 'METRICS_PORT',
    'health_check': 'HEALTH_CHECK_PORT',
    'redis': 'REDIS_PORT',
    'postgres': 'POSTGRES_PORT',
    'monitoring': 'MONITORING_PORT'
}