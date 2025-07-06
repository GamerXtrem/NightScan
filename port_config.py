"""
NightScan Port Configuration
Centralized port management to prevent conflicts.
"""

import os
from typing import Dict, Optional

# Default port assignments
DEFAULT_PORTS = {
    'web_app': 8000,
    'prediction_api': 8001, 
    'ml_service': 8002,
    'websocket': 8003,
    'metrics': 9090,
    'health_check': 8080,
    'redis': 6379,
    'postgres': 5432,
    'monitoring': 3000
}

# Environment variable mapping
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

def get_port(service: str) -> int:
    """Get port for a specific service from environment or default."""
    env_var = PORT_ENV_VARS.get(service)
    if env_var and env_var in os.environ:
        return int(os.environ[env_var])
    return DEFAULT_PORTS.get(service, 8000)

def get_all_ports() -> Dict[str, int]:
    """Get all configured ports."""
    return {service: get_port(service) for service in DEFAULT_PORTS.keys()}

def check_port_conflicts() -> Dict[str, list]:
    """Check for port conflicts in current configuration."""
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

# Port ranges for different environments
PORT_RANGES = {
    'development': {
        'start': 8000,
        'end': 8099
    },
    'testing': {
        'start': 9000, 
        'end': 9099
    },
    'production': {
        'start': 80,
        'end': 8999
    }
}

def allocate_port_range(environment: str = 'development') -> Dict[str, int]:
    """Allocate ports within a specific range for an environment."""
    range_config = PORT_RANGES.get(environment, PORT_RANGES['development'])
    start_port = range_config['start']
    
    allocated_ports = {}
    for i, service in enumerate(DEFAULT_PORTS.keys()):
        allocated_ports[service] = start_port + i
        
    return allocated_ports
