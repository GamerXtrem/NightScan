"""
NightScan API Routing Configuration
Resolves endpoint conflicts and standardizes API structure.
"""

from flask import Blueprint

# API versioning configuration
API_VERSION = 'v1'
API_PREFIX = f'/api/{API_VERSION}'

# Health check endpoints configuration
HEALTH_ENDPOINTS = {
    'web_app': '/health',
    'prediction_api': '/api/health', 
    'ml_service': '/ml/health'
}

# Service-specific routing
SERVICE_ROUTES = {
    'web': {
        'prefix': '',
        'health': '/health',
        'ready': '/ready',
        'metrics': '/metrics'
    },
    'prediction_api': {
        'prefix': '/api/v1',
        'health': '/api/v1/health',
        'ready': '/api/v1/ready',
        'predict': '/api/v1/predict'
    },
    'ml_service': {
        'prefix': '/ml',
        'health': '/ml/health',
        'predict': '/ml/predict'
    }
}

def create_service_blueprint(service_name: str, import_name: str):
    """Create a service-specific blueprint with proper routing."""
    config = SERVICE_ROUTES.get(service_name, {})
    prefix = config.get('prefix', f'/{service_name}')
    
    blueprint = Blueprint(
        f'{service_name}_service',
        import_name,
        url_prefix=prefix
    )
    
    return blueprint, config

# Endpoint conflict resolution mapping
ENDPOINT_MAPPING = {
    '/': {
        'web_app': '/',
        'demo_service': '/demo'
    },
    '/health': {
        'web_app': '/health',
        'api_v1': '/api/v1/health', 
        'prediction_api': '/api/health'
    },
    '/ready': {
        'web_app': '/ready',
        'api_v1': '/api/v1/ready',
        'prediction_api': '/api/ready'
    },
    '/metrics': {
        'web_app': '/metrics',
        'prediction_api': '/api/metrics'
    }
}

def get_service_endpoint(service: str, endpoint: str) -> str:
    """Get the correct endpoint for a specific service."""
    return ENDPOINT_MAPPING.get(endpoint, {}).get(service, endpoint)
