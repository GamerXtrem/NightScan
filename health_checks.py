"""Health check endpoints for all NightScan services"""

from flask import Blueprint, jsonify, current_app
from datetime import datetime
import psutil
import os
import logging

logger = logging.getLogger(__name__)

# Create health check blueprint
health_bp = Blueprint('health', __name__)


def get_system_stats():
    """Get system resource statistics"""
    try:
        # CPU info
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_count = psutil.cpu_count()
        
        # Memory info
        memory = psutil.virtual_memory()
        
        # Disk info
        disk = psutil.disk_usage('/')
        
        # Process info
        process = psutil.Process(os.getpid())
        process_memory = process.memory_info()
        
        return {
            'cpu': {
                'percent': cpu_percent,
                'count': cpu_count
            },
            'memory': {
                'total_mb': memory.total / (1024 * 1024),
                'available_mb': memory.available / (1024 * 1024),
                'percent': memory.percent
            },
            'disk': {
                'total_gb': disk.total / (1024 * 1024 * 1024),
                'free_gb': disk.free / (1024 * 1024 * 1024),
                'percent': disk.percent
            },
            'process': {
                'memory_mb': process_memory.rss / (1024 * 1024),
                'threads': process.num_threads()
            }
        }
    except Exception as e:
        logger.error(f"Failed to get system stats: {e}")
        return None


def check_database_health():
    """Check database connectivity and basic operations"""
    try:
        from web.app import db
        
        # Test basic query
        result = db.session.execute("SELECT 1")
        
        # Test table access
        db.session.execute("SELECT COUNT(*) FROM users")
        
        return {
            'status': 'healthy',
            'responsive': True
        }
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return {
            'status': 'unhealthy',
            'responsive': False,
            'error': str(e)
        }


def check_redis_health():
    """Check Redis connectivity"""
    try:
        import redis
        redis_url = os.environ.get('REDIS_URL', 'redis://localhost:6379/0')
        r = redis.from_url(redis_url)
        r.ping()
        
        return {
            'status': 'healthy',
            'responsive': True
        }
    except Exception as e:
        logger.warning(f"Redis health check failed: {e}")
        return {
            'status': 'unhealthy',
            'responsive': False,
            'error': str(e)
        }


def check_ml_models_health():
    """Check if ML models are loaded and ready"""
    try:
        # Check if models directory exists
        models_path = os.environ.get('MODEL_PATH', 'models/')
        if os.path.exists(models_path):
            model_files = [f for f in os.listdir(models_path) if f.endswith('.pth') or f.endswith('.pkl')]
            return {
                'status': 'healthy',
                'models_found': len(model_files),
                'models': model_files[:5]  # First 5 models
            }
        else:
            return {
                'status': 'unhealthy',
                'error': 'Models directory not found'
            }
    except Exception as e:
        return {
            'status': 'unhealthy',
            'error': str(e)
        }


@health_bp.route('/health', methods=['GET'])
def basic_health():
    """Basic health check endpoint
    
    Returns 200 if service is up, used by load balancers
    """
    return jsonify({
        'status': 'healthy',
        'service': current_app.config.get('SERVICE_NAME', 'nightscan-web'),
        'timestamp': datetime.utcnow().isoformat()
    })


@health_bp.route('/health/detailed', methods=['GET'])
def detailed_health():
    """Detailed health check with component status
    
    Returns comprehensive health information about all components
    """
    # Get component health
    database = check_database_health()
    redis = check_redis_health()
    models = check_ml_models_health()
    system = get_system_stats()
    
    # Determine overall status
    all_healthy = all([
        database.get('status') == 'healthy',
        redis.get('status') == 'healthy' or redis.get('status') == 'degraded',  # Redis is optional
        models.get('status') == 'healthy' or models.get('status') == 'degraded'  # Models optional for web
    ])
    
    overall_status = 'healthy' if all_healthy else 'unhealthy'
    
    return jsonify({
        'status': overall_status,
        'service': current_app.config.get('SERVICE_NAME', 'nightscan-web'),
        'version': current_app.config.get('VERSION', '1.0.0'),
        'timestamp': datetime.utcnow().isoformat(),
        'components': {
            'database': database,
            'redis': redis,
            'ml_models': models
        },
        'system': system
    }), 200 if overall_status == 'healthy' else 503


@health_bp.route('/health/live', methods=['GET'])
def liveness_probe():
    """Kubernetes liveness probe endpoint
    
    Returns 200 if the service is alive and can handle requests
    """
    return jsonify({
        'alive': True,
        'timestamp': datetime.utcnow().isoformat()
    })


@health_bp.route('/health/ready', methods=['GET'])
def readiness_probe():
    """Kubernetes readiness probe endpoint
    
    Returns 200 if the service is ready to accept traffic
    """
    # Check critical components
    database = check_database_health()
    
    if database.get('status') != 'healthy':
        return jsonify({
            'ready': False,
            'reason': 'Database not healthy',
            'timestamp': datetime.utcnow().isoformat()
        }), 503
    
    return jsonify({
        'ready': True,
        'timestamp': datetime.utcnow().isoformat()
    })


# Service-specific health checks
def create_prediction_health_bp():
    """Create health blueprint for prediction service"""
    pred_health_bp = Blueprint('pred_health', __name__)
    
    @pred_health_bp.route('/health', methods=['GET'])
    def prediction_health():
        models = check_ml_models_health()
        
        return jsonify({
            'status': models.get('status', 'unhealthy'),
            'service': 'nightscan-prediction',
            'timestamp': datetime.utcnow().isoformat(),
            'models': models
        })
    
    return pred_health_bp


def create_analytics_health_bp():
    """Create health blueprint for analytics service"""
    analytics_health_bp = Blueprint('analytics_health', __name__)
    
    @analytics_health_bp.route('/health', methods=['GET'])
    def analytics_health():
        database = check_database_health()
        redis = check_redis_health()
        
        # Analytics requires both DB and Redis
        healthy = (database.get('status') == 'healthy' and 
                  redis.get('status') == 'healthy')
        
        return jsonify({
            'status': 'healthy' if healthy else 'unhealthy',
            'service': 'nightscan-analytics',
            'timestamp': datetime.utcnow().isoformat(),
            'components': {
                'database': database,
                'redis': redis
            }
        })
    
    return analytics_health_bp