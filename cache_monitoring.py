"""Cache monitoring endpoints for NightScan"""

from flask import Blueprint, jsonify
from flask_login import login_required
from datetime import datetime
import psutil
import os

from cache_manager import get_cache_manager
from cache_utils import get_cache
from metrics import track_request_metrics

# Create monitoring blueprint
cache_monitor_bp = Blueprint('cache_monitor', __name__, url_prefix='/api/cache')


@cache_monitor_bp.route('/metrics', methods=['GET'])
@login_required
@track_request_metrics
def cache_metrics():
    """Get comprehensive cache metrics and statistics
    
    Returns:
        JSON with cache statistics including:
        - Hit/miss rates
        - Cache sizes
        - Performance metrics
        - Redis status
    """
    try:
        # Get cache manager stats
        cache_manager = get_cache_manager()
        manager_stats = cache_manager.get_stats()
        
        # Get prediction cache stats
        prediction_cache = get_cache()
        prediction_stats = prediction_cache.get_cache_stats()
        
        # Get system memory info
        memory_info = psutil.virtual_memory()
        
        # Combine all metrics
        metrics = {
            'timestamp': datetime.utcnow().isoformat(),
            'cache_manager': {
                'hits': manager_stats.get('hits', 0),
                'misses': manager_stats.get('misses', 0),
                'errors': manager_stats.get('errors', 0),
                'invalidations': manager_stats.get('invalidations', 0),
                'hit_rate': manager_stats.get('hit_rate', 0),
                'local_cache_size': manager_stats.get('local_cache_size', 0),
                'redis_keys': manager_stats.get('redis_keys', 0),
                'redis_memory': manager_stats.get('redis_memory', 'N/A')
            },
            'prediction_cache': {
                'enabled': prediction_stats.get('enabled', False),
                'redis_available': prediction_stats.get('redis_available', False),
                'default_ttl': prediction_stats.get('default_ttl', 0),
                'hit_rate': prediction_stats.get('hit_rate', 0),
                'redis_version': prediction_stats.get('redis_version', 'N/A')
            },
            'system': {
                'memory_used_percent': memory_info.percent,
                'memory_available_mb': memory_info.available / (1024 * 1024),
                'process_memory_mb': psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
            }
        }
        
        return jsonify({
            'success': True,
            'data': metrics
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@cache_monitor_bp.route('/health', methods=['GET'])
@login_required
@track_request_metrics
def cache_health():
    """Check cache system health
    
    Returns:
        JSON with health status of cache components
    """
    try:
        cache_manager = get_cache_manager()
        prediction_cache = get_cache()
        
        # Test cache operations
        test_key = 'health_check_test'
        test_value = {'timestamp': datetime.utcnow().isoformat()}
        
        # Test cache manager
        manager_healthy = False
        try:
            cache_manager.set(test_key, test_value, ttl=10)
            retrieved = cache_manager.get(test_key)
            if retrieved == test_value:
                manager_healthy = True
            cache_manager.delete(test_key)
        except:
            pass
        
        # Test prediction cache
        prediction_healthy = False
        if prediction_cache.cache_enabled:
            try:
                prediction_cache.redis_client.ping()
                prediction_healthy = True
            except:
                pass
        
        health_status = {
            'timestamp': datetime.utcnow().isoformat(),
            'overall_status': 'healthy' if (manager_healthy and prediction_healthy) else 'degraded',
            'components': {
                'cache_manager': {
                    'status': 'healthy' if manager_healthy else 'unhealthy',
                    'redis_connected': cache_manager.cache_enabled
                },
                'prediction_cache': {
                    'status': 'healthy' if prediction_healthy else 'unhealthy',
                    'redis_connected': prediction_cache.cache_enabled
                }
            }
        }
        
        status_code = 200 if health_status['overall_status'] == 'healthy' else 503
        
        return jsonify({
            'success': True,
            'data': health_status
        }), status_code
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'data': {
                'overall_status': 'error',
                'timestamp': datetime.utcnow().isoformat()
            }
        }), 500


@cache_monitor_bp.route('/clear', methods=['POST'])
@login_required
@track_request_metrics
def clear_cache():
    """Clear all cache entries (admin only)
    
    Returns:
        JSON confirmation of cache clearing
    """
    try:
        # Check if user is admin (you may want to implement proper admin check)
        if not getattr(current_user, 'is_admin', False):
            return jsonify({
                'success': False,
                'error': 'Admin access required'
            }), 403
        
        cache_manager = get_cache_manager()
        prediction_cache = get_cache()
        
        # Clear cache manager
        cleared_count = cache_manager.invalidate_pattern('*')
        
        # Clear prediction cache
        prediction_cache.clear_cache()
        
        return jsonify({
            'success': True,
            'data': {
                'message': 'Cache cleared successfully',
                'cleared_entries': cleared_count,
                'timestamp': datetime.utcnow().isoformat()
            }
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500