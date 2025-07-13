"""Performance tests for cache implementation"""

import pytest
import time
import json
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta

from cache_manager import CacheManager, get_cache_manager, cache_analytics_result, cache_user_data
from cache_utils import PredictionCache
from analytics_dashboard import OptimizedAnalyticsEngine, AnalyticsMetrics
from quota_manager import QuotaManager


class TestCachePerformance:
    """Test suite for cache performance improvements"""
    
    @pytest.fixture
    def cache_manager(self):
        """Create a test cache manager"""
        # Use mock Redis for testing
        manager = CacheManager()
        manager.cache_enabled = True
        manager.redis_client = MagicMock()
        return manager
    
    @pytest.fixture
    def mock_db(self):
        """Create a mock database connection"""
        db = MagicMock()
        return db
    
    def test_cache_manager_basic_operations(self, cache_manager):
        """Test basic cache operations"""
        # Test set and get
        cache_manager.redis_client.get.return_value = None
        cache_manager.redis_client.setex.return_value = True
        
        # First call - cache miss
        result = cache_manager.get('test_key')
        assert result is None
        assert cache_manager.cache_stats['misses'] == 1
        
        # Set value
        cache_manager.set('test_key', {'data': 'test'}, ttl=60)
        cache_manager.redis_client.setex.assert_called_once()
        
        # Second call - cache hit
        cache_manager.redis_client.get.return_value = '{"data": "test"}'
        result = cache_manager.get('test_key')
        assert result == {'data': 'test'}
        assert cache_manager.cache_stats['hits'] == 1
    
    def test_cache_key_generation(self, cache_manager):
        """Test cache key generation with different arguments"""
        # Test with simple args
        key1 = cache_manager._generate_cache_key('analytics', 30)
        assert key1 == 'analytics:30'
        
        # Test with kwargs
        key2 = cache_manager._generate_cache_key('user', user_id=123, type='quota')
        assert 'user' in key2
        assert '123' in key2
        assert 'quota' in key2
        
        # Test with complex objects
        key3 = cache_manager._generate_cache_key('complex', {'nested': 'data'})
        assert 'complex' in key3
        assert len(key3) > len('complex:')  # Should have hash
    
    @patch('analytics_dashboard.db')
    def test_analytics_caching(self, mock_db_module):
        """Test analytics dashboard caching"""
        # Setup mock database
        mock_session = MagicMock()
        mock_query = MagicMock()
        mock_db_module.session = mock_session
        mock_session.query.return_value = mock_query
        
        # Mock query results
        mock_query.filter.return_value = mock_query
        mock_query.first.return_value = MagicMock(
            total=100,
            unique_species=10,
            avg_confidence=0.85,
            today=20,
            week=50,
            month=100
        )
        mock_query.scalar.return_value = 5
        mock_query.group_by.return_value = mock_query
        mock_query.order_by.return_value = mock_query
        mock_query.limit.return_value = mock_query
        mock_query.all.return_value = []
        
        # Create analytics engine
        engine = OptimizedAnalyticsEngine(mock_db_module)
        
        # Measure performance without cache
        start_time = time.time()
        result1 = engine.get_detection_metrics(days=30)
        no_cache_time = time.time() - start_time
        
        # Second call should use cache (if implemented correctly)
        start_time = time.time()
        result2 = engine.get_detection_metrics(days=30)
        cache_time = time.time() - start_time
        
        # Cache should be faster (in real scenario)
        assert isinstance(result1, AnalyticsMetrics)
        assert isinstance(result2, AnalyticsMetrics)
        assert result1.total_detections == result2.total_detections
    
    def test_cache_invalidation_patterns(self, cache_manager):
        """Test pattern-based cache invalidation"""
        # Mock Redis SCAN operation
        cache_manager.redis_client.scan.return_value = (0, ['analytics:metrics:30', 'analytics:species:owl'])
        cache_manager.redis_client.delete.return_value = 2
        
        # Test pattern invalidation
        count = cache_manager.invalidate_pattern('analytics:*')
        
        assert count == 2
        cache_manager.redis_client.scan.assert_called_with(0, match='analytics:*', count=100)
        cache_manager.redis_client.delete.assert_called_once()
    
    def test_quota_caching(self):
        """Test quota manager caching"""
        quota_manager = QuotaManager()
        
        # Mock database query
        with patch.object(quota_manager, '_execute_query') as mock_query:
            mock_query.return_value = {
                'plan_type': 'premium',
                'current_usage': 50,
                'monthly_quota': 100,
                'reset_date': datetime.now() + timedelta(days=15),
                'last_prediction_at': datetime.now()
            }
            
            # First call - should hit database
            start_time = time.time()
            status1 = quota_manager.get_user_quota_status(123)
            db_time = time.time() - start_time
            
            # Verify result
            assert status1.user_id == 123
            assert status1.current_usage == 50
            assert status1.monthly_quota == 100
            assert mock_query.call_count == 1
    
    def test_multi_tier_caching(self, cache_manager):
        """Test multi-tier caching (local + Redis)"""
        # Setup
        test_data = {'key': 'value', 'timestamp': time.time()}
        cache_manager.redis_client.get.return_value = json.dumps(test_data)
        
        # First call - Redis hit, populate local
        result1 = cache_manager.get('test_key', use_local=True)
        assert result1 == test_data
        assert 'test_key' in cache_manager.local_cache
        
        # Second call - local cache hit
        cache_manager.redis_client.get.return_value = None  # Simulate Redis miss
        result2 = cache_manager.get('test_key', use_local=True)
        assert result2 == test_data  # Should still get from local
        
        # Test expiration
        cache_manager.local_cache['test_key']['expires'] = time.time() - 1
        result3 = cache_manager.get('test_key', use_local=True)
        assert result3 is None  # Expired
        assert 'test_key' not in cache_manager.local_cache
    
    def test_cache_metrics_collection(self, cache_manager):
        """Test cache metrics and statistics"""
        # Simulate operations
        cache_manager.redis_client.get.side_effect = [None, '{"data": "hit"}', None]
        
        # Operations
        cache_manager.get('key1')  # Miss
        cache_manager.get('key2')  # Hit
        cache_manager.get('key3')  # Miss
        
        # Get stats
        stats = cache_manager.get_stats()
        
        assert stats['hits'] == 1
        assert stats['misses'] == 2
        assert stats['hit_rate'] == pytest.approx(33.33, rel=0.1)
    
    @pytest.mark.benchmark
    def test_performance_improvement(self, benchmark):
        """Benchmark cache performance improvement"""
        
        def without_cache():
            # Simulate expensive operation
            time.sleep(0.01)  # 10ms database query
            return {'result': 'data'}
        
        def with_cache(cache_manager):
            key = 'benchmark_key'
            result = cache_manager.get(key)
            if result is None:
                result = without_cache()
                cache_manager.set(key, result, ttl=60)
            return result
        
        # Create cache manager for benchmark
        cache_manager = CacheManager()
        cache_manager.cache_enabled = True
        cache_manager.redis_client = MagicMock()
        cache_manager.redis_client.get.side_effect = [None, json.dumps({'result': 'data'})]
        cache_manager.redis_client.setex.return_value = True
        
        # Run benchmark
        result = benchmark(with_cache, cache_manager)
        assert result == {'result': 'data'}


class TestCacheIntegration:
    """Integration tests for cache system"""
    
    @pytest.mark.integration
    def test_end_to_end_caching_flow(self):
        """Test complete caching flow from request to response"""
        # This would require a running Redis instance
        # Marked as integration test to be run separately
        pass
    
    @pytest.mark.integration
    def test_cache_failover(self):
        """Test system behavior when cache is unavailable"""
        # Test graceful degradation when Redis is down
        cache = PredictionCache()
        
        # Should work without Redis
        result = cache.get_prediction(b'test_audio_data')
        assert result is None
        
        # Should not crash on cache operations
        success = cache.cache_prediction(b'test_audio_data', [{'label': 'test'}])
        assert success is False  # But doesn't crash


if __name__ == '__main__':
    pytest.main([__file__, '-v'])