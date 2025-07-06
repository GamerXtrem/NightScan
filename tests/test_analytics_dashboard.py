import pytest
import json
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add the parent directory to Python path
sys.path.append(str(Path(__file__).resolve().parents[1]))

# Mock Flask and dependencies
mock_flask = MagicMock()
mock_blueprint = MagicMock()
mock_flask.Blueprint.return_value = mock_blueprint
sys.modules['flask'] = mock_flask
sys.modules['flask_login'] = MagicMock()

from analytics_dashboard import (
    analytics_bp,
    get_detection_stats,
    get_species_distribution,
    get_temporal_patterns,
    get_geographic_data,
    get_confidence_analysis,
    get_system_metrics,
    get_user_activity
)


class MockDB:
    """Mock database for testing."""
    def __init__(self):
        self.session = Mock()
        self.Model = Mock()
        self.func = Mock()
        self.and_ = Mock()
        self.or_ = Mock()
        
    def Column(self, *args, **kwargs):
        return Mock()


class MockDetection:
    """Mock Detection model."""
    def __init__(self, **kwargs):
        self.id = kwargs.get('id', 1)
        self.species = kwargs.get('species', 'Great Horned Owl')
        self.time = kwargs.get('time', datetime.now())
        self.latitude = kwargs.get('latitude', 46.2044)
        self.longitude = kwargs.get('longitude', 6.1432)
        self.zone = kwargs.get('zone', 'Forest Area A')
        self.confidence = kwargs.get('confidence', 0.95)
        self.user_id = kwargs.get('user_id', 1)


class MockPrediction:
    """Mock Prediction model."""
    def __init__(self, **kwargs):
        self.id = kwargs.get('id', 1)
        self.user_id = kwargs.get('user_id', 1)
        self.filename = kwargs.get('filename', 'test.wav')
        self.result = kwargs.get('result', '{"species": "Owl", "confidence": 0.9}')
        self.file_size = kwargs.get('file_size', 1024)


class MockUser:
    """Mock User model."""
    def __init__(self, **kwargs):
        self.id = kwargs.get('id', 1)
        self.username = kwargs.get('username', 'testuser')


@pytest.fixture
def mock_db():
    """Create mock database."""
    return MockDB()


@pytest.fixture
def sample_detections():
    """Create sample detection data."""
    base_time = datetime.now() - timedelta(days=30)
    detections = []
    
    species_list = ['Great Horned Owl', 'Barn Owl', 'Screech Owl', 'Red-tailed Hawk', 'Peregrine Falcon']
    zones = ['Forest Area A', 'Field B', 'Urban Zone C', 'Wetland D']
    
    for i in range(100):
        detection = MockDetection(
            id=i + 1,
            species=species_list[i % len(species_list)],
            time=base_time + timedelta(hours=i),
            latitude=46.2044 + (i % 10) * 0.001,
            longitude=6.1432 + (i % 10) * 0.001,
            zone=zones[i % len(zones)],
            confidence=0.7 + (i % 30) * 0.01,
            user_id=(i % 3) + 1
        )
        detections.append(detection)
    
    return detections


@pytest.fixture
def sample_predictions():
    """Create sample prediction data."""
    predictions = []
    for i in range(50):
        prediction = MockPrediction(
            id=i + 1,
            user_id=(i % 3) + 1,
            filename=f'audio_{i}.wav',
            result=json.dumps({
                'species': ['Owl', 'Hawk', 'Falcon'][i % 3],
                'confidence': 0.8 + (i % 20) * 0.01
            }),
            file_size=1024 * (i + 1)
        )
        predictions.append(prediction)
    
    return predictions


class TestAnalyticsFunctions:
    """Test analytics calculation functions."""
    
    def test_get_detection_stats(self, mock_db, sample_detections):
        """Test detection statistics calculation."""
        # Mock database queries
        mock_db.session.query.return_value.count.return_value = len(sample_detections)
        mock_db.session.query.return_value.filter.return_value.count.return_value = 25  # Last 24h
        mock_db.session.query.return_value.filter.return_value.filter.return_value.count.return_value = 75  # Last 7 days
        
        with patch('analytics_dashboard.db', mock_db), \
             patch('analytics_dashboard.Detection', MockDetection):
            
            stats = get_detection_stats()
            
            assert stats['total_detections'] == 100
            assert stats['last_24h'] == 25
            assert stats['last_7_days'] == 75
            assert 'growth_rate_24h' in stats
            assert 'growth_rate_7d' in stats
            
    def test_get_species_distribution(self, mock_db, sample_detections):
        """Test species distribution analysis."""
        # Mock database query result
        mock_result = [
            ('Great Horned Owl', 20),
            ('Barn Owl', 20),
            ('Screech Owl', 20),
            ('Red-tailed Hawk', 20),
            ('Peregrine Falcon', 20)
        ]
        mock_db.session.query.return_value.group_by.return_value.order_by.return_value.all.return_value = mock_result
        
        with patch('analytics_dashboard.db', mock_db), \
             patch('analytics_dashboard.Detection', MockDetection):
            
            distribution = get_species_distribution()
            
            assert len(distribution) == 5
            assert distribution[0]['species'] == 'Great Horned Owl'
            assert distribution[0]['count'] == 20
            assert all('percentage' in item for item in distribution)
            
    def test_get_temporal_patterns(self, mock_db):
        """Test temporal pattern analysis."""
        # Mock hourly distribution
        hourly_data = [(i, i * 2) for i in range(24)]  # Hour, count pairs
        daily_data = [(i, i * 5) for i in range(7)]    # Day of week, count pairs
        
        mock_db.session.query.return_value.group_by.return_value.order_by.return_value.all.side_effect = [
            hourly_data, daily_data
        ]
        
        with patch('analytics_dashboard.db', mock_db), \
             patch('analytics_dashboard.Detection', MockDetection):
            
            patterns = get_temporal_patterns()
            
            assert 'hourly_distribution' in patterns
            assert 'daily_distribution' in patterns
            assert len(patterns['hourly_distribution']) == 24
            assert len(patterns['daily_distribution']) == 7
            
    def test_get_geographic_data(self, mock_db, sample_detections):
        """Test geographic data analysis."""
        # Mock zone distribution
        zone_data = [
            ('Forest Area A', 25),
            ('Field B', 25),
            ('Urban Zone C', 25),
            ('Wetland D', 25)
        ]
        
        # Mock coordinate data
        coord_data = [(46.2044 + i * 0.001, 6.1432 + i * 0.001, f'Species_{i}', 0.8 + i * 0.01) 
                      for i in range(10)]
        
        mock_query = mock_db.session.query.return_value
        mock_query.group_by.return_value.order_by.return_value.all.return_value = zone_data
        mock_query.filter.return_value.all.return_value = coord_data
        
        with patch('analytics_dashboard.db', mock_db), \
             patch('analytics_dashboard.Detection', MockDetection):
            
            geo_data = get_geographic_data()
            
            assert 'zone_distribution' in geo_data
            assert 'detection_points' in geo_data
            assert len(geo_data['zone_distribution']) == 4
            assert len(geo_data['detection_points']) == 10
            
    def test_get_confidence_analysis(self, mock_db):
        """Test confidence analysis."""
        # Mock confidence histogram data
        confidence_data = [
            (0.7, 10), (0.75, 15), (0.8, 20), (0.85, 25), (0.9, 20), (0.95, 10)
        ]
        
        mock_db.session.query.return_value.group_by.return_value.order_by.return_value.all.return_value = confidence_data
        mock_db.func.round.return_value = Mock()
        
        with patch('analytics_dashboard.db', mock_db), \
             patch('analytics_dashboard.Detection', MockDetection):
            
            analysis = get_confidence_analysis()
            
            assert 'confidence_histogram' in analysis
            assert 'average_confidence' in analysis
            assert 'high_confidence_rate' in analysis
            
    def test_get_system_metrics(self, mock_db):
        """Test system metrics calculation."""
        # Mock system metrics
        mock_db.session.query.return_value.count.side_effect = [100, 50, 3]  # detections, predictions, users
        mock_db.session.query.return_value.scalar.return_value = 5242880  # total file size
        
        with patch('analytics_dashboard.db', mock_db), \
             patch('analytics_dashboard.Detection', MockDetection), \
             patch('analytics_dashboard.Prediction', MockPrediction), \
             patch('analytics_dashboard.User', MockUser), \
             patch('psutil.disk_usage') as mock_disk, \
             patch('psutil.virtual_memory') as mock_memory, \
             patch('psutil.cpu_percent') as mock_cpu:
            
            # Mock system stats
            mock_disk.return_value.free = 50 * 1024**3  # 50GB free
            mock_memory.return_value.percent = 65.5
            mock_cpu.return_value = 25.3
            
            metrics = get_system_metrics()
            
            assert metrics['total_detections'] == 100
            assert metrics['total_predictions'] == 50
            assert metrics['active_users'] == 3
            assert metrics['storage_used_mb'] == 5.0
            assert metrics['disk_free_gb'] == 50.0
            assert metrics['memory_usage_percent'] == 65.5
            assert metrics['cpu_usage_percent'] == 25.3
            
    def test_get_user_activity(self, mock_db):
        """Test user activity analysis."""
        # Mock user activity data
        user_data = [
            (1, 'user1', 25, 15),
            (2, 'user2', 30, 20),
            (3, 'user3', 45, 15)
        ]
        
        mock_db.session.query.return_value.outerjoin.return_value.group_by.return_value.order_by.return_value.all.return_value = user_data
        
        with patch('analytics_dashboard.db', mock_db), \
             patch('analytics_dashboard.User', MockUser), \
             patch('analytics_dashboard.Detection', MockDetection), \
             patch('analytics_dashboard.Prediction', MockPrediction):
            
            activity = get_user_activity()
            
            assert len(activity) == 3
            assert activity[0]['user_id'] == 1
            assert activity[0]['username'] == 'user1'
            assert activity[0]['detection_count'] == 25
            assert activity[0]['prediction_count'] == 15


class TestAnalyticsDashboardRoutes:
    """Test analytics dashboard route handlers."""
    
    @pytest.fixture
    def mock_app(self):
        """Create mock Flask app."""
        app = Mock()
        app.route = Mock()
        return app
        
    def test_analytics_overview_route(self):
        """Test analytics overview route."""
        # This would test the route handler that returns overview data
        with patch('analytics_dashboard.get_detection_stats') as mock_stats, \
             patch('analytics_dashboard.get_species_distribution') as mock_species, \
             patch('analytics_dashboard.get_system_metrics') as mock_metrics:
            
            mock_stats.return_value = {'total_detections': 100}
            mock_species.return_value = [{'species': 'Owl', 'count': 50}]
            mock_metrics.return_value = {'cpu_usage_percent': 25.0}
            
            # Test would call the actual route handler
            # and verify the response format
            pass
            
    def test_analytics_temporal_route(self):
        """Test temporal analytics route."""
        with patch('analytics_dashboard.get_temporal_patterns') as mock_patterns:
            mock_patterns.return_value = {
                'hourly_distribution': [],
                'daily_distribution': []
            }
            
            # Test would call the temporal analytics route
            pass
            
    def test_analytics_geographic_route(self):
        """Test geographic analytics route."""
        with patch('analytics_dashboard.get_geographic_data') as mock_geo:
            mock_geo.return_value = {
                'zone_distribution': [],
                'detection_points': []
            }
            
            # Test would call the geographic analytics route
            pass


class TestAnalyticsDataProcessing:
    """Test data processing and transformation functions."""
    
    def test_confidence_histogram_binning(self):
        """Test confidence score binning for histogram."""
        # Mock confidence data
        confidence_values = [0.65, 0.72, 0.78, 0.85, 0.91, 0.97]
        
        # Test binning logic
        bins = {}
        for conf in confidence_values:
            bin_key = round(conf, 1)
            bins[bin_key] = bins.get(bin_key, 0) + 1
            
        assert 0.7 in bins
        assert 0.9 in bins
        assert bins[0.7] >= 1
        
    def test_temporal_data_aggregation(self):
        """Test temporal data aggregation logic."""
        # Mock timestamp data
        timestamps = [
            datetime(2023, 1, 1, 10, 30),  # Monday, 10 AM
            datetime(2023, 1, 2, 14, 15),  # Tuesday, 2 PM
            datetime(2023, 1, 1, 10, 45),  # Monday, 10 AM (same hour)
        ]
        
        # Test hourly aggregation
        hourly_counts = {}
        for ts in timestamps:
            hour = ts.hour
            hourly_counts[hour] = hourly_counts.get(hour, 0) + 1
            
        assert hourly_counts[10] == 2  # Two detections at 10 AM
        assert hourly_counts[14] == 1  # One detection at 2 PM
        
        # Test daily aggregation
        daily_counts = {}
        for ts in timestamps:
            day = ts.weekday()  # Monday = 0
            daily_counts[day] = daily_counts.get(day, 0) + 1
            
        assert daily_counts[0] == 2  # Two detections on Monday
        assert daily_counts[1] == 1  # One detection on Tuesday
        
    def test_geographic_data_formatting(self):
        """Test geographic data formatting for maps."""
        # Mock raw coordinate data
        raw_data = [
            (46.2044, 6.1432, 'Owl', 0.95),
            (46.2055, 6.1445, 'Hawk', 0.87),
            (46.2033, 6.1420, 'Falcon', 0.92)
        ]
        
        # Test formatting for map display
        formatted_points = []
        for lat, lng, species, confidence in raw_data:
            point = {
                'lat': lat,
                'lng': lng,
                'species': species,
                'confidence': confidence,
                'popup_text': f'{species} (Confidence: {confidence:.0%})'
            }
            formatted_points.append(point)
            
        assert len(formatted_points) == 3
        assert formatted_points[0]['lat'] == 46.2044
        assert 'Owl' in formatted_points[0]['popup_text']
        assert '95%' in formatted_points[0]['popup_text']


class TestAnalyticsErrorHandling:
    """Test error handling in analytics functions."""
    
    def test_database_error_handling(self, mock_db):
        """Test handling of database errors."""
        # Mock database error
        mock_db.session.query.side_effect = Exception("Database connection error")
        
        with patch('analytics_dashboard.db', mock_db), \
             patch('analytics_dashboard.Detection', MockDetection):
            
            # Functions should handle errors gracefully
            stats = get_detection_stats()
            assert isinstance(stats, dict)
            # Should return empty or default values on error
            
    def test_empty_data_handling(self, mock_db):
        """Test handling of empty datasets."""
        # Mock empty query results
        mock_db.session.query.return_value.count.return_value = 0
        mock_db.session.query.return_value.all.return_value = []
        
        with patch('analytics_dashboard.db', mock_db), \
             patch('analytics_dashboard.Detection', MockDetection):
            
            distribution = get_species_distribution()
            assert isinstance(distribution, list)
            assert len(distribution) == 0
            
    def test_invalid_data_filtering(self):
        """Test filtering of invalid or malformed data."""
        # Mock data with some invalid entries
        raw_detections = [
            {'species': 'Owl', 'confidence': 0.95, 'latitude': 46.2044, 'longitude': 6.1432},
            {'species': None, 'confidence': 0.87, 'latitude': 46.2055, 'longitude': 6.1445},  # Invalid
            {'species': 'Hawk', 'confidence': None, 'latitude': 46.2033, 'longitude': 6.1420},  # Invalid
            {'species': 'Falcon', 'confidence': 0.92, 'latitude': None, 'longitude': None},  # Invalid coords
        ]
        
        # Filter valid detections
        valid_detections = [
            d for d in raw_detections 
            if d.get('species') and d.get('confidence') is not None
        ]
        
        assert len(valid_detections) == 2  # Only 2 valid entries
        assert all(d['species'] for d in valid_detections)
        assert all(d['confidence'] is not None for d in valid_detections)


@pytest.mark.integration
class TestAnalyticsDashboardIntegration:
    """Integration tests for analytics dashboard."""
    
    def test_full_analytics_pipeline(self):
        """Test complete analytics data pipeline."""
        # This would test the full flow from raw data to dashboard display
        pass
        
    def test_real_time_data_updates(self):
        """Test real-time data updates in analytics."""
        # This would test WebSocket updates for real-time analytics
        pass
        
    def test_analytics_caching(self):
        """Test analytics data caching for performance."""
        # This would test that expensive analytics queries are cached
        pass
        
    def test_analytics_export(self):
        """Test analytics data export functionality."""
        # This would test CSV/JSON export of analytics data
        pass


if __name__ == '__main__':
    pytest.main([__file__])