import pytest
import time
from unittest.mock import Mock, patch, MagicMock
import sys
from pathlib import Path

# Add the parent directory to Python path
sys.path.append(str(Path(__file__).resolve().parents[1]))

# Mock prometheus_client before importing metrics
mock_prometheus = MagicMock()
mock_prometheus.Counter = MagicMock
mock_prometheus.Histogram = MagicMock
mock_prometheus.Gauge = MagicMock
mock_prometheus.generate_latest = MagicMock
mock_prometheus.CONTENT_TYPE_LATEST = 'text/plain; version=0.0.4; charset=utf-8'
sys.modules['prometheus_client'] = mock_prometheus

from metrics import (
    track_request_metrics,
    record_failed_login,
    record_quota_usage,
    record_detection_event,
    record_prediction_metrics,
    record_websocket_connection,
    record_email_sent,
    get_metrics,
    CONTENT_TYPE_LATEST,
    # Import metric instances for testing
    REQUEST_COUNT,
    REQUEST_DURATION,
    FAILED_LOGIN_COUNT,
    QUOTA_USAGE,
    DETECTION_COUNT,
    PREDICTION_PROCESSING_TIME,
    WEBSOCKET_CONNECTIONS,
    EMAIL_NOTIFICATIONS
)


class TestMetricsInitialization:
    """Test metrics initialization and configuration."""
    
    def test_metrics_are_created(self):
        """Test that all metrics are properly initialized."""
        # Verify that metric instances exist
        assert REQUEST_COUNT is not None
        assert REQUEST_DURATION is not None
        assert FAILED_LOGIN_COUNT is not None
        assert QUOTA_USAGE is not None
        assert DETECTION_COUNT is not None
        assert PREDICTION_PROCESSING_TIME is not None
        assert WEBSOCKET_CONNECTIONS is not None
        assert EMAIL_NOTIFICATIONS is not None
        
    def test_content_type_constant(self):
        """Test that content type constant is properly set."""
        assert CONTENT_TYPE_LATEST == 'text/plain; version=0.0.4; charset=utf-8'


class TestRequestMetrics:
    """Test request tracking metrics."""
    
    def test_track_request_metrics_decorator(self):
        """Test the request metrics tracking decorator."""
        
        @track_request_metrics
        def sample_endpoint():
            """Sample endpoint for testing."""
            time.sleep(0.1)  # Simulate processing time
            return "success"
        
        # Mock Flask request context
        with patch('metrics.request') as mock_request:
            mock_request.method = 'GET'
            mock_request.endpoint = 'test_endpoint'
            
            # Call the decorated function
            result = sample_endpoint()
            
            assert result == "success"
            
            # Verify metrics were recorded
            REQUEST_COUNT.labels.assert_called_with(
                method='GET',
                endpoint='test_endpoint',
                status_code=200
            )
            REQUEST_DURATION.labels.assert_called_with(
                method='GET',
                endpoint='test_endpoint'
            )
            
    def test_track_request_metrics_with_exception(self):
        """Test request metrics tracking when function raises exception."""
        
        @track_request_metrics
        def failing_endpoint():
            """Endpoint that raises an exception."""
            raise ValueError("Test exception")
        
        with patch('metrics.request') as mock_request:
            mock_request.method = 'POST'
            mock_request.endpoint = 'failing_endpoint'
            
            # Function should still raise the exception
            with pytest.raises(ValueError):
                failing_endpoint()
            
            # Verify metrics were still recorded
            REQUEST_COUNT.labels.assert_called_with(
                method='POST',
                endpoint='failing_endpoint',
                status_code=500
            )
            
    def test_track_request_metrics_without_flask_context(self):
        """Test request metrics tracking outside Flask request context."""
        
        @track_request_metrics
        def standalone_function():
            """Function called outside request context."""
            return "standalone"
        
        # Should handle gracefully when no request context
        result = standalone_function()
        assert result == "standalone"


class TestSecurityMetrics:
    """Test security-related metrics."""
    
    def test_record_failed_login(self):
        """Test failed login attempt recording."""
        # Test different failure reasons
        record_failed_login('invalid_credentials')
        FAILED_LOGIN_COUNT.labels.assert_called_with(reason='invalid_credentials')
        
        record_failed_login('invalid_captcha')
        FAILED_LOGIN_COUNT.labels.assert_called_with(reason='invalid_captcha')
        
        record_failed_login('missing_credentials')
        FAILED_LOGIN_COUNT.labels.assert_called_with(reason='missing_credentials')
        
    def test_record_failed_login_with_ip(self):
        """Test failed login recording with IP address."""
        record_failed_login('192.168.1.100')
        FAILED_LOGIN_COUNT.labels.assert_called_with(reason='192.168.1.100')


class TestSystemMetrics:
    """Test system resource and usage metrics."""
    
    def test_record_quota_usage(self):
        """Test quota usage recording."""
        # Test different usage percentages
        record_quota_usage(25.5)
        QUOTA_USAGE.set.assert_called_with(25.5)
        
        record_quota_usage(85.2)
        QUOTA_USAGE.set.assert_called_with(85.2)
        
        record_quota_usage(100.0)
        QUOTA_USAGE.set.assert_called_with(100.0)
        
    def test_record_quota_usage_bounds(self):
        """Test quota usage with boundary values."""
        # Test edge cases
        record_quota_usage(0.0)
        QUOTA_USAGE.set.assert_called_with(0.0)
        
        # Should handle values over 100%
        record_quota_usage(105.3)
        QUOTA_USAGE.set.assert_called_with(105.3)


class TestDetectionMetrics:
    """Test wildlife detection metrics."""
    
    def test_record_detection_event(self):
        """Test detection event recording."""
        # Test different species
        record_detection_event('Great Horned Owl', 0.95, 'Forest Area A')
        DETECTION_COUNT.labels.assert_called_with(
            species='Great Horned Owl',
            zone='Forest Area A'
        )
        
        record_detection_event('Barn Owl', 0.87, 'Field B')
        DETECTION_COUNT.labels.assert_called_with(
            species='Barn Owl',
            zone='Field B'
        )
        
    def test_record_detection_event_without_zone(self):
        """Test detection event recording without zone information."""
        record_detection_event('Red-tailed Hawk', 0.92, None)
        DETECTION_COUNT.labels.assert_called_with(
            species='Red-tailed Hawk',
            zone='unknown'
        )
        
        record_detection_event('Peregrine Falcon', 0.89, '')
        DETECTION_COUNT.labels.assert_called_with(
            species='Peregrine Falcon',
            zone='unknown'
        )
        
    def test_record_detection_event_with_special_characters(self):
        """Test detection with special characters in species name."""
        record_detection_event('Great Gray Owl (Strix nebulosa)', 0.88, 'Zone-1')
        DETECTION_COUNT.labels.assert_called_with(
            species='Great Gray Owl (Strix nebulosa)',
            zone='Zone-1'
        )


class TestPredictionMetrics:
    """Test audio prediction metrics."""
    
    def test_record_prediction_metrics(self):
        """Test prediction processing metrics."""
        # Test successful prediction
        record_prediction_metrics(2.5, 'completed', 1024)
        
        PREDICTION_PROCESSING_TIME.labels.assert_called_with(status='completed')
        PREDICTION_PROCESSING_TIME.labels.return_value.observe.assert_called_with(2.5)
        
    def test_record_prediction_metrics_failed(self):
        """Test prediction metrics for failed predictions."""
        record_prediction_metrics(0.8, 'error', 512)
        
        PREDICTION_PROCESSING_TIME.labels.assert_called_with(status='error')
        PREDICTION_PROCESSING_TIME.labels.return_value.observe.assert_called_with(0.8)
        
    def test_record_prediction_metrics_timeout(self):
        """Test prediction metrics for timeout cases."""
        record_prediction_metrics(30.0, 'timeout', 2048)
        
        PREDICTION_PROCESSING_TIME.labels.assert_called_with(status='timeout')
        PREDICTION_PROCESSING_TIME.labels.return_value.observe.assert_called_with(30.0)


class TestWebSocketMetrics:
    """Test WebSocket connection metrics."""
    
    def test_record_websocket_connection_connect(self):
        """Test WebSocket connection recording."""
        record_websocket_connection('connect')
        WEBSOCKET_CONNECTIONS.labels.assert_called_with(event='connect')
        
    def test_record_websocket_connection_disconnect(self):
        """Test WebSocket disconnection recording."""
        record_websocket_connection('disconnect')
        WEBSOCKET_CONNECTIONS.labels.assert_called_with(event='disconnect')
        
    def test_record_websocket_connection_authenticate(self):
        """Test WebSocket authentication recording."""
        record_websocket_connection('authenticate')
        WEBSOCKET_CONNECTIONS.labels.assert_called_with(event='authenticate')
        
    def test_record_websocket_connection_error(self):
        """Test WebSocket error recording."""
        record_websocket_connection('error')
        WEBSOCKET_CONNECTIONS.labels.assert_called_with(event='error')


class TestNotificationMetrics:
    """Test notification delivery metrics."""
    
    def test_record_email_sent_success(self):
        """Test successful email notification recording."""
        record_email_sent('detection', 'success')
        EMAIL_NOTIFICATIONS.labels.assert_called_with(
            type='detection',
            status='success'
        )
        
    def test_record_email_sent_failure(self):
        """Test failed email notification recording."""
        record_email_sent('system_alert', 'failed')
        EMAIL_NOTIFICATIONS.labels.assert_called_with(
            type='system_alert',
            status='failed'
        )
        
    def test_record_email_sent_prediction(self):
        """Test prediction complete email recording."""
        record_email_sent('prediction_complete', 'success')
        EMAIL_NOTIFICATIONS.labels.assert_called_with(
            type='prediction_complete',
            status='success'
        )


class TestMetricsExport:
    """Test metrics export functionality."""
    
    def test_get_metrics(self):
        """Test metrics export function."""
        # Mock prometheus generate_latest
        mock_prometheus.generate_latest.return_value = b'# Sample metrics data\nrequest_count_total 42\n'
        
        result = get_metrics()
        
        assert isinstance(result, str)
        mock_prometheus.generate_latest.assert_called_once()
        
    def test_get_metrics_content_type(self):
        """Test that metrics are returned in correct format."""
        mock_prometheus.generate_latest.return_value = b'metric_data'
        
        result = get_metrics()
        
        # Should return string (decoded bytes)
        assert isinstance(result, str)
        
    def test_get_metrics_empty(self):
        """Test metrics export when no data exists."""
        mock_prometheus.generate_latest.return_value = b''
        
        result = get_metrics()
        
        assert result == ''


class TestMetricsIntegration:
    """Integration tests for metrics functionality."""
    
    def test_multiple_metrics_recording(self):
        """Test recording multiple different metrics."""
        # Record various metrics
        record_detection_event('Owl', 0.95, 'Forest')
        record_prediction_metrics(1.5, 'completed', 1024)
        record_websocket_connection('connect')
        record_email_sent('detection', 'success')
        record_failed_login('invalid_credentials')
        record_quota_usage(75.0)
        
        # Verify all metrics were called
        assert DETECTION_COUNT.labels.called
        assert PREDICTION_PROCESSING_TIME.labels.called
        assert WEBSOCKET_CONNECTIONS.labels.called
        assert EMAIL_NOTIFICATIONS.labels.called
        assert FAILED_LOGIN_COUNT.labels.called
        assert QUOTA_USAGE.set.called
        
    def test_metric_labels_consistency(self):
        """Test that metric labels are consistent."""
        # Test that the same metric with same labels works correctly
        record_detection_event('Owl', 0.95, 'Forest')
        record_detection_event('Owl', 0.87, 'Forest')  # Same species and zone
        
        # Should be called twice with same labels
        assert DETECTION_COUNT.labels.call_count >= 2
        
    def test_metrics_with_unicode_data(self):
        """Test metrics with unicode characters."""
        # Test with unicode species name
        record_detection_event('Strix nebulosa', 0.92, 'Forêt')
        
        DETECTION_COUNT.labels.assert_called_with(
            species='Strix nebulosa',
            zone='Forêt'
        )


class TestMetricsErrorHandling:
    """Test error handling in metrics recording."""
    
    def test_metrics_with_none_values(self):
        """Test metrics recording with None values."""
        # Should handle None gracefully
        record_detection_event(None, 0.95, 'Forest')
        record_detection_event('Owl', None, 'Forest')
        
        # Should not crash, may use default values
        assert True  # If we get here, no exception was raised
        
    def test_metrics_with_invalid_types(self):
        """Test metrics with invalid data types."""
        # Test with invalid processing time
        record_prediction_metrics('invalid', 'completed', 1024)
        
        # Should handle gracefully
        assert True
        
    def test_metrics_recording_exception(self):
        """Test behavior when metrics recording fails."""
        # Mock metrics to raise exception
        with patch.object(DETECTION_COUNT, 'labels', side_effect=Exception("Metrics error")):
            # Should not crash the application
            record_detection_event('Owl', 0.95, 'Forest')
            
            # Application should continue working
            assert True


@pytest.mark.performance
class TestMetricsPerformance:
    """Performance tests for metrics recording."""
    
    def test_metrics_recording_performance(self):
        """Test that metrics recording is fast."""
        start_time = time.time()
        
        # Record many metrics quickly
        for i in range(1000):
            record_detection_event(f'Species_{i % 10}', 0.9, f'Zone_{i % 5}')
            
        end_time = time.time()
        
        # Should complete quickly (less than 1 second for 1000 operations)
        assert (end_time - start_time) < 1.0
        
    def test_decorator_performance_overhead(self):
        """Test that the metrics decorator has minimal overhead."""
        
        @track_request_metrics
        def fast_function():
            return "quick"
        
        def regular_function():
            return "quick"
        
        # Time the decorated function
        start_time = time.time()
        for _ in range(100):
            fast_function()
        decorated_time = time.time() - start_time
        
        # Time the regular function
        start_time = time.time()
        for _ in range(100):
            regular_function()
        regular_time = time.time() - start_time
        
        # Overhead should be minimal (less than 50% increase)
        overhead_ratio = decorated_time / regular_time if regular_time > 0 else 1
        assert overhead_ratio < 1.5


if __name__ == '__main__':
    pytest.main([__file__])