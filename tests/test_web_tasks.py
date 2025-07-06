import pytest
import json
import time
from unittest.mock import Mock, patch, MagicMock
import sys
from pathlib import Path

# Add the parent directory to Python path
sys.path.append(str(Path(__file__).resolve().parents[1]))

# Mock celery and dependencies
mock_celery = MagicMock()
mock_celery.Celery = MagicMock
mock_celery.task = lambda f: f  # Simple decorator mock
sys.modules['celery'] = mock_celery

# Mock Flask dependencies
mock_flask = MagicMock()
mock_flask_sqlalchemy = MagicMock()
sys.modules['flask'] = mock_flask
sys.modules['flask_sqlalchemy'] = mock_flask_sqlalchemy

from web.tasks import (
    run_prediction,
    send_detection_notifications,
    send_system_alert_notifications,
    create_celery_app
)


class MockPrediction:
    """Mock Prediction model for testing."""
    def __init__(self, id, user_id, filename):
        self.id = id
        self.user_id = user_id
        self.filename = filename
        self.result = None


class MockDB:
    """Mock database session."""
    def __init__(self):
        self.session = Mock()
        self.predictions = {}
        
    def get(self, model, pred_id):
        """Mock getting a prediction by ID."""
        return self.predictions.get(pred_id)
        
    def add_prediction(self, pred_id, user_id, filename):
        """Helper to add a mock prediction."""
        pred = MockPrediction(pred_id, user_id, filename)
        self.predictions[pred_id] = pred
        return pred


class TestCeleryAppCreation:
    """Test Celery app creation and configuration."""
    
    @patch('web.tasks.create_app')
    def test_create_celery_app(self, mock_create_app):
        """Test Celery app creation."""
        mock_app = Mock()
        mock_create_app.return_value = mock_app
        
        app = create_celery_app()
        
        assert app == mock_app
        mock_create_app.assert_called_once()


class TestRunPredictionTask:
    """Test the run_prediction Celery task."""
    
    @pytest.fixture
    def mock_prediction_setup(self):
        """Set up mock prediction environment."""
        mock_db = MockDB()
        mock_app = Mock()
        mock_app.app_context.return_value.__enter__ = Mock()
        mock_app.app_context.return_value.__exit__ = Mock()
        
        # Add a test prediction
        pred = mock_db.add_prediction(1, 123, "test_audio.wav")
        
        return mock_db, mock_app, pred
        
    @patch('web.tasks.create_celery_app')
    @patch('web.tasks.requests.post')
    @patch('web.tasks.time.time')
    def test_run_prediction_success(self, mock_time, mock_post, mock_create_app, mock_prediction_setup):
        """Test successful prediction processing."""
        mock_db, mock_app, pred = mock_prediction_setup
        
        # Mock time for processing time calculation
        mock_time.side_effect = [0.0, 2.5]  # Start and end times
        
        # Mock successful API response
        mock_response = Mock()
        mock_response.json.return_value = {
            'species': 'Great Horned Owl',
            'confidence': 0.95,
            'detections': [{'time': 1.5, 'frequency': 1200}]
        }
        mock_post.return_value = mock_response
        
        # Mock app context and database
        mock_create_app.return_value = mock_app
        
        with patch('web.tasks.db', mock_db), \
             patch('web.tasks.Prediction', MockPrediction), \
             patch('web.tasks.get_notification_service') as mock_notification:
            
            mock_notification_service = Mock()
            mock_notification.return_value = mock_notification_service
            mock_notification_service.send_prediction_complete_notification = Mock()
            
            # Mock database session
            mock_db.session.get.return_value = pred
            
            # Run the task
            run_prediction(1, "test_audio.wav", b"fake_wav_data", "http://api.example.com/predict")
            
            # Verify API was called correctly
            mock_post.assert_called_once()
            call_args = mock_post.call_args
            assert call_args[1]['files']['file'][0] == "test_audio.wav"
            assert call_args[1]['timeout'] == 30
            
            # Verify prediction result was saved
            mock_db.session.commit.assert_called_once()
            assert pred.result is not None
            result_data = json.loads(pred.result)
            assert result_data['species'] == 'Great Horned Owl'
            assert result_data['confidence'] == 0.95
            
    @patch('web.tasks.create_celery_app')
    @patch('web.tasks.requests.post')
    def test_run_prediction_api_error(self, mock_post, mock_create_app, mock_prediction_setup):
        """Test prediction with API error."""
        mock_db, mock_app, pred = mock_prediction_setup
        
        # Mock API error
        mock_post.side_effect = Exception("API connection failed")
        mock_create_app.return_value = mock_app
        
        with patch('web.tasks.db', mock_db), \
             patch('web.tasks.Prediction', MockPrediction):
            
            mock_db.session.get.return_value = pred
            
            # Run the task (should handle error gracefully)
            run_prediction(1, "test_audio.wav", b"fake_wav_data", "http://api.example.com/predict")
            
            # Verify error result was saved
            assert pred.result is not None
            result_data = json.loads(pred.result)
            assert 'error' in result_data
            
    @patch('web.tasks.create_celery_app')
    @patch('web.tasks.requests.post')
    def test_run_prediction_prediction_not_found(self, mock_post, mock_create_app):
        """Test prediction when prediction record not found."""
        mock_db = MockDB()
        mock_app = Mock()
        mock_app.app_context.return_value.__enter__ = Mock()
        mock_app.app_context.return_value.__exit__ = Mock()
        
        mock_create_app.return_value = mock_app
        
        with patch('web.tasks.db', mock_db), \
             patch('web.tasks.Prediction', MockPrediction):
            
            # Return None for prediction (not found)
            mock_db.session.get.return_value = None
            
            # Should not crash when prediction not found
            run_prediction(999, "test_audio.wav", b"fake_wav_data", "http://api.example.com/predict")
            
            # Should still call the API
            mock_post.assert_called_once()
            
    @patch('web.tasks.create_celery_app')
    @patch('web.tasks.requests.post')
    def test_run_prediction_with_notification(self, mock_post, mock_create_app, mock_prediction_setup):
        """Test prediction with notification sending."""
        mock_db, mock_app, pred = mock_prediction_setup
        
        # Mock successful API response
        mock_response = Mock()
        mock_response.json.return_value = {'species': 'Barn Owl', 'confidence': 0.87}
        mock_post.return_value = mock_response
        mock_create_app.return_value = mock_app
        
        with patch('web.tasks.db', mock_db), \
             patch('web.tasks.Prediction', MockPrediction), \
             patch('web.tasks.get_notification_service') as mock_notification, \
             patch('web.tasks.asyncio') as mock_asyncio:
            
            mock_notification_service = Mock()
            mock_notification.return_value = mock_notification_service
            mock_notification_service.send_prediction_complete_notification = Mock()
            
            # Mock asyncio event loop
            mock_loop = Mock()
            mock_asyncio.new_event_loop.return_value = mock_loop
            mock_asyncio.set_event_loop.return_value = None
            
            mock_db.session.get.return_value = pred
            
            # Run the task
            run_prediction(1, "test_audio.wav", b"fake_wav_data", "http://api.example.com/predict")
            
            # Verify notification service was called
            mock_notification.assert_called_once_with(mock_db)
            
            # Verify asyncio loop was used
            mock_asyncio.new_event_loop.assert_called_once()
            mock_loop.run_until_complete.assert_called_once()
            mock_loop.close.assert_called_once()


class TestDetectionNotificationsTask:
    """Test the send_detection_notifications Celery task."""
    
    @patch('web.tasks.create_celery_app')
    def test_send_detection_notifications_success(self, mock_create_app):
        """Test successful detection notification sending."""
        mock_app = Mock()
        mock_app.app_context.return_value.__enter__ = Mock()
        mock_app.app_context.return_value.__exit__ = Mock()
        mock_create_app.return_value = mock_app
        
        detection_data = {
            'id': 1,
            'species': 'Great Horned Owl',
            'zone': 'Forest Area A',
            'confidence': 0.95,
            'timestamp': '2023-01-01T12:00:00Z'
        }
        user_ids = [1, 2, 3]
        
        with patch('web.tasks.get_notification_service') as mock_notification, \
             patch('web.tasks.asyncio') as mock_asyncio:
            
            mock_notification_service = Mock()
            mock_notification.return_value = mock_notification_service
            mock_notification_service.send_detection_notification = Mock()
            
            # Mock asyncio event loop
            mock_loop = Mock()
            mock_asyncio.new_event_loop.return_value = mock_loop
            
            # Run the task
            send_detection_notifications(detection_data, user_ids)
            
            # Verify notification service was called correctly
            mock_notification_service.send_detection_notification.assert_called_once()
            call_args = mock_notification_service.send_detection_notification.call_args[0]
            assert call_args[0] == detection_data
            assert call_args[1] == user_ids
            
    @patch('web.tasks.create_celery_app')
    def test_send_detection_notifications_error(self, mock_create_app):
        """Test detection notification with error handling."""
        mock_app = Mock()
        mock_app.app_context.return_value.__enter__ = Mock()
        mock_app.app_context.return_value.__exit__ = Mock()
        mock_create_app.return_value = mock_app
        
        detection_data = {'id': 1, 'species': 'Owl'}
        
        with patch('web.tasks.get_notification_service') as mock_notification, \
             patch('web.tasks.asyncio') as mock_asyncio:
            
            # Mock notification service to raise an exception
            mock_notification.side_effect = Exception("Notification service error")
            
            # Should not crash on error
            send_detection_notifications(detection_data, [1])
            
            # Exception should be caught and handled gracefully
            assert True  # If we get here, exception was handled
            
    @patch('web.tasks.create_celery_app')
    def test_send_detection_notifications_no_users(self, mock_create_app):
        """Test detection notification with no target users."""
        mock_app = Mock()
        mock_app.app_context.return_value.__enter__ = Mock()
        mock_app.app_context.return_value.__exit__ = Mock()
        mock_create_app.return_value = mock_app
        
        detection_data = {'id': 1, 'species': 'Owl'}
        
        with patch('web.tasks.get_notification_service') as mock_notification:
            mock_notification_service = Mock()
            mock_notification.return_value = mock_notification_service
            
            # Run with None user_ids
            send_detection_notifications(detection_data, None)
            
            # Should still call notification service
            mock_notification_service.send_detection_notification.assert_called_once()


class TestSystemAlertNotificationsTask:
    """Test the send_system_alert_notifications Celery task."""
    
    @patch('web.tasks.create_celery_app')
    def test_send_system_alert_success(self, mock_create_app):
        """Test successful system alert notification."""
        mock_app = Mock()
        mock_app.app_context.return_value.__enter__ = Mock()
        mock_app.app_context.return_value.__exit__ = Mock()
        mock_create_app.return_value = mock_app
        
        alert_data = {
            'type': 'System Error',
            'message': 'Database connection lost',
            'priority': 'high',
            'timestamp': '2023-01-01T12:00:00Z'
        }
        user_ids = [1, 2]
        
        with patch('web.tasks.get_notification_service') as mock_notification, \
             patch('web.tasks.asyncio') as mock_asyncio:
            
            mock_notification_service = Mock()
            mock_notification.return_value = mock_notification_service
            mock_notification_service.send_system_alert = Mock()
            
            # Mock asyncio event loop
            mock_loop = Mock()
            mock_asyncio.new_event_loop.return_value = mock_loop
            
            # Run the task
            send_system_alert_notifications(alert_data, user_ids)
            
            # Verify notification service was called correctly
            mock_notification_service.send_system_alert.assert_called_once()
            call_args = mock_notification_service.send_system_alert.call_args[0]
            assert call_args[0] == alert_data
            assert call_args[1] == user_ids
            
    @patch('web.tasks.create_celery_app')
    def test_send_system_alert_all_users(self, mock_create_app):
        """Test system alert to all users."""
        mock_app = Mock()
        mock_app.app_context.return_value.__enter__ = Mock()
        mock_app.app_context.return_value.__exit__ = Mock()
        mock_create_app.return_value = mock_app
        
        alert_data = {
            'type': 'Maintenance',
            'message': 'System will be down for maintenance',
            'priority': 'normal'
        }
        
        with patch('web.tasks.get_notification_service') as mock_notification:
            mock_notification_service = Mock()
            mock_notification.return_value = mock_notification_service
            
            # Run with None user_ids (broadcast to all)
            send_system_alert_notifications(alert_data, None)
            
            # Should call with None user_ids for broadcast
            mock_notification_service.send_system_alert.assert_called_once()
            call_args = mock_notification_service.send_system_alert.call_args[0]
            assert call_args[1] is None


class TestTasksIntegration:
    """Integration tests for Celery tasks."""
    
    @patch('web.tasks.create_celery_app')
    @patch('web.tasks.requests.post')
    def test_prediction_task_end_to_end(self, mock_post, mock_create_app):
        """Test complete prediction task flow."""
        # Mock successful prediction workflow
        mock_app = Mock()
        mock_app.app_context.return_value.__enter__ = Mock()
        mock_app.app_context.return_value.__exit__ = Mock()
        mock_create_app.return_value = mock_app
        
        # Mock API response
        mock_response = Mock()
        mock_response.json.return_value = {
            'species': 'Screech Owl',
            'confidence': 0.82,
            'processing_time': 1.8
        }
        mock_post.return_value = mock_response
        
        # Mock database and prediction
        mock_db = MockDB()
        pred = mock_db.add_prediction(1, 123, "integration_test.wav")
        
        with patch('web.tasks.db', mock_db), \
             patch('web.tasks.Prediction', MockPrediction), \
             patch('web.tasks.get_notification_service') as mock_notification:
            
            mock_notification_service = Mock()
            mock_notification.return_value = mock_notification_service
            mock_db.session.get.return_value = pred
            
            # Run the complete workflow
            run_prediction(1, "integration_test.wav", b"fake_audio_data", "http://api.test.com/predict")
            
            # Verify all steps completed
            assert pred.result is not None
            result = json.loads(pred.result)
            assert result['species'] == 'Screech Owl'
            assert result['confidence'] == 0.82
            
    def test_notification_tasks_chaining(self):
        """Test chaining of different notification tasks."""
        # This would test scenarios where multiple notification tasks
        # are triggered in sequence (e.g., detection -> alert -> email)
        pass
        
    def test_task_retry_behavior(self):
        """Test task retry behavior on failures."""
        # This would test Celery retry mechanisms
        pass
        
    def test_task_concurrency(self):
        """Test task execution under concurrent conditions."""
        # This would test multiple tasks running simultaneously
        pass


class TestTaskErrorHandling:
    """Test error handling in Celery tasks."""
    
    @patch('web.tasks.create_celery_app')
    def test_database_connection_error(self, mock_create_app):
        """Test handling of database connection errors."""
        mock_app = Mock()
        # Mock database connection failure
        mock_app.app_context.side_effect = Exception("Database connection failed")
        mock_create_app.return_value = mock_app
        
        # Should handle database errors gracefully
        try:
            run_prediction(1, "test.wav", b"data", "http://api.test.com")
        except Exception:
            pytest.fail("Task should handle database errors gracefully")
            
    @patch('web.tasks.create_celery_app')
    @patch('web.tasks.requests.post')
    def test_api_timeout_handling(self, mock_post, mock_create_app):
        """Test handling of API timeout errors."""
        mock_app = Mock()
        mock_app.app_context.return_value.__enter__ = Mock()
        mock_app.app_context.return_value.__exit__ = Mock()
        mock_create_app.return_value = mock_app
        
        # Mock API timeout
        import requests
        mock_post.side_effect = requests.Timeout("API request timed out")
        
        mock_db = MockDB()
        pred = mock_db.add_prediction(1, 123, "timeout_test.wav")
        
        with patch('web.tasks.db', mock_db), \
             patch('web.tasks.Prediction', MockPrediction):
            
            mock_db.session.get.return_value = pred
            
            # Should handle timeout gracefully
            run_prediction(1, "timeout_test.wav", b"data", "http://slow-api.test.com")
            
            # Should record error in result
            assert pred.result is not None
            result = json.loads(pred.result)
            assert 'error' in result
            
    def test_invalid_task_parameters(self):
        """Test tasks with invalid parameters."""
        # Test with None parameters
        try:
            send_detection_notifications(None, None)
            send_system_alert_notifications(None, None)
        except Exception:
            pytest.fail("Tasks should handle None parameters gracefully")
            
        # Test with empty data
        try:
            send_detection_notifications({}, [])
            send_system_alert_notifications({}, [])
        except Exception:
            pytest.fail("Tasks should handle empty data gracefully")


if __name__ == '__main__':
    pytest.main([__file__])