import pytest
import asyncio
import json
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from datetime import datetime
import tempfile
import os
import sys
from pathlib import Path

# Add the parent directory to Python path
sys.path.append(str(Path(__file__).resolve().parents[1]))

# Mock dependencies that might not be available
sys.modules['socketio'] = MagicMock()

from notification_service import (
    NotificationService, 
    get_notification_service,
    EmailService,
    WebSocketService,
    PushNotificationService
)


class MockDB:
    """Mock database for testing."""
    def __init__(self):
        self.session = Mock()
        self.Model = Mock()
        
    def Column(self, *args, **kwargs):
        return Mock()
        
    def Integer(self):
        return Mock()
        
    def String(self, length):
        return Mock()
        
    def Boolean(self):
        return Mock()
        
    def Text(self):
        return Mock()
        
    def ForeignKey(self, key):
        return Mock()
        
    def relationship(self, *args, **kwargs):
        return Mock()


class MockNotificationPreference:
    """Mock notification preference model."""
    def __init__(self, **kwargs):
        self.user_id = kwargs.get('user_id', 1)
        self.email_notifications = kwargs.get('email_notifications', True)
        self.push_notifications = kwargs.get('push_notifications', True)
        self.email_address = kwargs.get('email_address', 'test@example.com')
        self.min_priority = kwargs.get('min_priority', 'normal')
        self.species_filter = kwargs.get('species_filter', '[]')
        self.zone_filter = kwargs.get('zone_filter', '[]')
        self.quiet_hours_start = kwargs.get('quiet_hours_start')
        self.quiet_hours_end = kwargs.get('quiet_hours_end')
        self.slack_webhook = kwargs.get('slack_webhook')
        self.discord_webhook = kwargs.get('discord_webhook')


@pytest.fixture
def mock_db():
    """Create mock database."""
    return MockDB()


@pytest.fixture
def notification_service(mock_db):
    """Create notification service instance for testing."""
    return NotificationService(db=mock_db)


@pytest.fixture
def sample_detection_data():
    """Sample detection data for testing."""
    return {
        'id': 1,
        'species': 'Great Horned Owl',
        'zone': 'Forest Area A',
        'confidence': 0.95,
        'latitude': 46.2044,
        'longitude': 6.1432,
        'timestamp': datetime.now().isoformat(),
        'description': 'Clear owl call detected'
    }


@pytest.fixture
def sample_notification_preferences():
    """Sample notification preferences."""
    return MockNotificationPreference(
        user_id=1,
        email_notifications=True,
        push_notifications=True,
        email_address='test@example.com',
        min_priority='normal',
        species_filter='["owl", "bat"]',
        zone_filter='["Forest Area A"]'
    )


class TestNotificationService:
    """Test the NotificationService class."""
    
    def test_initialization(self, mock_db):
        """Test service initialization."""
        service = NotificationService(db=mock_db)
        assert service.db == mock_db
        assert service.user_preferences == {}
        assert isinstance(service.templates, dict)
        
    def test_get_notification_service(self, mock_db):
        """Test singleton pattern for notification service."""
        service1 = get_notification_service(mock_db)
        service2 = get_notification_service(mock_db)
        assert service1 is service2
        
    def test_load_templates(self, notification_service):
        """Test template loading."""
        templates = notification_service._load_templates()
        assert 'detection_email' in templates
        assert 'prediction_complete_email' in templates
        assert 'system_alert_email' in templates
        
    @pytest.mark.asyncio
    async def test_get_user_preferences_cached(self, notification_service, sample_notification_preferences):
        """Test user preferences caching."""
        # Mock database query
        notification_service.db.session.query.return_value.filter_by.return_value.first.return_value = sample_notification_preferences
        
        # First call should hit database
        prefs1 = await notification_service._get_user_preferences(1)
        assert prefs1['email_notifications'] == True
        
        # Second call should use cache
        prefs2 = await notification_service._get_user_preferences(1)
        assert prefs1 == prefs2
        
        # Database should only be called once
        assert notification_service.db.session.query.call_count == 1
        
    @pytest.mark.asyncio
    async def test_get_user_preferences_not_found(self, notification_service):
        """Test default preferences when user not found."""
        # Mock database query returning None
        notification_service.db.session.query.return_value.filter_by.return_value.first.return_value = None
        
        prefs = await notification_service._get_user_preferences(999)
        assert prefs['email_notifications'] == False
        assert prefs['push_notifications'] == False
        
    @pytest.mark.asyncio
    async def test_should_send_notification_filters(self, notification_service, sample_detection_data):
        """Test notification filtering logic."""
        # Test species filter
        prefs = {
            'species_filter': ['bat', 'eagle'],
            'zone_filter': [],
            'min_priority': 'normal'
        }
        assert not notification_service._should_send_notification(sample_detection_data, prefs, 'normal')
        
        # Test zone filter
        prefs = {
            'species_filter': [],
            'zone_filter': ['Urban Area'],
            'min_priority': 'normal'
        }
        assert not notification_service._should_send_notification(sample_detection_data, prefs, 'normal')
        
        # Test priority filter
        prefs = {
            'species_filter': [],
            'zone_filter': [],
            'min_priority': 'high'
        }
        assert not notification_service._should_send_notification(sample_detection_data, prefs, 'normal')
        
    @pytest.mark.asyncio
    async def test_should_send_notification_passes(self, notification_service, sample_detection_data):
        """Test notification that should pass filters."""
        prefs = {
            'species_filter': ['owl'],
            'zone_filter': ['Forest Area A'],
            'min_priority': 'normal'
        }
        assert notification_service._should_send_notification(sample_detection_data, prefs, 'normal')
        
    @pytest.mark.asyncio
    async def test_is_quiet_hours(self, notification_service):
        """Test quiet hours detection."""
        prefs = {
            'quiet_hours_start': '22:00',
            'quiet_hours_end': '06:00'
        }
        
        # Mock current time
        with patch('notification_service.datetime') as mock_datetime:
            # Test during quiet hours (midnight)
            mock_datetime.now.return_value.hour = 0
            mock_datetime.now.return_value.minute = 0
            assert notification_service._is_quiet_hours(prefs)
            
            # Test outside quiet hours (noon)
            mock_datetime.now.return_value.hour = 12
            mock_datetime.now.return_value.minute = 0
            assert not notification_service._is_quiet_hours(prefs)
            
    @pytest.mark.asyncio
    async def test_send_detection_notification(self, notification_service, sample_detection_data, sample_notification_preferences):
        """Test sending detection notification."""
        notification_service.db.session.query.return_value.filter_by.return_value.first.return_value = sample_notification_preferences
        
        with patch.object(notification_service, '_send_email') as mock_email, \
             patch.object(notification_service, '_send_push_notification') as mock_push, \
             patch.object(notification_service, '_send_websocket_notification') as mock_websocket:
            
            await notification_service.send_detection_notification(sample_detection_data, [1])
            
            # Verify all notification methods were called
            mock_email.assert_called_once()
            mock_push.assert_called_once()
            mock_websocket.assert_called_once()
            
    @pytest.mark.asyncio
    async def test_send_prediction_complete_notification(self, notification_service, sample_notification_preferences):
        """Test prediction complete notification."""
        notification_service.db.session.query.return_value.filter_by.return_value.first.return_value = sample_notification_preferences
        
        prediction_data = {
            'filename': 'test_audio.wav',
            'status': 'completed',
            'processing_time': '2.5 seconds',
            'results': {'species': 'Owl', 'confidence': 0.9}
        }
        
        with patch.object(notification_service, '_send_email') as mock_email, \
             patch.object(notification_service, '_send_websocket_notification') as mock_websocket:
            
            await notification_service.send_prediction_complete_notification(prediction_data, 1)
            
            mock_email.assert_called_once()
            mock_websocket.assert_called_once()
            
    @pytest.mark.asyncio
    async def test_send_system_alert(self, notification_service, sample_notification_preferences):
        """Test system alert notification."""
        notification_service.db.session.query.return_value.filter_by.return_value.first.return_value = sample_notification_preferences
        
        alert_data = {
            'type': 'System Error',
            'message': 'Database connection lost',
            'priority': 'high',
            'timestamp': datetime.now().isoformat()
        }
        
        with patch.object(notification_service, '_send_email') as mock_email, \
             patch.object(notification_service, '_send_websocket_notification') as mock_websocket, \
             patch.object(notification_service, '_send_slack_notification') as mock_slack:
            
            # Set up slack webhook
            sample_notification_preferences.slack_webhook = 'https://hooks.slack.com/test'
            
            await notification_service.send_system_alert(alert_data, [1])
            
            mock_email.assert_called_once()
            mock_websocket.assert_called_once()
            mock_slack.assert_called_once()


class TestEmailService:
    """Test the EmailService class."""
    
    @pytest.fixture
    def email_service(self):
        """Create email service for testing."""
        config = {
            'smtp_server': 'smtp.example.com',
            'smtp_port': 587,
            'smtp_username': 'test@example.com',
            'smtp_password': 'password',
            'from_email': 'noreply@nightscan.com'
        }
        return EmailService(config)
        
    def test_email_service_initialization(self, email_service):
        """Test email service initialization."""
        assert email_service.smtp_server == 'smtp.example.com'
        assert email_service.smtp_port == 587
        assert email_service.from_email == 'noreply@nightscan.com'
        
    @pytest.mark.asyncio
    async def test_send_email_success(self, email_service):
        """Test successful email sending."""
        with patch('smtplib.SMTP') as mock_smtp:
            mock_server = Mock()
            mock_smtp.return_value.__enter__.return_value = mock_server
            
            await email_service.send_email(
                to_email='test@example.com',
                subject='Test Subject',
                html_content='<p>Test</p>',
                text_content='Test'
            )
            
            mock_server.starttls.assert_called_once()
            mock_server.login.assert_called_once()
            mock_server.send_message.assert_called_once()
            
    @pytest.mark.asyncio
    async def test_send_email_failure(self, email_service):
        """Test email sending failure."""
        with patch('smtplib.SMTP') as mock_smtp:
            mock_smtp.side_effect = Exception("SMTP Error")
            
            result = await email_service.send_email(
                to_email='test@example.com',
                subject='Test Subject',
                html_content='<p>Test</p>',
                text_content='Test'
            )
            
            assert result == False


class TestWebSocketService:
    """Test the WebSocketService class."""
    
    @pytest.fixture
    def websocket_service(self):
        """Create websocket service for testing."""
        return WebSocketService()
        
    @pytest.mark.asyncio
    async def test_emit_notification(self, websocket_service):
        """Test WebSocket notification emission."""
        with patch.object(websocket_service, 'sio') as mock_sio:
            mock_sio.emit = AsyncMock()
            
            await websocket_service.emit_notification(
                event_type='new_detection',
                data={'species': 'Owl'},
                user_ids=[1, 2]
            )
            
            mock_sio.emit.assert_called_once_with(
                'notification',
                {
                    'event_type': 'new_detection',
                    'data': {'species': 'Owl'},
                    'timestamp': mock_sio.emit.call_args[0][1]['timestamp']
                },
                room='user_1'
            )


class TestPushNotificationService:
    """Test the PushNotificationService class."""
    
    @pytest.fixture
    def push_service(self):
        """Create push notification service for testing."""
        config = {
            'expo_access_token': 'test_token'
        }
        return PushNotificationService(config)
        
    @pytest.mark.asyncio
    async def test_send_push_notification(self, push_service):
        """Test push notification sending."""
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_response = Mock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value={'data': [{'status': 'ok'}]})
            mock_post.return_value.__aenter__.return_value = mock_response
            
            result = await push_service.send_push_notification(
                push_tokens=['ExponentPushToken[test]'],
                title='Test Notification',
                body='Test Body',
                data={'type': 'test'}
            )
            
            assert result == True
            mock_post.assert_called_once()
            
    @pytest.mark.asyncio
    async def test_send_push_notification_failure(self, push_service):
        """Test push notification failure."""
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_response = Mock()
            mock_response.status = 400
            mock_post.return_value.__aenter__.return_value = mock_response
            
            result = await push_service.send_push_notification(
                push_tokens=['invalid_token'],
                title='Test Notification',
                body='Test Body'
            )
            
            assert result == False


@pytest.mark.integration
class TestNotificationServiceIntegration:
    """Integration tests for notification service."""
    
    def test_notification_flow_end_to_end(self):
        """Test complete notification flow."""
        # This would test the full flow from detection to notification
        # in a more realistic environment
        pass
        
    def test_template_rendering(self):
        """Test email template rendering with real data."""
        pass
        
    def test_notification_persistence(self):
        """Test notification history persistence."""
        pass


if __name__ == '__main__':
    pytest.main([__file__])