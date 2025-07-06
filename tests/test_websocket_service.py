import pytest
import asyncio
import json
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from datetime import datetime
import sys
from pathlib import Path

# Add the parent directory to Python path
sys.path.append(str(Path(__file__).resolve().parents[1]))

# Mock socketio before importing websocket_service
mock_socketio = MagicMock()
sys.modules['socketio'] = mock_socketio

from websocket_service import (
    WebSocketManager,
    FlaskWebSocketIntegration,
    get_websocket_manager
)


class TestWebSocketManager:
    """Test the WebSocketManager class."""
    
    @pytest.fixture
    def websocket_manager(self):
        """Create WebSocket manager for testing."""
        return WebSocketManager()
        
    def test_initialization(self, websocket_manager):
        """Test WebSocket manager initialization."""
        assert websocket_manager.active_connections == {}
        assert websocket_manager.user_rooms == {}
        assert websocket_manager.room_users == {}
        assert websocket_manager.authenticated_users == set()
        
    @pytest.mark.asyncio
    async def test_authenticate_user(self, websocket_manager):
        """Test user authentication."""
        sid = 'test_session_id'
        user_id = 123
        
        # Mock socketio join_room
        with patch.object(websocket_manager.sio, 'enter_room') as mock_join:
            await websocket_manager.authenticate_user(sid, user_id)
            
            assert user_id in websocket_manager.authenticated_users
            assert websocket_manager.user_rooms[user_id] == f'user_{user_id}'
            assert f'user_{user_id}' in websocket_manager.room_users
            assert sid in websocket_manager.room_users[f'user_{user_id}']
            
            mock_join.assert_called_once_with(sid, f'user_{user_id}')
            
    @pytest.mark.asyncio
    async def test_disconnect_user(self, websocket_manager):
        """Test user disconnection."""
        sid = 'test_session_id'
        user_id = 123
        
        # First authenticate
        await websocket_manager.authenticate_user(sid, user_id)
        
        # Then disconnect
        with patch.object(websocket_manager.sio, 'leave_room') as mock_leave:
            await websocket_manager.disconnect_user(sid, user_id)
            
            assert user_id not in websocket_manager.authenticated_users
            assert user_id not in websocket_manager.user_rooms
            
            mock_leave.assert_called_once_with(sid, f'user_{user_id}')
            
    @pytest.mark.asyncio
    async def test_subscribe_to_events(self, websocket_manager):
        """Test event subscription."""
        sid = 'test_session_id'
        event_types = ['new_detection', 'prediction_complete']
        
        await websocket_manager.subscribe_to_events(sid, event_types)
        
        assert websocket_manager.active_connections[sid]['subscriptions'] == set(event_types)
        
    @pytest.mark.asyncio
    async def test_unsubscribe_from_events(self, websocket_manager):
        """Test event unsubscription."""
        sid = 'test_session_id'
        event_types = ['new_detection', 'prediction_complete']
        
        # First subscribe
        await websocket_manager.subscribe_to_events(sid, event_types)
        
        # Then unsubscribe from one event
        await websocket_manager.unsubscribe_from_events(sid, ['new_detection'])
        
        assert websocket_manager.active_connections[sid]['subscriptions'] == {'prediction_complete'}
        
    @pytest.mark.asyncio
    async def test_broadcast_notification(self, websocket_manager):
        """Test notification broadcasting."""
        # Set up authenticated user
        sid = 'test_session_id'
        user_id = 123
        await websocket_manager.authenticate_user(sid, user_id)
        await websocket_manager.subscribe_to_events(sid, ['new_detection'])
        
        notification_data = {
            'species': 'Great Horned Owl',
            'zone': 'Forest Area A',
            'confidence': 0.95
        }
        
        with patch.object(websocket_manager.sio, 'emit') as mock_emit:
            await websocket_manager.broadcast_notification(
                'new_detection',
                notification_data,
                user_ids=[user_id]
            )
            
            mock_emit.assert_called_once()
            call_args = mock_emit.call_args
            assert call_args[0][0] == 'notification'
            assert call_args[1]['room'] == f'user_{user_id}'
            
    @pytest.mark.asyncio
    async def test_notify_new_detection(self, websocket_manager):
        """Test new detection notification."""
        detection_data = {
            'id': 1,
            'species': 'Barn Owl',
            'zone': 'Field A',
            'confidence': 0.88,
            'timestamp': datetime.now().isoformat()
        }
        user_id = 123
        
        with patch.object(websocket_manager, 'broadcast_notification') as mock_broadcast:
            await websocket_manager.notify_new_detection(detection_data, user_id)
            
            mock_broadcast.assert_called_once_with(
                'new_detection',
                detection_data,
                user_ids=[user_id]
            )
            
    @pytest.mark.asyncio
    async def test_notify_prediction_complete(self, websocket_manager):
        """Test prediction complete notification."""
        prediction_data = {
            'filename': 'audio_sample.wav',
            'status': 'completed',
            'results': {'species': 'Eagle', 'confidence': 0.92}
        }
        user_id = 123
        
        with patch.object(websocket_manager, 'broadcast_notification') as mock_broadcast:
            await websocket_manager.notify_prediction_complete(prediction_data, user_id)
            
            mock_broadcast.assert_called_once_with(
                'prediction_complete',
                prediction_data,
                user_ids=[user_id]
            )
            
    @pytest.mark.asyncio
    async def test_notify_system_status(self, websocket_manager):
        """Test system status notification."""
        status_data = {
            'type': 'maintenance',
            'message': 'System will be down for maintenance',
            'scheduled_time': datetime.now().isoformat()
        }
        
        with patch.object(websocket_manager, 'broadcast_notification') as mock_broadcast:
            await websocket_manager.notify_system_status(status_data)
            
            mock_broadcast.assert_called_once_with(
                'system_status',
                status_data,
                user_ids=None  # Broadcast to all users
            )
            
    def test_get_connection_stats(self, websocket_manager):
        """Test connection statistics."""
        # Add some mock connections
        websocket_manager.active_connections = {
            'sid1': {'user_id': 1, 'subscriptions': {'new_detection'}},
            'sid2': {'user_id': 2, 'subscriptions': {'prediction_complete'}},
            'sid3': {'user_id': None, 'subscriptions': set()}
        }
        websocket_manager.authenticated_users = {1, 2}
        
        stats = websocket_manager.get_connection_stats()
        
        assert stats['total_connections'] == 3
        assert stats['authenticated_connections'] == 2
        assert stats['active_subscriptions']['new_detection'] == 1
        assert stats['active_subscriptions']['prediction_complete'] == 1
        
    @pytest.mark.asyncio
    async def test_handle_connect(self, websocket_manager):
        """Test connection handling."""
        sid = 'test_session_id'
        environ = {'HTTP_USER_AGENT': 'test-client'}
        
        with patch.object(websocket_manager.sio, 'emit') as mock_emit:
            await websocket_manager.handle_connect(sid, environ)
            
            assert sid in websocket_manager.active_connections
            assert websocket_manager.active_connections[sid]['user_id'] is None
            assert websocket_manager.active_connections[sid]['subscriptions'] == set()
            
            mock_emit.assert_called_once_with('connected', {'status': 'connected'}, room=sid)
            
    @pytest.mark.asyncio
    async def test_handle_disconnect(self, websocket_manager):
        """Test disconnection handling."""
        sid = 'test_session_id'
        user_id = 123
        
        # Set up connection
        websocket_manager.active_connections[sid] = {
            'user_id': user_id,
            'subscriptions': set(['new_detection'])
        }
        websocket_manager.authenticated_users.add(user_id)
        websocket_manager.user_rooms[user_id] = f'user_{user_id}'
        websocket_manager.room_users[f'user_{user_id}'] = {sid}
        
        await websocket_manager.handle_disconnect(sid)
        
        assert sid not in websocket_manager.active_connections
        assert user_id not in websocket_manager.authenticated_users
        
    @pytest.mark.asyncio
    async def test_handle_authenticate_success(self, websocket_manager):
        """Test successful authentication handling."""
        sid = 'test_session_id'
        auth_data = {'user_id': 123, 'token': 'valid_token'}
        
        # Mock connection exists
        websocket_manager.active_connections[sid] = {
            'user_id': None,
            'subscriptions': set()
        }
        
        with patch.object(websocket_manager, '_validate_auth_token', return_value=True), \
             patch.object(websocket_manager.sio, 'emit') as mock_emit:
            
            await websocket_manager.handle_authenticate(sid, auth_data)
            
            mock_emit.assert_called_with(
                'authenticated',
                {'status': 'authenticated', 'user_id': 123},
                room=sid
            )
            
    @pytest.mark.asyncio
    async def test_handle_authenticate_failure(self, websocket_manager):
        """Test failed authentication handling."""
        sid = 'test_session_id'
        auth_data = {'user_id': 123, 'token': 'invalid_token'}
        
        # Mock connection exists
        websocket_manager.active_connections[sid] = {
            'user_id': None,
            'subscriptions': set()
        }
        
        with patch.object(websocket_manager, '_validate_auth_token', return_value=False), \
             patch.object(websocket_manager.sio, 'emit') as mock_emit:
            
            await websocket_manager.handle_authenticate(sid, auth_data)
            
            mock_emit.assert_called_with(
                'authentication_error',
                {'status': 'authentication_failed', 'message': 'Invalid credentials'},
                room=sid
            )
            
    @pytest.mark.asyncio
    async def test_handle_subscribe(self, websocket_manager):
        """Test subscription handling."""
        sid = 'test_session_id'
        sub_data = {'event_types': ['new_detection', 'system_status']}
        
        # Mock authenticated connection
        websocket_manager.active_connections[sid] = {
            'user_id': 123,
            'subscriptions': set()
        }
        
        with patch.object(websocket_manager.sio, 'emit') as mock_emit:
            await websocket_manager.handle_subscribe(sid, sub_data)
            
            assert websocket_manager.active_connections[sid]['subscriptions'] == {
                'new_detection', 'system_status'
            }
            
            mock_emit.assert_called_with(
                'subscribed',
                {'event_types': ['new_detection', 'system_status']},
                room=sid
            )
            
    @pytest.mark.asyncio
    async def test_handle_ping(self, websocket_manager):
        """Test ping handling."""
        sid = 'test_session_id'
        ping_data = {'timestamp': '2023-01-01T00:00:00Z'}
        
        with patch.object(websocket_manager.sio, 'emit') as mock_emit:
            await websocket_manager.handle_ping(sid, ping_data)
            
            mock_emit.assert_called_once()
            assert mock_emit.call_args[0][0] == 'pong'
            assert 'timestamp' in mock_emit.call_args[0][1]
            
    def test_validate_auth_token(self, websocket_manager):
        """Test token validation."""
        # Mock JWT validation
        with patch('jwt.decode') as mock_decode:
            mock_decode.return_value = {'user_id': 123, 'exp': 9999999999}
            
            result = websocket_manager._validate_auth_token('valid_token', 123)
            assert result == True
            
        # Test invalid token
        with patch('jwt.decode', side_effect=Exception("Invalid token")):
            result = websocket_manager._validate_auth_token('invalid_token', 123)
            assert result == False


class TestFlaskWebSocketIntegration:
    """Test Flask-WebSocket integration."""
    
    @pytest.fixture
    def mock_app(self):
        """Create mock Flask app."""
        app = Mock()
        app.config = {'SECRET_KEY': 'test_secret'}
        return app
        
    @pytest.fixture
    def flask_integration(self, mock_app):
        """Create Flask WebSocket integration."""
        return FlaskWebSocketIntegration(mock_app)
        
    def test_initialization(self, flask_integration, mock_app):
        """Test Flask integration initialization."""
        assert flask_integration.app == mock_app
        assert hasattr(flask_integration, 'websocket_manager')
        assert hasattr(flask_integration, 'sio')
        
    def test_setup_event_handlers(self, flask_integration):
        """Test event handler setup."""
        # This would test that all event handlers are properly registered
        # with the socketio instance
        handlers = flask_integration.sio.handlers['/']
        
        expected_events = ['connect', 'disconnect', 'authenticate', 'subscribe', 'unsubscribe', 'ping']
        for event in expected_events:
            assert event in handlers


class TestWebSocketManagerSingleton:
    """Test WebSocket manager singleton pattern."""
    
    def test_get_websocket_manager_singleton(self):
        """Test that get_websocket_manager returns the same instance."""
        manager1 = get_websocket_manager()
        manager2 = get_websocket_manager()
        
        assert manager1 is manager2
        assert isinstance(manager1, WebSocketManager)


@pytest.mark.integration
class TestWebSocketIntegration:
    """Integration tests for WebSocket functionality."""
    
    @pytest.mark.asyncio
    async def test_full_connection_flow(self):
        """Test complete connection, authentication, and notification flow."""
        websocket_manager = WebSocketManager()
        
        # Simulate connection
        sid = 'test_session'
        await websocket_manager.handle_connect(sid, {})
        
        # Simulate authentication
        auth_data = {'user_id': 123, 'token': 'test_token'}
        with patch.object(websocket_manager, '_validate_auth_token', return_value=True):
            await websocket_manager.handle_authenticate(sid, auth_data)
        
        # Simulate subscription
        sub_data = {'event_types': ['new_detection']}
        await websocket_manager.handle_subscribe(sid, sub_data)
        
        # Simulate notification
        detection_data = {'species': 'Owl', 'confidence': 0.9}
        with patch.object(websocket_manager.sio, 'emit') as mock_emit:
            await websocket_manager.notify_new_detection(detection_data, 123)
            mock_emit.assert_called()
        
        # Simulate disconnection
        await websocket_manager.handle_disconnect(sid)
        
        assert sid not in websocket_manager.active_connections
        
    @pytest.mark.asyncio
    async def test_multiple_user_notifications(self):
        """Test notifications to multiple users."""
        websocket_manager = WebSocketManager()
        
        # Set up multiple users
        users = [
            {'sid': 'user1_session', 'user_id': 1},
            {'sid': 'user2_session', 'user_id': 2},
            {'sid': 'user3_session', 'user_id': 3}
        ]
        
        for user in users:
            await websocket_manager.handle_connect(user['sid'], {})
            await websocket_manager.authenticate_user(user['sid'], user['user_id'])
            await websocket_manager.subscribe_to_events(user['sid'], ['new_detection'])
        
        # Broadcast to specific users
        detection_data = {'species': 'Eagle', 'confidence': 0.95}
        with patch.object(websocket_manager.sio, 'emit') as mock_emit:
            await websocket_manager.broadcast_notification(
                'new_detection',
                detection_data,
                user_ids=[1, 3]  # Only users 1 and 3
            )
            
            # Should be called twice (for users 1 and 3)
            assert mock_emit.call_count == 2
            
    @pytest.mark.asyncio
    async def test_system_broadcast(self):
        """Test system-wide broadcasting."""
        websocket_manager = WebSocketManager()
        
        # Set up multiple users
        users = [
            {'sid': 'user1_session', 'user_id': 1},
            {'sid': 'user2_session', 'user_id': 2}
        ]
        
        for user in users:
            await websocket_manager.handle_connect(user['sid'], {})
            await websocket_manager.authenticate_user(user['sid'], user['user_id'])
            await websocket_manager.subscribe_to_events(user['sid'], ['system_status'])
        
        # System broadcast (no specific user_ids)
        status_data = {'message': 'System maintenance in 10 minutes'}
        with patch.object(websocket_manager.sio, 'emit') as mock_emit:
            await websocket_manager.notify_system_status(status_data)
            
            # Should broadcast to all authenticated users
            assert mock_emit.call_count == 2


if __name__ == '__main__':
    pytest.main([__file__])