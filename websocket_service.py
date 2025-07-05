"""WebSocket service for real-time notifications in NightScan."""

import json
import logging
import asyncio
from typing import Dict, Set, Optional, Any, List
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum

try:
    import socketio
    from aioredis import Redis
    import aioredis
    SOCKETIO_AVAILABLE = True
except ImportError:
    SOCKETIO_AVAILABLE = False
    # Mock classes for environments without socketio
    class AsyncServer:
        def __init__(self, *args, **kwargs): pass
        async def emit(self, *args, **kwargs): pass
        def on(self, event): 
            def decorator(func): return func
            return decorator
    
    socketio = type('MockModule', (), {'AsyncServer': AsyncServer})()

from config import get_config

logger = logging.getLogger(__name__)


class EventType(Enum):
    """Types of events that can be sent via WebSocket."""
    NEW_DETECTION = "new_detection"
    PREDICTION_COMPLETE = "prediction_complete"
    SYSTEM_STATUS = "system_status"
    USER_ACTIVITY = "user_activity"
    ERROR_NOTIFICATION = "error_notification"


@dataclass
class NotificationEvent:
    """Structure for WebSocket notification events."""
    event_type: EventType
    data: Dict[str, Any]
    timestamp: datetime
    user_id: Optional[int] = None
    session_id: Optional[str] = None
    priority: str = "normal"  # low, normal, high, critical
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = asdict(self)
        result['event_type'] = self.event_type.value
        result['timestamp'] = self.timestamp.isoformat()
        return result


class WebSocketManager:
    """Manages WebSocket connections and real-time notifications."""
    
    def __init__(self, redis_url: Optional[str] = None):
        """Initialize WebSocket manager."""
        self.config = get_config()
        self.sio = None
        self.redis = None
        self.connected_users: Dict[int, Set[str]] = {}  # user_id -> set of session_ids
        self.session_users: Dict[str, int] = {}  # session_id -> user_id
        
        if SOCKETIO_AVAILABLE:
            self._setup_socketio()
            if redis_url or self.config.redis.enabled:
                asyncio.create_task(self._setup_redis(redis_url))
        else:
            logger.warning("SocketIO not available, WebSocket functionality disabled")
    
    def _setup_socketio(self):
        """Setup SocketIO server."""
        self.sio = socketio.AsyncServer(
            async_mode='asgi',
            cors_allowed_origins="*",  # Configure based on config.api.cors_origins
            logger=True,
            engineio_logger=True
        )
        
        # Register event handlers
        self.sio.on('connect', self._handle_connect)
        self.sio.on('disconnect', self._handle_disconnect)
        self.sio.on('authenticate', self._handle_authenticate)
        self.sio.on('subscribe', self._handle_subscribe)
        self.sio.on('unsubscribe', self._handle_unsubscribe)
        self.sio.on('ping', self._handle_ping)
    
    async def _setup_redis(self, redis_url: Optional[str] = None):
        """Setup Redis for pub/sub between multiple server instances."""
        try:
            url = redis_url or self.config.redis.url
            self.redis = await aioredis.from_url(url)
            
            # Subscribe to notification channel
            pubsub = self.redis.pubsub()
            await pubsub.subscribe('nightscan_notifications')
            
            # Start listening for Redis messages
            asyncio.create_task(self._redis_listener(pubsub))
            logger.info("Redis pub/sub initialized for WebSocket notifications")
            
        except Exception as e:
            logger.error(f"Failed to setup Redis for WebSocket: {e}")
            self.redis = None
    
    async def _redis_listener(self, pubsub):
        """Listen for Redis pub/sub messages."""
        try:
            async for message in pubsub.listen():
                if message['type'] == 'message':
                    try:
                        event_data = json.loads(message['data'])
                        event = NotificationEvent(**event_data)
                        await self._broadcast_event(event)
                    except Exception as e:
                        logger.error(f"Failed to process Redis message: {e}")
        except Exception as e:
            logger.error(f"Redis listener error: {e}")
    
    async def _handle_connect(self, sid, environ, auth):
        """Handle new WebSocket connection."""
        logger.info(f"WebSocket connection established: {sid}")
        await self.sio.emit('connected', {'message': 'Connected to NightScan'}, room=sid)
    
    async def _handle_disconnect(self, sid):
        """Handle WebSocket disconnection."""
        logger.info(f"WebSocket disconnected: {sid}")
        
        # Clean up user mappings
        if sid in self.session_users:
            user_id = self.session_users[sid]
            del self.session_users[sid]
            
            if user_id in self.connected_users:
                self.connected_users[user_id].discard(sid)
                if not self.connected_users[user_id]:
                    del self.connected_users[user_id]
    
    async def _handle_authenticate(self, sid, data):
        """Handle user authentication for WebSocket."""
        try:
            user_id = data.get('user_id')
            token = data.get('token')  # Would validate JWT token in real implementation
            
            if user_id and token:
                # In real implementation, validate token
                self.session_users[sid] = user_id
                
                if user_id not in self.connected_users:
                    self.connected_users[user_id] = set()
                self.connected_users[user_id].add(sid)
                
                await self.sio.emit('authenticated', {'status': 'success'}, room=sid)
                logger.info(f"User {user_id} authenticated on session {sid}")
            else:
                await self.sio.emit('authentication_error', {'error': 'Invalid credentials'}, room=sid)
                
        except Exception as e:
            logger.error(f"Authentication error: {e}")
            await self.sio.emit('authentication_error', {'error': 'Authentication failed'}, room=sid)
    
    async def _handle_subscribe(self, sid, data):
        """Handle subscription to specific event types."""
        try:
            event_types = data.get('event_types', [])
            
            for event_type in event_types:
                await self.sio.enter_room(sid, f"events_{event_type}")
            
            await self.sio.emit('subscribed', {'event_types': event_types}, room=sid)
            logger.info(f"Session {sid} subscribed to events: {event_types}")
            
        except Exception as e:
            logger.error(f"Subscription error: {e}")
            await self.sio.emit('subscription_error', {'error': 'Failed to subscribe'}, room=sid)
    
    async def _handle_unsubscribe(self, sid, data):
        """Handle unsubscription from event types."""
        try:
            event_types = data.get('event_types', [])
            
            for event_type in event_types:
                await self.sio.leave_room(sid, f"events_{event_type}")
            
            await self.sio.emit('unsubscribed', {'event_types': event_types}, room=sid)
            
        except Exception as e:
            logger.error(f"Unsubscription error: {e}")
    
    async def _handle_ping(self, sid, data):
        """Handle ping/pong for connection keepalive."""
        await self.sio.emit('pong', {'timestamp': datetime.utcnow().isoformat()}, room=sid)
    
    async def _broadcast_event(self, event: NotificationEvent):
        """Broadcast event to appropriate clients."""
        if not self.sio:
            return
        
        try:
            event_data = event.to_dict()
            
            # Broadcast to specific user if specified
            if event.user_id and event.user_id in self.connected_users:
                for sid in self.connected_users[event.user_id]:
                    await self.sio.emit('notification', event_data, room=sid)
            
            # Broadcast to subscribers of this event type
            room_name = f"events_{event.event_type.value}"
            await self.sio.emit('notification', event_data, room=room_name)
            
            logger.debug(f"Broadcasted event {event.event_type.value} to room {room_name}")
            
        except Exception as e:
            logger.error(f"Failed to broadcast event: {e}")
    
    async def notify_new_detection(self, detection_data: Dict[str, Any], user_id: Optional[int] = None):
        """Send notification for new wildlife detection."""
        event = NotificationEvent(
            event_type=EventType.NEW_DETECTION,
            data=detection_data,
            timestamp=datetime.utcnow(),
            user_id=user_id,
            priority="high"
        )
        
        await self._send_event(event)
    
    async def notify_prediction_complete(self, prediction_data: Dict[str, Any], user_id: Optional[int] = None):
        """Send notification when prediction is complete."""
        event = NotificationEvent(
            event_type=EventType.PREDICTION_COMPLETE,
            data=prediction_data,
            timestamp=datetime.utcnow(),
            user_id=user_id,
            priority="normal"
        )
        
        await self._send_event(event)
    
    async def notify_system_status(self, status_data: Dict[str, Any]):
        """Send system status notification."""
        event = NotificationEvent(
            event_type=EventType.SYSTEM_STATUS,
            data=status_data,
            timestamp=datetime.utcnow(),
            priority="normal"
        )
        
        await self._send_event(event)
    
    async def notify_error(self, error_data: Dict[str, Any], user_id: Optional[int] = None):
        """Send error notification."""
        event = NotificationEvent(
            event_type=EventType.ERROR_NOTIFICATION,
            data=error_data,
            timestamp=datetime.utcnow(),
            user_id=user_id,
            priority="critical"
        )
        
        await self._send_event(event)
    
    async def _send_event(self, event: NotificationEvent):
        """Send event via WebSocket and Redis pub/sub."""
        # Send via WebSocket if available
        if self.sio:
            await self._broadcast_event(event)
        
        # Publish to Redis for other server instances
        if self.redis:
            try:
                event_json = json.dumps(event.to_dict(), default=str)
                await self.redis.publish('nightscan_notifications', event_json)
            except Exception as e:
                logger.error(f"Failed to publish to Redis: {e}")
    
    def get_connected_users(self) -> List[int]:
        """Get list of currently connected user IDs."""
        return list(self.connected_users.keys())
    
    def get_connection_count(self) -> int:
        """Get total number of active connections."""
        return sum(len(sessions) for sessions in self.connected_users.values())
    
    async def send_to_user(self, user_id: int, event_type: str, data: Dict[str, Any]):
        """Send custom event to specific user."""
        if user_id in self.connected_users:
            for sid in self.connected_users[user_id]:
                await self.sio.emit(event_type, data, room=sid)


# Global WebSocket manager instance
_websocket_manager: Optional[WebSocketManager] = None


def get_websocket_manager() -> WebSocketManager:
    """Get or create global WebSocket manager instance."""
    global _websocket_manager
    
    if _websocket_manager is None:
        _websocket_manager = WebSocketManager()
    
    return _websocket_manager


# ASGI application for WebSocket
def create_websocket_app():
    """Create ASGI application with WebSocket support."""
    if not SOCKETIO_AVAILABLE:
        logger.error("Cannot create WebSocket app: SocketIO not available")
        return None
    
    manager = get_websocket_manager()
    if manager.sio:
        return socketio.ASGIApp(manager.sio, other_asgi_app=None)
    
    return None


# Integration helpers for Flask apps
class FlaskWebSocketIntegration:
    """Helper class to integrate WebSocket notifications with Flask."""
    
    def __init__(self, app=None):
        self.manager = get_websocket_manager()
        if app:
            self.init_app(app)
    
    def init_app(self, app):
        """Initialize with Flask app."""
        app.websocket_manager = self.manager
        
        # Add notification helpers to app context
        @app.context_processor
        def inject_websocket():
            return {
                'websocket_enabled': SOCKETIO_AVAILABLE and self.manager.sio is not None
            }
    
    async def emit_detection(self, detection_data: Dict[str, Any], user_id: Optional[int] = None):
        """Emit new detection event."""
        await self.manager.notify_new_detection(detection_data, user_id)
    
    async def emit_prediction(self, prediction_data: Dict[str, Any], user_id: Optional[int] = None):
        """Emit prediction complete event."""
        await self.manager.notify_prediction_complete(prediction_data, user_id)


# Example usage for testing
async def test_websocket_service():
    """Test WebSocket service functionality."""
    manager = get_websocket_manager()
    
    # Test notification creation
    await manager.notify_new_detection({
        'id': 123,
        'species': 'Great Horned Owl',
        'location': 'Sensor Zone A',
        'confidence': 0.95,
        'timestamp': datetime.utcnow().isoformat()
    })
    
    await manager.notify_system_status({
        'status': 'healthy',
        'active_sensors': 5,
        'predictions_today': 42
    })
    
    print(f"Connected users: {manager.get_connected_users()}")
    print(f"Total connections: {manager.get_connection_count()}")


if __name__ == "__main__":
    if SOCKETIO_AVAILABLE:
        asyncio.run(test_websocket_service())
    else:
        print("SocketIO not available. Install with: pip install python-socketio aioredis")