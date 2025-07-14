"""Push notification service for mobile devices."""

import json
import logging
import asyncio
import os
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime

try:
    import aioapns
    from aioapns import APNs, NotificationRequest, PushType
    APNS_AVAILABLE = True
except ImportError:
    APNS_AVAILABLE = False

try:
    from pyfcm import FCMNotification
    FCM_AVAILABLE = True
except ImportError:
    FCM_AVAILABLE = False

from config import get_config

logger = logging.getLogger(__name__)


class NotificationType(Enum):
    """Types of push notifications."""
    NEW_DETECTION = "new_detection"
    PREDICTION_COMPLETE = "prediction_complete"
    SYSTEM_ALERT = "system_alert"
    SYNC_COMPLETE = "sync_complete"
    LOW_BATTERY = "low_battery"


@dataclass
class PushNotification:
    """Structure for push notifications."""
    title: str
    body: str
    notification_type: NotificationType
    data: Dict[str, Any]
    user_id: int
    priority: str = "normal"  # low, normal, high
    badge_count: Optional[int] = None
    sound: str = "default"
    
    def to_apns_payload(self) -> Dict[str, Any]:
        """Convert to APNs payload format."""
        payload = {
            "aps": {
                "alert": {
                    "title": self.title,
                    "body": self.body
                },
                "sound": self.sound,
                "badge": self.badge_count,
                "category": self.notification_type.value
            },
            "data": self.data
        }
        
        if self.priority == "high":
            payload["aps"]["priority"] = 10
        
        return payload
    
    def to_fcm_payload(self) -> Dict[str, Any]:
        """Convert to FCM payload format."""
        return {
            "notification": {
                "title": self.title,
                "body": self.body,
                "sound": self.sound,
                "badge": str(self.badge_count) if self.badge_count else None
            },
            "data": {
                "type": self.notification_type.value,
                **{k: str(v) for k, v in self.data.items()}
            },
            "priority": "high" if self.priority == "high" else "normal"
        }


class DeviceToken:
    """Represents a device token for push notifications."""
    
    def __init__(self, user_id: int, token: str, platform: str, active: bool = True):
        self.user_id = user_id
        self.token = token
        self.platform = platform.lower()  # 'ios' or 'android'
        self.active = active
        self.created_at = datetime.utcnow()
        self.last_used = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'user_id': self.user_id,
            'token': self.token,
            'platform': self.platform,
            'active': self.active,
            'created_at': self.created_at.isoformat(),
            'last_used': self.last_used.isoformat()
        }


class APNSService:
    """Apple Push Notification Service integration."""
    
    def __init__(self, cert_file: str, key_file: str, bundle_id: str, use_sandbox: bool = False):
        """Initialize APNs service."""
        self.cert_file = cert_file
        self.key_file = key_file
        self.bundle_id = bundle_id
        self.use_sandbox = use_sandbox
        self.client = None
        
        if APNS_AVAILABLE:
            self._setup_client()
        else:
            logger.warning("APNs not available. Install with: pip install aioapns")
    
    def _setup_client(self):
        """Setup APNs client."""
        try:
            self.client = APNs(
                cert_file=self.cert_file,
                key_file=self.key_file,
                bundle_id=self.bundle_id,
                use_sandbox=self.use_sandbox
            )
            logger.info("APNs client initialized")
        except Exception as e:
            logger.error(f"Failed to setup APNs client: {e}")
            self.client = None
    
    async def send_notification(self, device_token: str, notification: PushNotification) -> bool:
        """Send notification via APNs."""
        if not self.client:
            return False
        
        try:
            payload = notification.to_apns_payload()
            request = NotificationRequest(
                device_token=device_token,
                message=payload,
                push_type=PushType.Alert
            )
            
            await self.client.send_notification(request)
            logger.debug(f"APNs notification sent to device {str(device_token)[:8]}***")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send APNs notification: {e}")
            return False


class FCMService:
    """Firebase Cloud Messaging service integration."""
    
    def __init__(self, api_key: str):
        """Initialize FCM service."""
        self.api_key = api_key
        self.client = None
        
        if FCM_AVAILABLE:
            self.client = FCMNotification(api_key=api_key)
            logger.info("FCM client initialized")
        else:
            logger.warning("FCM not available. Install with: pip install pyfcm")
    
    async def send_notification(self, device_token: str, notification: PushNotification) -> bool:
        """Send notification via FCM."""
        if not self.client:
            return False
        
        try:
            payload = notification.to_fcm_payload()
            
            result = self.client.notify_single_device(
                registration_id=device_token,
                message_title=notification.title,
                message_body=notification.body,
                data_message=payload["data"],
                sound=notification.sound,
                badge=notification.badge_count
            )
            
            if result and result.get('success'):
                logger.debug(f"FCM notification sent to device {str(device_token)[:8]}***")
                return True
            else:
                logger.error(f"FCM notification failed: {result}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to send FCM notification: {e}")
            return False


class PushNotificationService:
    """Main push notification service."""
    
    def __init__(self):
        """Initialize push notification service."""
        self.config = get_config()
        self.device_tokens: Dict[int, List[DeviceToken]] = {}  # user_id -> tokens
        
        # Initialize services based on configuration
        self.apns_service = None
        self.fcm_service = None
        
        self._setup_services()
    
    def _setup_services(self):
        """Setup push notification services based on config."""
        # APNs setup (would be configured via environment variables)
        apns_cert = os.environ.get("APNS_CERT_FILE")
        apns_key = os.environ.get("APNS_KEY_FILE")
        apns_bundle_id = os.environ.get("APNS_BUNDLE_ID", "com.nightscan.app")
        
        if apns_cert and apns_key and APNS_AVAILABLE:
            self.apns_service = APNSService(
                cert_file=apns_cert,
                key_file=apns_key,
                bundle_id=apns_bundle_id,
                use_sandbox=self.config.environment != "production"
            )
        
        # FCM setup
        fcm_api_key = os.environ.get("FCM_API_KEY")
        if fcm_api_key and FCM_AVAILABLE:
            self.fcm_service = FCMService(fcm_api_key)
    
    def register_device(self, user_id: int, token: str, platform: str) -> bool:
        """Register a device token for a user."""
        try:
            device_token = DeviceToken(user_id, token, platform)
            
            if user_id not in self.device_tokens:
                self.device_tokens[user_id] = []
            
            # Remove existing token if it exists (update)
            self.device_tokens[user_id] = [
                dt for dt in self.device_tokens[user_id] 
                if dt.token != token
            ]
            
            # Add new token
            self.device_tokens[user_id].append(device_token)
            
            logger.info(f"Device token registered for user {user_id} on {platform}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register device token: {e}")
            return False
    
    def unregister_device(self, user_id: int, token: str) -> bool:
        """Unregister a device token."""
        try:
            if user_id in self.device_tokens:
                self.device_tokens[user_id] = [
                    dt for dt in self.device_tokens[user_id]
                    if dt.token != token
                ]
                
                if not self.device_tokens[user_id]:
                    del self.device_tokens[user_id]
                
                logger.info(f"Device token unregistered for user {user_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to unregister device token: {e}")
            return False
    
    async def send_to_user(self, user_id: int, notification: PushNotification) -> Dict[str, Any]:
        """Send push notification to all devices of a user."""
        results = {
            'sent': 0,
            'failed': 0,
            'platforms': {'ios': 0, 'android': 0},
            'errors': []
        }
        
        if user_id not in self.device_tokens:
            logger.warning(f"No device tokens found for user {user_id}")
            return results
        
        tasks = []
        for device_token in self.device_tokens[user_id]:
            if not device_token.active:
                continue
            
            if device_token.platform == 'ios' and self.apns_service:
                task = self._send_ios_notification(device_token, notification)
                tasks.append((task, device_token, 'ios'))
            elif device_token.platform == 'android' and self.fcm_service:
                task = self._send_android_notification(device_token, notification)
                tasks.append((task, device_token, 'android'))
        
        # Execute all sends concurrently
        if tasks:
            task_results = await asyncio.gather(
                *[task for task, _, _ in tasks],
                return_exceptions=True
            )
            
            for i, result in enumerate(task_results):
                _, device_token, platform = tasks[i]
                
                if isinstance(result, Exception):
                    results['failed'] += 1
                    results['errors'].append(str(result))
                    logger.error(f"Failed to send to device {device_token.token[:8]}***: {result}")
                elif result:
                    results['sent'] += 1
                    results['platforms'][platform] += 1
                    device_token.last_used = datetime.utcnow()
                else:
                    results['failed'] += 1
        
        logger.info(f"Push notification sent to user {user_id}: {results}")
        return results
    
    async def _send_ios_notification(self, device_token: DeviceToken, notification: PushNotification) -> bool:
        """Send notification to iOS device."""
        if self.apns_service:
            return await self.apns_service.send_notification(device_token.token, notification)
        return False
    
    async def _send_android_notification(self, device_token: DeviceToken, notification: PushNotification) -> bool:
        """Send notification to Android device."""
        if self.fcm_service:
            return await self.fcm_service.send_notification(device_token.token, notification)
        return False
    
    async def send_detection_notification(self, user_id: int, detection_data: Dict[str, Any]) -> Dict[str, Any]:
        """Send notification for new wildlife detection."""
        species = detection_data.get('species', 'Unknown species')
        location = detection_data.get('zone', 'Unknown location')
        
        notification = PushNotification(
            title="New Wildlife Detection!",
            body=f"{species} detected at {location}",
            notification_type=NotificationType.NEW_DETECTION,
            data=detection_data,
            user_id=user_id,
            priority="high",
            sound="detection_alert.wav"
        )
        
        return await self.send_to_user(user_id, notification)
    
    async def send_prediction_complete_notification(self, user_id: int, prediction_data: Dict[str, Any]) -> Dict[str, Any]:
        """Send notification when prediction is complete."""
        filename = prediction_data.get('filename', 'audio file')
        
        notification = PushNotification(
            title="Analysis Complete",
            body=f"Analysis of {filename} is ready",
            notification_type=NotificationType.PREDICTION_COMPLETE,
            data=prediction_data,
            user_id=user_id,
            priority="normal"
        )
        
        return await self.send_to_user(user_id, notification)
    
    async def send_system_alert(self, user_ids: List[int], alert_data: Dict[str, Any]) -> Dict[str, Any]:
        """Send system alert to multiple users."""
        title = alert_data.get('title', 'System Alert')
        message = alert_data.get('message', 'System notification')
        
        notification = PushNotification(
            title=title,
            body=message,
            notification_type=NotificationType.SYSTEM_ALERT,
            data=alert_data,
            user_id=0,  # System notification
            priority="high"
        )
        
        total_results = {'sent': 0, 'failed': 0, 'platforms': {'ios': 0, 'android': 0}, 'errors': []}
        
        tasks = []
        for user_id in user_ids:
            notification.user_id = user_id
            tasks.append(self.send_to_user(user_id, notification))
        
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in results:
                if isinstance(result, dict):
                    total_results['sent'] += result.get('sent', 0)
                    total_results['failed'] += result.get('failed', 0)
                    for platform in ['ios', 'android']:
                        total_results['platforms'][platform] += result.get('platforms', {}).get(platform, 0)
                    total_results['errors'].extend(result.get('errors', []))
        
        return total_results
    
    def get_user_devices(self, user_id: int) -> List[Dict[str, Any]]:
        """Get device tokens for a user."""
        if user_id in self.device_tokens:
            return [dt.to_dict() for dt in self.device_tokens[user_id]]
        return []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get push notification service statistics."""
        total_devices = sum(len(tokens) for tokens in self.device_tokens.values())
        active_devices = sum(
            len([dt for dt in tokens if dt.active])
            for tokens in self.device_tokens.values()
        )
        
        platforms = {'ios': 0, 'android': 0}
        for tokens in self.device_tokens.values():
            for token in tokens:
                if token.active:
                    platforms[token.platform] = platforms.get(token.platform, 0) + 1
        
        return {
            'total_users': len(self.device_tokens),
            'total_devices': total_devices,
            'active_devices': active_devices,
            'platforms': platforms,
            'apns_available': self.apns_service is not None,
            'fcm_available': self.fcm_service is not None
        }


# Global push notification service instance
_push_service: Optional[PushNotificationService] = None


def get_push_service() -> PushNotificationService:
    """Get or create global push notification service instance."""
    global _push_service
    
    if _push_service is None:
        _push_service = PushNotificationService()
    
    return _push_service


# Example usage and testing
async def test_push_notifications():
    """Test push notification functionality."""
    service = get_push_service()
    
    # Register test device
    service.register_device(user_id=1, token="test_token_123", platform="ios")
    
    # Test detection notification
    await service.send_detection_notification(
        user_id=1,
        detection_data={
            'id': 123,
            'species': 'Great Horned Owl',
            'zone': 'Sensor A',
            'confidence': 0.95,
            'timestamp': datetime.utcnow().isoformat()
        }
    )
    
    # Log stats
    stats = service.get_stats()
    logger.debug(f"Push service stats: {stats}")


if __name__ == "__main__":
    import os
    # Set test environment variables
    os.environ["APNS_CERT_FILE"] = "/path/to/cert.pem"
    os.environ["APNS_KEY_FILE"] = "/path/to/key.pem"
    os.environ["FCM_API_KEY"] = "your_fcm_api_key"
    
    asyncio.run(test_push_notifications())