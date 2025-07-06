"""
Shared notification utilities for NightScan.
Eliminates duplicate notification functions across services.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

class NotificationCoordinator:
    """Coordinates notifications across different services to avoid duplication."""
    
    def __init__(self):
        self.notification_service = None
        self.websocket_service = None
        
    def register_notification_service(self, service):
        """Register the main notification service."""
        self.notification_service = service
        
    def register_websocket_service(self, service):
        """Register the WebSocket service."""
        self.websocket_service = service
        
    async def send_prediction_complete_notification(self, prediction_data: Dict[str, Any], user_id: int):
        """Centralized prediction complete notification."""
        if self.notification_service:
            await self.notification_service.send_prediction_complete_notification(
                prediction_data, user_id
            )
            
        if self.websocket_service:
            await self.websocket_service.notify_prediction_complete(
                prediction_data, user_id
            )
            
        logger.info(f"Prediction complete notification sent for user {user_id}")
        
    async def notify_prediction_complete(self, prediction_data: Dict[str, Any], user_id: int):
        """Alias for backward compatibility."""
        await self.send_prediction_complete_notification(prediction_data, user_id)

# Global coordinator instance
_coordinator = NotificationCoordinator()

def get_notification_coordinator():
    """Get the global notification coordinator."""
    return _coordinator

# Backward compatible functions
async def send_prediction_complete_notification(prediction_data: Dict[str, Any], user_id: int):
    """Backward compatible function."""
    coordinator = get_notification_coordinator()
    await coordinator.send_prediction_complete_notification(prediction_data, user_id)

async def notify_prediction_complete(prediction_data: Dict[str, Any], user_id: int):
    """Backward compatible function."""
    coordinator = get_notification_coordinator()
    await coordinator.notify_prediction_complete(prediction_data, user_id)
