"""Notification service for NightScan real-time alerts."""

import smtplib
import json
import logging
import requests
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.utils import formataddr

try:
    from flask_sqlalchemy import SQLAlchemy
    from flask import Flask
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False

from config import get_config

logger = logging.getLogger(__name__)


class NotificationChannel(Enum):
    """Available notification channels."""
    EMAIL = "email"
    PUSH = "push"
    SLACK = "slack"
    DISCORD = "discord"
    WEBHOOK = "webhook"
    SMS = "sms"


class NotificationPriority(Enum):
    """Notification priority levels."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class NotificationPreference:
    """User notification preferences."""
    user_id: int
    channels: List[NotificationChannel]
    email_address: Optional[str] = None
    push_endpoint: Optional[str] = None
    slack_webhook: Optional[str] = None
    discord_webhook: Optional[str] = None
    min_priority: NotificationPriority = NotificationPriority.NORMAL
    species_filter: Optional[List[str]] = None
    zone_filter: Optional[List[str]] = None
    quiet_hours_start: Optional[str] = None  # HH:MM format
    quiet_hours_end: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result['channels'] = [c.value for c in self.channels]
        result['min_priority'] = self.min_priority.value
        return result


@dataclass
class NotificationTemplate:
    """Email/notification templates."""
    subject: str
    html_body: str
    text_body: str
    priority: NotificationPriority = NotificationPriority.NORMAL


class NotificationService:
    """Comprehensive notification service for NightScan."""
    
    def __init__(self, db=None):
        """Initialize notification service."""
        self.config = get_config()
        self.db = db
        self.user_preferences: Dict[int, NotificationPreference] = {}
        self.templates = self._load_templates()
        
        # Email configuration
        self.smtp_config = {
            'server': self.config.notifications.smtp_server if hasattr(self.config, 'notifications') else 'localhost',
            'port': getattr(self.config.notifications, 'smtp_port', 587) if hasattr(self.config, 'notifications') else 587,
            'username': getattr(self.config.notifications, 'smtp_username', '') if hasattr(self.config, 'notifications') else '',
            'password': getattr(self.config.notifications, 'smtp_password', '') if hasattr(self.config, 'notifications') else '',
            'use_tls': getattr(self.config.notifications, 'smtp_use_tls', True) if hasattr(self.config, 'notifications') else True,
            'from_address': getattr(self.config.notifications, 'from_address', 'noreply@nightscan.example.com') if hasattr(self.config, 'notifications') else 'noreply@nightscan.example.com',
            'from_name': getattr(self.config.notifications, 'from_name', 'NightScan') if hasattr(self.config, 'notifications') else 'NightScan'
        }
        
        # Load user preferences
        if FLASK_AVAILABLE and self.db:
            self._load_user_preferences()
    
    def _load_templates(self) -> Dict[str, NotificationTemplate]:
        """Load notification templates."""
        return {
            'new_detection': NotificationTemplate(
                subject="🦉 New Wildlife Detection - {species}",
                html_body="""
                <html>
                <head>
                    <style>
                        body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; }
                        .container { max-width: 600px; margin: 0 auto; background: white; border-radius: 10px; overflow: hidden; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
                        .header { background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%); color: white; padding: 30px; text-align: center; }
                        .content { padding: 30px; }
                        .detection-card { background: #f8f9fa; border-left: 4px solid #4CAF50; padding: 20px; margin: 20px 0; border-radius: 5px; }
                        .footer { background: #f8f9fa; padding: 20px; text-align: center; color: #666; font-size: 12px; }
                        .btn { background: #4CAF50; color: white; padding: 12px 24px; text-decoration: none; border-radius: 5px; display: inline-block; margin: 10px 0; }
                    </style>
                </head>
                <body>
                    <div class="container">
                        <div class="header">
                            <h1>🌙 NightScan Detection Alert</h1>
                        </div>
                        <div class="content">
                            <h2>New Wildlife Detection!</h2>
                            <div class="detection-card">
                                <h3>🦎 {species}</h3>
                                <p><strong>Location:</strong> {zone}</p>
                                <p><strong>Time:</strong> {timestamp}</p>
                                <p><strong>Confidence:</strong> {confidence}%</p>
                                {description}
                            </div>
                            <p>This detection was automatically identified by our AI-powered wildlife monitoring system.</p>
                            <a href="{dashboard_url}" class="btn">View Live Dashboard</a>
                        </div>
                        <div class="footer">
                            <p>NightScan Wildlife Detection System</p>
                            <p>To manage your notification preferences, visit your dashboard settings.</p>
                        </div>
                    </div>
                </body>
                </html>
                """,
                text_body="""
                🌙 NightScan Detection Alert
                
                New Wildlife Detection: {species}
                
                Location: {zone}
                Time: {timestamp}
                Confidence: {confidence}%
                
                {description}
                
                View the live dashboard: {dashboard_url}
                
                ---
                NightScan Wildlife Detection System
                """,
                priority=NotificationPriority.HIGH
            ),
            
            'prediction_complete': NotificationTemplate(
                subject="✅ Audio Analysis Complete - {filename}",
                html_body="""
                <html>
                <head>
                    <style>
                        body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; }
                        .container { max-width: 600px; margin: 0 auto; background: white; border-radius: 10px; overflow: hidden; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
                        .header { background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%); color: white; padding: 30px; text-align: center; }
                        .content { padding: 30px; }
                        .result-card { background: #f8f9fa; border-left: 4px solid #2196F3; padding: 20px; margin: 20px 0; border-radius: 5px; }
                        .footer { background: #f8f9fa; padding: 20px; text-align: center; color: #666; font-size: 12px; }
                        .btn { background: #2196F3; color: white; padding: 12px 24px; text-decoration: none; border-radius: 5px; display: inline-block; margin: 10px 0; }
                    </style>
                </head>
                <body>
                    <div class="container">
                        <div class="header">
                            <h1>🎵 Audio Analysis Complete</h1>
                        </div>
                        <div class="content">
                            <h2>Your audio file has been processed!</h2>
                            <div class="result-card">
                                <h3>📁 {filename}</h3>
                                <p><strong>Status:</strong> {status}</p>
                                <p><strong>Processing Time:</strong> {processing_time}</p>
                                <p><strong>Results:</strong></p>
                                <div style="margin-left: 20px;">
                                    {results}
                                </div>
                            </div>
                            <a href="{dashboard_url}" class="btn">View Results</a>
                        </div>
                        <div class="footer">
                            <p>NightScan Wildlife Detection System</p>
                        </div>
                    </div>
                </body>
                </html>
                """,
                text_body="""
                🎵 NightScan Audio Analysis Complete
                
                File: {filename}
                Status: {status}
                Processing Time: {processing_time}
                
                Results:
                {results}
                
                View results: {dashboard_url}
                
                ---
                NightScan Wildlife Detection System
                """,
                priority=NotificationPriority.NORMAL
            ),
            
            'system_alert': NotificationTemplate(
                subject="🚨 NightScan System Alert - {alert_type}",
                html_body="""
                <html>
                <head>
                    <style>
                        body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; }
                        .container { max-width: 600px; margin: 0 auto; background: white; border-radius: 10px; overflow: hidden; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
                        .header { background: linear-gradient(135deg, #f44336 0%, #d32f2f 100%); color: white; padding: 30px; text-align: center; }
                        .content { padding: 30px; }
                        .alert-card { background: #ffebee; border-left: 4px solid #f44336; padding: 20px; margin: 20px 0; border-radius: 5px; }
                        .footer { background: #f8f9fa; padding: 20px; text-align: center; color: #666; font-size: 12px; }
                    </style>
                </head>
                <body>
                    <div class="container">
                        <div class="header">
                            <h1>🚨 System Alert</h1>
                        </div>
                        <div class="content">
                            <div class="alert-card">
                                <h3>{alert_type}</h3>
                                <p><strong>Time:</strong> {timestamp}</p>
                                <p><strong>Details:</strong> {message}</p>
                                <p><strong>Action Required:</strong> {action}</p>
                            </div>
                        </div>
                        <div class="footer">
                            <p>NightScan Wildlife Detection System</p>
                        </div>
                    </div>
                </body>
                </html>
                """,
                text_body="""
                🚨 NightScan System Alert
                
                Alert Type: {alert_type}
                Time: {timestamp}
                Details: {message}
                Action Required: {action}
                
                ---
                NightScan Wildlife Detection System
                """,
                priority=NotificationPriority.CRITICAL
            )
        }
    
    def _load_user_preferences(self):
        """Load user notification preferences from database."""
        # In a real implementation, this would load from database
        # For now, create default preferences
        pass
    
    def set_user_preferences(self, user_id: int, preferences: NotificationPreference):
        """Set notification preferences for a user."""
        self.user_preferences[user_id] = preferences
        
        # In a real implementation, save to database
        if FLASK_AVAILABLE and self.db:
            # Save to database
            pass
    
    def get_user_preferences(self, user_id: int) -> NotificationPreference:
        """Get notification preferences for a user."""
        return self.user_preferences.get(user_id, NotificationPreference(
            user_id=user_id,
            channels=[NotificationChannel.EMAIL],
            min_priority=NotificationPriority.NORMAL
        ))
    
    def should_send_notification(self, user_id: int, priority: NotificationPriority, 
                               species: Optional[str] = None, zone: Optional[str] = None) -> bool:
        """Check if notification should be sent to user based on preferences."""
        prefs = self.get_user_preferences(user_id)
        
        # Check priority
        priority_levels = [NotificationPriority.LOW, NotificationPriority.NORMAL, 
                          NotificationPriority.HIGH, NotificationPriority.CRITICAL]
        if priority_levels.index(priority) < priority_levels.index(prefs.min_priority):
            return False
        
        # Check species filter
        if prefs.species_filter and species and species.lower() not in [s.lower() for s in prefs.species_filter]:
            return False
        
        # Check zone filter
        if prefs.zone_filter and zone and zone.lower() not in [z.lower() for z in prefs.zone_filter]:
            return False
        
        # Check quiet hours
        if prefs.quiet_hours_start and prefs.quiet_hours_end:
            now = datetime.now().time()
            start_time = datetime.strptime(prefs.quiet_hours_start, '%H:%M').time()
            end_time = datetime.strptime(prefs.quiet_hours_end, '%H:%M').time()
            
            if start_time <= end_time:
                # Same day range
                if start_time <= now <= end_time:
                    return False
            else:
                # Overnight range
                if now >= start_time or now <= end_time:
                    return False
        
        return True
    
    async def send_detection_notification(self, detection_data: Dict[str, Any], 
                                        user_ids: Optional[List[int]] = None):
        """Send notification for new wildlife detection."""
        template = self.templates['new_detection']
        
        # Prepare template data
        template_data = {
            'species': detection_data.get('species', 'Unknown Species'),
            'zone': detection_data.get('zone', 'Unknown Zone'),
            'timestamp': detection_data.get('timestamp', datetime.now().isoformat()),
            'confidence': round(detection_data.get('confidence', 0) * 100, 1),
            'description': detection_data.get('description', ''),
            'dashboard_url': self._get_dashboard_url()
        }
        
        # Send to specified users or all users with preferences
        target_users = user_ids or list(self.user_preferences.keys())
        
        for user_id in target_users:
            if self.should_send_notification(user_id, template.priority, 
                                           template_data['species'], template_data['zone']):
                await self._send_notification_to_user(user_id, template, template_data)
    
    async def send_prediction_complete_notification(self, prediction_data: Dict[str, Any], user_id: int):
        """Send notification when prediction is complete."""
        template = self.templates['prediction_complete']
        
        template_data = {
            'filename': prediction_data.get('filename', 'Unknown File'),
            'status': prediction_data.get('status', 'Completed'),
            'processing_time': prediction_data.get('processing_time', 'Unknown'),
            'results': self._format_prediction_results(prediction_data.get('results', {})),
            'dashboard_url': self._get_dashboard_url()
        }
        
        if self.should_send_notification(user_id, template.priority):
            await self._send_notification_to_user(user_id, template, template_data)
    
    async def send_system_alert(self, alert_data: Dict[str, Any], user_ids: Optional[List[int]] = None):
        """Send system alert notification."""
        template = self.templates['system_alert']
        
        template_data = {
            'alert_type': alert_data.get('type', 'System Alert'),
            'timestamp': alert_data.get('timestamp', datetime.now().isoformat()),
            'message': alert_data.get('message', 'System alert occurred'),
            'action': alert_data.get('action', 'Please check the system dashboard')
        }
        
        target_users = user_ids or list(self.user_preferences.keys())
        
        for user_id in target_users:
            if self.should_send_notification(user_id, template.priority):
                await self._send_notification_to_user(user_id, template, template_data)
    
    async def _send_notification_to_user(self, user_id: int, template: NotificationTemplate, 
                                       template_data: Dict[str, Any]):
        """Send notification to specific user via their preferred channels."""
        prefs = self.get_user_preferences(user_id)
        
        # Send via each preferred channel
        for channel in prefs.channels:
            try:
                if channel == NotificationChannel.EMAIL and prefs.email_address:
                    await self._send_email(prefs.email_address, template, template_data)
                elif channel == NotificationChannel.PUSH and prefs.push_endpoint:
                    await self._send_push_notification(prefs.push_endpoint, template, template_data)
                elif channel == NotificationChannel.SLACK and prefs.slack_webhook:
                    await self._send_slack_notification(prefs.slack_webhook, template, template_data)
                elif channel == NotificationChannel.DISCORD and prefs.discord_webhook:
                    await self._send_discord_notification(prefs.discord_webhook, template, template_data)
                    
            except Exception as e:
                logger.error(f"Failed to send {channel.value} notification to user {user_id}: {e}")
    
    async def _send_email(self, email_address: str, template: NotificationTemplate, 
                         template_data: Dict[str, Any]):
        """Send email notification."""
        try:
            # Format email content
            subject = template.subject.format(**template_data)
            html_body = template.html_body.format(**template_data)
            text_body = template.text_body.format(**template_data)
            
            # Create message
            msg = MIMEMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = formataddr((self.smtp_config['from_name'], self.smtp_config['from_address']))
            msg['To'] = email_address
            
            # Add both text and HTML parts
            text_part = MIMEText(text_body, 'plain', 'utf-8')
            html_part = MIMEText(html_body, 'html', 'utf-8')
            
            msg.attach(text_part)
            msg.attach(html_part)
            
            # Send email
            if self.smtp_config['server'] and self.smtp_config['server'] != 'localhost':
                with smtplib.SMTP(self.smtp_config['server'], self.smtp_config['port']) as server:
                    if self.smtp_config['use_tls']:
                        server.starttls()
                    if self.smtp_config['username']:
                        server.login(self.smtp_config['username'], self.smtp_config['password'])
                    server.send_message(msg)
                
                logger.info(f"Email notification sent to {email_address}")
            else:
                logger.warning(f"SMTP not configured, would send email to {email_address}: {subject}")
                
        except Exception as e:
            logger.error(f"Failed to send email to {email_address}: {e}")
    
    async def _send_push_notification(self, endpoint: str, template: NotificationTemplate, 
                                    template_data: Dict[str, Any]):
        """Send push notification."""
        # Implementation for web push notifications
        payload = {
            'title': template.subject.format(**template_data),
            'body': template.text_body.format(**template_data)[:200] + '...',
            'icon': '/static/nightscan-icon.png',
            'badge': '/static/nightscan-badge.png',
            'data': template_data
        }
        
        logger.info(f"Would send push notification: {payload}")
    
    async def _send_slack_notification(self, webhook_url: str, template: NotificationTemplate, 
                                     template_data: Dict[str, Any]):
        """Send Slack notification."""
        try:
            payload = {
                'text': template.subject.format(**template_data),
                'attachments': [{
                    'color': self._get_color_for_priority(template.priority),
                    'fields': [
                        {'title': key.replace('_', ' ').title(), 'value': str(value), 'short': True}
                        for key, value in template_data.items()
                        if key not in ['dashboard_url', 'description']
                    ]
                }]
            }
            
            async with asyncio.create_task(self._async_post(webhook_url, payload)):
                logger.info(f"Slack notification sent")
                
        except Exception as e:
            logger.error(f"Failed to send Slack notification: {e}")
    
    async def _send_discord_notification(self, webhook_url: str, template: NotificationTemplate, 
                                       template_data: Dict[str, Any]):
        """Send Discord notification."""
        try:
            payload = {
                'content': template.subject.format(**template_data),
                'embeds': [{
                    'title': 'NightScan Notification',
                    'description': template.text_body.format(**template_data)[:1000],
                    'color': int(self._get_color_for_priority(template.priority).replace('#', ''), 16),
                    'timestamp': datetime.now().isoformat()
                }]
            }
            
            async with asyncio.create_task(self._async_post(webhook_url, payload)):
                logger.info(f"Discord notification sent")
                
        except Exception as e:
            logger.error(f"Failed to send Discord notification: {e}")
    
    async def _async_post(self, url: str, payload: Dict[str, Any]):
        """Async HTTP POST request."""
        # In a real implementation, use aiohttp
        import threading
        
        def post_request():
            try:
                response = requests.post(url, json=payload, timeout=10)
                response.raise_for_status()
            except Exception as e:
                logger.error(f"HTTP POST failed: {e}")
        
        thread = threading.Thread(target=post_request)
        thread.start()
    
    def _format_prediction_results(self, results: Dict[str, Any]) -> str:
        """Format prediction results for display."""
        if not results:
            return "No results available"
        
        formatted = []
        for key, value in results.items():
            if isinstance(value, (int, float)):
                formatted.append(f"• {key}: {value}")
            else:
                formatted.append(f"• {key}: {str(value)[:100]}")
        
        return '\n'.join(formatted)
    
    def _get_dashboard_url(self) -> str:
        """Get dashboard URL."""
        base_url = getattr(self.config, 'base_url', 'http://localhost:8000')
        return f"{base_url}/dashboard"
    
    def _get_color_for_priority(self, priority: NotificationPriority) -> str:
        """Get color code for priority level."""
        colors = {
            NotificationPriority.LOW: '#2196F3',
            NotificationPriority.NORMAL: '#4CAF50', 
            NotificationPriority.HIGH: '#FF9800',
            NotificationPriority.CRITICAL: '#f44336'
        }
        return colors.get(priority, '#4CAF50')


# Global notification service instance
_notification_service: Optional[NotificationService] = None


def get_notification_service(db=None) -> NotificationService:
    """Get or create global notification service instance."""
    global _notification_service
    
    if _notification_service is None:
        _notification_service = NotificationService(db)
    
    return _notification_service


# Convenience functions for common notifications
async def notify_new_detection(detection_data: Dict[str, Any], user_ids: Optional[List[int]] = None):
    """Send new detection notification."""
    service = get_notification_service()
    await service.send_detection_notification(detection_data, user_ids)


async def notify_prediction_complete(prediction_data: Dict[str, Any], user_id: int):
    """Send prediction complete notification."""
    service = get_notification_service()
    await service.send_prediction_complete_notification(prediction_data, user_id)


async def notify_system_alert(alert_data: Dict[str, Any], user_ids: Optional[List[int]] = None):
    """Send system alert notification."""
    service = get_notification_service()
    await service.send_system_alert(alert_data, user_ids)


# Example usage and testing
async def test_notifications():
    """Test notification service."""
    service = get_notification_service()
    
    # Set up test user preferences
    service.set_user_preferences(1, NotificationPreference(
        user_id=1,
        channels=[NotificationChannel.EMAIL],
        email_address="test@example.com",
        min_priority=NotificationPriority.NORMAL
    ))
    
    # Test detection notification
    await service.send_detection_notification({
        'species': 'Great Horned Owl',
        'zone': 'North Forest Zone',
        'timestamp': datetime.now().isoformat(),
        'confidence': 0.95,
        'description': 'Detected in sector 7-B with high confidence.'
    }, [1])
    
    # Test prediction complete notification
    await service.send_prediction_complete_notification({
        'filename': 'night_sounds_001.wav',
        'status': 'Completed Successfully',
        'processing_time': '2.3 seconds',
        'results': {
            'Primary Species': 'Great Horned Owl (95% confidence)',
            'Secondary Species': 'Barn Owl (67% confidence)',
            'Background Noise': 'Wind, leaves rustling'
        }
    }, 1)


if __name__ == "__main__":
    asyncio.run(test_notifications())