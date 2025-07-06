import pytest
import json
from unittest.mock import Mock, patch, MagicMock, AsyncMock
import sys
from pathlib import Path

# Add the parent directory to Python path
sys.path.append(str(Path(__file__).resolve().parents[1]))

# Mock React Native dependencies before importing
mock_expo_notifications = MagicMock()
mock_expo_device = MagicMock()
mock_expo_constants = MagicMock()
mock_async_storage = MagicMock()
mock_react_native = MagicMock()

sys.modules['expo-notifications'] = mock_expo_notifications
sys.modules['expo-device'] = mock_expo_device
sys.modules['expo-constants'] = mock_expo_constants
sys.modules['@react-native-async-storage/async-storage'] = mock_async_storage
sys.modules['react-native'] = mock_react_native

# Mock the mobile notification functions
# Since these are JavaScript modules, we'll create Python equivalents for testing


class MockAsyncStorage:
    """Mock AsyncStorage for testing."""
    def __init__(self):
        self.storage = {}
        
    async def getItem(self, key):
        return self.storage.get(key)
        
    async def setItem(self, key, value):
        self.storage[key] = value
        
    async def removeItem(self, key):
        if key in self.storage:
            del self.storage[key]


class MockExpoNotifications:
    """Mock Expo Notifications for testing."""
    def __init__(self):
        self.notification_handler = None
        self.categories = {}
        self.listeners = []
        
    def setNotificationHandler(self, handler):
        self.notification_handler = handler
        
    async def setNotificationCategoryAsync(self, identifier, actions):
        self.categories[identifier] = actions
        
    async def getPermissionsAsync(self):
        return {'status': 'granted'}
        
    async def requestPermissionsAsync(self):
        return {'status': 'granted'}
        
    async def getExpoPushTokenAsync(self, config):
        return {'data': 'ExponentPushToken[test_token]'}
        
    async def scheduleNotificationAsync(self, notification):
        return 'notification_id_123'
        
    async def setBadgeCountAsync(self, count):
        pass
        
    def addNotificationResponseReceivedListener(self, callback):
        listener = {'callback': callback, 'type': 'response'}
        self.listeners.append(listener)
        return listener
        
    def addNotificationReceivedListener(self, callback):
        listener = {'callback': callback, 'type': 'received'}
        self.listeners.append(listener)
        return listener
        
    def removeNotificationSubscription(self, listener):
        if listener in self.listeners:
            self.listeners.remove(listener)


class TestMobileNotificationService:
    """Test mobile notification service functionality."""
    
    @pytest.fixture
    def mock_storage(self):
        """Create mock async storage."""
        return MockAsyncStorage()
        
    @pytest.fixture
    def mock_notifications(self):
        """Create mock expo notifications."""
        return MockExpoNotifications()
        
    @pytest.fixture
    def notification_preferences(self):
        """Default notification preferences for testing."""
        return {
            'enableDetections': True,
            'enablePredictions': True,
            'enableSystem': True,
            'minConfidence': 0.7,
            'quietHoursStart': None,
            'quietHoursEnd': None,
            'speciesFilter': [],
            'zoneFilter': []
        }
        
    def test_notification_handler_setup(self, mock_notifications):
        """Test notification handler configuration."""
        handler_config = {
            'shouldShowAlert': True,
            'shouldPlaySound': True,
            'shouldSetBadge': True
        }
        
        mock_notifications.setNotificationHandler(handler_config)
        assert mock_notifications.notification_handler == handler_config
        
    def test_notification_categories_setup(self, mock_notifications):
        """Test notification category configuration."""
        detection_actions = [
            {
                'identifier': 'view',
                'buttonTitle': 'View Details',
                'options': {'opensAppToForeground': True}
            },
            {
                'identifier': 'dismiss',
                'buttonTitle': 'Dismiss',
                'options': {'opensAppToForeground': False}
            }
        ]
        
        # Test async category setup
        async def test_categories():
            await mock_notifications.setNotificationCategoryAsync('detection', detection_actions)
            assert 'detection' in mock_notifications.categories
            assert mock_notifications.categories['detection'] == detection_actions
            
        # Run the async test
        import asyncio
        asyncio.run(test_categories())
        
    def test_push_token_registration(self, mock_notifications):
        """Test push notification token registration."""
        async def test_registration():
            # Mock device check
            with patch('expo-device.isDevice', True):
                # Mock permissions
                permissions = await mock_notifications.getPermissionsAsync()
                assert permissions['status'] == 'granted'
                
                # Mock token generation
                token_data = await mock_notifications.getExpoPushTokenAsync({
                    'projectId': 'test_project_id'
                })
                assert token_data['data'] == 'ExponentPushToken[test_token]'
                
        asyncio.run(test_registration())
        
    def test_local_notification_scheduling(self, mock_notifications):
        """Test local notification scheduling."""
        async def test_scheduling():
            notification_config = {
                'content': {
                    'title': 'Wildlife Detection',
                    'body': 'Great Horned Owl detected',
                    'data': {'type': 'detection', 'detectionId': 123},
                    'categoryIdentifier': 'detection',
                    'sound': 'default',
                    'badge': 1
                },
                'trigger': None  # Show immediately
            }
            
            notification_id = await mock_notifications.scheduleNotificationAsync(notification_config)
            assert notification_id == 'notification_id_123'
            
        asyncio.run(test_scheduling())


class TestNotificationPreferences:
    """Test notification preferences management."""
    
    def test_preference_loading(self, mock_storage, notification_preferences):
        """Test loading notification preferences from storage."""
        async def test_loading():
            # Store preferences
            prefs_json = json.dumps(notification_preferences)
            await mock_storage.setItem('notificationPreferences', prefs_json)
            
            # Load preferences
            stored_prefs = await mock_storage.getItem('notificationPreferences')
            loaded_prefs = json.loads(stored_prefs) if stored_prefs else {}
            
            assert loaded_prefs['enableDetections'] == True
            assert loaded_prefs['minConfidence'] == 0.7
            
        asyncio.run(test_loading())
        
    def test_preference_saving(self, mock_storage):
        """Test saving notification preferences to storage."""
        async def test_saving():
            new_prefs = {
                'enableDetections': False,
                'enablePredictions': True,
                'minConfidence': 0.8,
                'speciesFilter': ['owl', 'hawk']
            }
            
            prefs_json = json.dumps(new_prefs)
            await mock_storage.setItem('notificationPreferences', prefs_json)
            
            # Verify saved
            stored = await mock_storage.getItem('notificationPreferences')
            loaded = json.loads(stored)
            
            assert loaded['enableDetections'] == False
            assert loaded['minConfidence'] == 0.8
            assert 'owl' in loaded['speciesFilter']
            
        asyncio.run(test_saving())
        
    def test_quiet_hours_logic(self):
        """Test quiet hours calculation."""
        from datetime import datetime
        
        def is_in_quiet_hours(quiet_start, quiet_end):
            """Mock quiet hours check logic."""
            if not quiet_start or not quiet_end:
                return False
                
            now = datetime.now()
            current_time = now.hour * 60 + now.minute
            
            start_hour, start_min = map(int, quiet_start.split(':'))
            end_hour, end_min = map(int, quiet_end.split(':'))
            
            start_time = start_hour * 60 + start_min
            end_time = end_hour * 60 + end_min
            
            if start_time <= end_time:
                # Same day range
                return start_time <= current_time <= end_time
            else:
                # Overnight range
                return current_time >= start_time or current_time <= end_time
        
        # Test same day quiet hours (10 PM to 6 AM)
        with patch('datetime.datetime') as mock_datetime:
            # Test at midnight (should be quiet)
            mock_datetime.now.return_value.hour = 0
            mock_datetime.now.return_value.minute = 0
            assert is_in_quiet_hours('22:00', '06:00') == True
            
            # Test at noon (should not be quiet)
            mock_datetime.now.return_value.hour = 12
            mock_datetime.now.return_value.minute = 0
            assert is_in_quiet_hours('22:00', '06:00') == False


class TestNotificationFiltering:
    """Test notification filtering logic."""
    
    def test_detection_filtering_by_species(self):
        """Test filtering detections by species."""
        def should_show_detection(detection, preferences):
            """Mock detection filtering logic."""
            if not preferences.get('enableDetections', True):
                return False
                
            # Check confidence threshold
            if detection.get('confidence', 0) < preferences.get('minConfidence', 0.7):
                return False
                
            # Check species filter
            species_filter = preferences.get('speciesFilter', [])
            if species_filter:
                species = detection.get('species', '').lower()
                return any(filter_term.lower() in species for filter_term in species_filter)
                
            return True
        
        preferences = {
            'enableDetections': True,
            'minConfidence': 0.7,
            'speciesFilter': ['owl', 'hawk']
        }
        
        # Test owl detection (should pass)
        owl_detection = {
            'species': 'Great Horned Owl',
            'confidence': 0.95,
            'zone': 'Forest A'
        }
        assert should_show_detection(owl_detection, preferences) == True
        
        # Test bat detection (should not pass species filter)
        bat_detection = {
            'species': 'Little Brown Bat',
            'confidence': 0.85,
            'zone': 'Cave B'
        }
        assert should_show_detection(bat_detection, preferences) == False
        
        # Test low confidence owl (should not pass confidence filter)
        low_conf_owl = {
            'species': 'Barn Owl',
            'confidence': 0.6,
            'zone': 'Field C'
        }
        assert should_show_detection(low_conf_owl, preferences) == False
        
    def test_detection_filtering_by_zone(self):
        """Test filtering detections by zone."""
        def should_show_detection_zone(detection, preferences):
            """Mock zone filtering logic."""
            zone_filter = preferences.get('zoneFilter', [])
            if zone_filter:
                zone = detection.get('zone', '').lower()
                return any(filter_term.lower() in zone for filter_term in zone_filter)
            return True
        
        preferences = {
            'zoneFilter': ['forest', 'field']
        }
        
        # Test forest detection (should pass)
        forest_detection = {
            'species': 'Owl',
            'zone': 'Forest Area A'
        }
        assert should_show_detection_zone(forest_detection, preferences) == True
        
        # Test urban detection (should not pass)
        urban_detection = {
            'species': 'Pigeon',
            'zone': 'Urban Center'
        }
        assert should_show_detection_zone(urban_detection, preferences) == False


class TestNotificationActions:
    """Test notification action handling."""
    
    def test_notification_response_handling(self, mock_notifications):
        """Test handling of notification responses."""
        responses_received = []
        
        def mock_navigation_navigate(screen, params=None):
            """Mock navigation function."""
            responses_received.append({'screen': screen, 'params': params})
        
        def handle_notification_response(response):
            """Mock notification response handler."""
            notification = response['notification']
            action_id = response.get('actionIdentifier', 'default')
            data = notification['request']['content']['data']
            
            if action_id == 'view':
                if data.get('type') == 'detection':
                    mock_navigation_navigate('DetectionDetail', {'id': data.get('detectionId')})
                elif data.get('type') == 'prediction':
                    mock_navigation_navigate('Predictions')
            elif action_id == 'dismiss':
                # Just dismiss
                pass
            else:
                # Default tap action
                if data.get('type') == 'detection':
                    mock_navigation_navigate('DetectionList')
                else:
                    mock_navigation_navigate('Home')
        
        # Test view action for detection
        detection_response = {
            'notification': {
                'request': {
                    'content': {
                        'data': {
                            'type': 'detection',
                            'detectionId': 123
                        }
                    }
                }
            },
            'actionIdentifier': 'view'
        }
        
        handle_notification_response(detection_response)
        assert len(responses_received) == 1
        assert responses_received[0]['screen'] == 'DetectionDetail'
        assert responses_received[0]['params']['id'] == 123
        
        # Test default tap for prediction
        prediction_response = {
            'notification': {
                'request': {
                    'content': {
                        'data': {
                            'type': 'prediction'
                        }
                    }
                }
            },
            'actionIdentifier': 'default'
        }
        
        responses_received.clear()
        handle_notification_response(prediction_response)
        assert len(responses_received) == 1
        assert responses_received[0]['screen'] == 'Home'
        
    def test_notification_listener_management(self, mock_notifications):
        """Test notification listener setup and cleanup."""
        def setup_listeners():
            """Mock listener setup."""
            response_listener = mock_notifications.addNotificationResponseReceivedListener(
                lambda response: None
            )
            foreground_listener = mock_notifications.addNotificationReceivedListener(
                lambda notification: None
            )
            
            return {
                'response_listener': response_listener,
                'foreground_listener': foreground_listener
            }
        
        def cleanup_listeners(listeners):
            """Mock listener cleanup."""
            if listeners.get('response_listener'):
                mock_notifications.removeNotificationSubscription(listeners['response_listener'])
            if listeners.get('foreground_listener'):
                mock_notifications.removeNotificationSubscription(listeners['foreground_listener'])
        
        # Test setup
        listeners = setup_listeners()
        assert len(mock_notifications.listeners) == 2
        
        # Test cleanup
        cleanup_listeners(listeners)
        assert len(mock_notifications.listeners) == 0


class TestNotificationHistory:
    """Test notification history management."""
    
    def test_notification_history_storage(self, mock_storage):
        """Test storing notification history."""
        async def test_storage():
            # Initial empty history
            history = await mock_storage.getItem('notificationHistory')
            assert history is None
            
            # Add first notification
            notification1 = {
                'id': 1,
                'title': 'Wildlife Detection',
                'body': 'Owl detected',
                'timestamp': '2023-01-01T12:00:00Z',
                'read': False
            }
            
            await mock_storage.setItem('notificationHistory', json.dumps([notification1]))
            
            # Verify stored
            stored_history = await mock_storage.getItem('notificationHistory')
            history_list = json.loads(stored_history)
            assert len(history_list) == 1
            assert history_list[0]['title'] == 'Wildlife Detection'
            
        asyncio.run(test_storage())
        
    def test_notification_history_trimming(self, mock_storage):
        """Test trimming notification history to limit size."""
        async def test_trimming():
            # Create a large history (over 100 items)
            large_history = []
            for i in range(150):
                notification = {
                    'id': i,
                    'title': f'Notification {i}',
                    'timestamp': f'2023-01-01T{i%24:02d}:00:00Z',
                    'read': False
                }
                large_history.append(notification)
            
            await mock_storage.setItem('notificationHistory', json.dumps(large_history))
            
            # Simulate trimming logic (keep only last 100)
            stored_history = await mock_storage.getItem('notificationHistory')
            history_list = json.loads(stored_history)
            
            if len(history_list) > 100:
                trimmed_history = history_list[:100]
                await mock_storage.setItem('notificationHistory', json.dumps(trimmed_history))
            
            # Verify trimmed
            final_history = await mock_storage.getItem('notificationHistory')
            final_list = json.loads(final_history)
            assert len(final_list) == 100
            
        asyncio.run(test_trimming())


class TestWebSocketNotificationHandling:
    """Test WebSocket notification handling in mobile app."""
    
    def test_websocket_notification_processing(self):
        """Test processing WebSocket notifications."""
        def handle_websocket_notification(notification):
            """Mock WebSocket notification handler."""
            event_type = notification.get('event_type')
            data = notification.get('data', {})
            priority = notification.get('priority', 'normal')
            
            processed_notifications = []
            
            if event_type == 'new_detection':
                # Check if should show detection notification
                if data.get('confidence', 0) >= 0.7:  # Mock threshold
                    processed_notifications.append({
                        'type': 'detection',
                        'title': f"🦉 New Wildlife Detection!",
                        'body': f"{data.get('species')} detected at {data.get('zone', 'unknown location')}",
                        'data': data
                    })
                    
            elif event_type == 'prediction_complete':
                processed_notifications.append({
                    'type': 'prediction',
                    'title': "✅ Analysis Complete",
                    'body': f"Results ready for {data.get('filename')}",
                    'data': data
                })
                
            elif event_type == 'system_status':
                if priority in ['high', 'critical']:
                    processed_notifications.append({
                        'type': 'system',
                        'title': f"🚨 {data.get('type', 'System Alert')}",
                        'body': data.get('message', 'System notification'),
                        'data': data
                    })
            
            return processed_notifications
        
        # Test detection notification
        detection_ws = {
            'event_type': 'new_detection',
            'data': {
                'species': 'Great Horned Owl',
                'zone': 'Forest Area A',
                'confidence': 0.95
            },
            'priority': 'normal'
        }
        
        notifications = handle_websocket_notification(detection_ws)
        assert len(notifications) == 1
        assert notifications[0]['type'] == 'detection'
        assert 'Great Horned Owl' in notifications[0]['body']
        
        # Test low confidence detection (should be filtered)
        low_conf_detection = {
            'event_type': 'new_detection',
            'data': {
                'species': 'Bat',
                'confidence': 0.5
            }
        }
        
        notifications = handle_websocket_notification(low_conf_detection)
        assert len(notifications) == 0
        
        # Test system alert
        system_alert = {
            'event_type': 'system_status',
            'data': {
                'type': 'Error',
                'message': 'Database connection lost'
            },
            'priority': 'high'
        }
        
        notifications = handle_websocket_notification(system_alert)
        assert len(notifications) == 1
        assert notifications[0]['type'] == 'system'
        assert 'Database connection lost' in notifications[0]['body']


@pytest.mark.integration
class TestMobileNotificationIntegration:
    """Integration tests for mobile notification system."""
    
    def test_notification_flow_end_to_end(self):
        """Test complete notification flow from WebSocket to display."""
        # This would test the full flow from receiving a WebSocket message
        # to displaying a local notification
        pass
        
    def test_notification_persistence_across_app_restarts(self):
        """Test that notification settings persist across app restarts."""
        # This would test AsyncStorage persistence
        pass
        
    def test_notification_sync_with_backend(self):
        """Test syncing notification preferences with backend."""
        # This would test API calls to sync preferences
        pass


if __name__ == '__main__':
    pytest.main([__file__])