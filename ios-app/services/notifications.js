import * as Notifications from 'expo-notifications';
import * as Device from 'expo-device';
import Constants from 'expo-constants';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { Platform } from 'react-native';

// Configure notification handler with enhanced settings
Notifications.setNotificationHandler({
  handleNotification: async () => ({
    shouldShowAlert: true,
    shouldPlaySound: true,
    shouldSetBadge: true,
  }),
});

// Notification categories for rich interactions
const notificationCategories = [
  {
    identifier: 'detection',
    actions: [
      {
        identifier: 'view',
        buttonTitle: 'View Details',
        options: {
          opensAppToForeground: true,
        },
      },
      {
        identifier: 'dismiss',
        buttonTitle: 'Dismiss',
        options: {
          opensAppToForeground: false,
        },
      },
    ],
  },
  {
    identifier: 'prediction',
    actions: [
      {
        identifier: 'view_results',
        buttonTitle: 'View Results',
        options: {
          opensAppToForeground: true,
        },
      },
    ],
  },
];

// Set notification categories
Notifications.setNotificationCategoryAsync('detection', notificationCategories[0].actions);
Notifications.setNotificationCategoryAsync('prediction', notificationCategories[1].actions);

export async function registerForPushNotificationsAsync() {
  if (!Device.isDevice) {
    console.log('Push notifications require a physical device');
    return null;
  }

  const { status: existingStatus } = await Notifications.getPermissionsAsync();
  let finalStatus = existingStatus;
  
  if (existingStatus !== 'granted') {
    const { status } = await Notifications.requestPermissionsAsync();
    finalStatus = status;
  }
  
  if (finalStatus !== 'granted') {
    console.log('Failed to get push token for push notification!');
    return null;
  }

  try {
    const projectId = Constants?.expoConfig?.extra?.eas?.projectId ?? Constants?.easConfig?.projectId;
    
    if (!projectId) {
      console.log('Project ID not found');
      return null;
    }

    const pushTokenData = await Notifications.getExpoPushTokenAsync({
      projectId,
    });

    const pushToken = pushTokenData.data;
    
    // Store token locally
    await AsyncStorage.setItem('pushToken', pushToken);
    
    console.log('Push token:', pushToken);
    return pushToken;
  } catch (error) {
    console.error('Error getting push token:', error);
    return null;
  }
}

export async function sendLocalNotification(title, body, data = {}, categoryIdentifier = null) {
  try {
    const notificationId = await Notifications.scheduleNotificationAsync({
      content: {
        title,
        body,
        data,
        categoryIdentifier,
        sound: 'default',
        badge: 1,
      },
      trigger: null, // Show immediately
    });
    
    return notificationId;
  } catch (error) {
    console.error('Error sending local notification:', error);
    return null;
  }
}

export function sendDetectionNotification(detection) {
  const title = `🦉 New Wildlife Detection!`;
  const body = `${detection.species} detected at ${detection.zone || 'sensor location'}`;
  const confidence = detection.confidence ? Math.round(detection.confidence * 100) : 0;
  
  return sendLocalNotification(
    title,
    `${body} (${confidence}% confidence)`,
    {
      type: 'detection',
      detectionId: detection.id,
      species: detection.species,
      zone: detection.zone,
      confidence: detection.confidence,
      timestamp: detection.timestamp || new Date().toISOString(),
    },
    'detection'
  );
}

export function sendPredictionCompleteNotification(prediction) {
  const title = `✅ Analysis Complete`;
  const body = `Results ready for ${prediction.filename}`;
  
  return sendLocalNotification(
    title,
    body,
    {
      type: 'prediction',
      predictionId: prediction.id,
      filename: prediction.filename,
      status: prediction.status,
    },
    'prediction'
  );
}

export function sendSystemNotification(alert) {
  const title = `🚨 ${alert.type || 'System Alert'}`;
  const body = alert.message || 'System notification';
  
  return sendLocalNotification(
    title,
    body,
    {
      type: 'system',
      alertType: alert.type,
      priority: alert.priority || 'normal',
    }
  );
}

// Enhanced detection tracking with user preferences
let lastDetectionId = 0;
let notificationPreferences = {
  enableDetections: true,
  enablePredictions: true,
  enableSystem: true,
  minConfidence: 0.7,
  quietHoursStart: null,
  quietHoursEnd: null,
  speciesFilter: [],
  zoneFilter: [],
};

export async function loadNotificationPreferences() {
  try {
    const stored = await AsyncStorage.getItem('notificationPreferences');
    if (stored) {
      notificationPreferences = { ...notificationPreferences, ...JSON.parse(stored) };
    }
  } catch (error) {
    console.error('Failed to load notification preferences:', error);
  }
}

export async function saveNotificationPreferences(prefs) {
  try {
    notificationPreferences = { ...notificationPreferences, ...prefs };
    await AsyncStorage.setItem('notificationPreferences', JSON.stringify(notificationPreferences));
  } catch (error) {
    console.error('Failed to save notification preferences:', error);
  }
}

export function getNotificationPreferences() {
  return notificationPreferences;
}

function isInQuietHours() {
  if (!notificationPreferences.quietHoursStart || !notificationPreferences.quietHoursEnd) {
    return false;
  }
  
  const now = new Date();
  const currentTime = now.getHours() * 60 + now.getMinutes();
  
  const [startHour, startMin] = notificationPreferences.quietHoursStart.split(':').map(Number);
  const [endHour, endMin] = notificationPreferences.quietHoursEnd.split(':').map(Number);
  
  const startTime = startHour * 60 + startMin;
  const endTime = endHour * 60 + endMin;
  
  if (startTime <= endTime) {
    // Same day range
    return currentTime >= startTime && currentTime <= endTime;
  } else {
    // Overnight range
    return currentTime >= startTime || currentTime <= endTime;
  }
}

function shouldShowDetectionNotification(detection) {
  if (!notificationPreferences.enableDetections) return false;
  if (isInQuietHours()) return false;
  
  // Check confidence threshold
  if (detection.confidence && detection.confidence < notificationPreferences.minConfidence) {
    return false;
  }
  
  // Check species filter
  if (notificationPreferences.speciesFilter.length > 0) {
    const species = detection.species?.toLowerCase() || '';
    const matchesFilter = notificationPreferences.speciesFilter.some(filter => 
      species.includes(filter.toLowerCase())
    );
    if (!matchesFilter) return false;
  }
  
  // Check zone filter
  if (notificationPreferences.zoneFilter.length > 0) {
    const zone = detection.zone?.toLowerCase() || '';
    const matchesFilter = notificationPreferences.zoneFilter.some(filter => 
      zone.includes(filter.toLowerCase())
    );
    if (!matchesFilter) return false;
  }
  
  return true;
}

export function checkForNewDetections(detections) {
  const maxId = detections.reduce((acc, d) => Math.max(acc, d.id), 0);
  if (lastDetectionId && maxId > lastDetectionId) {
    detections
      .filter((d) => d.id > lastDetectionId)
      .forEach((d) => {
        if (shouldShowDetectionNotification(d)) {
          sendDetectionNotification(d).catch((error) => {
            console.error('Failed to send detection notification:', error);
          });
        }
      });
  }
  lastDetectionId = maxId;
}

// Enhanced notification handling for WebSocket events
export function handleWebSocketNotification(notification) {
  const { event_type, data, priority } = notification;
  
  switch (event_type) {
    case 'new_detection':
      if (shouldShowDetectionNotification(data)) {
        sendDetectionNotification(data);
      }
      break;
      
    case 'prediction_complete':
      if (notificationPreferences.enablePredictions && !isInQuietHours()) {
        sendPredictionCompleteNotification(data);
      }
      break;
      
    case 'system_status':
    case 'error_notification':
      if (notificationPreferences.enableSystem && (priority === 'high' || priority === 'critical')) {
        sendSystemNotification(data);
      }
      break;
  }
}

// Notification action handlers
export function setupNotificationActionHandlers(navigation) {
  // Handle notification responses (when user taps notification or action buttons)
  const notificationListener = Notifications.addNotificationResponseReceivedListener(response => {
    const { notification, actionIdentifier } = response;
    const { data } = notification.request.content;
    
    switch (actionIdentifier) {
      case 'view':
      case 'view_results':
        if (data.type === 'detection' && data.detectionId) {
          navigation.navigate('DetectionDetail', { id: data.detectionId });
        } else if (data.type === 'prediction') {
          navigation.navigate('Predictions');
        }
        break;
        
      case 'dismiss':
        // Just dismiss, no action needed
        break;
        
      default:
        // Default tap action
        if (data.type === 'detection') {
          navigation.navigate('DetectionList');
        } else if (data.type === 'prediction') {
          navigation.navigate('Predictions');
        } else {
          navigation.navigate('Home');
        }
        break;
    }
  });

  // Handle notifications received while app is in foreground
  const foregroundListener = Notifications.addNotificationReceivedListener(notification => {
    console.log('Notification received in foreground:', notification);
    // You can show in-app notification here if desired
  });

  return {
    notificationListener,
    foregroundListener,
  };
}

// Clean up listeners
export function removeNotificationListeners(listeners) {
  if (listeners.notificationListener) {
    Notifications.removeNotificationSubscription(listeners.notificationListener);
  }
  if (listeners.foregroundListener) {
    Notifications.removeNotificationSubscription(listeners.foregroundListener);
  }
}

// Badge management
export async function updateNotificationBadge(count = 0) {
  try {
    await Notifications.setBadgeCountAsync(count);
  } catch (error) {
    console.error('Failed to update badge count:', error);
  }
}

export async function clearNotificationBadge() {
  await updateNotificationBadge(0);
}

// Get notification history
export async function getNotificationHistory() {
  try {
    const stored = await AsyncStorage.getItem('notificationHistory');
    return stored ? JSON.parse(stored) : [];
  } catch (error) {
    console.error('Failed to get notification history:', error);
    return [];
  }
}

// Store notification in history
export async function storeNotificationInHistory(notification) {
  try {
    const history = await getNotificationHistory();
    const newEntry = {
      id: Date.now() + Math.random(),
      ...notification,
      receivedAt: new Date().toISOString(),
      read: false,
    };
    
    history.unshift(newEntry);
    
    // Keep only last 100 notifications
    const trimmed = history.slice(0, 100);
    
    await AsyncStorage.setItem('notificationHistory', JSON.stringify(trimmed));
  } catch (error) {
    console.error('Failed to store notification in history:', error);
  }
}

// Initialize notification service
export async function initializeNotificationService() {
  try {
    await loadNotificationPreferences();
    const pushToken = await registerForPushNotificationsAsync();
    
    console.log('Notification service initialized');
    return { pushToken, preferences: notificationPreferences };
  } catch (error) {
    console.error('Failed to initialize notification service:', error);
    return { pushToken: null, preferences: notificationPreferences };
  }
}
