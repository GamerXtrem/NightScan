// WebSocket service for real-time notifications in NightScan mobile app
import { io } from 'socket.io-client';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { Alert } from 'react-native';

const WEBSOCKET_URL = process.env.WEBSOCKET_URL || 'ws://localhost:8000';
const RECONNECT_ATTEMPTS = 5;
const RECONNECT_DELAY = 1000; // 1 second

class WebSocketService {
  constructor() {
    this.socket = null;
    this.isConnected = false;
    this.reconnectAttempts = 0;
    this.eventListeners = new Map();
    this.userToken = null;
    this.userId = null;
    this.subscriptions = new Set();
  }

  async connect(userToken = null, userId = null) {
    try {
      this.userToken = userToken;
      this.userId = userId;

      this.socket = io(WEBSOCKET_URL, {
        transports: ['websocket'],
        autoConnect: true,
        reconnection: true,
        reconnectionAttempts: RECONNECT_ATTEMPTS,
        reconnectionDelay: RECONNECT_DELAY,
      });

      this._setupEventHandlers();
      
      return new Promise((resolve, reject) => {
        const timeout = setTimeout(() => {
          reject(new Error('Connection timeout'));
        }, 10000);

        this.socket.on('connect', () => {
          clearTimeout(timeout);
          this.isConnected = true;
          this.reconnectAttempts = 0;
          console.log('WebSocket connected');
          
          // Authenticate if we have credentials
          if (this.userToken && this.userId) {
            this.authenticate();
          }
          
          resolve();
        });

        this.socket.on('connect_error', (error) => {
          clearTimeout(timeout);
          console.error('WebSocket connection error:', error);
          reject(error);
        });
      });
    } catch (error) {
      console.error('Failed to connect WebSocket:', error);
      throw error;
    }
  }

  disconnect() {
    if (this.socket) {
      this.socket.disconnect();
      this.socket = null;
      this.isConnected = false;
      console.log('WebSocket disconnected');
    }
  }

  _setupEventHandlers() {
    this.socket.on('connect', () => {
      this.isConnected = true;
      this.reconnectAttempts = 0;
      console.log('WebSocket connected');
      this._emitToListeners('connected', {});
    });

    this.socket.on('disconnect', (reason) => {
      this.isConnected = false;
      console.log('WebSocket disconnected:', reason);
      this._emitToListeners('disconnected', { reason });
    });

    this.socket.on('reconnect', (attemptNumber) => {
      console.log('WebSocket reconnected after', attemptNumber, 'attempts');
      this._emitToListeners('reconnected', { attemptNumber });
      
      // Re-authenticate and re-subscribe
      if (this.userToken && this.userId) {
        this.authenticate();
      }
      this._resubscribeAll();
    });

    this.socket.on('reconnect_error', (error) => {
      this.reconnectAttempts++;
      console.error('WebSocket reconnection error:', error);
      this._emitToListeners('reconnect_error', { error, attempts: this.reconnectAttempts });
    });

    this.socket.on('authenticated', (data) => {
      console.log('WebSocket authenticated:', data);
      this._emitToListeners('authenticated', data);
      
      // Subscribe to default events
      this.subscribe(['new_detection', 'prediction_complete', 'system_status']);
    });

    this.socket.on('authentication_error', (data) => {
      console.error('WebSocket authentication error:', data);
      this._emitToListeners('authentication_error', data);
    });

    this.socket.on('notification', (data) => {
      console.log('Received notification:', data);
      this._handleNotification(data);
    });

    this.socket.on('subscribed', (data) => {
      console.log('Subscribed to events:', data.event_types);
      data.event_types.forEach(type => this.subscriptions.add(type));
    });

    this.socket.on('pong', (data) => {
      // Handle ping/pong for connection health
      this._emitToListeners('pong', data);
    });
  }

  authenticate() {
    if (this.socket && this.isConnected && this.userToken && this.userId) {
      this.socket.emit('authenticate', {
        user_id: this.userId,
        token: this.userToken
      });
    }
  }

  subscribe(eventTypes) {
    if (this.socket && this.isConnected) {
      this.socket.emit('subscribe', {
        event_types: eventTypes
      });
      eventTypes.forEach(type => this.subscriptions.add(type));
    }
  }

  unsubscribe(eventTypes) {
    if (this.socket && this.isConnected) {
      this.socket.emit('unsubscribe', {
        event_types: eventTypes
      });
      eventTypes.forEach(type => this.subscriptions.delete(type));
    }
  }

  _resubscribeAll() {
    if (this.subscriptions.size > 0) {
      this.subscribe(Array.from(this.subscriptions));
    }
  }

  ping() {
    if (this.socket && this.isConnected) {
      this.socket.emit('ping', { timestamp: new Date().toISOString() });
    }
  }

  _handleNotification(data) {
    const { event_type, data: notificationData, priority, timestamp } = data;

    // Store notification locally
    this._storeNotification(data);

    // Emit to registered listeners
    this._emitToListeners('notification', data);
    this._emitToListeners(`notification_${event_type}`, notificationData);

    // Handle different notification types
    switch (event_type) {
      case 'new_detection':
        this._handleNewDetection(notificationData);
        break;
      case 'prediction_complete':
        this._handlePredictionComplete(notificationData);
        break;
      case 'system_status':
        this._handleSystemStatus(notificationData);
        break;
      case 'error_notification':
        this._handleErrorNotification(notificationData);
        break;
    }

    // Show local notification if app is in background
    this._showLocalNotification(data);
  }

  _handleNewDetection(data) {
    console.log('New detection:', data);
    
    // Update local cache with new detection
    this._updateDetectionCache(data);

    // Show alert for high-priority detections
    if (data.confidence && data.confidence > 0.9) {
      Alert.alert(
        'New Wildlife Detection!',
        `${data.species} detected at ${data.zone || 'unknown location'}`,
        [
          { text: 'View', onPress: () => this._emitToListeners('view_detection', data) },
          { text: 'OK', style: 'default' }
        ]
      );
    }
  }

  _handlePredictionComplete(data) {
    console.log('Prediction complete:', data);
    
    // Update UI to show completed prediction
    this._emitToListeners('prediction_ready', data);
  }

  _handleSystemStatus(data) {
    console.log('System status update:', data);
    
    // Update system status in app state
    this._emitToListeners('system_status_update', data);
  }

  _handleErrorNotification(data) {
    console.error('System error notification:', data);
    
    // Show error alert
    Alert.alert(
      'System Alert',
      data.message || 'A system error occurred',
      [{ text: 'OK' }]
    );
  }

  async _storeNotification(notification) {
    try {
      const stored = await AsyncStorage.getItem('notifications');
      const notifications = stored ? JSON.parse(stored) : [];
      
      // Add new notification
      notifications.unshift({
        ...notification,
        id: Date.now() + Math.random(),
        read: false,
        received_at: new Date().toISOString()
      });
      
      // Keep only last 100 notifications
      const trimmed = notifications.slice(0, 100);
      
      await AsyncStorage.setItem('notifications', JSON.stringify(trimmed));
    } catch (error) {
      console.error('Failed to store notification:', error);
    }
  }

  async _updateDetectionCache(detection) {
    try {
      const stored = await AsyncStorage.getItem('nightscan_cache_detections_1_all');
      if (stored) {
        const data = JSON.parse(stored);
        if (data.payload && data.payload.detections) {
          // Add new detection to the beginning
          data.payload.detections.unshift(detection);
          
          // Update timestamp
          data.timestamp = new Date().toISOString();
          
          await AsyncStorage.setItem('nightscan_cache_detections_1_all', JSON.stringify(data));
        }
      }
    } catch (error) {
      console.error('Failed to update detection cache:', error);
    }
  }

  _showLocalNotification(data) {
    // This would integrate with React Native's push notification system
    // For now, we'll just log it
    console.log('Local notification:', {
      title: this._getNotificationTitle(data.event_type),
      body: this._getNotificationBody(data),
      data: data.data
    });
  }

  _getNotificationTitle(eventType) {
    switch (eventType) {
      case 'new_detection':
        return 'New Wildlife Detection!';
      case 'prediction_complete':
        return 'Analysis Complete';
      case 'system_status':
        return 'System Update';
      case 'error_notification':
        return 'System Alert';
      default:
        return 'NightScan Notification';
    }
  }

  _getNotificationBody(data) {
    const { event_type, data: notificationData } = data;
    
    switch (event_type) {
      case 'new_detection':
        return `${notificationData.species} detected at ${notificationData.zone || 'sensor location'}`;
      case 'prediction_complete':
        return `Analysis of ${notificationData.filename} is ready`;
      case 'system_status':
        return notificationData.message || 'System status updated';
      case 'error_notification':
        return notificationData.message || 'System error occurred';
      default:
        return 'New notification received';
    }
  }

  // Event listener management
  addEventListener(event, callback) {
    if (!this.eventListeners.has(event)) {
      this.eventListeners.set(event, new Set());
    }
    this.eventListeners.get(event).add(callback);
  }

  removeEventListener(event, callback) {
    if (this.eventListeners.has(event)) {
      this.eventListeners.get(event).delete(callback);
    }
  }

  _emitToListeners(event, data) {
    if (this.eventListeners.has(event)) {
      this.eventListeners.get(event).forEach(callback => {
        try {
          callback(data);
        } catch (error) {
          console.error(`Error in event listener for ${event}:`, error);
        }
      });
    }
  }

  // Utility methods
  getConnectionState() {
    return {
      connected: this.isConnected,
      authenticated: this.userId !== null,
      subscriptions: Array.from(this.subscriptions),
      reconnectAttempts: this.reconnectAttempts
    };
  }

  async getStoredNotifications() {
    try {
      const stored = await AsyncStorage.getItem('notifications');
      return stored ? JSON.parse(stored) : [];
    } catch (error) {
      console.error('Failed to get stored notifications:', error);
      return [];
    }
  }

  async markNotificationAsRead(notificationId) {
    try {
      const stored = await AsyncStorage.getItem('notifications');
      if (stored) {
        const notifications = JSON.parse(stored);
        const updated = notifications.map(n => 
          n.id === notificationId ? { ...n, read: true } : n
        );
        await AsyncStorage.setItem('notifications', JSON.stringify(updated));
      }
    } catch (error) {
      console.error('Failed to mark notification as read:', error);
    }
  }

  async clearNotifications() {
    try {
      await AsyncStorage.removeItem('notifications');
    } catch (error) {
      console.error('Failed to clear notifications:', error);
    }
  }
}

// Global WebSocket service instance
const websocketService = new WebSocketService();

export default websocketService;