// API Gateway service with JWT authentication for NightScan mobile app
import AsyncStorage from '@react-native-async-storage/async-storage';
import NetInfo from '@react-native-community/netinfo';

// Configuration - Use environment variables or config file
const config = {
  GATEWAY_URL: process.env.API_GATEWAY_URL || 'http://localhost:8080',
  PI_SERVICE_URL: 'http://192.168.4.1:5000',
  PI_LOCATION_URL: 'http://192.168.4.1:5001',
};

// Storage keys
const TOKEN_KEY = 'nightscan_access_token';
const REFRESH_TOKEN_KEY = 'nightscan_refresh_token';
const USER_KEY = 'nightscan_user';
const CACHE_PREFIX = 'nightscan_cache_';
const SYNC_QUEUE_KEY = 'nightscan_sync_queue';
const LAST_SYNC_KEY = 'nightscan_last_sync';

class APIGatewayService {
  constructor() {
    this.accessToken = null;
    this.refreshToken = null;
    this.user = null;
    this.isOnline = true;
    this.syncQueue = [];
    this.tokenRefreshPromise = null;
    
    this.initializeService();
  }

  async initializeService() {
    // Load stored tokens
    await this.loadStoredAuth();
    
    // Initialize network listener
    this.initializeNetworkListener();
    
    // Load sync queue
    await this.loadSyncQueue();
  }

  async loadStoredAuth() {
    try {
      const [accessToken, refreshToken, userStr] = await Promise.all([
        AsyncStorage.getItem(TOKEN_KEY),
        AsyncStorage.getItem(REFRESH_TOKEN_KEY),
        AsyncStorage.getItem(USER_KEY),
      ]);
      
      this.accessToken = accessToken;
      this.refreshToken = refreshToken;
      this.user = userStr ? JSON.parse(userStr) : null;
    } catch (error) {
      console.error('Failed to load stored auth:', error);
    }
  }

  async saveAuth(accessToken, refreshToken, user) {
    this.accessToken = accessToken;
    this.refreshToken = refreshToken;
    this.user = user;
    
    try {
      await Promise.all([
        AsyncStorage.setItem(TOKEN_KEY, accessToken),
        AsyncStorage.setItem(REFRESH_TOKEN_KEY, refreshToken),
        AsyncStorage.setItem(USER_KEY, JSON.stringify(user)),
      ]);
    } catch (error) {
      console.error('Failed to save auth:', error);
    }
  }

  async clearAuth() {
    this.accessToken = null;
    this.refreshToken = null;
    this.user = null;
    
    try {
      await Promise.all([
        AsyncStorage.removeItem(TOKEN_KEY),
        AsyncStorage.removeItem(REFRESH_TOKEN_KEY),
        AsyncStorage.removeItem(USER_KEY),
      ]);
    } catch (error) {
      console.error('Failed to clear auth:', error);
    }
  }

  initializeNetworkListener() {
    NetInfo.addEventListener(state => {
      const wasOffline = !this.isOnline;
      this.isOnline = state.isConnected;
      
      // If we just came back online, sync pending changes
      if (wasOffline && this.isOnline) {
        this.processSyncQueue();
      }
    });
  }

  async loadSyncQueue() {
    try {
      const queueData = await AsyncStorage.getItem(SYNC_QUEUE_KEY);
      this.syncQueue = queueData ? JSON.parse(queueData) : [];
    } catch (error) {
      console.error('Failed to load sync queue:', error);
      this.syncQueue = [];
    }
  }

  async saveSyncQueue() {
    try {
      await AsyncStorage.setItem(SYNC_QUEUE_KEY, JSON.stringify(this.syncQueue));
    } catch (error) {
      console.error('Failed to save sync queue:', error);
    }
  }

  async makeRequest(endpoint, options = {}) {
    const url = endpoint.startsWith('http') ? endpoint : `${config.GATEWAY_URL}${endpoint}`;
    const { skipAuth = false, ...requestOptions } = options;
    
    // Add auth header if we have a token and not skipping auth
    if (this.accessToken && !skipAuth) {
      requestOptions.headers = {
        ...requestOptions.headers,
        'Authorization': `Bearer ${this.accessToken}`,
      };
    }
    
    try {
      const response = await fetch(url, requestOptions);
      
      // Handle token expiration
      if (response.status === 401 && !skipAuth && this.refreshToken) {
        // Try to refresh the token
        const refreshed = await this.refreshAccessToken();
        if (refreshed) {
          // Retry the request with new token
          requestOptions.headers['Authorization'] = `Bearer ${this.accessToken}`;
          return fetch(url, requestOptions);
        }
      }
      
      return response;
    } catch (error) {
      // Network error - queue for later if applicable
      if (!this.isOnline && options.method !== 'GET') {
        await this.addToSyncQueue({
          endpoint,
          options: requestOptions,
        });
      }
      throw error;
    }
  }

  async refreshAccessToken() {
    // Prevent multiple simultaneous refresh attempts
    if (this.tokenRefreshPromise) {
      return this.tokenRefreshPromise;
    }
    
    this.tokenRefreshPromise = this._doRefreshToken();
    const result = await this.tokenRefreshPromise;
    this.tokenRefreshPromise = null;
    
    return result;
  }

  async _doRefreshToken() {
    if (!this.refreshToken) {
      return false;
    }
    
    try {
      const response = await fetch(`${config.GATEWAY_URL}/api/auth/refresh`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          refreshToken: this.refreshToken,
        }),
      });
      
      if (!response.ok) {
        // Refresh failed - clear auth
        await this.clearAuth();
        return false;
      }
      
      const data = await response.json();
      if (data.success && data.data.accessToken) {
        this.accessToken = data.data.accessToken;
        await AsyncStorage.setItem(TOKEN_KEY, this.accessToken);
        return true;
      }
      
      return false;
    } catch (error) {
      console.error('Token refresh error:', error);
      return false;
    }
  }

  async addToSyncQueue(request) {
    const queueItem = {
      id: Date.now() + Math.random(),
      timestamp: new Date().toISOString(),
      ...request,
    };
    
    this.syncQueue.push(queueItem);
    await this.saveSyncQueue();
    
    // Try to process immediately if online
    if (this.isOnline) {
      await this.processSyncQueue();
    }
  }

  async processSyncQueue() {
    if (!this.isOnline || this.syncQueue.length === 0) {
      return;
    }

    const processedItems = [];
    
    for (const item of this.syncQueue) {
      try {
        const response = await this.makeRequest(item.endpoint, item.options);
        if (response.ok) {
          processedItems.push(item.id);
        }
      } catch (error) {
        console.error('Failed to sync item:', item, error);
      }
    }

    // Remove successfully processed items
    this.syncQueue = this.syncQueue.filter(item => !processedItems.includes(item.id));
    await this.saveSyncQueue();

    if (processedItems.length > 0) {
      await AsyncStorage.setItem(LAST_SYNC_KEY, new Date().toISOString());
    }
  }

  // Authentication methods
  async login(username, password) {
    if (!this.isOnline) {
      throw new Error('Login requires internet connection');
    }

    const response = await this.makeRequest('/api/auth/login', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ username, password }),
      skipAuth: true,
    });
    
    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.error || 'Login failed');
    }
    
    const data = await response.json();
    if (data.success && data.data) {
      await this.saveAuth(
        data.data.accessToken,
        data.data.refreshToken,
        data.data.user
      );
      return data.data;
    }
    
    throw new Error('Invalid login response');
  }

  async register(username, password, email) {
    if (!this.isOnline) {
      throw new Error('Registration requires internet connection');
    }

    const response = await this.makeRequest('/api/auth/register', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ username, password, email }),
      skipAuth: true,
    });
    
    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.error || 'Registration failed');
    }
    
    const data = await response.json();
    if (data.success && data.data) {
      await this.saveAuth(
        data.data.accessToken,
        data.data.refreshToken,
        data.data.user
      );
      return data.data;
    }
    
    throw new Error('Invalid registration response');
  }

  async logout() {
    try {
      if (this.accessToken) {
        await this.makeRequest('/api/auth/logout', {
          method: 'POST',
        });
      }
    } catch (error) {
      console.error('Logout error:', error);
    } finally {
      await this.clearAuth();
    }
  }

  async verifyToken() {
    if (!this.accessToken) {
      return false;
    }
    
    try {
      const response = await this.makeRequest('/api/auth/verify', {
        method: 'GET',
      });
      
      return response.ok;
    } catch (error) {
      return false;
    }
  }

  // API methods
  async fetchDetections(page = 1, species = null) {
    const params = new URLSearchParams({ page: page.toString() });
    if (species) {
      params.append('species', species);
    }
    
    const cacheKey = `detections_${page}_${species || 'all'}`;
    
    // Try cache first if offline
    if (!this.isOnline) {
      const cached = await this.getCachedData(cacheKey);
      if (cached) {
        return { ...cached, fromCache: true };
      }
      throw new Error('No internet connection and no cached data available');
    }
    
    try {
      const response = await this.makeRequest(`/api/v1/detections?${params}`);
      
      if (!response.ok) {
        throw new Error('Failed to fetch detections');
      }
      
      const data = await response.json();
      
      // Cache the response
      await this.setCachedData(cacheKey, data);
      
      return { ...data, fromCache: false };
    } catch (error) {
      // Fallback to cache if request fails
      const cached = await this.getCachedData(cacheKey);
      if (cached) {
        return { ...cached, fromCache: true };
      }
      throw error;
    }
  }

  async uploadMedia(uri, mimeType = 'audio/wav') {
    const form = new FormData();
    form.append('file', {
      uri,
      type: mimeType,
      name: uri.split('/').pop() || 'upload.wav',
    });

    if (!this.isOnline) {
      // Queue for later sync
      await this.addToSyncQueue({
        endpoint: '/api/predict',
        options: {
          method: 'POST',
          body: form,
        },
      });
      
      return {
        queued: true,
        message: 'Upload queued for when connection is restored',
      };
    }

    const response = await this.makeRequest('/api/predict', {
      method: 'POST',
      body: form,
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.error || 'Upload failed');
    }

    const result = await response.json();
    
    // Cache successful prediction
    const cacheKey = `prediction_${uri.split('/').pop()}_${Date.now()}`;
    await this.setCachedData(cacheKey, result);
    
    return result;
  }

  async getQuotaStatus() {
    const response = await this.makeRequest('/api/v1/quota/status');
    
    if (!response.ok) {
      throw new Error('Failed to fetch quota status');
    }
    
    return await response.json();
  }

  async getAnalytics(days = 30) {
    const response = await this.makeRequest(`/api/analytics/metrics?days=${days}`);
    
    if (!response.ok) {
      throw new Error('Failed to fetch analytics');
    }
    
    return await response.json();
  }

  // Cache management
  async getCachedData(key) {
    try {
      const cached = await AsyncStorage.getItem(CACHE_PREFIX + key);
      if (cached) {
        const data = JSON.parse(cached);
        // Check if cache is still valid (24 hours)
        const cacheTime = new Date(data.timestamp);
        const now = new Date();
        const hoursDiff = (now - cacheTime) / (1000 * 60 * 60);
        
        if (hoursDiff < 24) {
          return data.payload;
        }
      }
    } catch (error) {
      console.error('Cache read error:', error);
    }
    return null;
  }

  async setCachedData(key, data) {
    try {
      const cacheData = {
        timestamp: new Date().toISOString(),
        payload: data,
      };
      await AsyncStorage.setItem(CACHE_PREFIX + key, JSON.stringify(cacheData));
    } catch (error) {
      console.error('Cache write error:', error);
    }
  }

  async clearCache() {
    try {
      const keys = await AsyncStorage.getAllKeys();
      const cacheKeys = keys.filter(key => key.startsWith(CACHE_PREFIX));
      await AsyncStorage.multiRemove(cacheKeys);
    } catch (error) {
      console.error('Failed to clear cache:', error);
    }
  }

  // Status methods
  async getSyncStatus() {
    const lastSync = await AsyncStorage.getItem(LAST_SYNC_KEY);
    return {
      isOnline: this.isOnline,
      isAuthenticated: !!this.accessToken,
      user: this.user,
      queuedItems: this.syncQueue.length,
      lastSync: lastSync ? new Date(lastSync) : null,
    };
  }

  async forceSyncNow() {
    if (this.isOnline) {
      await this.processSyncQueue();
    } else {
      throw new Error('Cannot sync while offline');
    }
  }

  // Pi-specific functions (direct connection, no gateway)
  async getCameraStatus() {
    const response = await fetch(`${config.PI_SERVICE_URL}/camera/status`);
    if (!response.ok) {
      throw new Error(`Camera status request failed: ${response.status}`);
    }
    return await response.json();
  }

  async startCameraPreview() {
    const response = await fetch(`${config.PI_SERVICE_URL}/camera/preview/start`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
    });
    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.error || 'Failed to start camera preview');
    }
    return await response.json();
  }

  async stopCameraPreview() {
    const response = await fetch(`${config.PI_SERVICE_URL}/camera/preview/stop`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
    });
    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.error || 'Failed to stop camera preview');
    }
    return await response.json();
  }

  async captureImage() {
    const response = await fetch(`${config.PI_SERVICE_URL}/camera/capture`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
    });
    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.error || 'Failed to capture image');
    }
    return await response.json();
  }

  async getPiHealth() {
    const response = await fetch(`${config.PI_SERVICE_URL}/health`);
    if (!response.ok) {
      throw new Error(`Pi health check failed: ${response.status}`);
    }
    return await response.json();
  }

  // Location management (Pi direct)
  async getLocationStatus() {
    const response = await fetch(`${config.PI_LOCATION_URL}/api/location/status`);
    if (!response.ok) {
      throw new Error(`Location status request failed: ${response.status}`);
    }
    return await response.json();
  }

  async getCurrentLocation() {
    const response = await fetch(`${config.PI_LOCATION_URL}/api/location`);
    if (!response.ok) {
      throw new Error(`Get location request failed: ${response.status}`);
    }
    return await response.json();
  }

  async updatePiLocation(locationData) {
    const response = await fetch(`${config.PI_LOCATION_URL}/api/location`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(locationData),
    });
    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.error || 'Failed to update Pi location');
    }
    return await response.json();
  }
}

// Create singleton instance
const apiGateway = new APIGatewayService();

// Export modern API
export default apiGateway;

// Export named functions for backward compatibility
export const login = (username, password) => apiGateway.login(username, password);
export const register = (username, password, email) => apiGateway.register(username, password, email);
export const logout = () => apiGateway.logout();
export const fetchDetections = (page, species) => apiGateway.fetchDetections(page, species);
export const uploadMedia = (uri, mimeType) => apiGateway.uploadMedia(uri, mimeType);
export const getCameraStatus = () => apiGateway.getCameraStatus();
export const startCameraPreview = () => apiGateway.startCameraPreview();
export const stopCameraPreview = () => apiGateway.stopCameraPreview();
export const captureImage = () => apiGateway.captureImage();
export const getPiHealth = () => apiGateway.getPiHealth();
export const getLocationStatus = () => apiGateway.getLocationStatus();
export const getCurrentLocation = () => apiGateway.getCurrentLocation();
export const updatePiLocation = (data) => apiGateway.updatePiLocation(data);

// Export new functions
export const getQuotaStatus = () => apiGateway.getQuotaStatus();
export const getAnalytics = (days) => apiGateway.getAnalytics(days);
export const getSyncStatus = () => apiGateway.getSyncStatus();
export const clearCache = () => apiGateway.clearCache();
export const forceSyncNow = () => apiGateway.forceSyncNow();
export const verifyToken = () => apiGateway.verifyToken();