// Enhanced API service with offline support for NightScan mobile app
import AsyncStorage from '@react-native-async-storage/async-storage';
import NetInfo from '@react-native-community/netinfo';

export const BASE_URL = 'http://localhost:8000';
export const API_BASE_URL = 'http://localhost:8000/api/v1';
export const PREDICT_API_URL = process.env.PREDICT_API_URL || 'http://localhost:8001/api/v1/predict';

const CACHE_PREFIX = 'nightscan_cache_';
const SYNC_QUEUE_KEY = 'nightscan_sync_queue';
const LAST_SYNC_KEY = 'nightscan_last_sync';

class ApiService {
  constructor() {
    this.isOnline = true;
    this.syncQueue = [];
    this.initializeNetworkListener();
    this.loadSyncQueue();
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

  async addToSyncQueue(request) {
    const queueItem = {
      id: Date.now() + Math.random(),
      timestamp: new Date().toISOString(),
      ...request
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
        await this.executeRequest(item);
        processedItems.push(item.id);
      } catch (error) {
        console.error('Failed to sync item:', item, error);
        // Keep item in queue for retry
      }
    }

    // Remove successfully processed items
    this.syncQueue = this.syncQueue.filter(item => !processedItems.includes(item.id));
    await this.saveSyncQueue();

    if (processedItems.length > 0) {
      await AsyncStorage.setItem(LAST_SYNC_KEY, new Date().toISOString());
    }
  }

  async executeRequest(request) {
    const { method, endpoint, data, formData, useAuth } = request;
    const url = endpoint.startsWith('http') ? endpoint : `${API_BASE_URL}${endpoint}`;
    
    const options = {
      method: method || 'GET',
      headers: {
        'Content-Type': formData ? undefined : 'application/json',
      },
      credentials: useAuth ? 'include' : 'omit',
    };

    if (data && !formData) {
      options.body = JSON.stringify(data);
    } else if (formData) {
      options.body = formData;
    }

    const response = await fetch(url, options);
    
    if (!response.ok) {
      throw new Error(`Request failed: ${response.status} ${response.statusText}`);
    }
    
    return await response.json();
  }

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
        payload: data
      };
      await AsyncStorage.setItem(CACHE_PREFIX + key, JSON.stringify(cacheData));
    } catch (error) {
      console.error('Cache write error:', error);
    }
  }

  // Legacy API functions with offline support
  async login(username, password) {
    if (!this.isOnline) {
      throw new Error('Login requires internet connection');
    }

    const resp = await fetch(`${BASE_URL}/login`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/x-www-form-urlencoded',
      },
      body: `username=${encodeURIComponent(username)}&password=${encodeURIComponent(password)}`,
      credentials: 'include',
    });
    
    if (!resp.ok) {
      throw new Error('Login failed');
    }
  }

  async register(username, password) {
    if (!this.isOnline) {
      throw new Error('Registration requires internet connection');
    }

    const resp = await fetch(`${BASE_URL}/register`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/x-www-form-urlencoded',
      },
      body: `username=${encodeURIComponent(username)}&password=${encodeURIComponent(password)}`,
      credentials: 'include',
    });
    
    if (!resp.ok) {
      throw new Error('Registration failed');
    }
  }

  async fetchDetections(page = 1, species = null) {
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
      const params = new URLSearchParams({ page: page.toString() });
      if (species) {
        params.append('species', species);
      }
      
      const resp = await fetch(`${API_BASE_URL}/detections?${params}`, {
        credentials: 'include',
      });
      
      if (!resp.ok) {
        throw new Error('Failed to fetch detections');
      }
      
      const data = await resp.json();
      
      // Cache the response
      await this.setCachedData(cacheKey, data);
      
      return { ...data, fromCache: false };
    } catch (error) {
      // Fallback to cache if request fails
      const cached = await this.getCachedData(cacheKey);
      if (cached) {
        return { ...cached, fromCache: true };
      }
      
      console.error('API Error:', error);
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
        method: 'POST',
        endpoint: PREDICT_API_URL,
        formData: form,
        useAuth: true
      });
      
      return {
        queued: true,
        message: 'Upload queued for when connection is restored'
      };
    }

    try {
      const resp = await fetch(PREDICT_API_URL, {
        method: 'POST',
        body: form,
        credentials: 'include',
      });

      if (!resp.ok) {
        const errorData = await resp.json();
        throw new Error(errorData.error || 'Upload failed');
      }

      const result = await resp.json();
      
      // Cache successful prediction
      const cacheKey = `prediction_${uri.split('/').pop()}_${Date.now()}`;
      await this.setCachedData(cacheKey, result);
      
      return result;
    } catch (error) {
      // Queue for retry if it's a network error
      if (!error.message.includes('400') && !error.message.includes('413')) {
        await this.addToSyncQueue({
          method: 'POST',
          endpoint: PREDICT_API_URL,
          formData: form,
          useAuth: true
        });
      }
      
      console.error('Upload Error:', error);
      throw error;
    }
  }

  async getHealth() {
    try {
      const response = await fetch(`${API_BASE_URL}/health`);
      return await response.json();
    } catch (error) {
      return { status: 'unreachable', error: error.message };
    }
  }

  async getSyncStatus() {
    const lastSync = await AsyncStorage.getItem(LAST_SYNC_KEY);
    return {
      isOnline: this.isOnline,
      queuedItems: this.syncQueue.length,
      lastSync: lastSync ? new Date(lastSync) : null
    };
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

  async forceSyncNow() {
    if (this.isOnline) {
      await this.processSyncQueue();
    } else {
      throw new Error('Cannot sync while offline');
    }
  }
}

const apiService = new ApiService();

// Export legacy functions for backward compatibility
export async function login(username, password) {
  return apiService.login(username, password);
}

export async function register(username, password) {
  return apiService.register(username, password);
}

export async function fetchDetections(page, species) {
  return apiService.fetchDetections(page, species);
}

export async function uploadMedia(uri, mimeType) {
  return apiService.uploadMedia(uri, mimeType);
}

// Export the enhanced service
export default apiService;