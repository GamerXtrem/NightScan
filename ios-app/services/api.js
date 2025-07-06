// Enhanced API service with offline support for NightScan mobile app
import AsyncStorage from '@react-native-async-storage/async-storage';
import NetInfo from '@react-native-community/netinfo';

export const BASE_URL = 'http://localhost:8000';
export const API_BASE_URL = 'http://localhost:8000/api/v1';
export const PREDICT_API_URL = process.env.PREDICT_API_URL || 'http://localhost:8001/api/v1/predict';
export const PI_SERVICE_URL = 'http://192.168.4.1:5000';

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

  // Camera preview functions
  async getCameraStatus() {
    try {
      const response = await fetch(`${PI_SERVICE_URL}/camera/status`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
      });
      
      if (!response.ok) {
        throw new Error(`Camera status request failed: ${response.status}`);
      }
      
      return await response.json();
    } catch (error) {
      console.error('Camera status error:', error);
      throw error;
    }
  }

  async startCameraPreview() {
    try {
      const response = await fetch(`${PI_SERVICE_URL}/camera/preview/start`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Failed to start camera preview');
      }
      
      return await response.json();
    } catch (error) {
      console.error('Start camera preview error:', error);
      throw error;
    }
  }

  async stopCameraPreview() {
    try {
      const response = await fetch(`${PI_SERVICE_URL}/camera/preview/stop`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Failed to stop camera preview');
      }
      
      return await response.json();
    } catch (error) {
      console.error('Stop camera preview error:', error);
      throw error;
    }
  }

  async captureImage() {
    try {
      const response = await fetch(`${PI_SERVICE_URL}/camera/capture`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Failed to capture image');
      }
      
      return await response.json();
    } catch (error) {
      console.error('Capture image error:', error);
      throw error;
    }
  }

  async getPiHealth() {
    try {
      const response = await fetch(`${PI_SERVICE_URL}/health`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
      });
      
      if (!response.ok) {
        throw new Error(`Pi health check failed: ${response.status}`);
      }
      
      return await response.json();
    } catch (error) {
      console.error('Pi health check error:', error);
      throw error;
    }
  }

  // Audio threshold functions
  async getAudioThresholdStatus() {
    try {
      const response = await fetch(`${PI_SERVICE_URL}/audio/threshold/status`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
      });
      
      if (!response.ok) {
        throw new Error(`Audio threshold status request failed: ${response.status}`);
      }
      
      return await response.json();
    } catch (error) {
      console.error('Audio threshold status error:', error);
      throw error;
    }
  }

  async updateAudioThresholdConfig(config) {
    try {
      const response = await fetch(`${PI_SERVICE_URL}/audio/threshold/config`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(config),
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Failed to update audio threshold config');
      }
      
      return await response.json();
    } catch (error) {
      console.error('Update audio threshold config error:', error);
      throw error;
    }
  }

  async applyAudioThresholdPreset(presetName) {
    try {
      const response = await fetch(`${PI_SERVICE_URL}/audio/threshold/preset/${presetName}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Failed to apply audio threshold preset');
      }
      
      return await response.json();
    } catch (error) {
      console.error('Apply audio threshold preset error:', error);
      throw error;
    }
  }

  async calibrateAudioThreshold(durationSeconds = 5) {
    try {
      const response = await fetch(`${PI_SERVICE_URL}/audio/threshold/calibrate`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ duration_seconds: durationSeconds }),
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Failed to calibrate audio threshold');
      }
      
      return await response.json();
    } catch (error) {
      console.error('Calibrate audio threshold error:', error);
      throw error;
    }
  }

  async testAudioThreshold(durationSeconds = 3) {
    try {
      const response = await fetch(`${PI_SERVICE_URL}/audio/threshold/test`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ duration_seconds: durationSeconds }),
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Failed to test audio threshold');
      }
      
      return await response.json();
    } catch (error) {
      console.error('Test audio threshold error:', error);
      throw error;
    }
  }

  async getAudioThresholdPresets() {
    try {
      const response = await fetch(`${PI_SERVICE_URL}/audio/threshold/presets`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
      });
      
      if (!response.ok) {
        throw new Error(`Audio threshold presets request failed: ${response.status}`);
      }
      
      return await response.json();
    } catch (error) {
      console.error('Audio threshold presets error:', error);
      throw error;
    }
  }

  async getLiveAudioLevels() {
    try {
      const response = await fetch(`${PI_SERVICE_URL}/audio/threshold/live`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
      });
      
      if (!response.ok) {
        throw new Error(`Live audio levels request failed: ${response.status}`);
      }
      
      return await response.json();
    } catch (error) {
      console.error('Live audio levels error:', error);
      throw error;
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

// Export camera preview functions
export async function getCameraStatus() {
  return apiService.getCameraStatus();
}

export async function startCameraPreview() {
  return apiService.startCameraPreview();
}

export async function stopCameraPreview() {
  return apiService.stopCameraPreview();
}

export async function captureImage() {
  return apiService.captureImage();
}

export async function getPiHealth() {
  return apiService.getPiHealth();
}

// Export audio threshold functions
export async function getAudioThresholdStatus() {
  return apiService.getAudioThresholdStatus();
}

export async function updateAudioThresholdConfig(config) {
  return apiService.updateAudioThresholdConfig(config);
}

export async function applyAudioThresholdPreset(presetName) {
  return apiService.applyAudioThresholdPreset(presetName);
}

export async function calibrateAudioThreshold(durationSeconds) {
  return apiService.calibrateAudioThreshold(durationSeconds);
}

export async function testAudioThreshold(durationSeconds) {
  return apiService.testAudioThreshold(durationSeconds);
}

export async function getAudioThresholdPresets() {
  return apiService.getAudioThresholdPresets();
}

export async function getLiveAudioLevels() {
  return apiService.getLiveAudioLevels();
}

// Export the enhanced service
export default apiService;