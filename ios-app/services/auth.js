import AsyncStorage from '@react-native-async-storage/async-storage';
import { Platform } from 'react-native';
import NetInfo from '@react-native-community/netinfo';

const AUTH_STORAGE_KEY = 'nightscan_auth';
const USER_PIN_KEY = 'nightscan_user_pin';
const DEVICE_MAC_KEY = 'nightscan_device_mac';

class AuthService {
  constructor() {
    this.authState = {
      isAuthenticated: false,
      isOwner: false,
      deviceMAC: null,
      piMAC: null,
      userPIN: null,
    };
    this.initializeAuth();
  }

  async initializeAuth() {
    try {
      // Load stored auth data
      const storedAuth = await AsyncStorage.getItem(AUTH_STORAGE_KEY);
      if (storedAuth) {
        this.authState = { ...this.authState, ...JSON.parse(storedAuth) };
      }

      // Get device MAC address
      const deviceMAC = await this.getDeviceMAC();
      this.authState.deviceMAC = deviceMAC;

      console.log('Auth service initialized:', this.authState);
    } catch (error) {
      console.error('Failed to initialize auth service:', error);
    }
  }

  /**
   * Get device MAC address (simulated for React Native)
   * In a real implementation, you'd use a native module
   */
  async getDeviceMAC() {
    try {
      // For development, generate a consistent MAC based on device info
      const deviceInfo = await this.getDeviceInfo();
      const hash = this.simpleHash(deviceInfo);
      const mac = this.formatAsMAC(hash);
      
      console.log('Device MAC:', mac);
      return mac;
    } catch (error) {
      console.error('Failed to get device MAC:', error);
      return this.generateFallbackMAC();
    }
  }

  async getDeviceInfo() {
    const netInfo = await NetInfo.fetch();
    return `${Platform.OS}-${Platform.Version}-${netInfo.type}`;
  }

  simpleHash(str) {
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
      const char = str.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash = hash & hash; // Convert to 32bit integer
    }
    return Math.abs(hash);
  }

  formatAsMAC(hash) {
    const hex = hash.toString(16).padStart(12, '0').slice(-12);
    return hex.match(/.{2}/g).join(':').toUpperCase();
  }

  generateFallbackMAC() {
    const randomHex = () => Math.floor(Math.random() * 256).toString(16).padStart(2, '0');
    return Array.from({ length: 6 }, randomHex).join(':').toUpperCase();
  }

  /**
   * Hash PIN for secure storage
   */
  hashPIN(pin) {
    return this.simpleHash(pin + 'nightscan_salt').toString();
  }

  /**
   * Check if this is the first time connecting to Pi
   */
  async checkPairingStatus() {
    try {
      const response = await fetch('http://192.168.4.1:5000/api/auth/status', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ deviceMAC: this.authState.deviceMAC }),
      });

      if (!response.ok) {
        throw new Error('Pi not reachable');
      }

      const data = await response.json();
      return {
        requiresPairing: data.requiresPairing,
        isOwner: data.isOwner,
        requiresPIN: data.requiresPIN,
      };
    } catch (error) {
      console.error('Failed to check pairing status:', error);
      // If Pi is not reachable, assume it needs pairing
      return { requiresPairing: true, isOwner: false, requiresPIN: false };
    }
  }

  /**
   * Perform initial pairing with PIN setup
   */
  async performPairing(userPIN) {
    try {
      const hashedPIN = this.hashPIN(userPIN);
      
      const response = await fetch('http://192.168.4.1:5000/api/auth/pair', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          deviceMAC: this.authState.deviceMAC,
          hashedPIN: hashedPIN,
        }),
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.message || 'Pairing failed');
      }

      const data = await response.json();
      
      // Update auth state
      this.authState = {
        ...this.authState,
        isAuthenticated: true,
        isOwner: true,
        userPIN: hashedPIN,
        piMAC: data.piMAC,
      };

      // Store auth data
      await this.saveAuthState();
      
      return { success: true, message: 'Pairing réussi' };
    } catch (error) {
      console.error('Pairing failed:', error);
      return { success: false, message: error.message };
    }
  }

  /**
   * Authenticate with PIN for unknown MAC
   */
  async authenticateWithPIN(pin) {
    try {
      const hashedPIN = this.hashPIN(pin);
      
      const response = await fetch('http://192.168.4.1:5000/api/auth/authenticate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          deviceMAC: this.authState.deviceMAC,
          hashedPIN: hashedPIN,
        }),
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.message || 'Authentification échouée');
      }

      const data = await response.json();
      
      // Update auth state
      this.authState = {
        ...this.authState,
        isAuthenticated: true,
        isOwner: data.isOwner,
        userPIN: hashedPIN,
      };

      await this.saveAuthState();
      
      return { success: true, message: 'Authentification réussie' };
    } catch (error) {
      console.error('Authentication failed:', error);
      return { success: false, message: error.message };
    }
  }

  /**
   * Add a new user by MAC address (owner only)
   */
  async addUserByMAC(macAddress, description = '') {
    try {
      if (!this.authState.isOwner) {
        throw new Error('Seul le propriétaire peut ajouter des utilisateurs');
      }

      const response = await fetch('http://192.168.4.1:5000/api/auth/add-user', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          ownerMAC: this.authState.deviceMAC,
          newUserMAC: macAddress.toUpperCase(),
          description: description,
          ownerPIN: this.authState.userPIN,
        }),
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.message || 'Ajout utilisateur échoué');
      }

      return { success: true, message: 'Utilisateur ajouté avec succès' };
    } catch (error) {
      console.error('Add user failed:', error);
      return { success: false, message: error.message };
    }
  }

  /**
   * Reset Pi to factory state (owner only)
   */
  async resetPi(pin) {
    try {
      const hashedPIN = this.hashPIN(pin);
      
      if (hashedPIN !== this.authState.userPIN) {
        throw new Error('PIN incorrect');
      }

      const response = await fetch('http://192.168.4.1:5000/api/auth/reset', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          ownerMAC: this.authState.deviceMAC,
          hashedPIN: hashedPIN,
        }),
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.message || 'Réinitialisation échouée');
      }

      // Clear local auth state
      await this.clearAuthState();
      
      return { success: true, message: 'Pi réinitialisé avec succès' };
    } catch (error) {
      console.error('Reset failed:', error);
      return { success: false, message: error.message };
    }
  }

  /**
   * Get list of authorized users (owner only)
   */
  async getAuthorizedUsers() {
    try {
      if (!this.authState.isOwner) {
        throw new Error('Seul le propriétaire peut voir les utilisateurs');
      }

      const response = await fetch('http://192.168.4.1:5000/api/auth/users', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          ownerMAC: this.authState.deviceMAC,
          ownerPIN: this.authState.userPIN,
        }),
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.message || 'Récupération utilisateurs échouée');
      }

      const data = await response.json();
      return { success: true, users: data.users };
    } catch (error) {
      console.error('Get users failed:', error);
      return { success: false, message: error.message, users: [] };
    }
  }

  /**
   * Remove user (owner only)
   */
  async removeUser(macAddress) {
    try {
      if (!this.authState.isOwner) {
        throw new Error('Seul le propriétaire peut supprimer des utilisateurs');
      }

      const response = await fetch('http://192.168.4.1:5000/api/auth/remove-user', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          ownerMAC: this.authState.deviceMAC,
          userMAC: macAddress,
          ownerPIN: this.authState.userPIN,
        }),
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.message || 'Suppression utilisateur échouée');
      }

      return { success: true, message: 'Utilisateur supprimé avec succès' };
    } catch (error) {
      console.error('Remove user failed:', error);
      return { success: false, message: error.message };
    }
  }

  async saveAuthState() {
    try {
      await AsyncStorage.setItem(AUTH_STORAGE_KEY, JSON.stringify(this.authState));
    } catch (error) {
      console.error('Failed to save auth state:', error);
    }
  }

  async clearAuthState() {
    try {
      await AsyncStorage.removeItem(AUTH_STORAGE_KEY);
      this.authState = {
        isAuthenticated: false,
        isOwner: false,
        deviceMAC: this.authState.deviceMAC, // Keep device MAC
        piMAC: null,
        userPIN: null,
      };
    } catch (error) {
      console.error('Failed to clear auth state:', error);
    }
  }

  // Getters for current state
  getAuthState() {
    return { ...this.authState };
  }

  isAuthenticated() {
    return this.authState.isAuthenticated;
  }

  isOwner() {
    return this.authState.isOwner;
  }

  getDeviceMACAddress() {
    return this.authState.deviceMAC;
  }
}

// Create singleton instance
const authService = new AuthService();

export default authService;