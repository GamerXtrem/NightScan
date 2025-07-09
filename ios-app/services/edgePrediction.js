/**
 * Edge Prediction Service for NightScan
 * 
 * This service handles on-device inference with fallback to cloud prediction
 * when confidence is below threshold or when edge models are unavailable.
 */

import AsyncStorage from '@react-native-async-storage/async-storage';
import { Platform } from 'react-native';
import { uploadMedia } from './api';

// Configuration constants
const CONFIDENCE_THRESHOLD = 0.8; // 80% confidence threshold
const EDGE_MODELS_CACHE_KEY = 'nightscan_edge_models';
const EDGE_PREDICTIONS_CACHE_KEY = 'nightscan_edge_predictions';
const EDGE_STATS_KEY = 'nightscan_edge_stats';

class EdgePredictionService {
  constructor() {
    this.isInitialized = false;
    this.audioModel = null;
    this.photoModel = null;
    this.stats = {
      totalPredictions: 0,
      edgePredictions: 0,
      cloudPredictions: 0,
      averageConfidence: 0,
      averageEdgeTime: 0,
      averageCloudTime: 0
    };
    this.loadStats();
  }

  /**
   * Initialize edge prediction service
   * Downloads or loads lightweight models if available
   */
  async initialize() {
    if (this.isInitialized) return true;

    try {
      console.log('🔄 Initializing Edge Prediction Service...');
      
      // Check if TensorFlow Lite is available
      const tfliteAvailable = await this.checkTensorFlowLite();
      
      if (!tfliteAvailable) {
        console.log('⚠️ TensorFlow Lite not available, using cloud-only mode');
        this.isInitialized = true;
        return false;
      }

      // Load or download edge models
      await this.loadEdgeModels();
      
      this.isInitialized = true;
      console.log('✅ Edge Prediction Service initialized');
      return true;
    } catch (error) {
      console.error('❌ Failed to initialize Edge Prediction Service:', error);
      this.isInitialized = true; // Set to true to prevent retry loops
      return false;
    }
  }

  /**
   * Check if TensorFlow Lite is available
   */
  async checkTensorFlowLite() {
    try {
      // For now, simulate TensorFlow Lite availability
      // In real implementation, this would check if react-native-tensorflow is installed
      
      // Mock check - in real app, you would do:
      // const { TensorFlowLite } = require('react-native-tensorflow');
      // return TensorFlowLite.isAvailable();
      
      return Platform.OS === 'ios' || Platform.OS === 'android';
    } catch (error) {
      console.log('TensorFlow Lite not available:', error);
      return false;
    }
  }

  /**
   * Load edge models from cache or download from server
   */
  async loadEdgeModels() {
    try {
      const cachedModels = await AsyncStorage.getItem(EDGE_MODELS_CACHE_KEY);
      
      if (cachedModels) {
        const models = JSON.parse(cachedModels);
        console.log('📦 Loading cached edge models');
        
        // Validate cached models
        if (await this.validateCachedModels(models)) {
          this.audioModel = models.audioModel;
          this.photoModel = models.photoModel;
          return;
        }
      }

      // Download fresh models
      await this.downloadEdgeModels();
    } catch (error) {
      console.error('Failed to load edge models:', error);
      throw error;
    }
  }

  /**
   * Download lightweight models from server
   */
  async downloadEdgeModels() {
    try {
      console.log('⬇️ Downloading edge models...');
      
      // Mock model download - in real implementation, this would:
      // 1. Fetch model URLs from server
      // 2. Download TensorFlow Lite models
      // 3. Store models in filesystem
      // 4. Cache model metadata
      
      const mockModels = {
        audioModel: {
          version: '1.0.0',
          size: 4.2, // MB
          accuracy: 0.85,
          classes: ['bird_song', 'mammal_call', 'insect_sound', 'environmental_sound', 'unknown'],
          downloadedAt: new Date().toISOString(),
          modelPath: 'nightscan_audio_lite_v1.tflite'
        },
        photoModel: {
          version: '1.0.0',
          size: 8.7, // MB
          accuracy: 0.82,
          classes: ['bat', 'owl', 'raccoon', 'opossum', 'deer', 'fox', 'unknown'],
          downloadedAt: new Date().toISOString(),
          modelPath: 'nightscan_photo_lite_v1.tflite'
        }
      };

      await AsyncStorage.setItem(EDGE_MODELS_CACHE_KEY, JSON.stringify(mockModels));
      this.audioModel = mockModels.audioModel;
      this.photoModel = mockModels.photoModel;
      
      console.log('✅ Edge models downloaded and cached');
    } catch (error) {
      console.error('Failed to download edge models:', error);
      throw error;
    }
  }

  /**
   * Validate cached models are still valid
   */
  async validateCachedModels(models) {
    try {
      // Check if models are not too old (e.g., 7 days)
      const maxAge = 7 * 24 * 60 * 60 * 1000; // 7 days in milliseconds
      const now = new Date().getTime();
      
      if (models.audioModel && models.photoModel) {
        const audioAge = now - new Date(models.audioModel.downloadedAt).getTime();
        const photoAge = now - new Date(models.photoModel.downloadedAt).getTime();
        
        return audioAge < maxAge && photoAge < maxAge;
      }
      
      return false;
    } catch (error) {
      console.error('Error validating cached models:', error);
      return false;
    }
  }

  /**
   * Main prediction function with edge/cloud hybrid logic
   */
  async predict(uri, mimeType = 'audio/wav') {
    const startTime = Date.now();
    
    try {
      // Ensure service is initialized
      if (!this.isInitialized) {
        await this.initialize();
      }

      // Determine file type
      const fileType = this.getFileType(uri, mimeType);
      
      // Try edge prediction first
      const edgeResult = await this.tryEdgePrediction(uri, fileType);
      
      if (edgeResult && edgeResult.confidence >= CONFIDENCE_THRESHOLD) {
        // High confidence edge prediction
        const processingTime = Date.now() - startTime;
        this.updateStats('edge', processingTime, edgeResult.confidence);
        
        return {
          ...edgeResult,
          source: 'edge',
          processingTime,
          fallbackUsed: false
        };
      }

      // Fall back to cloud prediction
      console.log('🌐 Falling back to cloud prediction...');
      const cloudResult = await this.fallbackToCloud(uri, mimeType);
      
      const processingTime = Date.now() - startTime;
      this.updateStats('cloud', processingTime, cloudResult.confidence || 0);
      
      return {
        ...cloudResult,
        source: 'cloud',
        processingTime,
        fallbackUsed: true,
        edgeAttempted: !!edgeResult
      };

    } catch (error) {
      console.error('Prediction failed:', error);
      throw error;
    }
  }

  /**
   * Determine file type from URI and mime type
   */
  getFileType(uri, mimeType) {
    if (mimeType.startsWith('audio/') || uri.includes('.wav') || uri.includes('.mp3')) {
      return 'audio';
    }
    if (mimeType.startsWith('image/') || uri.includes('.jpg') || uri.includes('.jpeg')) {
      return 'photo';
    }
    return 'unknown';
  }

  /**
   * Attempt edge prediction
   */
  async tryEdgePrediction(uri, fileType) {
    try {
      const model = fileType === 'audio' ? this.audioModel : this.photoModel;
      
      if (!model) {
        console.log(`❌ No ${fileType} model available for edge prediction`);
        return null;
      }

      console.log(`🔍 Attempting edge prediction for ${fileType}...`);
      
      // Mock edge prediction - in real implementation, this would:
      // 1. Load the TensorFlow Lite model
      // 2. Preprocess the input file
      // 3. Run inference
      // 4. Post-process results
      
      const mockPrediction = this.mockEdgePrediction(fileType);
      
      console.log(`📊 Edge prediction result: ${mockPrediction.predictedClass} (${(mockPrediction.confidence * 100).toFixed(1)}%)`);
      
      return mockPrediction;
    } catch (error) {
      console.error('Edge prediction failed:', error);
      return null;
    }
  }

  /**
   * Mock edge prediction for development
   */
  mockEdgePrediction(fileType) {
    const audioClasses = ['bird_song', 'mammal_call', 'insect_sound', 'environmental_sound', 'unknown'];
    const photoClasses = ['bat', 'owl', 'raccoon', 'opossum', 'deer', 'fox', 'unknown'];
    
    const classes = fileType === 'audio' ? audioClasses : photoClasses;
    const randomClass = classes[Math.floor(Math.random() * classes.length)];
    
    // Simulate varying confidence levels
    const confidence = Math.random() * 0.4 + 0.6; // 60-100% confidence
    
    return {
      predictedClass: randomClass,
      confidence: confidence,
      modelType: fileType,
      topPredictions: [
        { class: randomClass, confidence: confidence },
        { class: classes[(classes.indexOf(randomClass) + 1) % classes.length], confidence: confidence * 0.7 },
        { class: classes[(classes.indexOf(randomClass) + 2) % classes.length], confidence: confidence * 0.4 }
      ]
    };
  }

  /**
   * Fallback to cloud prediction
   */
  async fallbackToCloud(uri, mimeType) {
    try {
      console.log('☁️ Executing cloud prediction...');
      
      // Use existing cloud prediction API
      const result = await uploadMedia(uri, mimeType);
      
      return {
        ...result,
        confidence: result.confidence || 0
      };
    } catch (error) {
      console.error('Cloud prediction failed:', error);
      throw error;
    }
  }

  /**
   * Update prediction statistics
   */
  updateStats(source, processingTime, confidence) {
    this.stats.totalPredictions++;
    
    if (source === 'edge') {
      this.stats.edgePredictions++;
      this.stats.averageEdgeTime = (this.stats.averageEdgeTime + processingTime) / 2;
    } else {
      this.stats.cloudPredictions++;
      this.stats.averageCloudTime = (this.stats.averageCloudTime + processingTime) / 2;
    }
    
    this.stats.averageConfidence = (this.stats.averageConfidence + confidence) / 2;
    
    // Save stats to storage
    this.saveStats();
  }

  /**
   * Load statistics from storage
   */
  async loadStats() {
    try {
      const savedStats = await AsyncStorage.getItem(EDGE_STATS_KEY);
      if (savedStats) {
        this.stats = { ...this.stats, ...JSON.parse(savedStats) };
      }
    } catch (error) {
      console.error('Failed to load stats:', error);
    }
  }

  /**
   * Save statistics to storage
   */
  async saveStats() {
    try {
      await AsyncStorage.setItem(EDGE_STATS_KEY, JSON.stringify(this.stats));
    } catch (error) {
      console.error('Failed to save stats:', error);
    }
  }

  /**
   * Get edge prediction statistics
   */
  getStats() {
    const edgePercentage = this.stats.totalPredictions > 0 
      ? (this.stats.edgePredictions / this.stats.totalPredictions) * 100 
      : 0;
    
    return {
      ...this.stats,
      edgePercentage: edgePercentage.toFixed(1),
      modelsAvailable: {
        audio: !!this.audioModel,
        photo: !!this.photoModel
      }
    };
  }

  /**
   * Clear edge prediction cache
   */
  async clearCache() {
    try {
      await AsyncStorage.multiRemove([
        EDGE_MODELS_CACHE_KEY,
        EDGE_PREDICTIONS_CACHE_KEY,
        EDGE_STATS_KEY
      ]);
      
      this.audioModel = null;
      this.photoModel = null;
      this.stats = {
        totalPredictions: 0,
        edgePredictions: 0,
        cloudPredictions: 0,
        averageConfidence: 0,
        averageEdgeTime: 0,
        averageCloudTime: 0
      };
      
      console.log('✅ Edge prediction cache cleared');
    } catch (error) {
      console.error('Failed to clear cache:', error);
    }
  }

  /**
   * Force model refresh
   */
  async refreshModels() {
    try {
      await AsyncStorage.removeItem(EDGE_MODELS_CACHE_KEY);
      await this.loadEdgeModels();
      console.log('✅ Edge models refreshed');
    } catch (error) {
      console.error('Failed to refresh models:', error);
      throw error;
    }
  }

  /**
   * Get model information
   */
  getModelInfo() {
    return {
      audio: this.audioModel,
      photo: this.photoModel,
      isInitialized: this.isInitialized
    };
  }

  /**
   * Update confidence threshold
   */
  updateConfidenceThreshold(threshold) {
    if (threshold >= 0 && threshold <= 1) {
      CONFIDENCE_THRESHOLD = threshold;
      console.log(`✅ Confidence threshold updated to ${threshold}`);
    } else {
      console.error('Invalid confidence threshold. Must be between 0 and 1.');
    }
  }
}

// Create singleton instance
const edgePredictionService = new EdgePredictionService();

// Export functions for easy usage
export const initializeEdgePrediction = () => edgePredictionService.initialize();
export const predictWithEdge = (uri, mimeType) => edgePredictionService.predict(uri, mimeType);
export const getEdgeStats = () => edgePredictionService.getStats();
export const clearEdgeCache = () => edgePredictionService.clearCache();
export const refreshEdgeModels = () => edgePredictionService.refreshModels();
export const getModelInfo = () => edgePredictionService.getModelInfo();
export const updateConfidenceThreshold = (threshold) => edgePredictionService.updateConfidenceThreshold(threshold);

export default edgePredictionService;