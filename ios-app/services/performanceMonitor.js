/**
 * Moniteur de Performance pour les Prédictions Edge
 * 
 * Ce service surveille les performances du système de prédiction edge
 * et optimise automatiquement les ressources utilisées.
 */

import AsyncStorage from '@react-native-async-storage/async-storage';
import { Platform } from 'react-native';
import { DeviceEventEmitter } from 'react-native';

// Configuration constants
const PERFORMANCE_STATS_KEY = 'nightscan_performance_stats';
const PERFORMANCE_SETTINGS_KEY = 'nightscan_performance_settings';
const MAX_PERFORMANCE_SAMPLES = 1000;

class PerformanceMonitor {
  constructor() {
    this.isMonitoring = false;
    this.stats = {
      totalPredictions: 0,
      edgePredictions: 0,
      cloudPredictions: 0,
      
      // Temps de traitement
      avgEdgeTime: 0,
      avgCloudTime: 0,
      minEdgeTime: Infinity,
      maxEdgeTime: 0,
      
      // Utilisation mémoire
      memoryUsage: [],
      peakMemory: 0,
      
      // Utilisation batterie
      batteryUsage: [],
      
      // Précision
      confidenceScores: [],
      
      // Erreurs
      errors: [],
      
      // Performances par type de fichier
      audioPerformance: {
        count: 0,
        avgTime: 0,
        avgConfidence: 0
      },
      photoPerformance: {
        count: 0,
        avgTime: 0,
        avgConfidence: 0
      },
      
      // Historique des performances
      performanceHistory: []
    };
    
    this.settings = {
      monitoringEnabled: true,
      memoryThreshold: 100 * 1024 * 1024, // 100MB
      batteryThreshold: 20, // 20%
      lowPerformanceMode: false,
      adaptiveQuality: true,
      maxConcurrentInferences: 2
    };
    
    this.loadStoredData();
    this.startMonitoring();
  }

  async loadStoredData() {
    try {
      const storedStats = await AsyncStorage.getItem(PERFORMANCE_STATS_KEY);
      const storedSettings = await AsyncStorage.getItem(PERFORMANCE_SETTINGS_KEY);
      
      if (storedStats) {
        this.stats = { ...this.stats, ...JSON.parse(storedStats) };
      }
      
      if (storedSettings) {
        this.settings = { ...this.settings, ...JSON.parse(storedSettings) };
      }
    } catch (error) {
      console.error('Failed to load performance data:', error);
    }
  }

  async saveData() {
    try {
      await AsyncStorage.setItem(PERFORMANCE_STATS_KEY, JSON.stringify(this.stats));
      await AsyncStorage.setItem(PERFORMANCE_SETTINGS_KEY, JSON.stringify(this.settings));
    } catch (error) {
      console.error('Failed to save performance data:', error);
    }
  }

  recordPrediction(predictionData) {
    if (!this.settings.monitoringEnabled) return;
    
    const {
      source,
      processingTime,
      confidence,
      modelType,
      success,
      error
    } = predictionData;
    
    this.stats.totalPredictions++;
    
    if (source === 'edge') {
      this.stats.edgePredictions++;
      this.stats.avgEdgeTime = (this.stats.avgEdgeTime * (this.stats.edgePredictions - 1) + processingTime) / this.stats.edgePredictions;
      this.stats.minEdgeTime = Math.min(this.stats.minEdgeTime, processingTime);
      this.stats.maxEdgeTime = Math.max(this.stats.maxEdgeTime, processingTime);
    } else {
      this.stats.cloudPredictions++;
      this.stats.avgCloudTime = (this.stats.avgCloudTime * (this.stats.cloudPredictions - 1) + processingTime) / this.stats.cloudPredictions;
    }
    
    this.stats.confidenceScores.push(confidence);
    if (this.stats.confidenceScores.length > MAX_PERFORMANCE_SAMPLES) {
      this.stats.confidenceScores.shift();
    }
    
    // Mettre à jour les statistiques par type
    if (modelType === 'audio') {
      const typeStats = this.stats.audioPerformance;
      typeStats.count++;
      typeStats.avgTime = (typeStats.avgTime * (typeStats.count - 1) + processingTime) / typeStats.count;
      typeStats.avgConfidence = (typeStats.avgConfidence * (typeStats.count - 1) + confidence) / typeStats.count;
    } else if (modelType === 'photo') {
      const typeStats = this.stats.photoPerformance;
      typeStats.count++;
      typeStats.avgTime = (typeStats.avgTime * (typeStats.count - 1) + processingTime) / typeStats.count;
      typeStats.avgConfidence = (typeStats.avgConfidence * (typeStats.count - 1) + confidence) / typeStats.count;
    }
    
    this.saveData();
  }

  generatePerformanceReport() {
    const edgeSuccessRate = this.stats.totalPredictions > 0 ? this.stats.edgePredictions / this.stats.totalPredictions : 0;
    const avgConfidence = this.stats.confidenceScores.length > 0 
      ? this.stats.confidenceScores.reduce((a, b) => a + b, 0) / this.stats.confidenceScores.length 
      : 0;
    
    return {
      summary: {
        totalPredictions: this.stats.totalPredictions,
        edgePredictions: this.stats.edgePredictions,
        cloudPredictions: this.stats.cloudPredictions,
        edgeSuccessRate: edgeSuccessRate,
        avgConfidence: avgConfidence
      },
      performance: {
        avgEdgeTime: this.stats.avgEdgeTime,
        avgCloudTime: this.stats.avgCloudTime,
        minEdgeTime: this.stats.minEdgeTime === Infinity ? 0 : this.stats.minEdgeTime,
        maxEdgeTime: this.stats.maxEdgeTime
      },
      byType: {
        audio: this.stats.audioPerformance,
        photo: this.stats.photoPerformance
      }
    };
  }

  resetStats() {
    this.stats = {
      totalPredictions: 0,
      edgePredictions: 0,
      cloudPredictions: 0,
      avgEdgeTime: 0,
      avgCloudTime: 0,
      minEdgeTime: Infinity,
      maxEdgeTime: 0,
      memoryUsage: [],
      peakMemory: 0,
      batteryUsage: [],
      confidenceScores: [],
      errors: [],
      audioPerformance: { count: 0, avgTime: 0, avgConfidence: 0 },
      photoPerformance: { count: 0, avgTime: 0, avgConfidence: 0 },
      performanceHistory: []
    };
    
    this.saveData();
  }

  getStats() {
    return { ...this.stats };
  }

  getSettings() {
    return { ...this.settings };
  }
}

// Instance singleton
const performanceMonitor = new PerformanceMonitor();

// Exports
export const recordPrediction = (data) => performanceMonitor.recordPrediction(data);
export const generatePerformanceReport = () => performanceMonitor.generatePerformanceReport();
export const resetPerformanceStats = () => performanceMonitor.resetStats();
export const getPerformanceStats = () => performanceMonitor.getStats();

export default performanceMonitor;