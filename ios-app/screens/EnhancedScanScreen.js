import React, { useState, useEffect, useCallback } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  Alert,
  ActivityIndicator,
  ScrollView,
  Dimensions,
  Platform,
  Modal,
  Switch
} from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { BlurView } from 'expo-blur';
import { Ionicons } from '@expo/vector-icons';
import * as DocumentPicker from 'expo-document-picker';
import AsyncStorage from '@react-native-async-storage/async-storage';

import { 
  initializeEdgePrediction, 
  predictWithEdge, 
  getEdgeStats,
  getModelInfo,
  updateConfidenceThreshold,
  refreshEdgeModels,
  clearEdgeCache
} from '../services/edgePrediction';

const { width, height } = Dimensions.get('window');

export default function EnhancedScanScreen({ navigation }) {
  const [isProcessing, setIsProcessing] = useState(false);
  const [predictionResult, setPredictionResult] = useState(null);
  const [edgeStats, setEdgeStats] = useState(null);
  const [modelInfo, setModelInfo] = useState(null);
  const [edgeInitialized, setEdgeInitialized] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const [edgeEnabled, setEdgeEnabled] = useState(true);
  const [confidenceThreshold, setConfidenceThreshold] = useState(0.8);
  const [processingSteps, setProcessingSteps] = useState([]);

  useEffect(() => {
    initializeEdgeSystem();
    loadUserPreferences();
  }, []);

  const initializeEdgeSystem = useCallback(async () => {
    try {
      console.log('🔄 Initializing edge prediction system...');
      const initialized = await initializeEdgePrediction();
      setEdgeInitialized(initialized);
      
      if (initialized) {
        const stats = await getEdgeStats();
        const models = await getModelInfo();
        setEdgeStats(stats);
        setModelInfo(models);
        console.log('✅ Edge prediction system initialized');
      }
    } catch (error) {
      console.error('❌ Failed to initialize edge system:', error);
    }
  }, []);

  const loadUserPreferences = async () => {
    try {
      const preferences = await AsyncStorage.getItem('nightscan_edge_preferences');
      if (preferences) {
        const prefs = JSON.parse(preferences);
        setEdgeEnabled(prefs.edgeEnabled ?? true);
        setConfidenceThreshold(prefs.confidenceThreshold ?? 0.8);
        await updateConfidenceThreshold(prefs.confidenceThreshold ?? 0.8);
      }
    } catch (error) {
      console.error('Failed to load preferences:', error);
    }
  };

  const saveUserPreferences = async (newPrefs) => {
    try {
      const preferences = {
        edgeEnabled,
        confidenceThreshold,
        ...newPrefs
      };
      await AsyncStorage.setItem('nightscan_edge_preferences', JSON.stringify(preferences));
    } catch (error) {
      console.error('Failed to save preferences:', error);
    }
  };

  const handleFilePick = async () => {
    try {
      const result = await DocumentPicker.getDocumentAsync({
        type: ['audio/*', 'image/*'],
        copyToCacheDirectory: true
      });

      if (result.type === 'cancel' || result.canceled) {
        return;
      }

      const file = result.assets ? result.assets[0] : result;
      await processPrediction(file);
    } catch (error) {
      console.error('File pick error:', error);
      Alert.alert('Erreur', 'Impossible de sélectionner le fichier');
    }
  };

  const processPrediction = async (file) => {
    setIsProcessing(true);
    setPredictionResult(null);
    setProcessingSteps([]);

    const addStep = (step) => {
      setProcessingSteps(prev => [...prev, {
        id: Date.now(),
        text: step,
        timestamp: new Date().toLocaleTimeString()
      }]);
    };

    try {
      addStep('📁 Analyse du fichier...');
      
      const startTime = Date.now();
      
      let result;
      if (edgeEnabled && edgeInitialized) {
        addStep('🔍 Tentative de prédiction locale...');
        result = await predictWithEdge(file.uri, file.mimeType);
      } else {
        addStep('☁️ Prédiction sur le serveur...');
        const { uploadMedia } = await import('../services/api');
        result = await uploadMedia(file.uri, file.mimeType);
        result.source = 'cloud';
        result.fallbackUsed = false;
      }

      const processingTime = Date.now() - startTime;
      
      // Enrichir les résultats
      const enrichedResult = {
        ...result,
        fileName: file.name,
        fileSize: file.size,
        mimeType: file.mimeType,
        processingTime: processingTime,
        timestamp: new Date().toISOString()
      };

      if (result.source === 'edge') {
        addStep(`📱 Prédiction locale terminée (${(result.confidence * 100).toFixed(1)}%)`);
        if (result.fallbackUsed) {
          addStep('⚠️ Confiance faible - Utilisation du serveur');
        }
      } else {
        addStep('✅ Prédiction serveur terminée');
      }

      setPredictionResult(enrichedResult);
      
      // Mettre à jour les statistiques
      const updatedStats = await getEdgeStats();
      setEdgeStats(updatedStats);

      // Sauvegarder dans l'historique
      await saveToHistory(enrichedResult);

    } catch (error) {
      console.error('Prediction error:', error);
      addStep('❌ Erreur de prédiction');
      Alert.alert('Erreur', `Prédiction échouée: ${error.message}`);
    } finally {
      setIsProcessing(false);
    }
  };

  const saveToHistory = async (result) => {
    try {
      const history = await AsyncStorage.getItem('nightscan_prediction_history');
      const historyArray = history ? JSON.parse(history) : [];
      
      historyArray.unshift(result);
      
      // Garder seulement les 50 dernières prédictions
      if (historyArray.length > 50) {
        historyArray.splice(50);
      }
      
      await AsyncStorage.setItem('nightscan_prediction_history', JSON.stringify(historyArray));
    } catch (error) {
      console.error('Failed to save to history:', error);
    }
  };

  const handleRefreshModels = async () => {
    try {
      Alert.alert(
        'Actualiser les modèles',
        'Voulez-vous télécharger les dernières versions des modèles ?',
        [
          { text: 'Annuler', style: 'cancel' },
          { 
            text: 'Actualiser', 
            onPress: async () => {
              setIsProcessing(true);
              await refreshEdgeModels();
              await initializeEdgeSystem();
              setIsProcessing(false);
              Alert.alert('Succès', 'Modèles actualisés avec succès');
            }
          }
        ]
      );
    } catch (error) {
      console.error('Failed to refresh models:', error);
      Alert.alert('Erreur', 'Impossible d\'actualiser les modèles');
    }
  };

  const handleClearCache = async () => {
    try {
      Alert.alert(
        'Vider le cache',
        'Voulez-vous vider le cache des modèles et des prédictions ?',
        [
          { text: 'Annuler', style: 'cancel' },
          { 
            text: 'Vider', 
            style: 'destructive',
            onPress: async () => {
              await clearEdgeCache();
              setEdgeStats(null);
              setModelInfo(null);
              setEdgeInitialized(false);
              Alert.alert('Succès', 'Cache vidé avec succès');
            }
          }
        ]
      );
    } catch (error) {
      console.error('Failed to clear cache:', error);
      Alert.alert('Erreur', 'Impossible de vider le cache');
    }
  };

  const renderPredictionResult = () => {
    if (!predictionResult) return null;

    const { source, confidence, predictedClass, topPredictions, processingTime, fallbackUsed } = predictionResult;

    return (
      <BlurView intensity={90} style={styles.resultContainer}>
        <View style={styles.resultHeader}>
          <Text style={styles.resultTitle}>Résultat de la Prédiction</Text>
          <View style={styles.sourceIndicator}>
            <Ionicons 
              name={source === 'edge' ? 'phone-portrait' : 'cloud'} 
              size={16} 
              color={source === 'edge' ? '#4CAF50' : '#2196F3'} 
            />
            <Text style={[styles.sourceText, { color: source === 'edge' ? '#4CAF50' : '#2196F3' }]}>
              {source === 'edge' ? 'Local' : 'Cloud'}
            </Text>
          </View>
        </View>

        <View style={styles.mainPrediction}>
          <Text style={styles.predictedClass}>{predictedClass}</Text>
          <Text style={styles.confidence}>{(confidence * 100).toFixed(1)}% de confiance</Text>
          {fallbackUsed && (
            <Text style={styles.fallbackNotice}>
              ⚠️ Confiance faible - Serveur utilisé
            </Text>
          )}
        </View>

        <View style={styles.processingInfo}>
          <Text style={styles.processingTime}>
            Temps de traitement: {processingTime}ms
          </Text>
        </View>

        {topPredictions && topPredictions.length > 1 && (
          <View style={styles.topPredictions}>
            <Text style={styles.topPredictionsTitle}>Top Prédictions:</Text>
            {topPredictions.slice(0, 3).map((pred, index) => (
              <View key={index} style={styles.predictionItem}>
                <Text style={styles.predictionClass}>{pred.class}</Text>
                <Text style={styles.predictionConfidence}>
                  {(pred.confidence * 100).toFixed(1)}%
                </Text>
              </View>
            ))}
          </View>
        )}
      </BlurView>
    );
  };

  const renderProcessingSteps = () => {
    if (!isProcessing && processingSteps.length === 0) return null;

    return (
      <View style={styles.processingSteps}>
        <Text style={styles.processingTitle}>Étapes de traitement:</Text>
        <ScrollView style={styles.stepsContainer}>
          {processingSteps.map((step) => (
            <View key={step.id} style={styles.stepItem}>
              <Text style={styles.stepText}>{step.text}</Text>
              <Text style={styles.stepTime}>{step.timestamp}</Text>
            </View>
          ))}
        </ScrollView>
      </View>
    );
  };

  const renderStats = () => {
    if (!edgeStats) return null;

    return (
      <View style={styles.statsContainer}>
        <Text style={styles.statsTitle}>Statistiques Edge</Text>
        <View style={styles.statsGrid}>
          <View style={styles.statItem}>
            <Text style={styles.statValue}>{edgeStats.totalPredictions}</Text>
            <Text style={styles.statLabel}>Total</Text>
          </View>
          <View style={styles.statItem}>
            <Text style={styles.statValue}>{edgeStats.edgePercentage}%</Text>
            <Text style={styles.statLabel}>Local</Text>
          </View>
          <View style={styles.statItem}>
            <Text style={styles.statValue}>{Math.round(edgeStats.averageEdgeTime)}ms</Text>
            <Text style={styles.statLabel}>Temps Local</Text>
          </View>
          <View style={styles.statItem}>
            <Text style={styles.statValue}>{Math.round(edgeStats.averageCloudTime)}ms</Text>
            <Text style={styles.statLabel}>Temps Cloud</Text>
          </View>
        </View>
      </View>
    );
  };

  const renderSettings = () => (
    <Modal
      visible={showSettings}
      animationType="slide"
      transparent={true}
      onRequestClose={() => setShowSettings(false)}
    >
      <View style={styles.modalOverlay}>
        <BlurView intensity={90} style={styles.settingsModal}>
          <View style={styles.settingsHeader}>
            <Text style={styles.settingsTitle}>Paramètres Edge</Text>
            <TouchableOpacity onPress={() => setShowSettings(false)}>
              <Ionicons name="close" size={24} color="#666" />
            </TouchableOpacity>
          </View>

          <ScrollView style={styles.settingsContent}>
            <View style={styles.settingItem}>
              <Text style={styles.settingLabel}>Prédiction locale</Text>
              <Switch
                value={edgeEnabled}
                onValueChange={async (value) => {
                  setEdgeEnabled(value);
                  await saveUserPreferences({ edgeEnabled: value });
                }}
              />
            </View>

            <View style={styles.settingItem}>
              <Text style={styles.settingLabel}>
                Seuil de confiance: {(confidenceThreshold * 100).toFixed(0)}%
              </Text>
              <Text style={styles.settingDescription}>
                Prédictions sous ce seuil utilisent le serveur
              </Text>
            </View>

            {modelInfo && (
              <View style={styles.modelInfo}>
                <Text style={styles.modelInfoTitle}>Modèles disponibles:</Text>
                <Text style={styles.modelStatus}>
                  Audio: {modelInfo.audio ? '✅' : '❌'}
                </Text>
                <Text style={styles.modelStatus}>
                  Photo: {modelInfo.photo ? '✅' : '❌'}
                </Text>
              </View>
            )}

            <View style={styles.settingActions}>
              <TouchableOpacity style={styles.actionButton} onPress={handleRefreshModels}>
                <Ionicons name="refresh" size={20} color="#2196F3" />
                <Text style={styles.actionButtonText}>Actualiser modèles</Text>
              </TouchableOpacity>

              <TouchableOpacity style={styles.actionButton} onPress={handleClearCache}>
                <Ionicons name="trash" size={20} color="#F44336" />
                <Text style={[styles.actionButtonText, { color: '#F44336' }]}>
                  Vider le cache
                </Text>
              </TouchableOpacity>
            </View>
          </ScrollView>
        </BlurView>
      </View>
    </Modal>
  );

  return (
    <LinearGradient colors={['#1a237e', '#3949ab', '#5c6bc0']} style={styles.container}>
      <ScrollView contentContainerStyle={styles.scrollContent}>
        <View style={styles.header}>
          <Text style={styles.title}>🌙 NightScan</Text>
          <Text style={styles.subtitle}>Prédiction Edge + Cloud</Text>
          <TouchableOpacity 
            style={styles.settingsButton}
            onPress={() => setShowSettings(true)}
          >
            <Ionicons name="settings" size={24} color="#fff" />
          </TouchableOpacity>
        </View>

        {renderStats()}

        <View style={styles.uploadSection}>
          <TouchableOpacity
            style={styles.uploadButton}
            onPress={handleFilePick}
            disabled={isProcessing}
          >
            <LinearGradient
              colors={['#4CAF50', '#66BB6A']}
              style={styles.uploadButtonGradient}
            >
              {isProcessing ? (
                <ActivityIndicator color="#fff" size="large" />
              ) : (
                <>
                  <Ionicons name="add-circle" size={32} color="#fff" />
                  <Text style={styles.uploadButtonText}>
                    Sélectionner un fichier
                  </Text>
                </>
              )}
            </LinearGradient>
          </TouchableOpacity>
        </View>

        {renderProcessingSteps()}
        {renderPredictionResult()}

        <View style={styles.footer}>
          <Text style={styles.footerText}>
            {edgeInitialized ? 
              `🟢 Prédiction edge ${edgeEnabled ? 'activée' : 'désactivée'}` : 
              '🔴 Prédiction edge non disponible'
            }
          </Text>
        </View>
      </ScrollView>

      {renderSettings()}
    </LinearGradient>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  scrollContent: {
    flexGrow: 1,
    paddingHorizontal: 20,
    paddingTop: Platform.OS === 'ios' ? 50 : 30,
  },
  header: {
    alignItems: 'center',
    marginBottom: 30,
    position: 'relative',
  },
  title: {
    fontSize: 32,
    fontWeight: 'bold',
    color: '#fff',
    marginBottom: 5,
  },
  subtitle: {
    fontSize: 16,
    color: '#e8eaf6',
    opacity: 0.9,
  },
  settingsButton: {
    position: 'absolute',
    right: 0,
    top: 0,
  },
  statsContainer: {
    backgroundColor: 'rgba(255, 255, 255, 0.1)',
    borderRadius: 15,
    padding: 20,
    marginBottom: 20,
  },
  statsTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#fff',
    marginBottom: 15,
    textAlign: 'center',
  },
  statsGrid: {
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  statItem: {
    alignItems: 'center',
  },
  statValue: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#fff',
  },
  statLabel: {
    fontSize: 12,
    color: '#e8eaf6',
    marginTop: 5,
  },
  uploadSection: {
    marginBottom: 30,
  },
  uploadButton: {
    borderRadius: 15,
    overflow: 'hidden',
    elevation: 5,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.25,
    shadowRadius: 3.84,
  },
  uploadButtonGradient: {
    paddingVertical: 20,
    paddingHorizontal: 30,
    alignItems: 'center',
    justifyContent: 'center',
    minHeight: 80,
  },
  uploadButtonText: {
    fontSize: 18,
    fontWeight: '600',
    color: '#fff',
    marginTop: 10,
  },
  processingSteps: {
    backgroundColor: 'rgba(255, 255, 255, 0.1)',
    borderRadius: 15,
    padding: 15,
    marginBottom: 20,
  },
  processingTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#fff',
    marginBottom: 10,
  },
  stepsContainer: {
    maxHeight: 150,
  },
  stepItem: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: 5,
  },
  stepText: {
    fontSize: 14,
    color: '#e8eaf6',
    flex: 1,
  },
  stepTime: {
    fontSize: 12,
    color: '#c5cae9',
  },
  resultContainer: {
    borderRadius: 15,
    padding: 20,
    marginBottom: 20,
    overflow: 'hidden',
  },
  resultHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 15,
  },
  resultTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#fff',
  },
  sourceIndicator: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: 'rgba(255, 255, 255, 0.2)',
    paddingHorizontal: 10,
    paddingVertical: 5,
    borderRadius: 15,
  },
  sourceText: {
    fontSize: 12,
    fontWeight: '600',
    marginLeft: 5,
  },
  mainPrediction: {
    alignItems: 'center',
    marginBottom: 20,
  },
  predictedClass: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#fff',
    textAlign: 'center',
    marginBottom: 10,
  },
  confidence: {
    fontSize: 16,
    color: '#e8eaf6',
    marginBottom: 5,
  },
  fallbackNotice: {
    fontSize: 14,
    color: '#FFC107',
    textAlign: 'center',
  },
  processingInfo: {
    alignItems: 'center',
    marginBottom: 15,
  },
  processingTime: {
    fontSize: 14,
    color: '#c5cae9',
  },
  topPredictions: {
    backgroundColor: 'rgba(255, 255, 255, 0.1)',
    borderRadius: 10,
    padding: 15,
  },
  topPredictionsTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#fff',
    marginBottom: 10,
  },
  predictionItem: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: 5,
  },
  predictionClass: {
    fontSize: 14,
    color: '#e8eaf6',
  },
  predictionConfidence: {
    fontSize: 14,
    color: '#c5cae9',
    fontWeight: '600',
  },
  footer: {
    alignItems: 'center',
    marginTop: 30,
    marginBottom: 20,
  },
  footerText: {
    fontSize: 14,
    color: '#e8eaf6',
  },
  // Modal styles
  modalOverlay: {
    flex: 1,
    backgroundColor: 'rgba(0, 0, 0, 0.5)',
    justifyContent: 'center',
    alignItems: 'center',
  },
  settingsModal: {
    width: width * 0.9,
    maxHeight: height * 0.8,
    borderRadius: 20,
    overflow: 'hidden',
  },
  settingsHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: 20,
    borderBottomWidth: 1,
    borderBottomColor: 'rgba(255, 255, 255, 0.1)',
  },
  settingsTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#333',
  },
  settingsContent: {
    padding: 20,
  },
  settingItem: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 20,
  },
  settingLabel: {
    fontSize: 16,
    color: '#333',
    flex: 1,
  },
  settingDescription: {
    fontSize: 14,
    color: '#666',
    marginTop: 5,
  },
  modelInfo: {
    backgroundColor: 'rgba(255, 255, 255, 0.1)',
    borderRadius: 10,
    padding: 15,
    marginBottom: 20,
  },
  modelInfoTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#333',
    marginBottom: 10,
  },
  modelStatus: {
    fontSize: 14,
    color: '#666',
    marginBottom: 5,
  },
  settingActions: {
    marginTop: 20,
  },
  actionButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    padding: 15,
    borderRadius: 10,
    backgroundColor: 'rgba(255, 255, 255, 0.1)',
    marginBottom: 10,
  },
  actionButtonText: {
    fontSize: 16,
    color: '#2196F3',
    marginLeft: 10,
    fontWeight: '600',
  },
});