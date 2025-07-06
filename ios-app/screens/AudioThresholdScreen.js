import React, { useState, useEffect, useRef } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  Alert,
  ActivityIndicator,
  ScrollView,
  Dimensions,
  AppState
} from 'react-native';
import Slider from '@react-native-community/slider';
import { Ionicons } from '@expo/vector-icons';
import { LinearGradient } from 'expo-linear-gradient';

const { width } = Dimensions.get('window');
const PI_SERVICE_URL = 'http://192.168.4.1:5000';

export default function AudioThresholdScreen({ navigation }) {
  const [loading, setLoading] = useState(true);
  const [config, setConfig] = useState(null);
  const [presets, setPresets] = useState({});
  const [currentPreset, setCurrentPreset] = useState('balanced');
  const [liveLevel, setLiveLevel] = useState(-60);
  const [isCalibrating, setIsCalibrating] = useState(false);
  const [isTesting, setIsTesting] = useState(false);
  const [testResult, setTestResult] = useState(null);
  const [error, setError] = useState(null);
  
  const liveLevelInterval = useRef(null);
  const appState = useRef(AppState.currentState);

  useEffect(() => {
    const subscription = AppState.addEventListener('change', nextAppState => {
      if (nextAppState.match(/inactive|background/)) {
        stopLiveMonitoring();
      } else if (nextAppState === 'active' && appState.current.match(/inactive|background/)) {
        loadAudioConfig();
      }
      appState.current = nextAppState;
    });

    return () => {
      subscription?.remove();
      stopLiveMonitoring();
    };
  }, []);

  useEffect(() => {
    loadAudioConfig();
    return () => stopLiveMonitoring();
  }, []);

  const loadAudioConfig = async () => {
    try {
      setError(null);
      setLoading(true);

      // Load current status and presets
      const [statusResponse, presetsResponse] = await Promise.all([
        fetch(`${PI_SERVICE_URL}/audio/threshold/status`),
        fetch(`${PI_SERVICE_URL}/audio/threshold/presets`)
      ]);

      if (statusResponse.ok && presetsResponse.ok) {
        const statusData = await statusResponse.json();
        const presetsData = await presetsResponse.json();

        setConfig(statusData.config);
        setPresets(presetsData.presets);
        setCurrentPreset(presetsData.current_preset || 'balanced');
        
        // Start live monitoring
        startLiveMonitoring();
      } else {
        throw new Error('Impossible de charger la configuration audio');
      }
    } catch (err) {
      console.error('Load config error:', err);
      setError('Pi non accessible. Vérifiez la connexion.');
    } finally {
      setLoading(false);
    }
  };

  const startLiveMonitoring = () => {
    if (liveLevelInterval.current) return;

    liveLevelInterval.current = setInterval(async () => {
      try {
        const response = await fetch(`${PI_SERVICE_URL}/audio/threshold/live`);
        if (response.ok) {
          const data = await response.json();
          setLiveLevel(data.current_db);
        }
      } catch (err) {
        // Silently fail for live updates
      }
    }, 200); // Update every 200ms
  };

  const stopLiveMonitoring = () => {
    if (liveLevelInterval.current) {
      clearInterval(liveLevelInterval.current);
      liveLevelInterval.current = null;
    }
  };

  const updateThreshold = async (newThreshold) => {
    try {
      const response = await fetch(`${PI_SERVICE_URL}/audio/threshold/config`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          threshold_db: newThreshold
        }),
      });

      if (response.ok) {
        const data = await response.json();
        setConfig(data.current_config);
      }
    } catch (err) {
      console.error('Update threshold error:', err);
    }
  };

  const applyPreset = async (presetName) => {
    try {
      setLoading(true);
      const response = await fetch(`${PI_SERVICE_URL}/audio/threshold/preset/${presetName}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
      });

      if (response.ok) {
        const data = await response.json();
        setConfig(data.current_config);
        setCurrentPreset(presetName);
        Alert.alert('Succès', `Preset "${presetName}" appliqué`);
      } else {
        throw new Error('Échec de l\'application du preset');
      }
    } catch (err) {
      console.error('Apply preset error:', err);
      Alert.alert('Erreur', 'Impossible d\'appliquer le preset');
    } finally {
      setLoading(false);
    }
  };

  const calibrateNoise = async () => {
    Alert.alert(
      'Calibration du bruit de fond',
      'Assurez-vous que l\'environnement est aussi silencieux que possible pendant 5 secondes.',
      [
        { text: 'Annuler', style: 'cancel' },
        {
          text: 'Démarrer',
          onPress: async () => {
            try {
              setIsCalibrating(true);
              const response = await fetch(`${PI_SERVICE_URL}/audio/threshold/calibrate`, {
                method: 'POST',
                headers: {
                  'Content-Type': 'application/json',
                },
                body: JSON.stringify({ duration_seconds: 5 }),
              });

              if (response.ok) {
                const data = await response.json();
                setConfig(data.current_config);
                Alert.alert('Succès', `Calibration terminée. Bruit de fond: ${data.noise_floor_db.toFixed(1)}dB`);
              } else {
                throw new Error('Échec de la calibration');
              }
            } catch (err) {
              console.error('Calibration error:', err);
              Alert.alert('Erreur', 'Échec de la calibration');
            } finally {
              setIsCalibrating(false);
            }
          }
        }
      ]
    );
  };

  const testThreshold = async () => {
    try {
      setIsTesting(true);
      setTestResult(null);
      
      const response = await fetch(`${PI_SERVICE_URL}/audio/threshold/test`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ duration_seconds: 3 }),
      });

      if (response.ok) {
        const data = await response.json();
        setTestResult(data.test_result);
      } else {
        throw new Error('Échec du test');
      }
    } catch (err) {
      console.error('Test error:', err);
      Alert.alert('Erreur', 'Échec du test de seuil');
    } finally {
      setIsTesting(false);
    }
  };

  const getLevelColor = (db) => {
    if (db > -20) return '#ff4444'; // Rouge - très fort
    if (db > -30) return '#ff8800'; // Orange - fort
    if (db > -40) return '#ffdd00'; // Jaune - modéré
    if (db > -50) return '#88dd00'; // Vert clair - faible
    return '#44aa44'; // Vert - très faible
  };

  const getLevelHeight = (db) => {
    // Convertir dB (-80 à 0) en pourcentage de hauteur (0 à 100%)
    const minDb = -80;
    const maxDb = 0;
    const percentage = Math.max(0, Math.min(100, ((db - minDb) / (maxDb - minDb)) * 100));
    return percentage;
  };

  const renderLevelMeter = () => {
    const levelHeight = getLevelHeight(liveLevel);
    const thresholdHeight = config ? getLevelHeight(config.threshold_db) : 50;
    const levelColor = getLevelColor(liveLevel);

    return (
      <View style={styles.levelMeterContainer}>
        <Text style={styles.levelMeterTitle}>Niveau Audio en Temps Réel</Text>
        
        <View style={styles.levelMeter}>
          {/* Background bars */}
          <LinearGradient
            colors={['#ff4444', '#ff8800', '#ffdd00', '#88dd00', '#44aa44']}
            style={styles.levelBackground}
            start={{ x: 0, y: 0 }}
            end={{ x: 0, y: 1 }}
          />
          
          {/* Current level */}
          <View 
            style={[
              styles.levelIndicator,
              {
                height: `${levelHeight}%`,
                backgroundColor: levelColor
              }
            ]} 
          />
          
          {/* Threshold line */}
          {config && (
            <View 
              style={[
                styles.thresholdLine,
                { bottom: `${thresholdHeight}%` }
              ]}
            />
          )}
        </View>
        
        <View style={styles.levelLabels}>
          <Text style={styles.levelValue}>{liveLevel.toFixed(1)} dB</Text>
          {config && (
            <Text style={styles.thresholdValue}>
              Seuil: {config.threshold_db.toFixed(1)} dB
            </Text>
          )}
        </View>
      </View>
    );
  };

  const renderPresets = () => {
    return (
      <View style={styles.presetsContainer}>
        <Text style={styles.sectionTitle}>Préréglages Environnement</Text>
        <View style={styles.presetButtons}>
          {Object.entries(presets).map(([key, preset]) => (
            <TouchableOpacity
              key={key}
              style={[
                styles.presetButton,
                currentPreset === key && styles.presetButtonActive
              ]}
              onPress={() => applyPreset(key)}
            >
              <Ionicons 
                name={
                  key === 'quiet' ? 'moon' :
                  key === 'balanced' ? 'partly-sunny' : 'sunny'
                }
                size={24}
                color={currentPreset === key ? '#fff' : '#007AFF'}
              />
              <Text style={[
                styles.presetButtonText,
                currentPreset === key && styles.presetButtonTextActive
              ]}>
                {key === 'quiet' ? 'Silencieux' :
                 key === 'balanced' ? 'Équilibré' : 'Bruyant'}
              </Text>
              <Text style={[
                styles.presetDescription,
                currentPreset === key && styles.presetDescriptionActive
              ]}>
                {preset.description}
              </Text>
            </TouchableOpacity>
          ))}
        </View>
      </View>
    );
  };

  const renderThresholdSlider = () => {
    if (!config) return null;

    return (
      <View style={styles.sliderContainer}>
        <Text style={styles.sectionTitle}>Ajustement Manuel du Seuil</Text>
        
        <View style={styles.sliderRow}>
          <Ionicons name="volume-low" size={24} color="#666" />
          <Slider
            style={styles.slider}
            minimumValue={-60}
            maximumValue={-10}
            value={config.threshold_db}
            onValueChange={updateThreshold}
            minimumTrackTintColor="#007AFF"
            maximumTrackTintColor="#ddd"
            thumbStyle={styles.sliderThumb}
            step={1}
          />
          <Ionicons name="volume-high" size={24} color="#666" />
        </View>
        
        <Text style={styles.sliderValue}>
          {config.threshold_db.toFixed(1)} dB
        </Text>
        
        <Text style={styles.sliderHint}>
          Plus bas = plus sensible, Plus haut = moins sensible
        </Text>
      </View>
    );
  };

  const renderTestResults = () => {
    if (!testResult) return null;

    const { total_detections, detection_rate, recommendation } = testResult;
    
    return (
      <View style={styles.testResultsContainer}>
        <Text style={styles.sectionTitle}>Résultats du Test</Text>
        <View style={styles.testResults}>
          <Text style={styles.testStat}>
            Détections: {total_detections}
          </Text>
          <Text style={styles.testStat}>
            Taux: {(detection_rate * 100).toFixed(1)}%
          </Text>
          <Text style={styles.testRecommendation}>
            {recommendation}
          </Text>
        </View>
      </View>
    );
  };

  if (loading) {
    return (
      <View style={styles.loadingContainer}>
        <ActivityIndicator size="large" color="#007AFF" />
        <Text style={styles.loadingText}>Chargement configuration audio...</Text>
      </View>
    );
  }

  if (error) {
    return (
      <View style={styles.errorContainer}>
        <Ionicons name="warning" size={64} color="#ff4444" />
        <Text style={styles.errorText}>{error}</Text>
        <TouchableOpacity style={styles.retryButton} onPress={loadAudioConfig}>
          <Text style={styles.retryButtonText}>Réessayer</Text>
        </TouchableOpacity>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <View style={styles.header}>
        <TouchableOpacity
          style={styles.backButton}
          onPress={() => navigation.goBack()}
        >
          <Ionicons name="arrow-back" size={24} color="#007AFF" />
        </TouchableOpacity>
        <Text style={styles.headerTitle}>Sensibilité Microphone</Text>
      </View>

      <ScrollView style={styles.content} showsVerticalScrollIndicator={false}>
        {renderLevelMeter()}
        {renderPresets()}
        {renderThresholdSlider()}
        
        <View style={styles.controlsContainer}>
          <TouchableOpacity
            style={[styles.controlButton, styles.calibrateButton]}
            onPress={calibrateNoise}
            disabled={isCalibrating}
          >
            {isCalibrating ? (
              <ActivityIndicator size="small" color="#fff" />
            ) : (
              <Ionicons name="settings" size={20} color="#fff" />
            )}
            <Text style={styles.controlButtonText}>
              {isCalibrating ? 'Calibration...' : 'Calibrer Bruit'}
            </Text>
          </TouchableOpacity>

          <TouchableOpacity
            style={[styles.controlButton, styles.testButton]}
            onPress={testThreshold}
            disabled={isTesting}
          >
            {isTesting ? (
              <ActivityIndicator size="small" color="#fff" />
            ) : (
              <Ionicons name="play" size={20} color="#fff" />
            )}
            <Text style={styles.controlButtonText}>
              {isTesting ? 'Test en cours...' : 'Tester Seuil'}
            </Text>
          </TouchableOpacity>
        </View>

        {renderTestResults()}
      </ScrollView>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f8f9fa',
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: 16,
    backgroundColor: '#fff',
    paddingTop: 50,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  backButton: {
    marginRight: 16,
  },
  headerTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#333',
  },
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  loadingText: {
    marginTop: 16,
    fontSize: 16,
    color: '#666',
  },
  errorContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 20,
  },
  errorText: {
    fontSize: 16,
    color: '#ff4444',
    textAlign: 'center',
    marginVertical: 20,
  },
  retryButton: {
    backgroundColor: '#007AFF',
    paddingHorizontal: 20,
    paddingVertical: 10,
    borderRadius: 8,
  },
  retryButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: 'bold',
  },
  content: {
    flex: 1,
    padding: 16,
  },
  levelMeterContainer: {
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 16,
    marginBottom: 16,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  levelMeterTitle: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 16,
    textAlign: 'center',
  },
  levelMeter: {
    height: 200,
    width: 60,
    backgroundColor: '#f0f0f0',
    borderRadius: 30,
    overflow: 'hidden',
    alignSelf: 'center',
    position: 'relative',
  },
  levelBackground: {
    position: 'absolute',
    width: '100%',
    height: '100%',
    opacity: 0.3,
  },
  levelIndicator: {
    position: 'absolute',
    bottom: 0,
    width: '100%',
    borderRadius: 30,
  },
  thresholdLine: {
    position: 'absolute',
    width: '120%',
    height: 2,
    backgroundColor: '#ff4444',
    left: '-10%',
  },
  levelLabels: {
    marginTop: 16,
    alignItems: 'center',
  },
  levelValue: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#333',
  },
  thresholdValue: {
    fontSize: 14,
    color: '#666',
    marginTop: 4,
  },
  presetsContainer: {
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 16,
    marginBottom: 16,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  sectionTitle: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 16,
  },
  presetButtons: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    gap: 8,
  },
  presetButton: {
    flex: 1,
    backgroundColor: '#f8f9fa',
    borderRadius: 12,
    padding: 16,
    alignItems: 'center',
    borderWidth: 2,
    borderColor: '#e9ecef',
  },
  presetButtonActive: {
    backgroundColor: '#007AFF',
    borderColor: '#007AFF',
  },
  presetButtonText: {
    fontSize: 14,
    fontWeight: 'bold',
    color: '#007AFF',
    marginTop: 8,
  },
  presetButtonTextActive: {
    color: '#fff',
  },
  presetDescription: {
    fontSize: 12,
    color: '#666',
    textAlign: 'center',
    marginTop: 4,
  },
  presetDescriptionActive: {
    color: '#fff',
    opacity: 0.9,
  },
  sliderContainer: {
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 16,
    marginBottom: 16,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  sliderRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 16,
  },
  slider: {
    flex: 1,
    height: 40,
  },
  sliderThumb: {
    backgroundColor: '#007AFF',
    width: 20,
    height: 20,
  },
  sliderValue: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#007AFF',
    textAlign: 'center',
    marginTop: 8,
  },
  sliderHint: {
    fontSize: 12,
    color: '#666',
    textAlign: 'center',
    marginTop: 8,
  },
  controlsContainer: {
    flexDirection: 'row',
    gap: 12,
    marginBottom: 16,
  },
  controlButton: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 16,
    borderRadius: 12,
    gap: 8,
  },
  calibrateButton: {
    backgroundColor: '#FF9500',
  },
  testButton: {
    backgroundColor: '#28a745',
  },
  controlButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: 'bold',
  },
  testResultsContainer: {
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 16,
    marginBottom: 16,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  testResults: {
    gap: 8,
  },
  testStat: {
    fontSize: 16,
    color: '#333',
  },
  testRecommendation: {
    fontSize: 14,
    color: '#007AFF',
    fontStyle: 'italic',
    marginTop: 8,
  },
});