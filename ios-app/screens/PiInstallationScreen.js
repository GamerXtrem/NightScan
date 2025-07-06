import React, { useState, useEffect, useRef } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  Alert,
  ActivityIndicator,
  Vibration,
  AppState
} from 'react-native';
import { WebView } from 'react-native-webview';
import { Audio } from 'expo-av';
import { Ionicons } from '@expo/vector-icons';
import { getPiHealth, getCameraStatus, startCameraPreview, stopCameraPreview } from '../services/api';

const PI_SERVICE_URL = 'http://192.168.4.1:5000';

export default function PiInstallationScreen({ navigation }) {
  const [step, setStep] = useState(1); // Installation workflow steps
  const [isConnecting, setIsConnecting] = useState(false);
  const [isPreviewActive, setIsPreviewActive] = useState(false);
  const [piConnected, setPiConnected] = useState(false);
  const [cameraReady, setCameraReady] = useState(false);
  const [sound, setSound] = useState();
  const [error, setError] = useState(null);
  const webViewRef = useRef(null);
  const appState = useRef(AppState.currentState);

  const INSTALLATION_STEPS = [
    {
      id: 1,
      title: "Réveil du Pi",
      description: "Approchez-vous du Pi et appuyez sur le bouton pour le réveiller",
      action: "wake"
    },
    {
      id: 2,
      title: "Connexion au Pi",
      description: "Connexion au réseau Wi-Fi du Pi",
      action: "connect"
    },
    {
      id: 3,
      title: "Prévisualisation caméra", 
      description: "Utilisez la prévisualisation pour ajuster l'angle de la caméra",
      action: "preview"
    },
    {
      id: 4,
      title: "Test final",
      description: "Vérifiez que tout fonctionne correctement",
      action: "test"
    }
  ];

  useEffect(() => {
    const subscription = AppState.addEventListener('change', nextAppState => {
      if (nextAppState.match(/inactive|background/)) {
        // App went to background - stop preview
        stopPreview();
      }
      appState.current = nextAppState;
    });

    return () => {
      subscription?.remove();
      cleanup();
    };
  }, []);

  useEffect(() => {
    // Auto-advance based on current conditions
    checkSystemStatus();
  }, [step]);

  const cleanup = async () => {
    if (isPreviewActive) {
      await stopPreview();
    }
    if (sound) {
      await sound.unloadAsync();
    }
  };

  const checkSystemStatus = async () => {
    try {
      // Check Pi connection
      const healthResponse = await getPiHealth();
      setPiConnected(true);
      
      if (step < 3) {
        setStep(3); // Jump to preview step if Pi is connected
      }

      // Check camera status
      const cameraResponse = await getCameraStatus();
      setCameraReady(cameraResponse.camera_available);
      
    } catch (error) {
      setPiConnected(false);
      setCameraReady(false);
      
      if (step > 2) {
        setStep(2); // Step back if connection is lost
      }
    }
  };

  const playWakeTone = async () => {
    try {
      const { sound: s } = await Audio.Sound.createAsync(
        require('../assets/wake_tone.wav')
      );
      setSound(s);
      await s.playAsync();
      
      // Provide haptic feedback
      Vibration.vibrate([0, 100, 100, 100]);
      
      // Advance to next step
      setStep(2);
      
      // Check for Pi response after delay
      setTimeout(() => {
        checkSystemStatus();
      }, 3000);
      
    } catch (error) {
      console.log('Wake tone playback failed:', error);
      Alert.alert('Erreur', 'Impossible de jouer le son de réveil');
    }
  };

  const startPreview = async () => {
    if (!cameraReady) {
      Alert.alert('Erreur', 'Caméra non disponible sur le Pi');
      return;
    }

    setIsConnecting(true);
    setError(null);
    
    try {
      await startCameraPreview();
      setIsPreviewActive(true);
      setStep(4); // Advance to test step
      
      // Refresh the WebView to start showing the stream
      if (webViewRef.current) {
        webViewRef.current.reload();
      }
    } catch (error) {
      console.error('Start preview error:', error);
      Alert.alert('Erreur', 'Impossible de démarrer la prévisualisation');
    } finally {
      setIsConnecting(false);
    }
  };

  const stopPreview = async () => {
    if (!isPreviewActive) return;

    try {
      await stopCameraPreview();
      setIsPreviewActive(false);
    } catch (error) {
      console.log('Stop preview error:', error);
      setIsPreviewActive(false);
    }
  };

  const completeInstallation = () => {
    Alert.alert(
      'Installation terminée',
      'Votre NightScan Pi est maintenant correctement installé et configuré !',
      [
        {
          text: 'Parfait !',
          onPress: () => navigation.goBack()
        }
      ]
    );
  };

  const getCurrentStepData = () => {
    return INSTALLATION_STEPS.find(s => s.id === step) || INSTALLATION_STEPS[0];
  };

  const renderStepProgress = () => {
    return (
      <View style={styles.progressContainer}>
        {INSTALLATION_STEPS.map((stepData, index) => (
          <View key={stepData.id} style={styles.progressStep}>
            <View style={[
              styles.progressDot,
              step > stepData.id && styles.completedDot,
              step === stepData.id && styles.activeDot
            ]}>
              {step > stepData.id ? (
                <Ionicons name="checkmark" size={16} color="#fff" />
              ) : (
                <Text style={styles.progressNumber}>{stepData.id}</Text>
              )}
            </View>
            {index < INSTALLATION_STEPS.length - 1 && (
              <View style={[
                styles.progressLine,
                step > stepData.id && styles.completedLine
              ]} />
            )}
          </View>
        ))}
      </View>
    );
  };

  const renderStepContent = () => {
    const currentStep = getCurrentStepData();

    switch (currentStep.action) {
      case 'wake':
        return (
          <View style={styles.stepContent}>
            <Ionicons name="volume-high" size={64} color="#FF9500" />
            <Text style={styles.stepTitle}>{currentStep.title}</Text>
            <Text style={styles.stepDescription}>{currentStep.description}</Text>
            <TouchableOpacity
              style={[styles.primaryButton, styles.wakeButton]}
              onPress={playWakeTone}
            >
              <Ionicons name="volume-high" size={20} color="#fff" />
              <Text style={styles.buttonText}>Réveiller le Pi</Text>
            </TouchableOpacity>
          </View>
        );

      case 'connect':
        return (
          <View style={styles.stepContent}>
            {piConnected ? (
              <>
                <Ionicons name="wifi" size={64} color="#28a745" />
                <Text style={styles.stepTitle}>Connexion établie</Text>
                <Text style={styles.stepDescription}>
                  Connecté au Pi avec succès !
                </Text>
              </>
            ) : (
              <>
                <ActivityIndicator size="large" color="#007AFF" />
                <Text style={styles.stepTitle}>{currentStep.title}</Text>
                <Text style={styles.stepDescription}>
                  Recherche du réseau NightScan...
                </Text>
                <TouchableOpacity
                  style={styles.secondaryButton}
                  onPress={checkSystemStatus}
                >
                  <Text style={styles.secondaryButtonText}>Vérifier la connexion</Text>
                </TouchableOpacity>
              </>
            )}
          </View>
        );

      case 'preview':
        return (
          <View style={styles.stepContent}>
            <Text style={styles.stepTitle}>{currentStep.title}</Text>
            <Text style={styles.stepDescription}>{currentStep.description}</Text>
            
            <View style={styles.previewContainer}>
              {isPreviewActive ? (
                <WebView
                  ref={webViewRef}
                  source={{ uri: `${PI_SERVICE_URL}/camera/preview/stream` }}
                  style={styles.webView}
                  onError={(error) => {
                    console.log('WebView error:', error);
                    setError('Erreur de streaming vidéo');
                  }}
                  scrollEnabled={false}
                  bounces={false}
                  scalesPageToFit={true}
                  startInLoadingState={true}
                  renderLoading={() => (
                    <View style={styles.loadingOverlay}>
                      <ActivityIndicator size="large" color="#007AFF" />
                      <Text style={styles.loadingText}>Chargement du stream...</Text>
                    </View>
                  )}
                />
              ) : (
                <View style={styles.noPreviewContainer}>
                  <Ionicons name="camera" size={48} color="#ccc" />
                  <Text style={styles.noPreviewText}>
                    Prévisualisation non active
                  </Text>
                </View>
              )}
            </View>

            <View style={styles.controlsContainer}>
              <TouchableOpacity
                style={[
                  styles.button,
                  isPreviewActive ? styles.stopButton : styles.startButton,
                  isConnecting && styles.disabledButton
                ]}
                onPress={isPreviewActive ? stopPreview : startPreview}
                disabled={isConnecting || !cameraReady}
              >
                {isConnecting ? (
                  <ActivityIndicator size="small" color="#fff" />
                ) : (
                  <Text style={styles.buttonText}>
                    {isPreviewActive ? 'Arrêter' : 'Démarrer prévisualisation'}
                  </Text>
                )}
              </TouchableOpacity>

              {isPreviewActive && (
                <TouchableOpacity
                  style={[styles.button, styles.nextButton]}
                  onPress={() => setStep(4)}
                >
                  <Text style={styles.buttonText}>Continuer</Text>
                </TouchableOpacity>
              )}
            </View>
          </View>
        );

      case 'test':
        return (
          <View style={styles.stepContent}>
            <Ionicons name="checkmark-circle" size={64} color="#28a745" />
            <Text style={styles.stepTitle}>Installation presque terminée</Text>
            <Text style={styles.stepDescription}>
              Votre caméra est maintenant correctement positionnée. 
              L'installation est terminée !
            </Text>
            
            <View style={styles.statusGrid}>
              <View style={styles.statusItem}>
                <Ionicons 
                  name={piConnected ? "checkmark-circle" : "close-circle"} 
                  size={24} 
                  color={piConnected ? "#28a745" : "#dc3545"} 
                />
                <Text style={styles.statusText}>Pi connecté</Text>
              </View>
              <View style={styles.statusItem}>
                <Ionicons 
                  name={cameraReady ? "checkmark-circle" : "close-circle"} 
                  size={24} 
                  color={cameraReady ? "#28a745" : "#dc3545"} 
                />
                <Text style={styles.statusText}>Caméra opérationnelle</Text>
              </View>
            </View>

            <TouchableOpacity
              style={[styles.primaryButton, styles.completeButton]}
              onPress={completeInstallation}
            >
              <Ionicons name="checkmark" size={20} color="#fff" />
              <Text style={styles.buttonText}>Terminer l'installation</Text>
            </TouchableOpacity>
          </View>
        );

      default:
        return null;
    }
  };

  return (
    <View style={styles.container}>
      <View style={styles.header}>
        <TouchableOpacity
          style={styles.backButton}
          onPress={() => navigation.goBack()}
        >
          <Ionicons name="arrow-back" size={24} color="#007AFF" />
        </TouchableOpacity>
        <Text style={styles.headerTitle}>Installation NightScan Pi</Text>
      </View>

      {renderStepProgress()}
      
      <View style={styles.content}>
        {renderStepContent()}
      </View>

      {error && (
        <View style={styles.errorContainer}>
          <Text style={styles.errorText}>{error}</Text>
        </View>
      )}
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
  progressContainer: {
    flexDirection: 'row',
    justifyContent: 'center',
    alignItems: 'center',
    paddingVertical: 20,
    backgroundColor: '#fff',
    marginBottom: 10,
  },
  progressStep: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  progressDot: {
    width: 32,
    height: 32,
    borderRadius: 16,
    backgroundColor: '#ddd',
    justifyContent: 'center',
    alignItems: 'center',
  },
  activeDot: {
    backgroundColor: '#007AFF',
  },
  completedDot: {
    backgroundColor: '#28a745',
  },
  progressNumber: {
    color: '#666',
    fontWeight: 'bold',
    fontSize: 14,
  },
  progressLine: {
    width: 40,
    height: 2,
    backgroundColor: '#ddd',
    marginHorizontal: 8,
  },
  completedLine: {
    backgroundColor: '#28a745',
  },
  content: {
    flex: 1,
    padding: 20,
  },
  stepContent: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
  },
  stepTitle: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#333',
    marginTop: 20,
    marginBottom: 10,
    textAlign: 'center',
  },
  stepDescription: {
    fontSize: 16,
    color: '#666',
    textAlign: 'center',
    marginBottom: 30,
    lineHeight: 22,
    paddingHorizontal: 20,
  },
  previewContainer: {
    width: '100%',
    height: 200,
    backgroundColor: '#000',
    borderRadius: 12,
    overflow: 'hidden',
    marginBottom: 20,
  },
  webView: {
    flex: 1,
  },
  loadingOverlay: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: 'rgba(0, 0, 0, 0.8)',
  },
  loadingText: {
    color: '#fff',
    marginTop: 10,
  },
  noPreviewContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  noPreviewText: {
    color: '#ccc',
    fontSize: 16,
    marginTop: 10,
  },
  controlsContainer: {
    width: '100%',
    gap: 12,
  },
  primaryButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#007AFF',
    paddingVertical: 16,
    paddingHorizontal: 24,
    borderRadius: 12,
    minWidth: 200,
    gap: 8,
  },
  secondaryButton: {
    backgroundColor: 'transparent',
    borderWidth: 2,
    borderColor: '#007AFF',
    paddingVertical: 12,
    paddingHorizontal: 24,
    borderRadius: 12,
    marginTop: 16,
  },
  button: {
    paddingVertical: 12,
    paddingHorizontal: 24,
    borderRadius: 8,
    alignItems: 'center',
    justifyContent: 'center',
  },
  wakeButton: {
    backgroundColor: '#FF9500',
  },
  startButton: {
    backgroundColor: '#28a745',
  },
  stopButton: {
    backgroundColor: '#dc3545',
  },
  nextButton: {
    backgroundColor: '#007AFF',
  },
  completeButton: {
    backgroundColor: '#28a745',
  },
  disabledButton: {
    backgroundColor: '#ccc',
    opacity: 0.6,
  },
  buttonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: 'bold',
  },
  secondaryButtonText: {
    color: '#007AFF',
    fontSize: 16,
    fontWeight: 'bold',
  },
  statusGrid: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    width: '100%',
    marginVertical: 20,
  },
  statusItem: {
    alignItems: 'center',
    gap: 8,
  },
  statusText: {
    fontSize: 14,
    color: '#666',
  },
  errorContainer: {
    backgroundColor: '#ffebee',
    padding: 16,
    marginHorizontal: 20,
    marginBottom: 20,
    borderRadius: 8,
    borderLeftWidth: 4,
    borderLeftColor: '#f44336',
  },
  errorText: {
    color: '#d32f2f',
    fontSize: 14,
  },
});