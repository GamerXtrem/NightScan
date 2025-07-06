import React, { useState, useEffect, useRef } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  Alert,
  ActivityIndicator,
  Image,
  Dimensions,
  AppState
} from 'react-native';
import { WebView } from 'react-native-webview';
import { Audio } from 'expo-av';

const { width, height } = Dimensions.get('window');
const PI_SERVICE_URL = 'http://192.168.4.1:5000';

export default function CameraPreviewScreen({ navigation, route }) {
  const [isConnecting, setIsConnecting] = useState(false);
  const [isPreviewActive, setIsPreviewActive] = useState(false);
  const [piStatus, setPiStatus] = useState(null);
  const [sound, setSound] = useState();
  const [error, setError] = useState(null);
  const webViewRef = useRef(null);
  const appState = useRef(AppState.currentState);
  const [appStateVisible, setAppStateVisible] = useState(appState.current);

  useEffect(() => {
    const subscription = AppState.addEventListener('change', nextAppState => {
      if (
        appState.current.match(/inactive|background/) &&
        nextAppState === 'active'
      ) {
        // App came to foreground
        checkPiStatus();
      } else if (nextAppState.match(/inactive|background/)) {
        // App went to background - stop preview
        stopPreview();
      }

      appState.current = nextAppState;
      setAppStateVisible(appState.current);
    });

    return () => subscription?.remove();
  }, []);

  useEffect(() => {
    // Check Pi status when screen loads
    checkPiStatus();
    
    // Auto-trigger wake sound if coming from wake-up flow
    if (route.params?.autoWake) {
      setTimeout(() => {
        playWakeTone();
      }, 1000);
    }

    return () => {
      // Cleanup when leaving screen
      stopPreview();
      sound?.unloadAsync();
    };
  }, []);

  const checkPiStatus = async () => {
    try {
      const response = await fetch(`${PI_SERVICE_URL}/camera/status`, {
        timeout: 5000,
      });
      
      if (response.ok) {
        const status = await response.json();
        setPiStatus(status);
        setError(null);
      } else {
        throw new Error('Pi not responding');
      }
    } catch (error) {
      console.log('Pi status check failed:', error);
      setError('Pi non accessible. Assurez-vous d\'être connecté au réseau NightScan.');
    }
  };

  const playWakeTone = async () => {
    try {
      const { sound: s } = await Audio.Sound.createAsync(
        require('../assets/wake_tone.wav')
      );
      setSound(s);
      await s.playAsync();
      
      // After playing wake tone, check Pi status
      setTimeout(() => {
        checkPiStatus();
      }, 2000);
    } catch (error) {
      console.log('Wake tone playback failed:', error);
    }
  };

  const startPreview = async () => {
    if (!piStatus?.camera_available) {
      Alert.alert('Erreur', 'Caméra non disponible sur le Pi');
      return;
    }

    setIsConnecting(true);
    setError(null);
    
    try {
      const response = await fetch(`${PI_SERVICE_URL}/camera/preview/start`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
      });

      if (response.ok) {
        setIsPreviewActive(true);
        // Refresh the WebView to start showing the stream
        if (webViewRef.current) {
          webViewRef.current.reload();
        }
      } else {
        const error = await response.json();
        throw new Error(error.error || 'Échec du démarrage de la prévisualisation');
      }
    } catch (error) {
      console.error('Start preview error:', error);
      Alert.alert('Erreur', error.message || 'Impossible de démarrer la prévisualisation');
    } finally {
      setIsConnecting(false);
    }
  };

  const stopPreview = async () => {
    if (!isPreviewActive) return;

    try {
      await fetch(`${PI_SERVICE_URL}/camera/preview/stop`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
      });
      
      setIsPreviewActive(false);
    } catch (error) {
      console.log('Stop preview error:', error);
      // Don't show error to user, just update state
      setIsPreviewActive(false);
    }
  };

  const captureImage = async () => {
    if (!isPreviewActive) {
      Alert.alert('Info', 'Démarrez d\'abord la prévisualisation');
      return;
    }

    try {
      const response = await fetch(`${PI_SERVICE_URL}/camera/capture`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
      });

      if (response.ok) {
        const result = await response.json();
        Alert.alert('Succès', `Image capturée (${Math.round(result.file_size / 1024)}KB)`);
      } else {
        const error = await response.json();
        throw new Error(error.error || 'Échec de la capture');
      }
    } catch (error) {
      console.error('Capture error:', error);
      Alert.alert('Erreur', error.message || 'Impossible de capturer l\'image');
    }
  };

  const renderContent = () => {
    if (error) {
      return (
        <View style={styles.errorContainer}>
          <Text style={styles.errorText}>{error}</Text>
          <TouchableOpacity style={styles.retryButton} onPress={checkPiStatus}>
            <Text style={styles.retryButtonText}>Réessayer</Text>
          </TouchableOpacity>
        </View>
      );
    }

    if (!piStatus) {
      return (
        <View style={styles.loadingContainer}>
          <ActivityIndicator size="large" color="#007AFF" />
          <Text style={styles.loadingText}>Connexion au Pi...</Text>
        </View>
      );
    }

    return (
      <View style={styles.contentContainer}>
        <View style={styles.statusContainer}>
          <Text style={styles.statusText}>
            Statut: {piStatus.camera_available ? 'Caméra OK' : 'Caméra indisponible'}
          </Text>
          {piStatus.sensor_type && (
            <Text style={styles.statusText}>
              Capteur: {piStatus.sensor_type}
            </Text>
          )}
        </View>

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
              onLoadStart={() => console.log('Stream loading...')}
              onLoadEnd={() => console.log('Stream loaded')}
              scrollEnabled={false}
              bounces={false}
              scalesPageToFit={true}
              startInLoadingState={true}
              renderLoading={() => (
                <View style={styles.loadingOverlay}>
                  <ActivityIndicator size="large" color="#007AFF" />
                  <Text>Chargement du stream...</Text>
                </View>
              )}
            />
          ) : (
            <View style={styles.noPreviewContainer}>
              <Text style={styles.noPreviewText}>
                Prévisualisation non active
              </Text>
              <Text style={styles.helpText}>
                Appuyez sur "Démarrer prévisualisation" pour voir le flux caméra
              </Text>
            </View>
          )}
        </View>

        <View style={styles.controlsContainer}>
          <TouchableOpacity
            style={[styles.button, styles.wakeButton]}
            onPress={playWakeTone}
          >
            <Text style={styles.buttonText}>🔊 Réveiller Pi</Text>
          </TouchableOpacity>

          <TouchableOpacity
            style={[
              styles.button,
              isPreviewActive ? styles.stopButton : styles.startButton,
              isConnecting && styles.disabledButton
            ]}
            onPress={isPreviewActive ? stopPreview : startPreview}
            disabled={isConnecting || !piStatus?.camera_available}
          >
            {isConnecting ? (
              <ActivityIndicator size="small" color="#fff" />
            ) : (
              <Text style={styles.buttonText}>
                {isPreviewActive ? '⏹️ Arrêter' : '▶️ Démarrer prévisualisation'}
              </Text>
            )}
          </TouchableOpacity>

          <TouchableOpacity
            style={[styles.button, styles.captureButton]}
            onPress={captureImage}
            disabled={!isPreviewActive}
          >
            <Text style={styles.buttonText}>📷 Capturer</Text>
          </TouchableOpacity>
        </View>
      </View>
    );
  };

  return (
    <View style={styles.container}>
      <View style={styles.header}>
        <TouchableOpacity
          style={styles.backButton}
          onPress={() => navigation.goBack()}
        >
          <Text style={styles.backButtonText}>← Retour</Text>
        </TouchableOpacity>
        <Text style={styles.headerTitle}>Prévisualisation Caméra</Text>
      </View>

      {renderContent()}

      <View style={styles.footer}>
        <Text style={styles.footerText}>
          Utilisez cette vue pour ajuster l'angle de la caméra lors de l'installation
        </Text>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#000',
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: 16,
    backgroundColor: '#1a1a1a',
    paddingTop: 50,
  },
  backButton: {
    marginRight: 16,
  },
  backButtonText: {
    color: '#007AFF',
    fontSize: 16,
  },
  headerTitle: {
    color: '#fff',
    fontSize: 18,
    fontWeight: 'bold',
  },
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  loadingText: {
    color: '#fff',
    marginTop: 16,
    fontSize: 16,
  },
  errorContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 20,
  },
  errorText: {
    color: '#ff4444',
    fontSize: 16,
    textAlign: 'center',
    marginBottom: 20,
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
  },
  contentContainer: {
    flex: 1,
  },
  statusContainer: {
    backgroundColor: '#1a1a1a',
    padding: 12,
    borderBottomWidth: 1,
    borderBottomColor: '#333',
  },
  statusText: {
    color: '#fff',
    fontSize: 14,
    marginBottom: 4,
  },
  previewContainer: {
    flex: 1,
    backgroundColor: '#000',
    margin: 8,
    borderRadius: 8,
    overflow: 'hidden',
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
  noPreviewContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 20,
  },
  noPreviewText: {
    color: '#ccc',
    fontSize: 18,
    fontWeight: 'bold',
    marginBottom: 10,
  },
  helpText: {
    color: '#888',
    fontSize: 14,
    textAlign: 'center',
  },
  controlsContainer: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    padding: 16,
    backgroundColor: '#1a1a1a',
    flexWrap: 'wrap',
  },
  button: {
    paddingHorizontal: 16,
    paddingVertical: 12,
    borderRadius: 8,
    minWidth: 80,
    alignItems: 'center',
    marginVertical: 4,
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
  captureButton: {
    backgroundColor: '#007AFF',
  },
  disabledButton: {
    backgroundColor: '#555',
    opacity: 0.6,
  },
  buttonText: {
    color: '#fff',
    fontSize: 14,
    fontWeight: 'bold',
  },
  footer: {
    padding: 16,
    backgroundColor: '#1a1a1a',
  },
  footerText: {
    color: '#888',
    fontSize: 12,
    textAlign: 'center',
  },
});