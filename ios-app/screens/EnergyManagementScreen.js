import React, { useState, useEffect, useRef } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  Alert,
  ActivityIndicator,
  ScrollView,
  Switch,
  AppState
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';

const PI_SERVICE_URL = 'http://192.168.4.1:5000';

export default function EnergyManagementScreen({ navigation }) {
  const [loading, setLoading] = useState(true);
  const [energyStatus, setEnergyStatus] = useState(null);
  const [wifiActive, setWifiActive] = useState(false);
  const [error, setError] = useState(null);
  const [activatingWifi, setActivatingWifi] = useState(false);
  
  const statusInterval = useRef(null);
  const appState = useRef(AppState.currentState);

  useEffect(() => {
    const subscription = AppState.addEventListener('change', nextAppState => {
      if (nextAppState.match(/inactive|background/)) {
        stopStatusMonitoring();
      } else if (nextAppState === 'active' && appState.current.match(/inactive|background/)) {
        loadEnergyStatus();
      }
      appState.current = nextAppState;
    });

    return () => {
      subscription?.remove();
      stopStatusMonitoring();
    };
  }, []);

  useEffect(() => {
    loadEnergyStatus();
    return () => stopStatusMonitoring();
  }, []);

  const loadEnergyStatus = async () => {
    try {
      setError(null);
      setLoading(true);

      const [energyResponse, wifiResponse] = await Promise.all([
        fetch(`${PI_SERVICE_URL}/energy/status`),
        fetch(`${PI_SERVICE_URL}/energy/wifi/status`)
      ]);

      if (energyResponse.ok && wifiResponse.ok) {
        const energyData = await energyResponse.json();
        const wifiData = await wifiResponse.json();

        setEnergyStatus(energyData);
        setWifiActive(wifiData.wifi_active);
        
        startStatusMonitoring();
      } else {
        throw new Error('Impossible de charger le statut énergétique');
      }
    } catch (err) {
      console.error('Load energy status error:', err);
      setError('Pi non accessible. Vérifiez la connexion.');
    } finally {
      setLoading(false);
    }
  };

  const startStatusMonitoring = () => {
    if (statusInterval.current) return;

    statusInterval.current = setInterval(async () => {
      try {
        const [energyResponse, wifiResponse] = await Promise.all([
          fetch(`${PI_SERVICE_URL}/energy/status`),
          fetch(`${PI_SERVICE_URL}/energy/wifi/status`)
        ]);

        if (energyResponse.ok && wifiResponse.ok) {
          const energyData = await energyResponse.json();
          const wifiData = await wifiResponse.json();
          
          setEnergyStatus(energyData);
          setWifiActive(wifiData.wifi_active);
        }
      } catch (err) {
        // Silently fail for status updates
      }
    }, 5000); // Update every 5 seconds
  };

  const stopStatusMonitoring = () => {
    if (statusInterval.current) {
      clearInterval(statusInterval.current);
      statusInterval.current = null;
    }
  };

  const toggleWifi = async () => {
    if (activatingWifi) return;

    try {
      setActivatingWifi(true);
      
      if (wifiActive) {
        // Deactivate WiFi
        const response = await fetch(`${PI_SERVICE_URL}/energy/wifi/deactivate`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
        });

        if (response.ok) {
          setWifiActive(false);
          Alert.alert('Succès', 'WiFi désactivé');
        } else {
          throw new Error('Échec de la désactivation WiFi');
        }
      } else {
        // Show duration selection
        Alert.alert(
          'Activer WiFi',
          'Durée d\'activation :',
          [
            { text: 'Annuler', style: 'cancel' },
            { text: '10 min', onPress: () => activateWifi(10) },
            { text: '30 min', onPress: () => activateWifi(30) },
            { text: '60 min', onPress: () => activateWifi(60) }
          ]
        );
      }
    } catch (err) {
      console.error('WiFi toggle error:', err);
      Alert.alert('Erreur', 'Impossible de modifier le statut WiFi');
    } finally {
      setActivatingWifi(false);
    }
  };

  const activateWifi = async (durationMinutes) => {
    try {
      const response = await fetch(`${PI_SERVICE_URL}/energy/wifi/activate`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ duration_minutes: durationMinutes }),
      });

      if (response.ok) {
        setWifiActive(true);
        Alert.alert('Succès', `WiFi activé pour ${durationMinutes} minutes`);
      } else {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Échec de l\'activation WiFi');
      }
    } catch (err) {
      console.error('WiFi activation error:', err);
      Alert.alert('Erreur', 'Impossible d\'activer le WiFi');
    }
  };

  const extendWifi = async () => {
    if (!wifiActive) return;

    Alert.alert(
      'Prolonger WiFi',
      'Temps supplémentaire :',
      [
        { text: 'Annuler', style: 'cancel' },
        { text: '10 min', onPress: () => performExtendWifi(10) },
        { text: '30 min', onPress: () => performExtendWifi(30) }
      ]
    );
  };

  const performExtendWifi = async (additionalMinutes) => {
    try {
      const response = await fetch(`${PI_SERVICE_URL}/energy/wifi/extend`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ additional_minutes: additionalMinutes }),
      });

      if (response.ok) {
        Alert.alert('Succès', `WiFi prolongé de ${additionalMinutes} minutes`);
      } else {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Échec de la prolongation WiFi');
      }
    } catch (err) {
      console.error('WiFi extend error:', err);
      Alert.alert('Erreur', 'Impossible de prolonger le WiFi');
    }
  };

  const getOperationModeIcon = (mode) => {
    switch (mode) {
      case 'camera_active': return 'camera';
      case 'audio_only': return 'mic';
      case 'sleep': return 'moon';
      case 'minimal': return 'battery-half';
      default: return 'help';
    }
  };

  const getOperationModeText = (mode) => {
    switch (mode) {
      case 'camera_active': return 'Caméra + Audio';
      case 'audio_only': return 'Audio seulement';
      case 'sleep': return 'Veille';
      case 'minimal': return 'Minimal';
      default: return 'Inconnu';
    }
  };

  const getOperationModeColor = (mode) => {
    switch (mode) {
      case 'camera_active': return '#4CAF50';
      case 'audio_only': return '#FF9500';
      case 'sleep': return '#6C757D';
      case 'minimal': return '#007AFF';
      default: return '#6C757D';
    }
  };

  const renderEnergyStatus = () => {
    if (!energyStatus) return null;

    const modeColor = getOperationModeColor(energyStatus.operation_mode);

    return (
      <View style={styles.statusContainer}>
        <Text style={styles.sectionTitle}>État Énergétique</Text>
        
        <View style={styles.statusCard}>
          <View style={[styles.modeIndicator, { backgroundColor: modeColor }]}>
            <Ionicons 
              name={getOperationModeIcon(energyStatus.operation_mode)} 
              size={24} 
              color="#fff" 
            />
          </View>
          <View style={styles.modeInfo}>
            <Text style={styles.modeTitle}>
              {getOperationModeText(energyStatus.operation_mode)}
            </Text>
            <Text style={styles.modeDescription}>
              {energyStatus.camera_active ? 'Caméra active' : 'Caméra inactive'} • 
              {energyStatus.audio_only ? ' Audio activé' : ' Audio variable'}
            </Text>
          </View>
        </View>

        <View style={styles.detailsGrid}>
          <View style={styles.detailItem}>
            <Ionicons name="camera" size={20} color="#666" />
            <Text style={styles.detailLabel}>Caméra</Text>
            <View style={[
              styles.statusDot, 
              { backgroundColor: energyStatus.camera_active ? '#4CAF50' : '#6C757D' }
            ]} />
          </View>
          
          <View style={styles.detailItem}>
            <Ionicons name="wifi" size={20} color="#666" />
            <Text style={styles.detailLabel}>WiFi</Text>
            <View style={[
              styles.statusDot, 
              { backgroundColor: wifiActive ? '#4CAF50' : '#6C757D' }
            ]} />
          </View>
          
          <View style={styles.detailItem}>
            <Ionicons name="cog" size={20} color="#666" />
            <Text style={styles.detailLabel}>Processus</Text>
            <View style={[
              styles.statusDot, 
              { backgroundColor: energyStatus.main_process_running ? '#4CAF50' : '#6C757D' }
            ]} />
          </View>
        </View>
      </View>
    );
  };

  const renderWifiControl = () => {
    return (
      <View style={styles.controlContainer}>
        <Text style={styles.sectionTitle}>Contrôle WiFi</Text>
        
        <View style={styles.wifiCard}>
          <View style={styles.wifiHeader}>
            <View style={styles.wifiInfo}>
              <Ionicons 
                name="wifi" 
                size={24} 
                color={wifiActive ? '#4CAF50' : '#6C757D'} 
              />
              <Text style={[
                styles.wifiStatus,
                { color: wifiActive ? '#4CAF50' : '#6C757D' }
              ]}>
                {wifiActive ? 'Actif' : 'Inactif'}
              </Text>
            </View>
            
            <Switch
              value={wifiActive}
              onValueChange={toggleWifi}
              disabled={activatingWifi}
              trackColor={{ false: '#D1D5DB', true: '#4CAF50' }}
              thumbColor={wifiActive ? '#fff' : '#f4f3f4'}
            />
          </View>
          
          <Text style={styles.wifiDescription}>
            {wifiActive 
              ? 'WiFi est actuellement activé. Il se désactivera automatiquement après expiration.'
              : 'WiFi est désactivé pour économiser l\'énergie. Activez-le pour la configuration.'}
          </Text>
          
          {wifiActive && (
            <TouchableOpacity 
              style={styles.extendButton}
              onPress={extendWifi}
            >
              <Ionicons name="time" size={16} color="#007AFF" />
              <Text style={styles.extendButtonText}>Prolonger</Text>
            </TouchableOpacity>
          )}
        </View>
      </View>
    );
  };

  if (loading) {
    return (
      <View style={styles.loadingContainer}>
        <ActivityIndicator size="large" color="#007AFF" />
        <Text style={styles.loadingText}>Chargement statut énergétique...</Text>
      </View>
    );
  }

  if (error) {
    return (
      <View style={styles.errorContainer}>
        <Ionicons name="warning" size={64} color="#ff4444" />
        <Text style={styles.errorText}>{error}</Text>
        <TouchableOpacity style={styles.retryButton} onPress={loadEnergyStatus}>
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
        <Text style={styles.headerTitle}>Gestion Énergétique</Text>
      </View>

      <ScrollView style={styles.content} showsVerticalScrollIndicator={false}>
        {renderEnergyStatus()}
        {renderWifiControl()}
        
        <View style={styles.infoContainer}>
          <Text style={styles.infoTitle}>Économie d'Énergie</Text>
          <Text style={styles.infoText}>
            • La caméra s'active automatiquement 30 minutes avant le coucher du soleil et 30 minutes après le lever{'\n'}
            • Le WiFi ne s'active que sur demande pour économiser l'énergie{'\n'}
            • Le système entre en veille pendant les heures de jour (11h-17h)
          </Text>
        </View>
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
  sectionTitle: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 16,
  },
  statusContainer: {
    marginBottom: 20,
  },
  statusCard: {
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 16,
    flexDirection: 'row',
    alignItems: 'center',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
    marginBottom: 16,
  },
  modeIndicator: {
    width: 50,
    height: 50,
    borderRadius: 25,
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: 16,
  },
  modeInfo: {
    flex: 1,
  },
  modeTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 4,
  },
  modeDescription: {
    fontSize: 14,
    color: '#666',
  },
  detailsGrid: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 16,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  detailItem: {
    alignItems: 'center',
    flex: 1,
  },
  detailLabel: {
    fontSize: 12,
    color: '#666',
    marginTop: 4,
    marginBottom: 8,
  },
  statusDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
  },
  controlContainer: {
    marginBottom: 20,
  },
  wifiCard: {
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 16,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  wifiHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 12,
  },
  wifiInfo: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  wifiStatus: {
    fontSize: 16,
    fontWeight: 'bold',
    marginLeft: 8,
  },
  wifiDescription: {
    fontSize: 14,
    color: '#666',
    lineHeight: 20,
  },
  extendButton: {
    flexDirection: 'row',
    alignItems: 'center',
    alignSelf: 'flex-start',
    marginTop: 12,
    paddingVertical: 8,
    paddingHorizontal: 12,
    backgroundColor: '#f0f8ff',
    borderRadius: 8,
  },
  extendButtonText: {
    color: '#007AFF',
    fontSize: 14,
    fontWeight: '500',
    marginLeft: 4,
  },
  infoContainer: {
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 16,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  infoTitle: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 12,
  },
  infoText: {
    fontSize: 14,
    color: '#666',
    lineHeight: 20,
  },
});