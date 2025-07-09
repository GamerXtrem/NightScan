import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  Alert,
  ActivityIndicator,
  ScrollView,
  TextInput,
  Switch,
  Modal,
  Platform,
  Dimensions
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import * as Location from 'expo-location';
import MapView, { Marker } from 'react-native-maps';
import { api } from '../services/api';
import { theme } from '../theme';

const { width, height } = Dimensions.get('window');

const LocationSettingsScreen = ({ navigation, route }) => {
  const [currentLocation, setCurrentLocation] = useState(null);
  const [phoneLocation, setPhoneLocation] = useState(null);
  const [loading, setLoading] = useState(false);
  const [mapLoading, setMapLoading] = useState(false);
  const [locationHistory, setLocationHistory] = useState([]);
  const [autoUpdate, setAutoUpdate] = useState(false);
  const [mapVisible, setMapVisible] = useState(false);
  const [manualCoords, setManualCoords] = useState({
    latitude: '',
    longitude: ''
  });
  const [locationPermission, setLocationPermission] = useState(null);

  useEffect(() => {
    loadCurrentLocation();
    loadLocationHistory();
    checkLocationPermission();
  }, []);

  const checkLocationPermission = async () => {
    try {
      const { status } = await Location.getForegroundPermissionsAsync();
      setLocationPermission(status === 'granted');
    } catch (error) {
      console.error('Erreur vérification permission:', error);
    }
  };

  const requestLocationPermission = async () => {
    try {
      const { status } = await Location.requestForegroundPermissionsAsync();
      setLocationPermission(status === 'granted');
      
      if (status === 'granted') {
        Alert.alert(
          'Permission accordée',
          'Vous pouvez maintenant utiliser la géolocalisation de votre téléphone.'
        );
      } else {
        Alert.alert(
          'Permission refusée',
          'La géolocalisation est nécessaire pour définir automatiquement la position du Pi.'
        );
      }
    } catch (error) {
      console.error('Erreur demande permission:', error);
      Alert.alert('Erreur', 'Impossible de demander la permission de géolocalisation.');
    }
  };

  const loadCurrentLocation = async () => {
    try {
      setLoading(true);
      const response = await api.get('/location');
      
      if (response.data.success) {
        setCurrentLocation(response.data.data);
        setManualCoords({
          latitude: response.data.data.latitude.toString(),
          longitude: response.data.data.longitude.toString()
        });
      }
    } catch (error) {
      console.error('Erreur chargement localisation:', error);
      Alert.alert('Erreur', 'Impossible de charger la localisation actuelle.');
    } finally {
      setLoading(false);
    }
  };

  const loadLocationHistory = async () => {
    try {
      const response = await api.get('/location/history?limit=5');
      
      if (response.data.success) {
        setLocationHistory(response.data.data.history);
      }
    } catch (error) {
      console.error('Erreur chargement historique:', error);
    }
  };

  const getCurrentPhoneLocation = async () => {
    try {
      if (!locationPermission) {
        Alert.alert(
          'Permission requise',
          'La permission de géolocalisation est nécessaire.',
          [
            { text: 'Annuler', style: 'cancel' },
            { text: 'Autoriser', onPress: requestLocationPermission }
          ]
        );
        return;
      }

      setLoading(true);
      
      const location = await Location.getCurrentPositionAsync({
        accuracy: Location.Accuracy.High,
        timeout: 10000
      });

      const phoneData = {
        latitude: location.coords.latitude,
        longitude: location.coords.longitude,
        accuracy: location.coords.accuracy,
        timestamp: location.timestamp
      };

      setPhoneLocation(phoneData);
      
      // Proposer de mettre à jour automatiquement
      Alert.alert(
        'Position détectée',
        `Latitude: ${phoneData.latitude.toFixed(6)}\nLongitude: ${phoneData.longitude.toFixed(6)}\nPrécision: ${phoneData.accuracy.toFixed(0)}m\n\nVoulez-vous configurer le Pi à cette position ?`,
        [
          { text: 'Annuler', style: 'cancel' },
          { text: 'Configurer', onPress: () => updatePiLocationFromPhone(phoneData) }
        ]
      );
      
    } catch (error) {
      console.error('Erreur géolocalisation:', error);
      Alert.alert(
        'Erreur de géolocalisation',
        'Impossible d\'obtenir la position actuelle. Vérifiez que le GPS est activé.'
      );
    } finally {
      setLoading(false);
    }
  };

  const updatePiLocationFromPhone = async (phoneData) => {
    try {
      setLoading(true);
      
      const response = await api.post('/location/phone', phoneData);
      
      if (response.data.success) {
        setCurrentLocation(response.data.data);
        Alert.alert(
          'Position mise à jour',
          `La localisation du Pi a été mise à jour avec la position de votre téléphone.`
        );
        loadLocationHistory();
      } else {
        Alert.alert('Erreur', response.data.error || 'Impossible de mettre à jour la position.');
      }
    } catch (error) {
      console.error('Erreur mise à jour:', error);
      Alert.alert('Erreur', 'Impossible de mettre à jour la localisation du Pi.');
    } finally {
      setLoading(false);
    }
  };

  const updateManualLocation = async () => {
    try {
      const latitude = parseFloat(manualCoords.latitude);
      const longitude = parseFloat(manualCoords.longitude);
      
      if (isNaN(latitude) || isNaN(longitude)) {
        Alert.alert('Erreur', 'Veuillez entrer des coordonnées valides.');
        return;
      }

      if (latitude < -90 || latitude > 90) {
        Alert.alert('Erreur', 'La latitude doit être entre -90 et 90.');
        return;
      }

      if (longitude < -180 || longitude > 180) {
        Alert.alert('Erreur', 'La longitude doit être entre -180 et 180.');
        return;
      }

      setLoading(true);
      
      const response = await api.post('/location', {
        latitude,
        longitude,
        source: 'manual'
      });
      
      if (response.data.success) {
        setCurrentLocation(response.data.data);
        Alert.alert(
          'Position mise à jour',
          'La localisation du Pi a été mise à jour manuellement.'
        );
        loadLocationHistory();
      } else {
        Alert.alert('Erreur', response.data.error || 'Impossible de mettre à jour la position.');
      }
    } catch (error) {
      console.error('Erreur mise à jour manuelle:', error);
      Alert.alert('Erreur', 'Impossible de mettre à jour la localisation.');
    } finally {
      setLoading(false);
    }
  };

  const resetToDefault = async () => {
    Alert.alert(
      'Remettre à zéro',
      'Voulez-vous remettre la localisation à la valeur par défaut (Zurich) ?',
      [
        { text: 'Annuler', style: 'cancel' },
        { text: 'Confirmer', onPress: async () => {
          try {
            setLoading(true);
            const response = await api.post('/location/reset');
            
            if (response.data.success) {
              setCurrentLocation(response.data.data);
              setManualCoords({
                latitude: response.data.data.latitude.toString(),
                longitude: response.data.data.longitude.toString()
              });
              Alert.alert('Succès', 'Localisation remise à la valeur par défaut.');
              loadLocationHistory();
            }
          } catch (error) {
            console.error('Erreur reset:', error);
            Alert.alert('Erreur', 'Impossible de remettre à zéro.');
          } finally {
            setLoading(false);
          }
        }}
      ]
    );
  };

  const openMap = () => {
    setMapLoading(true);
    setMapVisible(true);
  };

  const selectLocationOnMap = (coordinate) => {
    setManualCoords({
      latitude: coordinate.latitude.toString(),
      longitude: coordinate.longitude.toString()
    });
  };

  const formatDate = (dateString) => {
    if (!dateString) return 'Date inconnue';
    return new Date(dateString).toLocaleString('fr-FR');
  };

  const getSourceIcon = (source) => {
    switch (source) {
      case 'phone': return 'phone-portrait';
      case 'manual': return 'create';
      case 'gps': return 'navigate';
      case 'reset': return 'refresh';
      default: return 'location';
    }
  };

  const getSourceLabel = (source) => {
    switch (source) {
      case 'phone': return 'Téléphone';
      case 'manual': return 'Manuel';
      case 'gps': return 'GPS';
      case 'reset': return 'Remise à zéro';
      default: return 'Inconnu';
    }
  };

  return (
    <ScrollView style={styles.container}>
      {/* Header */}
      <View style={styles.header}>
        <TouchableOpacity onPress={() => navigation.goBack()}>
          <Ionicons name="arrow-back" size={24} color={theme.colors.white} />
        </TouchableOpacity>
        <Text style={styles.headerTitle}>Localisation du Pi</Text>
        <TouchableOpacity onPress={loadCurrentLocation}>
          <Ionicons name="refresh" size={24} color={theme.colors.white} />
        </TouchableOpacity>
      </View>

      {loading && (
        <View style={styles.loadingContainer}>
          <ActivityIndicator size="large" color={theme.colors.primary} />
          <Text style={styles.loadingText}>Chargement...</Text>
        </View>
      )}

      {/* Position actuelle */}
      {currentLocation && (
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Position Actuelle</Text>
          <View style={styles.locationCard}>
            <View style={styles.locationRow}>
              <Ionicons name="location" size={20} color={theme.colors.primary} />
              <Text style={styles.locationText}>
                {currentLocation.latitude.toFixed(6)}, {currentLocation.longitude.toFixed(6)}
              </Text>
            </View>
            
            {currentLocation.address && (
              <View style={styles.locationRow}>
                <Ionicons name="home" size={20} color={theme.colors.gray} />
                <Text style={styles.addressText}>{currentLocation.address}</Text>
              </View>
            )}
            
            <View style={styles.locationRow}>
              <Ionicons name={getSourceIcon(currentLocation.source)} size={20} color={theme.colors.gray} />
              <Text style={styles.sourceText}>
                {getSourceLabel(currentLocation.source)} • {formatDate(currentLocation.updated_at)}
              </Text>
            </View>
          </View>
        </View>
      )}

      {/* Géolocalisation téléphone */}
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Géolocalisation du Téléphone</Text>
        
        {!locationPermission ? (
          <View style={styles.permissionCard}>
            <Ionicons name="warning" size={40} color={theme.colors.warning} />
            <Text style={styles.permissionText}>
              Permission de géolocalisation requise pour utiliser cette fonctionnalité.
            </Text>
            <TouchableOpacity 
              style={styles.permissionButton}
              onPress={requestLocationPermission}
            >
              <Text style={styles.permissionButtonText}>Autoriser la Géolocalisation</Text>
            </TouchableOpacity>
          </View>
        ) : (
          <TouchableOpacity 
            style={styles.phoneLocationButton}
            onPress={getCurrentPhoneLocation}
            disabled={loading}
          >
            <Ionicons name="phone-portrait" size={24} color={theme.colors.white} />
            <Text style={styles.phoneLocationButtonText}>
              Utiliser la Position du Téléphone
            </Text>
          </TouchableOpacity>
        )}
      </View>

      {/* Saisie manuelle */}
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Saisie Manuelle</Text>
        
        <View style={styles.inputContainer}>
          <Text style={styles.inputLabel}>Latitude</Text>
          <TextInput
            style={styles.input}
            value={manualCoords.latitude}
            onChangeText={(text) => setManualCoords({...manualCoords, latitude: text})}
            placeholder="46.9480"
            keyboardType="numeric"
            placeholderTextColor={theme.colors.gray}
          />
        </View>
        
        <View style={styles.inputContainer}>
          <Text style={styles.inputLabel}>Longitude</Text>
          <TextInput
            style={styles.input}
            value={manualCoords.longitude}
            onChangeText={(text) => setManualCoords({...manualCoords, longitude: text})}
            placeholder="7.4474"
            keyboardType="numeric"
            placeholderTextColor={theme.colors.gray}
          />
        </View>
        
        <View style={styles.buttonRow}>
          <TouchableOpacity 
            style={styles.mapButton}
            onPress={openMap}
          >
            <Ionicons name="map" size={20} color={theme.colors.white} />
            <Text style={styles.mapButtonText}>Carte</Text>
          </TouchableOpacity>
          
          <TouchableOpacity 
            style={styles.updateButton}
            onPress={updateManualLocation}
            disabled={loading}
          >
            <Ionicons name="save" size={20} color={theme.colors.white} />
            <Text style={styles.updateButtonText}>Mettre à Jour</Text>
          </TouchableOpacity>
        </View>
      </View>

      {/* Historique */}
      {locationHistory.length > 0 && (
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Historique des Positions</Text>
          
          {locationHistory.map((location, index) => (
            <View key={index} style={styles.historyItem}>
              <View style={styles.historyHeader}>
                <Ionicons 
                  name={getSourceIcon(location.source)} 
                  size={16} 
                  color={theme.colors.primary} 
                />
                <Text style={styles.historySource}>
                  {getSourceLabel(location.source)}
                </Text>
                <Text style={styles.historyDate}>
                  {formatDate(location.changed_at)}
                </Text>
              </View>
              
              <Text style={styles.historyCoords}>
                {location.latitude.toFixed(6)}, {location.longitude.toFixed(6)}
              </Text>
              
              {location.address && (
                <Text style={styles.historyAddress}>{location.address}</Text>
              )}
            </View>
          ))}
        </View>
      )}

      {/* Actions */}
      <View style={styles.section}>
        <TouchableOpacity 
          style={styles.resetButton}
          onPress={resetToDefault}
        >
          <Ionicons name="refresh" size={20} color={theme.colors.danger} />
          <Text style={styles.resetButtonText}>Remettre à Zéro</Text>
        </TouchableOpacity>
      </View>

      {/* Modal carte */}
      <Modal
        visible={mapVisible}
        animationType="slide"
        onRequestClose={() => setMapVisible(false)}
      >
        <View style={styles.mapContainer}>
          <View style={styles.mapHeader}>
            <TouchableOpacity onPress={() => setMapVisible(false)}>
              <Ionicons name="close" size={24} color={theme.colors.white} />
            </TouchableOpacity>
            <Text style={styles.mapTitle}>Sélectionner une Position</Text>
            <TouchableOpacity onPress={() => setMapVisible(false)}>
              <Text style={styles.mapDone}>Terminé</Text>
            </TouchableOpacity>
          </View>
          
          <MapView
            style={styles.map}
            initialRegion={{
              latitude: currentLocation ? currentLocation.latitude : 46.9480,
              longitude: currentLocation ? currentLocation.longitude : 7.4474,
              latitudeDelta: 0.01,
              longitudeDelta: 0.01,
            }}
            onPress={(event) => selectLocationOnMap(event.nativeEvent.coordinate)}
            onMapReady={() => setMapLoading(false)}
          >
            {currentLocation && (
              <Marker
                coordinate={{
                  latitude: currentLocation.latitude,
                  longitude: currentLocation.longitude
                }}
                title="Position actuelle du Pi"
                pinColor={theme.colors.primary}
              />
            )}
            
            {manualCoords.latitude && manualCoords.longitude && (
              <Marker
                coordinate={{
                  latitude: parseFloat(manualCoords.latitude),
                  longitude: parseFloat(manualCoords.longitude)
                }}
                title="Nouvelle position"
                pinColor={theme.colors.success}
              />
            )}
          </MapView>
          
          {mapLoading && (
            <View style={styles.mapLoading}>
              <ActivityIndicator size="large" color={theme.colors.primary} />
            </View>
          )}
        </View>
      </Modal>
    </ScrollView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: theme.colors.background,
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    backgroundColor: theme.colors.primary,
    paddingHorizontal: 20,
    paddingTop: Platform.OS === 'ios' ? 60 : 40,
    paddingBottom: 20,
  },
  headerTitle: {
    color: theme.colors.white,
    fontSize: 18,
    fontWeight: 'bold',
  },
  loadingContainer: {
    alignItems: 'center',
    padding: 20,
  },
  loadingText: {
    marginTop: 10,
    color: theme.colors.gray,
  },
  section: {
    margin: 20,
    marginBottom: 10,
  },
  sectionTitle: {
    fontSize: 16,
    fontWeight: 'bold',
    color: theme.colors.text,
    marginBottom: 15,
  },
  locationCard: {
    backgroundColor: theme.colors.white,
    borderRadius: 12,
    padding: 15,
    ...theme.shadows.small,
  },
  locationRow: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 10,
  },
  locationText: {
    marginLeft: 10,
    fontSize: 16,
    fontWeight: 'bold',
    color: theme.colors.text,
  },
  addressText: {
    marginLeft: 10,
    fontSize: 14,
    color: theme.colors.gray,
    flex: 1,
  },
  sourceText: {
    marginLeft: 10,
    fontSize: 12,
    color: theme.colors.gray,
  },
  permissionCard: {
    backgroundColor: theme.colors.white,
    borderRadius: 12,
    padding: 20,
    alignItems: 'center',
    ...theme.shadows.small,
  },
  permissionText: {
    textAlign: 'center',
    color: theme.colors.text,
    marginVertical: 15,
  },
  permissionButton: {
    backgroundColor: theme.colors.primary,
    paddingHorizontal: 20,
    paddingVertical: 10,
    borderRadius: 8,
  },
  permissionButtonText: {
    color: theme.colors.white,
    fontWeight: 'bold',
  },
  phoneLocationButton: {
    backgroundColor: theme.colors.success,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    padding: 15,
    borderRadius: 12,
    ...theme.shadows.small,
  },
  phoneLocationButtonText: {
    color: theme.colors.white,
    fontWeight: 'bold',
    marginLeft: 10,
    fontSize: 16,
  },
  inputContainer: {
    marginBottom: 15,
  },
  inputLabel: {
    fontSize: 14,
    fontWeight: 'bold',
    color: theme.colors.text,
    marginBottom: 5,
  },
  input: {
    backgroundColor: theme.colors.white,
    borderRadius: 8,
    padding: 15,
    fontSize: 16,
    color: theme.colors.text,
    ...theme.shadows.small,
  },
  buttonRow: {
    flexDirection: 'row',
    gap: 10,
  },
  mapButton: {
    backgroundColor: theme.colors.info,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    padding: 15,
    borderRadius: 8,
    flex: 1,
  },
  mapButtonText: {
    color: theme.colors.white,
    fontWeight: 'bold',
    marginLeft: 5,
  },
  updateButton: {
    backgroundColor: theme.colors.primary,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    padding: 15,
    borderRadius: 8,
    flex: 1,
  },
  updateButtonText: {
    color: theme.colors.white,
    fontWeight: 'bold',
    marginLeft: 5,
  },
  historyItem: {
    backgroundColor: theme.colors.white,
    borderRadius: 8,
    padding: 15,
    marginBottom: 10,
    ...theme.shadows.small,
  },
  historyHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 5,
  },
  historySource: {
    marginLeft: 5,
    fontWeight: 'bold',
    color: theme.colors.text,
    flex: 1,
  },
  historyDate: {
    fontSize: 12,
    color: theme.colors.gray,
  },
  historyCoords: {
    fontSize: 14,
    color: theme.colors.text,
    marginBottom: 5,
  },
  historyAddress: {
    fontSize: 12,
    color: theme.colors.gray,
  },
  resetButton: {
    backgroundColor: theme.colors.white,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    padding: 15,
    borderRadius: 8,
    borderWidth: 1,
    borderColor: theme.colors.danger,
  },
  resetButtonText: {
    color: theme.colors.danger,
    fontWeight: 'bold',
    marginLeft: 5,
  },
  mapContainer: {
    flex: 1,
  },
  mapHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    backgroundColor: theme.colors.primary,
    paddingHorizontal: 20,
    paddingTop: Platform.OS === 'ios' ? 60 : 40,
    paddingBottom: 20,
  },
  mapTitle: {
    color: theme.colors.white,
    fontSize: 18,
    fontWeight: 'bold',
  },
  mapDone: {
    color: theme.colors.white,
    fontSize: 16,
    fontWeight: 'bold',
  },
  map: {
    flex: 1,
  },
  mapLoading: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: 'rgba(255, 255, 255, 0.8)',
  },
});

export default LocationSettingsScreen;