import React, { useState, useEffect, useRef } from 'react';
import {
  View,
  StyleSheet,
  ScrollView,
  RefreshControl,
  Dimensions,
  AppState,
  ImageBackground,
} from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { Ionicons } from '@expo/vector-icons';
import { Audio } from 'expo-av';

import GlassCard from '../components/ui/GlassCard';
import GlassButton from '../components/ui/GlassButton';
import Typography, { Heading, Body, Label } from '../components/ui/Typography';
import Theme from '../theme/DesignSystem';

const { width: screenWidth } = Dimensions.get('window');

export default function ModernHomeScreen({ navigation }) {
  const [sound, setSound] = useState();
  const [refreshing, setRefreshing] = useState(false);
  const [systemStatus, setSystemStatus] = useState(null);
  const [recentDetections, setRecentDetections] = useState([]);
  const [energyStatus, setEnergyStatus] = useState(null);
  
  const statusInterval = useRef(null);
  const appState = useRef(AppState.currentState);

  useEffect(() => {
    const subscription = AppState.addEventListener('change', nextAppState => {
      if (nextAppState === 'active' && appState.current.match(/inactive|background/)) {
        loadDashboardData();
      }
      appState.current = nextAppState;
    });

    loadDashboardData();
    startStatusMonitoring();

    return () => {
      subscription?.remove();
      stopStatusMonitoring();
      if (sound) {
        sound.unloadAsync().catch(() => {});
      }
    };
  }, []);

  const loadDashboardData = async () => {
    try {
      // Load system status, recent detections, and energy status
      // This would call your API endpoints
      await Promise.all([
        loadSystemStatus(),
        loadRecentDetections(),
        loadEnergyStatus(),
      ]);
    } catch (error) {
      console.error('Dashboard data loading error:', error);
    }
  };

  const loadSystemStatus = async () => {
    // Mock data - replace with actual API call
    setSystemStatus({
      isOnline: true,
      lastSeen: new Date(),
      batteryLevel: 78,
      cameraActive: true,
      audioActive: true,
      wifiActive: false,
    });
  };

  const loadRecentDetections = async () => {
    // Mock data - replace with actual API call
    setRecentDetections([
      { id: 1, species: 'Renard', time: '23:45', confidence: 0.92 },
      { id: 2, species: 'Chouette', time: '22:15', confidence: 0.88 },
      { id: 3, species: 'Hérisson', time: '21:30', confidence: 0.95 },
    ]);
  };

  const loadEnergyStatus = async () => {
    // Mock data - replace with actual API call
    setEnergyStatus({
      mode: 'camera_active',
      batteryLevel: 78,
      estimatedHours: 24,
    });
  };

  const startStatusMonitoring = () => {
    if (statusInterval.current) return;
    
    statusInterval.current = setInterval(() => {
      loadSystemStatus();
    }, 30000); // Update every 30 seconds
  };

  const stopStatusMonitoring = () => {
    if (statusInterval.current) {
      clearInterval(statusInterval.current);
      statusInterval.current = null;
    }
  };

  const onRefresh = React.useCallback(async () => {
    setRefreshing(true);
    await loadDashboardData();
    setRefreshing(false);
  }, []);

  const playWakeTone = async () => {
    try {
      const { sound: s } = await Audio.Sound.createAsync(
        require('../assets/wake_tone.wav')
      );
      setSound(s);
      await s.replayAsync();
    } catch {
      // ignore playback errors
    }
  };

  const getStatusColor = (isActive) => {
    return isActive ? Theme.colors.success : Theme.colors.neutral[400];
  };

  const getEnergyModeText = (mode) => {
    switch (mode) {
      case 'camera_active': return 'Caméra Active';
      case 'audio_only': return 'Audio Seulement';
      case 'sleep': return 'En Veille';
      default: return 'Inconnu';
    }
  };

  const getEnergyModeIcon = (mode) => {
    switch (mode) {
      case 'camera_active': return 'camera';
      case 'audio_only': return 'mic';
      case 'sleep': return 'moon';
      default: return 'help';
    }
  };

  const renderHeader = () => (
    <View style={styles.header}>
      <View>
        <Heading level={1} color={Theme.colors.neutral[0]} weight="bold">
          NightScan
        </Heading>
        <Body color={Theme.colors.neutral[200]} style={styles.subtitle}>
          Surveillance Nocturne Intelligente
        </Body>
      </View>
      
      <GlassButton
        icon="notifications"
        variant="glass"
        size="medium"
        onPress={() => navigation.navigate('Detections')}
        style={styles.notificationButton}
      />
    </View>
  );

  const renderSystemStatus = () => {
    if (!systemStatus) return null;

    return (
      <GlassCard variant="medium" style={styles.statusCard}>
        <View style={styles.statusHeader}>
          <View style={styles.statusTitleRow}>
            <Ionicons name="hardware-chip" size={24} color={Theme.colors.primary[500]} />
            <Heading level={5} style={styles.cardTitle}>
              État du Système
            </Heading>
          </View>
          <View style={[
            styles.statusIndicator,
            { backgroundColor: systemStatus.isOnline ? Theme.colors.success : Theme.colors.error }
          ]} />
        </View>

        <View style={styles.statusGrid}>
          <View style={styles.statusItem}>
            <Ionicons name="camera" size={20} color={getStatusColor(systemStatus.cameraActive)} />
            <Label size="small" color={Theme.colors.neutral[600]}>
              Caméra
            </Label>
          </View>
          
          <View style={styles.statusItem}>
            <Ionicons name="mic" size={20} color={getStatusColor(systemStatus.audioActive)} />
            <Label size="small" color={Theme.colors.neutral[600]}>
              Audio
            </Label>
          </View>
          
          <View style={styles.statusItem}>
            <Ionicons name="wifi" size={20} color={getStatusColor(systemStatus.wifiActive)} />
            <Label size="small" color={Theme.colors.neutral[600]}>
              WiFi
            </Label>
          </View>
          
          <View style={styles.statusItem}>
            <Ionicons name="battery-half" size={20} color={getStatusColor(systemStatus.batteryLevel > 20)} />
            <Label size="small" color={Theme.colors.neutral[600]}>
              {systemStatus.batteryLevel}%
            </Label>
          </View>
        </View>
      </GlassCard>
    );
  };

  const renderQuickActions = () => (
    <GlassCard variant="medium" style={styles.quickActionsCard}>
      <Heading level={5} style={styles.cardTitle}>
        Actions Rapides
      </Heading>
      
      <View style={styles.quickActionsGrid}>
        <GlassButton
          title="Scanner Audio"
          icon="scan"
          variant="glass"
          size="medium"
          onPress={() => navigation.navigate('Scan')}
          style={styles.quickActionButton}
        />
        
        <GlassButton
          title="Réveiller Pi"
          icon="power"
          variant="glass"
          size="medium"
          onPress={playWakeTone}
          style={styles.quickActionButton}
        />
        
        <GlassButton
          title="Carte Live"
          icon="map"
          variant="glass"
          size="medium"
          onPress={() => navigation.navigate('Map')}
          style={styles.quickActionButton}
        />
        
        <GlassButton
          title="Paramètres"
          icon="settings"
          variant="glass"
          size="medium"
          onPress={() => navigation.navigate('Settings')}
          style={styles.quickActionButton}
        />
      </View>
    </GlassCard>
  );

  const renderEnergyStatus = () => {
    if (!energyStatus) return null;

    return (
      <GlassCard variant="medium" style={styles.energyCard}>
        <View style={styles.energyHeader}>
          <View style={styles.energyTitleRow}>
            <Ionicons 
              name={getEnergyModeIcon(energyStatus.mode)} 
              size={24} 
              color={Theme.colors.secondary[500]} 
            />
            <Heading level={5} style={styles.cardTitle}>
              Gestion Énergétique
            </Heading>
          </View>
          
          <GlassButton
            icon="settings"
            variant="ghost"
            size="small"
            onPress={() => navigation.navigate('EnergyManagement')}
          />
        </View>

        <View style={styles.energyContent}>
          <View style={styles.energyModeRow}>
            <Label color={Theme.colors.neutral[600]}>Mode Actuel:</Label>
            <Body weight="medium" color={Theme.colors.secondary[600]}>
              {getEnergyModeText(energyStatus.mode)}
            </Body>
          </View>
          
          <View style={styles.energyStatsRow}>
            <View style={styles.energyStat}>
              <Body weight="bold" color={Theme.colors.primary[500]}>
                {energyStatus.batteryLevel}%
              </Body>
              <Label size="small" color={Theme.colors.neutral[600]}>
                Batterie
              </Label>
            </View>
            
            <View style={styles.energyStat}>
              <Body weight="bold" color={Theme.colors.primary[500]}>
                {energyStatus.estimatedHours}h
              </Body>
              <Label size="small" color={Theme.colors.neutral[600]}>
                Autonomie
              </Label>
            </View>
          </View>
        </View>
      </GlassCard>
    );
  };

  const renderRecentDetections = () => (
    <GlassCard variant="medium" style={styles.detectionsCard}>
      <View style={styles.detectionsHeader}>
        <Heading level={5} style={styles.cardTitle}>
          Détections Récentes
        </Heading>
        
        <GlassButton
          title="Voir Tout"
          variant="ghost"
          size="small"
          onPress={() => navigation.navigate('Detections')}
        />
      </View>

      {recentDetections.length > 0 ? (
        <View style={styles.detectionsList}>
          {recentDetections.slice(0, 3).map((detection) => (
            <View key={detection.id} style={styles.detectionItem}>
              <View style={styles.detectionInfo}>
                <Body weight="medium">{detection.species}</Body>
                <Label size="small" color={Theme.colors.neutral[600]}>
                  {detection.time} • {Math.round(detection.confidence * 100)}% confiance
                </Label>
              </View>
              <Ionicons name="chevron-forward" size={16} color={Theme.colors.neutral[400]} />
            </View>
          ))}
        </View>
      ) : (
        <View style={styles.emptyState}>
          <Ionicons name="moon" size={48} color={Theme.colors.neutral[400]} />
          <Body color={Theme.colors.neutral[600]} align="center">
            Aucune détection récente{'\n'}La nuit est calme...
          </Body>
        </View>
      )}
    </GlassCard>
  );

  return (
    <ImageBackground
      source={{ uri: 'data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNDAwIiBoZWlnaHQ9IjQwMCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48ZGVmcz48bGluZWFyR3JhZGllbnQgaWQ9ImEiIHgxPSIwJSIgeTE9IjAlIiB4Mj0iMTAwJSIgeTI9IjEwMCUiPjxzdG9wIG9mZnNldD0iMCUiIHN0b3AtY29sb3I9IiMxODJBM0EiLz48c3RvcCBvZmZzZXQ9IjEwMCUiIHN0b3AtY29sb3I9IiMwRjE0MjAiLz48L2xpbmVhckdyYWRpZW50PjwvZGVmcz48cmVjdCB3aWR0aD0iMTAwJSIgaGVpZ2h0PSIxMDAlIiBmaWxsPSJ1cmwoI2EpIi8+PC9zdmc+' }}
      style={styles.backgroundImage}
      resizeMode="cover"
    >
      <LinearGradient
        colors={['rgba(15, 20, 32, 0.8)', 'rgba(24, 42, 58, 0.9)']}
        style={styles.gradientOverlay}
      >
        <ScrollView
          style={styles.container}
          contentContainerStyle={styles.scrollContent}
          refreshControl={
            <RefreshControl
              refreshing={refreshing}
              onRefresh={onRefresh}
              tintColor={Theme.colors.neutral[0]}
            />
          }
          showsVerticalScrollIndicator={false}
        >
          {renderHeader()}
          {renderSystemStatus()}
          {renderQuickActions()}
          {renderEnergyStatus()}
          {renderRecentDetections()}
        </ScrollView>
      </LinearGradient>
    </ImageBackground>
  );
}

const styles = StyleSheet.create({
  backgroundImage: {
    flex: 1,
  },
  gradientOverlay: {
    flex: 1,
  },
  container: {
    flex: 1,
  },
  scrollContent: {
    paddingTop: Theme.layout.safeAreaTop + Theme.spacing[4],
    paddingHorizontal: Theme.spacing[4],
    paddingBottom: Theme.spacing[8],
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
    marginBottom: Theme.spacing[6],
  },
  subtitle: {
    marginTop: Theme.spacing[1],
  },
  notificationButton: {
    width: 44,
    height: 44,
  },
  
  // Status Card
  statusCard: {
    marginBottom: Theme.spacing[4],
  },
  statusHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: Theme.spacing[4],
  },
  statusTitleRow: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  statusIndicator: {
    width: 12,
    height: 12,
    borderRadius: 6,
  },
  statusGrid: {
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  statusItem: {
    alignItems: 'center',
    flex: 1,
  },
  
  // Quick Actions
  quickActionsCard: {
    marginBottom: Theme.spacing[4],
  },
  quickActionsGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    marginTop: Theme.spacing[3],
    gap: Theme.spacing[2],
  },
  quickActionButton: {
    width: (screenWidth - Theme.spacing[4] * 2 - Theme.spacing[4] * 2 - Theme.spacing[2]) / 2,
  },
  
  // Energy Card
  energyCard: {
    marginBottom: Theme.spacing[4],
  },
  energyHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: Theme.spacing[3],
  },
  energyTitleRow: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  energyContent: {
    gap: Theme.spacing[3],
  },
  energyModeRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  energyStatsRow: {
    flexDirection: 'row',
    justifyContent: 'space-around',
  },
  energyStat: {
    alignItems: 'center',
  },
  
  // Detections Card
  detectionsCard: {
    marginBottom: Theme.spacing[4],
  },
  detectionsHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: Theme.spacing[3],
  },
  detectionsList: {
    gap: Theme.spacing[2],
  },
  detectionItem: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: Theme.spacing[2],
    borderBottomWidth: 1,
    borderBottomColor: 'rgba(255, 255, 255, 0.1)',
  },
  detectionInfo: {
    flex: 1,
  },
  emptyState: {
    alignItems: 'center',
    paddingVertical: Theme.spacing[6],
    gap: Theme.spacing[3],
  },
  
  // Common
  cardTitle: {
    marginLeft: Theme.spacing[2],
  },
});