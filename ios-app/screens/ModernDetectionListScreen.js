import React, { useState, useEffect, useContext } from 'react';
import {
  View,
  StyleSheet,
  FlatList,
  TouchableOpacity,
  TextInput,
  Share,
  ImageBackground,
  Alert,
} from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { Ionicons } from '@expo/vector-icons';
import AsyncStorage from '@react-native-async-storage/async-storage';

import GlassCard from '../components/ui/GlassCard';
import GlassButton from '../components/ui/GlassButton';
import Typography, { Heading, Body, Label } from '../components/ui/Typography';
import Theme from '../theme/DesignSystem';
import { fetchDetections } from '../services/api';
import { AppContext } from '../AppContext';

export default function ModernDetectionListScreen({ navigation }) {
  const { zoneFilter, setZoneFilter } = useContext(AppContext);
  const [detections, setDetections] = useState([]);
  const [query, setQuery] = useState('');
  const [refreshing, setRefreshing] = useState(false);
  const [activeTab, setActiveTab] = useState('list'); // 'list' or 'map'

  const STORAGE_KEY = 'detections';

  useEffect(() => {
    async function load() {
      try {
        const raw = await AsyncStorage.getItem(STORAGE_KEY);
        if (raw) {
          setDetections(JSON.parse(raw));
        }
      } catch {
        // ignore read errors
      }
      refresh();
    }
    load();
  }, []);

  async function refresh() {
    setRefreshing(true);
    try {
      const list = await fetchDetections();
      setDetections(list);
      await AsyncStorage.setItem(STORAGE_KEY, JSON.stringify(list));
    } catch {
      // ignore network errors
    } finally {
      setRefreshing(false);
    }
  }

  const filtered = detections.filter(
    (d) =>
      d.species.toLowerCase().includes(query.toLowerCase()) &&
      (!zoneFilter || (d.zone || '').toLowerCase().includes(zoneFilter.toLowerCase()))
  );

  async function handleExport() {
    const header = 'id,species,time,latitude,longitude,zone';
    const rows = detections.map((d) =>
      [d.id, d.species, d.time, d.latitude, d.longitude, d.zone || ''].join(',')
    );
    const csv = [header, ...rows].join('\n');
    try {
      await Share.share({ 
        message: csv,
        title: 'Détections NightScan'
      });
    } catch {
      Alert.alert('Erreur', 'Impossible d\'exporter les données');
    }
  }

  const renderHeader = () => (
    <View style={styles.header}>
      <Heading level={1} color={Theme.colors.neutral[0]} weight="bold">
        Détections
      </Heading>
      <Body color={Theme.colors.neutral[200]} style={styles.subtitle}>
        {filtered.length} détection{filtered.length !== 1 ? 's' : ''} trouvée{filtered.length !== 1 ? 's' : ''}
      </Body>
      
      {/* Tab Switcher */}
      <View style={styles.tabSwitcher}>
        <GlassButton
          title="Liste"
          icon="list"
          variant={activeTab === 'list' ? 'primary' : 'ghost'}
          size="small"
          onPress={() => setActiveTab('list')}
          style={styles.tabButton}
        />
        <GlassButton
          title="Carte"
          icon="map"
          variant={activeTab === 'map' ? 'primary' : 'ghost'}
          size="small"
          onPress={() => setActiveTab('map')}
          style={styles.tabButton}
        />
      </View>
    </View>
  );

  const renderFilters = () => (
    <GlassCard variant="medium" style={styles.filtersCard}>
      <View style={styles.filtersHeader}>
        <Ionicons name="filter" size={20} color={Theme.colors.primary[500]} />
        <Heading level={6} style={styles.filtersTitle}>
          Filtres et Actions
        </Heading>
      </View>

      <View style={styles.searchInputs}>
        <View style={styles.inputContainer}>
          <Ionicons 
            name="search" 
            size={18} 
            color={Theme.colors.neutral[400]} 
            style={styles.inputIcon}
          />
          <TextInput
            style={styles.searchInput}
            placeholder="Rechercher une espèce..."
            placeholderTextColor={Theme.colors.neutral[500]}
            value={query}
            onChangeText={setQuery}
          />
        </View>

        <View style={styles.inputContainer}>
          <Ionicons 
            name="location" 
            size={18} 
            color={Theme.colors.neutral[400]} 
            style={styles.inputIcon}
          />
          <TextInput
            style={styles.searchInput}
            placeholder="Filtrer par zone..."
            placeholderTextColor={Theme.colors.neutral[500]}
            value={zoneFilter}
            onChangeText={setZoneFilter}
          />
        </View>
      </View>

      <View style={styles.actionButtons}>
        <GlassButton
          title="Actualiser"
          icon="refresh"
          variant="secondary"
          size="small"
          onPress={refresh}
          style={styles.actionButton}
          disabled={refreshing}
        />
        
        <GlassButton
          title="Exporter CSV"
          icon="download"
          variant="primary"
          size="small"
          onPress={handleExport}
          style={styles.actionButton}
          disabled={detections.length === 0}
        />
      </View>
    </GlassCard>
  );

  const renderDetectionItem = ({ item }) => (
    <TouchableOpacity
      onPress={() => navigation.navigate('DetectionDetail', { detection: item })}
      style={styles.detectionItemWrapper}
    >
      <GlassCard variant="light" style={styles.detectionItem}>
        <View style={styles.detectionHeader}>
          <View style={styles.speciesInfo}>
            <Heading level={6} color={Theme.colors.neutral[0]} weight="medium">
              {item.species}
            </Heading>
            {item.zone && (
              <View style={styles.zoneTag}>
                <Ionicons name="location" size={12} color={Theme.colors.secondary[400]} />
                <Label size="small" color={Theme.colors.secondary[400]} style={styles.zoneText}>
                  {item.zone}
                </Label>
              </View>
            )}
          </View>
          
          <View style={styles.timeInfo}>
            <Label size="small" color={Theme.colors.neutral[400]}>
              {new Date(item.time).toLocaleDateString('fr-FR')}
            </Label>
            <Label size="small" color={Theme.colors.neutral[300]}>
              {new Date(item.time).toLocaleTimeString('fr-FR', { 
                hour: '2-digit', 
                minute: '2-digit' 
              })}
            </Label>
          </View>
        </View>

        {(item.latitude && item.longitude) && (
          <View style={styles.coordinates}>
            <Ionicons name="navigate" size={14} color={Theme.colors.info} />
            <Label size="small" color={Theme.colors.neutral[400]}>
              {item.latitude.toFixed(6)}, {item.longitude.toFixed(6)}
            </Label>
          </View>
        )}

        <View style={styles.detectionFooter}>
          <View style={styles.confidenceIndicator}>
            <View style={[styles.confidenceDot, { backgroundColor: Theme.colors.success }]} />
            <Label size="small" color={Theme.colors.success}>
              Détection confirmée
            </Label>
          </View>
          
          <Ionicons name="chevron-forward" size={16} color={Theme.colors.neutral[500]} />
        </View>
      </GlassCard>
    </TouchableOpacity>
  );

  const renderEmptyState = () => (
    <GlassCard variant="medium" style={styles.emptyCard}>
      <View style={styles.emptyContent}>
        <Ionicons name="search" size={48} color={Theme.colors.neutral[600]} />
        <Heading level={5} color={Theme.colors.neutral[300]} style={styles.emptyTitle}>
          Aucune détection trouvée
        </Heading>
        <Body color={Theme.colors.neutral[500]} style={styles.emptyText}>
          {detections.length === 0 
            ? "Aucune détection n'a encore été enregistrée"
            : "Aucune détection ne correspond aux filtres actuels"
          }
        </Body>
        
        {detections.length === 0 && (
          <GlassButton
            title="Actualiser les données"
            icon="refresh"
            variant="primary"
            size="medium"
            onPress={refresh}
            style={styles.emptyButton}
            disabled={refreshing}
          />
        )}
      </View>
    </GlassCard>
  );

  const renderMapView = () => (
    <GlassCard variant="medium" style={styles.mapCard}>
      <View style={styles.mapContainer}>
        <Ionicons name="map" size={48} color={Theme.colors.neutral[600]} />
        <Heading level={5} color={Theme.colors.neutral[300]} style={styles.mapTitle}>
          Carte Interactive
        </Heading>
        <Body color={Theme.colors.neutral[500]} style={styles.mapText}>
          Visualisation géographique des détections
        </Body>
        <Body color={Theme.colors.neutral[600]} style={styles.mapPlaceholder}>
          [Carte des détections sera affichée ici]
        </Body>
        
        {/* Legend */}
        <View style={styles.mapLegend}>
          <View style={styles.legendItem}>
            <View style={[styles.legendDot, { backgroundColor: Theme.colors.success }]} />
            <Label size="small" color={Theme.colors.neutral[400]}>
              Détections récentes
            </Label>
          </View>
          <View style={styles.legendItem}>
            <View style={[styles.legendDot, { backgroundColor: Theme.colors.warning }]} />
            <Label size="small" color={Theme.colors.neutral[400]}>
              Zone surveillée
            </Label>
          </View>
        </View>
      </View>
    </GlassCard>
  );

  const renderContent = () => {
    if (activeTab === 'map') {
      return renderMapView();
    }
    
    if (filtered.length === 0) {
      return renderEmptyState();
    }
    
    return (
      <FlatList
        style={styles.list}
        contentContainerStyle={styles.listContent}
        data={filtered}
        renderItem={renderDetectionItem}
        keyExtractor={(item) => item.id.toString()}
        refreshing={refreshing}
        onRefresh={refresh}
        showsVerticalScrollIndicator={false}
      />
    );
  };

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
        <View style={styles.container}>
          {renderHeader()}
          {activeTab === 'list' && renderFilters()}
          {renderContent()}
        </View>
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
    paddingTop: Theme.layout.safeAreaTop + Theme.spacing[4],
    paddingHorizontal: Theme.spacing[4],
  },
  
  // Header
  header: {
    marginBottom: Theme.spacing[6],
  },
  subtitle: {
    marginTop: Theme.spacing[1],
  },
  
  // Tab Switcher
  tabSwitcher: {
    flexDirection: 'row',
    marginTop: Theme.spacing[4],
    gap: Theme.spacing[2],
  },
  tabButton: {
    flex: 1,
  },
  
  // Filters Card
  filtersCard: {
    marginBottom: Theme.spacing[4],
  },
  filtersHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: Theme.spacing[4],
  },
  filtersTitle: {
    marginLeft: Theme.spacing[2],
  },
  searchInputs: {
    gap: Theme.spacing[3],
    marginBottom: Theme.spacing[4],
  },
  inputContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: 'rgba(255, 255, 255, 0.1)',
    borderRadius: Theme.borderRadius.md,
    paddingHorizontal: Theme.spacing[3],
    paddingVertical: Theme.spacing[2],
    borderWidth: 1,
    borderColor: 'rgba(255, 255, 255, 0.2)',
  },
  inputIcon: {
    marginRight: Theme.spacing[2],
  },
  searchInput: {
    flex: 1,
    color: Theme.colors.neutral[0],
    fontSize: Theme.typography.sizes.md,
    fontFamily: Theme.typography.fonts.regular,
  },
  actionButtons: {
    flexDirection: 'row',
    gap: Theme.spacing[2],
  },
  actionButton: {
    flex: 1,
  },
  
  // Detection List
  list: {
    flex: 1,
  },
  listContent: {
    paddingBottom: Theme.spacing[8] + 60, // Tab bar height
  },
  detectionItemWrapper: {
    marginBottom: Theme.spacing[3],
  },
  detectionItem: {
    padding: Theme.spacing[4],
  },
  detectionHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
    marginBottom: Theme.spacing[3],
  },
  speciesInfo: {
    flex: 1,
  },
  zoneTag: {
    flexDirection: 'row',
    alignItems: 'center',
    marginTop: Theme.spacing[1],
  },
  zoneText: {
    marginLeft: Theme.spacing[1],
  },
  timeInfo: {
    alignItems: 'flex-end',
  },
  coordinates: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: Theme.spacing[3],
  },
  detectionFooter: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  confidenceIndicator: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  confidenceDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
    marginRight: Theme.spacing[2],
  },
  
  // Empty State
  emptyCard: {
    marginTop: Theme.spacing[8],
  },
  emptyContent: {
    alignItems: 'center',
    paddingVertical: Theme.spacing[6],
  },
  emptyTitle: {
    marginTop: Theme.spacing[4],
    marginBottom: Theme.spacing[2],
  },
  emptyText: {
    textAlign: 'center',
    marginBottom: Theme.spacing[6],
    paddingHorizontal: Theme.spacing[4],
  },
  emptyButton: {
    minWidth: 180,
  },
  
  // Map View
  mapCard: {
    marginTop: Theme.spacing[4],
    flex: 1,
  },
  mapContainer: {
    alignItems: 'center',
    paddingVertical: Theme.spacing[8],
    minHeight: 400,
    justifyContent: 'center',
  },
  mapTitle: {
    marginTop: Theme.spacing[4],
    marginBottom: Theme.spacing[2],
  },
  mapText: {
    textAlign: 'center',
    marginBottom: Theme.spacing[4],
  },
  mapPlaceholder: {
    textAlign: 'center',
    fontStyle: 'italic',
    marginBottom: Theme.spacing[6],
  },
  mapLegend: {
    flexDirection: 'row',
    gap: Theme.spacing[6],
    marginTop: Theme.spacing[4],
  },
  legendItem: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: Theme.spacing[2],
  },
  legendDot: {
    width: 12,
    height: 12,
    borderRadius: 6,
  },
});