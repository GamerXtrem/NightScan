import React, { useState, useEffect, useContext } from 'react';
import {
  View,
  Text,
  StyleSheet,
  FlatList,
  TouchableOpacity,
  TextInput,
  Button,
  Share,
} from 'react-native';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { fetchDetections } from '../services/api';
import { AppContext } from '../AppContext';

export default function DetectionListScreen({ navigation }) {
  const { zoneFilter, setZoneFilter } = useContext(AppContext);
  const [detections, setDetections] = useState([]);
  const [query, setQuery] = useState('');
  const [refreshing, setRefreshing] = useState(false);

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
      await Share.share({ message: csv });
    } catch {
      // ignore share errors
    }
  }

  const renderItem = ({ item }) => (
    <TouchableOpacity
      onPress={() => navigation.navigate('DetectionDetail', { detection: item })}
    >
      <View style={styles.item}>
        <Text style={styles.species}>{item.species}</Text>
        <Text style={styles.time}>{item.time}</Text>
      </View>
    </TouchableOpacity>
  );

  return (
    <View style={styles.container}>
      <TextInput
        style={styles.search}
        placeholder="Search species"
        value={query}
        onChangeText={setQuery}
      />
      <TextInput
        style={styles.search}
        placeholder="Filter by zone"
        value={zoneFilter}
        onChangeText={setZoneFilter}
      />
      <Button title="Export CSV" onPress={handleExport} />
      <FlatList
        style={styles.list}
        data={filtered}
        renderItem={renderItem}
        keyExtractor={(item) => item.id.toString()}
        refreshing={refreshing}
        onRefresh={refresh}
      />
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  search: {
    paddingHorizontal: 16,
    paddingVertical: 8,
  },
  list: {
    flex: 1,
    paddingHorizontal: 16,
  },
  item: {
    paddingVertical: 8,
    borderBottomWidth: StyleSheet.hairlineWidth,
    borderColor: '#ccc',
  },
  species: {
    fontSize: 16,
  },
  time: {
    color: '#555',
  },
});
