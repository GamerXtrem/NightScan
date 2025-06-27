import React, { useEffect, useState, useContext } from 'react';
import { View, StyleSheet, TextInput } from 'react-native';
import MapView, { Marker } from 'react-native-maps';
import { fetchDetections } from '../services/api';
import { AppContext } from '../AppContext';

export default function MapScreen() {
  const { zoneFilter, setZoneFilter } = useContext(AppContext);
  const [allMarkers, setAllMarkers] = useState([]);

  useEffect(() => {
    fetchDetections()
      .then((list) =>
        setAllMarkers(
          list.map((d) => ({
            id: d.id,
            zone: d.zone,
            title: d.species,
            coordinate: { latitude: d.latitude, longitude: d.longitude },
          }))
        )
      )
      .catch(() => {});
  }, []);

  const markers = allMarkers.filter((m) =>
    !zoneFilter || (m.zone || '').toLowerCase().includes(zoneFilter.toLowerCase())
  );

  return (
    <View style={styles.container}>
      <TextInput
        style={styles.filter}
        placeholder="Filter by zone"
        value={zoneFilter}
        onChangeText={setZoneFilter}
      />
      <MapView
        style={styles.map}
        initialRegion={{
          latitude: 37.78825,
          longitude: -122.4324,
          latitudeDelta: 0.0922,
          longitudeDelta: 0.0421,
        }}
      >
        {markers.map((m) => (
          <Marker
            key={m.id}
            testID={`marker-${m.id}`}
            coordinate={m.coordinate}
            title={m.title}
          />
        ))}
      </MapView>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  filter: {
    position: 'absolute',
    top: 10,
    left: 10,
    right: 10,
    backgroundColor: '#fff',
    paddingHorizontal: 8,
    paddingVertical: 4,
    zIndex: 1,
  },
  map: {
    ...StyleSheet.absoluteFillObject,
  },
});
