import React, { useEffect, useState } from 'react';
import { View, StyleSheet } from 'react-native';
import MapView, { Marker } from 'react-native-maps';
import { fetchDetections } from '../services/api';

export default function MapScreen() {
  const [markers, setMarkers] = useState([]);

  useEffect(() => {
    fetchDetections()
      .then((list) =>
        setMarkers(
          list.map((d) => ({
            id: d.id,
            title: d.species,
            coordinate: { latitude: d.latitude, longitude: d.longitude },
          }))
        )
      )
      .catch(() => {});
  }, []);

  return (
    <View style={styles.container}>
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
          <Marker key={m.id} coordinate={m.coordinate} title={m.title} />
        ))}
      </MapView>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  map: {
    ...StyleSheet.absoluteFillObject,
  },
});
