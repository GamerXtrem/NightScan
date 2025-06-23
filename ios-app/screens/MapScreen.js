import React from 'react';
import { View, StyleSheet } from 'react-native';
import MapView, { Marker } from 'react-native-maps';

export default function MapScreen() {
  const markers = [
    { id: 1, title: 'Fox', coordinate: { latitude: 37.78825, longitude: -122.4324 } },
    { id: 2, title: 'Deer', coordinate: { latitude: 37.78925, longitude: -122.4344 } },
  ];

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
