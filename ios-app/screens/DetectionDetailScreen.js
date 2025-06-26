import React from 'react';
import { View, Text, StyleSheet, Image } from 'react-native';
import MapView, { Marker } from 'react-native-maps';

export default function DetectionDetailScreen({ route }) {
  const { detection } = route.params;

  return (
    <View style={styles.container}>
      <Text style={styles.title}>{detection.species}</Text>
      <Text style={styles.time}>{detection.time}</Text>
      <MapView
        style={styles.map}
        initialRegion={{
          latitude: detection.latitude,
          longitude: detection.longitude,
          latitudeDelta: 0.01,
          longitudeDelta: 0.01,
        }}
      >
        <Marker
          coordinate={{
            latitude: detection.latitude,
            longitude: detection.longitude,
          }}
          title={detection.species}
        />
      </MapView>
      <Image style={styles.image} source={{ uri: detection.image }} />
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  title: {
    fontSize: 20,
    fontWeight: 'bold',
    padding: 16,
  },
  time: {
    paddingHorizontal: 16,
    marginBottom: 8,
  },
  map: {
    height: 200,
  },
  image: {
    flex: 1,
    resizeMode: 'contain',
  },
});
