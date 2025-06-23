import React from 'react';
import { View, Text, StyleSheet, Button } from 'react-native';

export default function HomeScreen({ navigation }) {
  return (
    <View style={styles.container}>
      <Text style={styles.title}>Welcome to NightScan iOS</Text>
      <Button
        title="Start Scan"
        onPress={() => navigation.navigate('Scan')}
      />
      <View style={styles.spacing} />
      <Button
        title="View Map"
        onPress={() => navigation.navigate('Map')}
      />
      <View style={styles.spacing} />
      <Button
        title="View Detections"
        onPress={() => navigation.navigate('Detections')}
      />
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  title: {
    fontSize: 18,
    marginBottom: 12,
  },
  spacing: {
    height: 12,
  },
});
