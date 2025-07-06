import React, { useState, useEffect } from 'react';
import { View, Text, StyleSheet, Button } from 'react-native';
import { Audio } from 'expo-av';

export default function HomeScreen({ navigation }) {
  const [sound, setSound] = useState();

  useEffect(() => {
    return sound
      ? () => {
          sound.unloadAsync().catch(() => {});
        }
      : undefined;
  }, [sound]);

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
      <View style={styles.spacing} />
      <Button title="Login" onPress={() => navigation.navigate('Login')} />
      <View style={styles.spacing} />
      <Button title="Register" onPress={() => navigation.navigate('Register')} />
      <View style={styles.spacing} />
      <Button title="Réveiller NightScanPi" onPress={playWakeTone} />
      <View style={styles.spacing} />
      <Button 
        title="📷 Prévisualisation Caméra" 
        onPress={() => navigation.navigate('CameraPreview')} 
      />
      <View style={styles.spacing} />
      <Button 
        title="🔧 Installation Pi" 
        onPress={() => navigation.navigate('PiInstallation')} 
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
