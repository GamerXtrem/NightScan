import React, { useState } from 'react';
import { View, StyleSheet, Button, Alert } from 'react-native';
import * as DocumentPicker from 'expo-document-picker';
import { uploadMedia } from '../services/api';

export default function ScanScreen() {
  const [uploading, setUploading] = useState(false);

  const handlePick = async () => {
    const result = await DocumentPicker.getDocumentAsync({});
    if (result.type === 'cancel' || result.canceled) {
      return;
    }
    const file = result.assets ? result.assets[0] : result;
    setUploading(true);
    try {
      await uploadMedia(file.uri, file.mimeType);
      Alert.alert('Upload complete');
    } catch {
      Alert.alert('Upload failed');
    } finally {
      setUploading(false);
    }
  };

  return (
    <View style={styles.container}>
      <Button
        title={uploading ? 'Uploading...' : 'Select File'}
        onPress={handlePick}
        disabled={uploading}
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
});
