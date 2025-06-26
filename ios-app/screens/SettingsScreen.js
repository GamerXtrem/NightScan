import React, { useContext, useState, useEffect } from 'react';
import { View, Switch, Text, StyleSheet } from 'react-native';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { AppContext } from '../AppContext';

const PREFS_KEY = 'settings';

export default function SettingsScreen() {
  const { darkMode, setDarkMode } = useContext(AppContext);
  const [notifications, setNotifications] = useState(false);

  useEffect(() => {
    async function load() {
      try {
        const raw = await AsyncStorage.getItem(PREFS_KEY);
        if (raw) {
          const prefs = JSON.parse(raw);
          setNotifications(!!prefs.notifications);
        }
      } catch (e) {
        // ignore read errors
      }
    }
    load();
  }, []);

  useEffect(() => {
    const prefs = { darkMode, notifications };
    AsyncStorage.setItem(PREFS_KEY, JSON.stringify(prefs)).catch(() => {});
  }, [darkMode, notifications]);

  return (
    <View style={styles.container}>
      <View style={styles.row}>
        <Text style={styles.label}>Dark Mode</Text>
        <Switch value={darkMode} onValueChange={setDarkMode} />
      </View>
      <View style={styles.row}>
        <Text style={styles.label}>Notifications</Text>
        <Switch value={notifications} onValueChange={setNotifications} />
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    padding: 16,
  },
  row: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingVertical: 12,
  },
  label: {
    fontSize: 16,
  },
});
