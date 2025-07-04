import React, { createContext, useState, useEffect } from 'react';
import AsyncStorage from '@react-native-async-storage/async-storage';
import {
  registerForPushNotificationsAsync,
  checkForNewDetections,
} from './services/notifications';
import { fetchDetections } from './services/api';

const PREFS_KEY = 'settings';

export const AppContext = createContext({
  darkMode: false,
  notifications: false,
  zoneFilter: '',
  setDarkMode: () => {},
  setNotifications: () => {},
  setZoneFilter: () => {},
});

export function AppProvider({ children }) {
  const [darkMode, setDarkMode] = useState(false);
  const [notifications, setNotifications] = useState(false);
  const [zoneFilter, setZoneFilter] = useState('');

  useEffect(() => {
    async function load() {
      try {
        const raw = await AsyncStorage.getItem(PREFS_KEY);
        if (raw) {
          const prefs = JSON.parse(raw);
          setDarkMode(!!prefs.darkMode);
          setNotifications(!!prefs.notifications);
          if (prefs.zoneFilter) setZoneFilter(prefs.zoneFilter);
        }
      } catch {
        // ignore read errors
      }
    }
    load();
  }, []);

  useEffect(() => {
    const prefs = { darkMode, notifications, zoneFilter };
    AsyncStorage.setItem(PREFS_KEY, JSON.stringify(prefs)).catch(() => {});
  }, [darkMode, notifications, zoneFilter]);

  useEffect(() => {
    let timer;
    async function poll() {
      try {
        const list = await fetchDetections();
        checkForNewDetections(list);
      } catch {
        // ignore errors
      }
    }
    if (notifications) {
      registerForPushNotificationsAsync().catch(() => {});
      poll();
      timer = setInterval(poll, 60000);
    }
    return () => {
      if (timer) clearInterval(timer);
    };
  }, [notifications]);

  return (
    <AppContext.Provider
      value={{
        darkMode,
        notifications,
        zoneFilter,
        setDarkMode,
        setNotifications,
        setZoneFilter,
      }}
    >
      {children}
    </AppContext.Provider>
  );
}
