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
  setDarkMode: () => {},
  setNotifications: () => {},
});

export function AppProvider({ children }) {
  const [darkMode, setDarkMode] = useState(false);
  const [notifications, setNotifications] = useState(false);

  useEffect(() => {
    async function load() {
      try {
        const raw = await AsyncStorage.getItem(PREFS_KEY);
        if (raw) {
          const prefs = JSON.parse(raw);
          setDarkMode(!!prefs.darkMode);
          setNotifications(!!prefs.notifications);
        }
      } catch {
        // ignore read errors
      }
    }
    load();
  }, []);

  useEffect(() => {
    const prefs = { darkMode, notifications };
    AsyncStorage.setItem(PREFS_KEY, JSON.stringify(prefs)).catch(() => {});
  }, [darkMode, notifications]);

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
    <AppContext.Provider value={{ darkMode, notifications, setDarkMode, setNotifications }}>
      {children}
    </AppContext.Provider>
  );
}
