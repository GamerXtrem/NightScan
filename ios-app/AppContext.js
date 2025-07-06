import React, { createContext, useState, useEffect } from 'react';
import AsyncStorage from '@react-native-async-storage/async-storage';
import {
  registerForPushNotificationsAsync,
  checkForNewDetections,
} from './services/notifications';
import { fetchDetections } from './services/api';
import authService from './services/auth';

const PREFS_KEY = 'settings';

export const AppContext = createContext({
  darkMode: false,
  notifications: false,
  zoneFilter: '',
  isAuthenticated: false,
  authState: null,
  setDarkMode: () => {},
  setNotifications: () => {},
  setZoneFilter: () => {},
  setAuthState: () => {},
});

export function AppProvider({ children }) {
  const [darkMode, setDarkMode] = useState(false);
  const [notifications, setNotifications] = useState(false);
  const [zoneFilter, setZoneFilter] = useState('');
  const [authState, setAuthState] = useState(null);
  const [isAuthenticated, setIsAuthenticated] = useState(false);

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

  // Initialize auth service and monitor auth state
  useEffect(() => {
    const updateAuthState = () => {
      const currentAuthState = authService.getAuthState();
      setAuthState(currentAuthState);
      setIsAuthenticated(authService.isAuthenticated());
    };

    // Initial auth state
    updateAuthState();

    // Set up a periodic check for auth state changes
    const authCheckInterval = setInterval(updateAuthState, 5000);

    return () => {
      clearInterval(authCheckInterval);
    };
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
        isAuthenticated,
        authState,
        setDarkMode,
        setNotifications,
        setZoneFilter,
        setAuthState,
      }}
    >
      {children}
    </AppContext.Provider>
  );
}
