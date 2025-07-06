import React, { useContext, useEffect, useRef } from 'react';
import { NavigationContainer, DefaultTheme, DarkTheme } from '@react-navigation/native';
import { StatusBar } from 'expo-status-bar';
import { createNativeStackNavigator } from '@react-navigation/native-stack';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import { Ionicons } from '@expo/vector-icons';
import HomeScreen from './screens/HomeScreen';
import ScanScreen from './screens/ScanScreen';
import MapScreen from './screens/MapScreen';
import DetectionListScreen from './screens/DetectionListScreen';
import DetectionDetailScreen from './screens/DetectionDetailScreen';
import LoginScreen from './screens/LoginScreen';
import RegisterScreen from './screens/RegisterScreen';
import SettingsScreen from './screens/SettingsScreen';
import { AppProvider, AppContext } from './AppContext';
import { initializeNotificationService, setupNotificationActionHandlers, removeNotificationListeners, handleWebSocketNotification } from './services/notifications';
import websocketService from './services/websocket';

const Stack = createNativeStackNavigator();
const Tab = createBottomTabNavigator();

function MainTabs() {
  return (
    <Tab.Navigator
      screenOptions={({ route }) => ({
        tabBarIcon: ({ focused, color, size }) => {
          let iconName;

          if (route.name === 'Map') {
            iconName = focused ? 'map' : 'map-outline';
          } else if (route.name === 'Detections') {
            iconName = focused ? 'list' : 'list-outline';
          } else if (route.name === 'Scan') {
            iconName = focused ? 'scan' : 'scan-outline';
          } else if (route.name === 'Settings') {
            iconName = focused ? 'settings' : 'settings-outline';
          }

          return <Ionicons name={iconName} size={size} color={color} />;
        },
        tabBarActiveTintColor: '#4CAF50',
        tabBarInactiveTintColor: 'gray',
        tabBarStyle: {
          backgroundColor: 'white',
          borderTopWidth: 0,
          elevation: 10,
          shadowOpacity: 0.1,
          shadowRadius: 10,
          shadowOffset: { width: 0, height: -5 },
        },
      })}
    >
      <Tab.Screen 
        name="Map" 
        component={MapScreen}
        options={{
          title: 'Live Map',
          headerShown: false,
        }}
      />
      <Tab.Screen 
        name="Detections" 
        component={DetectionListScreen}
        options={{
          title: 'Detections',
          headerShown: false,
        }}
      />
      <Tab.Screen 
        name="Scan" 
        component={ScanScreen}
        options={{
          title: 'Scan Audio',
          headerShown: false,
        }}
      />
      <Tab.Screen 
        name="Settings" 
        component={SettingsScreen}
        options={{
          title: 'Settings',
          headerShown: false,
        }}
      />
    </Tab.Navigator>
  );
}

function RootNavigator() {
  const { darkMode, isAuthenticated } = useContext(AppContext);
  const navigationRef = useRef();
  const notificationListeners = useRef(null);

  useEffect(() => {
    // Initialize notification service
    const initNotifications = async () => {
      try {
        const { pushToken, preferences } = await initializeNotificationService();
        console.log('Notifications initialized:', { pushToken, preferences });
        
        // Set up notification action handlers
        if (navigationRef.current) {
          notificationListeners.current = setupNotificationActionHandlers(navigationRef.current);
        }
      } catch (error) {
        console.error('Failed to initialize notifications:', error);
      }
    };

    initNotifications();

    // Initialize WebSocket service if authenticated
    if (isAuthenticated) {
      const initWebSocket = async () => {
        try {
          await websocketService.connect();
          
          // Set up WebSocket notification handler
          websocketService.addEventListener('notification', handleWebSocketNotification);
          
          console.log('WebSocket initialized');
        } catch (error) {
          console.error('Failed to initialize WebSocket:', error);
        }
      };

      initWebSocket();
    }

    // Cleanup function
    return () => {
      if (notificationListeners.current) {
        removeNotificationListeners(notificationListeners.current);
      }
      
      if (isAuthenticated) {
        websocketService.removeEventListener('notification', handleWebSocketNotification);
        websocketService.disconnect();
      }
    };
  }, [isAuthenticated]);

  return (
    <NavigationContainer 
      ref={navigationRef}
      theme={darkMode ? DarkTheme : DefaultTheme}
    >
      <Stack.Navigator
        screenOptions={{
          headerStyle: {
            backgroundColor: darkMode ? '#1a1a1a' : '#4CAF50',
          },
          headerTintColor: '#fff',
          headerTitleStyle: {
            fontWeight: 'bold',
          },
        }}
      >
        <Stack.Screen 
          name="Home" 
          component={HomeScreen}
          options={{
            title: 'NightScan',
            headerShown: false,
          }}
        />
        <Stack.Screen 
          name="Login" 
          component={LoginScreen}
          options={{
            title: 'Sign In',
            presentation: 'modal',
          }}
        />
        <Stack.Screen 
          name="Register" 
          component={RegisterScreen}
          options={{
            title: 'Create Account',
            presentation: 'modal',
          }}
        />
        <Stack.Screen
          name="Main"
          component={MainTabs}
          options={{ 
            headerShown: false,
            gestureEnabled: false,
          }}
        />
        <Stack.Screen 
          name="DetectionDetail" 
          component={DetectionDetailScreen}
          options={({ route }) => ({
            title: route.params?.species || 'Detection Details',
            headerBackTitleVisible: false,
          })}
        />
      </Stack.Navigator>
      <StatusBar style={darkMode ? "light" : "dark"} />
    </NavigationContainer>
  );
}

export default function App() {
  return (
    <AppProvider>
      <RootNavigator />
    </AppProvider>
  );
}
