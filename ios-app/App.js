import React, { useContext, useEffect, useRef } from 'react';
import { NavigationContainer, DefaultTheme, DarkTheme } from '@react-navigation/native';
import { StatusBar } from 'expo-status-bar';
import { createNativeStackNavigator } from '@react-navigation/native-stack';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import { Ionicons } from '@expo/vector-icons';
import { registerRootComponent } from 'expo';

import ModernHomeScreen from './screens/ModernHomeScreen';
import MapScreen from './screens/MapScreen';
import ModernDetectionListScreen from './screens/ModernDetectionListScreen';
import DetectionDetailScreen from './screens/DetectionDetailScreen';
import LoginScreen from './screens/LoginScreen';
import RegisterScreen from './screens/RegisterScreen';
import ModernSettingsScreen from './screens/ModernSettingsScreen';
import PinSetupScreen from './screens/PinSetupScreen';
import PinEntryScreen from './screens/PinEntryScreen';
import CameraPreviewScreen from './screens/CameraPreviewScreen';
import PiInstallationScreen from './screens/PiInstallationScreen';
import AudioThresholdScreen from './screens/AudioThresholdScreen';
import EnergyManagementScreen from './screens/EnergyManagementScreen';
import { AppProvider, AppContext } from './AppContext';
import { initializeNotificationService, setupNotificationActionHandlers, removeNotificationListeners, handleWebSocketNotification } from './services/notifications';
import websocketService from './services/websocket';
import AuthWrapper from './components/AuthWrapper';

const Stack = createNativeStackNavigator();
const Tab = createBottomTabNavigator();

function MainTabs() {
  return (
    <Tab.Navigator
      screenOptions={({ route }) => ({
        tabBarIcon: ({ focused, color, size }) => {
          let iconName;

          if (route.name === 'Accueil') {
            iconName = focused ? 'home' : 'home-outline';
          } else if (route.name === 'Détections') {
            iconName = focused ? 'list' : 'list-outline';
          } else if (route.name === 'Paramètres') {
            iconName = focused ? 'settings' : 'settings-outline';
          }

          return <Ionicons name={iconName} size={size} color={color} />;
        },
        tabBarActiveTintColor: '#4CAF50',
        tabBarInactiveTintColor: 'rgba(255, 255, 255, 0.6)',
        tabBarStyle: {
          backgroundColor: 'rgba(0, 0, 0, 0.8)',
          borderTopWidth: 0,
          elevation: 10,
          shadowOpacity: 0.3,
          shadowRadius: 15,
          shadowOffset: { width: 0, height: -5 },
          position: 'absolute',
          borderTopLeftRadius: 20,
          borderTopRightRadius: 20,
          paddingTop: 5,
        },
        tabBarLabelStyle: {
          fontSize: 12,
          fontWeight: '500',
        },
      })}
    >
      <Tab.Screen 
        name="Accueil" 
        component={ModernHomeScreen}
        options={{
          title: 'Accueil',
          headerShown: false,
        }}
      />
      <Tab.Screen 
        name="Détections" 
        component={ModernDetectionListScreen}
        options={{
          title: 'Détections',
          headerShown: false,
        }}
      />
      <Tab.Screen 
        name="Paramètres" 
        component={ModernSettingsScreen}
        options={{
          title: 'Paramètres',
          headerShown: false,
        }}
      />
    </Tab.Navigator>
  );
}

function RootNavigator() {
  const { darkMode, isAuthenticated, authState } = useContext(AppContext);
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
        initialRouteName="Main"
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
          name="Main"
          options={{ 
            headerShown: false,
            gestureEnabled: false,
          }}
        >
          {(props) => (
            <AuthWrapper {...props}>
              <MainTabs />
            </AuthWrapper>
          )}
        </Stack.Screen>
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
          name="PinSetup" 
          component={PinSetupScreen}
          options={{
            title: 'Configuration PIN',
            headerShown: false,
            gestureEnabled: false,
          }}
        />
        <Stack.Screen 
          name="PinEntry" 
          component={PinEntryScreen}
          options={{
            title: 'Authentification',
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
        <Stack.Screen 
          name="CameraPreview" 
          component={CameraPreviewScreen}
          options={{
            title: 'Camera Preview',
            headerShown: false,
            presentation: 'modal',
          }}
        />
        <Stack.Screen 
          name="PiInstallation" 
          component={PiInstallationScreen}
          options={{
            title: 'Pi Installation',
            headerShown: false,
            presentation: 'modal',
          }}
        />
        <Stack.Screen 
          name="AudioThreshold" 
          component={AudioThresholdScreen}
          options={{
            title: 'Audio Threshold',
            headerShown: false,
            presentation: 'modal',
          }}
        />
        <Stack.Screen 
          name="EnergyManagement" 
          component={EnergyManagementScreen}
          options={{
            title: 'Energy Management',
            headerShown: false,
            presentation: 'modal',
          }}
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

// Register the main component for Expo
registerRootComponent(App);