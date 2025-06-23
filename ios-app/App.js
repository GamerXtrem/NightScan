import React from 'react';
import { NavigationContainer } from '@react-navigation/native';
import { createNativeStackNavigator } from '@react-navigation/native-stack';
import HomeScreen from './screens/HomeScreen';
import ScanScreen from './screens/ScanScreen';
import MapScreen from './screens/MapScreen';
import DetectionListScreen from './screens/DetectionListScreen';

const Stack = createNativeStackNavigator();

export default function App() {
  return (
    <NavigationContainer>
      <Stack.Navigator>
        <Stack.Screen name="Home" component={HomeScreen} />
        <Stack.Screen name="Scan" component={ScanScreen} />
        <Stack.Screen name="Map" component={MapScreen} />
        <Stack.Screen name="Detections" component={DetectionListScreen} />
      </Stack.Navigator>
    </NavigationContainer>
  );
}
