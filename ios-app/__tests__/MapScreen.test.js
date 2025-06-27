import React from 'react';
import { render, waitFor } from '@testing-library/react-native';
import MapScreen from '../screens/MapScreen';
import * as api from '../services/api';
import { AppContext } from '../AppContext';

jest.mock('../services/api');
jest.mock('react-native-maps', () => {
  const React = require('react');
  const { View } = require('react-native');
  const MockMapView = (props) => <View {...props} />;
  const MockMarker = (props) => <View {...props} />;
  return {
    __esModule: true,
    default: MockMapView,
    Marker: MockMarker,
  };
});

test('renders markers from API', async () => {
  api.fetchDetections.mockResolvedValue([
    { id: 1, species: 'Bat A', latitude: 1, longitude: 2, zone: 'East' },
    { id: 2, species: 'Bat B', latitude: 3, longitude: 4, zone: 'West' },
  ]);
  const Wrapper = ({ children }) => (
    <AppContext.Provider
      value={{
        darkMode: false,
        notifications: false,
        zoneFilter: '',
        setDarkMode: jest.fn(),
        setNotifications: jest.fn(),
        setZoneFilter: jest.fn(),
      }}
    >
      {children}
    </AppContext.Provider>
  );
  const { queryByTestId } = render(<MapScreen />, { wrapper: Wrapper });
  await waitFor(() => {
    expect(api.fetchDetections).toHaveBeenCalled();
    expect(queryByTestId('marker-1')).toBeTruthy();
    expect(queryByTestId('marker-2')).toBeTruthy();
  });
});

test('filters markers by zone', async () => {
  api.fetchDetections.mockResolvedValue([
    { id: 1, species: 'Bat A', latitude: 1, longitude: 2, zone: 'East' },
    { id: 2, species: 'Bat B', latitude: 3, longitude: 4, zone: 'West' },
  ]);
  const Wrapper = ({ children }) => {
    const [zoneFilter, setZoneFilter] = React.useState('East');
    return (
      <AppContext.Provider
        value={{
          darkMode: false,
          notifications: false,
          zoneFilter,
          setDarkMode: jest.fn(),
          setNotifications: jest.fn(),
          setZoneFilter,
        }}
      >
        {children}
      </AppContext.Provider>
    );
  };
  const { queryByTestId } = render(<MapScreen />, { wrapper: Wrapper });
  await waitFor(() => {
    expect(queryByTestId('marker-1')).toBeTruthy();
    expect(queryByTestId('marker-2')).toBeNull();
  });
});
