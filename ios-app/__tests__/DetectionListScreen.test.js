import React from 'react';
import { render, waitFor, fireEvent } from '@testing-library/react-native';
import { Share } from 'react-native';

import DetectionListScreen from '../screens/DetectionListScreen';
import * as api from '../services/api';
import { AppContext } from '../AppContext';

jest.mock('../services/api');

Share.share = jest.fn();

function Wrapper({ children }) {
  const [zoneFilter, setZoneFilter] = React.useState('');
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
}

test('filters detection list by zone', async () => {
  api.fetchDetections.mockResolvedValue([
    { id: 1, species: 'Bat A', time: 't1', zone: 'East' },
    { id: 2, species: 'Bat B', time: 't2', zone: 'West' },
  ]);
  const navigation = { navigate: jest.fn() };
  const { getByPlaceholderText, queryByText } = render(
    <DetectionListScreen navigation={navigation} />,
    { wrapper: Wrapper }
  );
  await waitFor(() => expect(api.fetchDetections).toHaveBeenCalled());
  fireEvent.changeText(getByPlaceholderText('Filter by zone'), 'West');
  await waitFor(() => {
    expect(queryByText('Bat B')).toBeTruthy();
    expect(queryByText('Bat A')).toBeNull();
  });
});

test('exports detections as CSV', async () => {
  api.fetchDetections.mockResolvedValue([
    { id: 1, species: 'Bat A', time: 't1', latitude: 1, longitude: 2, zone: 'East' },
  ]);
  const navigation = { navigate: jest.fn() };
  const { getByText } = render(
    <DetectionListScreen navigation={navigation} />,
    { wrapper: Wrapper }
  );
  await waitFor(() => expect(api.fetchDetections).toHaveBeenCalled());
  fireEvent.press(getByText('Export CSV'));
  expect(Share.share).toHaveBeenCalled();
});
