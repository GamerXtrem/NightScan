import React from 'react';
import { render } from '@testing-library/react-native';
import HomeScreen from '../screens/HomeScreen';

jest.mock('expo-av', () => ({
  Audio: {
    Sound: {
      createAsync: jest.fn(() => Promise.resolve({
        sound: { unloadAsync: jest.fn(), playAsync: jest.fn() }
      })),
    },
  },
}));

it('shows welcome message', () => {
  const navigation = { navigate: jest.fn() };
  const { getByText } = render(<HomeScreen navigation={navigation} />);
  expect(getByText('Welcome to NightScan iOS')).toBeTruthy();
});
