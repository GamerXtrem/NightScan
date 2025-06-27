import React from 'react';
import { render } from '@testing-library/react-native';
import HomeScreen from '../screens/HomeScreen';

it('shows welcome message', () => {
  const navigation = { navigate: jest.fn() };
  const { getByText } = render(<HomeScreen navigation={navigation} />);
  expect(getByText('Welcome to NightScan iOS')).toBeTruthy();
});
