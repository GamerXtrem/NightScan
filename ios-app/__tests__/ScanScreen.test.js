import React from 'react';
import { render } from '@testing-library/react-native';
import ScanScreen from '../screens/ScanScreen';

jest.mock('expo-document-picker', () => ({
  getDocumentAsync: jest.fn().mockResolvedValue({ type: 'cancel' }),
}));

it('renders select file button', () => {
  const { getByText } = render(<ScanScreen />);
  expect(getByText('Select File')).toBeTruthy();
});
