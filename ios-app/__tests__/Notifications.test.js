import { checkForNewDetections } from '../services/notifications';
import * as ExpoNotif from 'expo-notifications';

ExpoNotif.scheduleNotificationAsync.mockResolvedValue();

test('sends notification for new detections', () => {
  checkForNewDetections([{ id: 1, species: 'A' }]);
  checkForNewDetections([
    { id: 1, species: 'A' },
    { id: 2, species: 'B' },
  ]);
  expect(ExpoNotif.scheduleNotificationAsync).toHaveBeenCalledTimes(1);
});
