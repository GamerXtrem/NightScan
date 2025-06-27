import * as Notifications from 'expo-notifications';
import * as Device from 'expo-device';

Notifications.setNotificationHandler({
  handleNotification: async () => ({
    shouldShowAlert: true,
    shouldPlaySound: false,
    shouldSetBadge: false,
  }),
});

export async function registerForPushNotificationsAsync() {
  if (!Device.isDevice) {
    return false;
  }
  const { status: existingStatus } = await Notifications.getPermissionsAsync();
  let finalStatus = existingStatus;
  if (existingStatus !== 'granted') {
    const { status } = await Notifications.requestPermissionsAsync();
    finalStatus = status;
  }
  return finalStatus === 'granted';
}

export function sendDetectionNotification(detection) {
  return Notifications.scheduleNotificationAsync({
    content: {
      title: 'New detection',
      body: detection.species,
    },
    trigger: null,
  });
}

let lastDetectionId = 0;

export function checkForNewDetections(detections) {
  const maxId = detections.reduce((acc, d) => Math.max(acc, d.id), 0);
  if (lastDetectionId && maxId > lastDetectionId) {
    detections
      .filter((d) => d.id > lastDetectionId)
      .forEach((d) => {
        sendDetectionNotification(d).catch(() => {});
      });
  }
  lastDetectionId = maxId;
}
