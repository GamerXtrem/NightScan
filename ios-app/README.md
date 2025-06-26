# NightScan iOS App

This folder contains a React Native application for iOS.
It provides a basic navigation setup with a home screen followed by a bottom tab
interface. The tabs include the map, the detection list and a scan view.
Additional placeholders can be filled in with authentication and settings.
The application can also be extended to support Android.

## Getting Started

Install dependencies and run the app using Expo:

```bash
cd ios-app
npm install
npx expo install react-native-maps
npx expo install @react-native-async-storage/async-storage
npx expo start
```
After the Metro bundler starts, scan the QR code in your terminal with the Expo Go app to launch NightScan on your device.

