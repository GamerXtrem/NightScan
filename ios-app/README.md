# NightScan iOS App

This folder contains a React Native application for iOS.
It provides a basic navigation setup with a home screen followed by a bottom tab
interface (map, detection list and scan views).
Login and registration screens are already implemented and use `services/api.js`
to contact the Flask backend. The detection list fetches data from
`/api/detections` and caches it for offline viewing.
The application can also be extended to support Android.

## Getting Started

Install dependencies and run the app using Expo:

```bash
cd ios-app
npm install
npx expo install react-native-maps
npx expo install @react-native-async-storage/async-storage
npx expo install expo-notifications expo-device
npx expo start
```
After the Metro bundler starts, scan the QR code in your terminal with the Expo Go app to launch NightScan on your device.

## Building a Release

To generate a standalone iOS build you can rely on Expo's EAS tooling. Install EAS CLI and run:

```bash
npm install -g eas-cli
eas build --platform ios
```

You will need an Apple developer account to sign the application. The resulting IPA can be deployed to TestFlight or the App Store.

### Wake tone asset

The repository does not store the wake tone audio file directly. It will be generated automatically when you run `npm start` thanks to the `prestart` script in `package.json`.
If you need to generate it manually, run:

```bash
python ../NightScanPi/Program/generate_wake_tone.py assets/wake_tone.wav
```


## Testing on iPhone with a QR Code
1. Install **Expo Go** from the App Store.
2. Connect your iPhone and development machine to the same Wi-Fi network.
3. Run `npx expo start` to print a QR code in the terminal.
4. Open Expo Go and tap **Scan QR Code**, or use the iPhone Camera app.
5. Point it at the terminal QR code. After a short build step the app loads inside Expo Go.
6. Keep the terminal window running so code changes reload automatically.

## Server interaction

Authentication requests are sent to the Flask backend using the helper
functions in `services/api.js`. After logging in, the detection list requests
`/api/detections` and stores the result with `AsyncStorage`.

## Offline caching

The detection list is stored in AsyncStorage so it remains available when the device is offline. Pull down on the list to refresh the cache when a network connection is available.

## Running Tests

Unit tests are written with Jest and React Native Testing Library. After installing dependencies, run:

```bash
npm test
```

All test suites should pass and verify the main screens' behaviour.

