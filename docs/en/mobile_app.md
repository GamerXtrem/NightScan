# Mobile App

This guide summarizes how to develop a React Native client for NightScan.

## Environment setup

Install Node.js along with the React Native CLI. Android development requires the SDK and an emulator from Android Studio. On macOS you may also install Xcode for the iOS simulator.

```bash
npm install -g react-native-cli
```

Ensure `node` and `watchman` are available in your PATH.

## Project initialization

Create a new project and start the development server:

```bash
npx react-native init NightScanApp
cd NightScanApp
npx react-native start
```

Launch the application with `npx react-native run-android` or `run-ios` depending on your platform.

## Authentication flow

The Flask server exposes `/register` and `/login` endpoints. Send `POST` requests with `username` and `password` to authenticate. The response includes a session cookie which must be sent with further requests.

After logging in you can request `/` to obtain the HTML page listing previous predictions for the active account.

## Using the prediction API

Uploads are forwarded to the API defined by the `PREDICT_API_URL` environment variable. Store this URL in your app configuration and post WAV files to it using `fetch` and `FormData`:

```javascript
const form = new FormData();
form.append('file', {
  uri: fileUri,
  type: 'audio/wav',
  name: 'sound.wav',
});

fetch(PREDICT_API_URL, {
  method: 'POST',
  body: form,
  credentials: 'include',
});
```

`credentials: 'include'` ensures the session cookie obtained during login is sent along with the upload request.

## Testing

Use the Android or iOS simulator for manual checks and run your preferred test framework (for example `npm test`) to automate unit tests. Confirm that uploads work and that the prediction list updates.

## Publishing

Generate release builds following the standard React Native commands. For Android run `npx react-native run-android --variant=release` and sign the APK. For iOS build through Xcode or `npx react-native run-ios --configuration Release`.

