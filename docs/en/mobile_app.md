# Mobile App

This guide summarizes how to develop a React Native client for NightScan.

## Application overview

NightScan for iOS and Android displays the wildlife detections produced by the
field sensors at night. Acoustic and photo sensors upload raw data to a central
server via SIM cards. An AI model then identifies the species and stores a JSON
record (species name, GPS coordinates, timestamp, etc.) in a MySQL database.
The mobile application fetches these entries and presents them in a clear
interface without exposing the original audio or images.

### Target users

The app is geared toward wildlife photographers, hobby naturalists and
researchers who need an intuitive way to review the observations. The interface
is deliberately straightforward so that anyone can quickly see when and where an
animal was detected.

### Main features

- **Interactive map** – pins mark each detection on a map background (Google
  Maps, OpenStreetMap, etc.). Tapping a pin opens a popup showing the detected
  species and the exact time.
- **Detection feed** – a chronological list complements the map with the latest
  observations, including species name and a short location label.
- **Real-time notifications** – the app can receive push notifications whenever a
  new detection is processed by the server.
- **Dynamic filters** – detections can be filtered by species and by geographic
  area; the map and feed update instantly according to the criteria.
- **Sharing and export** – each record can be shared by email and the app can
  export all detections as CSV or KMZ for further analysis.

### User interface

NightScan uses a light theme with a simple bottom navigation bar to access the
Map, List, Filters and Settings sections. Each entry highlights only the
essential information—species and time—without revealing the media files. Native
gestures such as pinch‑to‑zoom and pull‑to‑refresh keep the experience familiar
on both iOS and Android.

## Environment setup

Install Node.js along with the React Native CLI. Android development requires the SDK and an emulator from Android Studio. On macOS you may also install Xcode for the iOS simulator.

```bash
npm install -g react-native-cli
```

Ensure `node` and `watchman` are available in your PATH.

## Project initialization

Create a new project and start the development server. A very small starter
project is already available in the repository under `ios-app/` if you want to
skip this step:

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

