# NightScanPi Task Checklist

This file lists the main actions required to develop the `NightScanPi` scripts.
It mirrors the original French notes so contributors can keep track of pending
work in English.

## 1. Installation and setup
- [x] Flash **Raspberry Pi OS Lite** onto the SD card.
- [x] Enable **SSH** and prepare `wifi_config.py` to receive the SSID and
  password from the mobile app.
- [x] Install the system packages: `python3-pip`, `ffmpeg`, `sox`,
  `libatlas-base-dev`.
- [x] Install the Python modules: `numpy`, `opencv-python`, `soundfile`, `flask`.

## 2. Core scripts
- [x] **main.py** &ndash; orchestrates detection, active hours and calls the
  other scripts.
- [x] **audio_capture.py** &ndash; record 8 s of audio for each PIR or threshold
  event and save as `.wav`.
- [x] **camera_trigger.py** &ndash; take an infrared photo when motion or sound
  is detected.
- [x] **spectrogram_gen.py** &ndash; after 12 hours convert `.wav` files to
  `.npy` spectrograms and delete the `.wav` files if the SD card exceeds 70%.
- [x] **wifi_config.py** &ndash; apply the Wi‑Fi credentials sent by the mobile
  app.
- [x] **sync.py** &ndash; automatically upload spectrograms and photos over Wi‑Fi
  or SIM; fall back to manual copy via SD card when offline.
- [x] **utils/energy_manager.py** &ndash; use the TPL5110 to power the Pi only
  between 18:00 and 10:00.
- [x] Add unit tests for `camera_trigger.py`.

## 3. Power management
- [x] Implement start/stop scheduling in `energy_manager.py` to reduce
  consumption.
- [x] Ensure spectrogram generation runs after noon so it doesn't interfere with
  night captures.

This checklist can be extended as the project progresses.

## 4. Additional tasks
- [x] Document wiring and specs in the `Hardware/` directory.
- [x] Integrate PIR and audio threshold detection in `main.py`.
- [x] Create a service to receive Wi‑Fi credentials from the mobile app and apply
  `wifi_config.py`.
- [x] Add SIM module support for data transfer when Wi‑Fi is unavailable.
- [x] Provide an automated install script for the Raspberry Pi (packages and
  configuration).
- [x] Add unit tests for `audio_capture.py` and `main.py`.
- [x] Supply a sample configuration file and enable error logging.
