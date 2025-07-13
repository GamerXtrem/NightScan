# NightScanPi Task List

This list groups the actions to implement for the `NightScanPi` scripts according to the folder documentation.

## 1. Installation and Configuration
- [x] Flash **Raspberry Pi OS Lite** on the SD card.
- [x] Enable **SSH** and prepare `wifi_config.py` to receive SSID and password from the mobile application.
- [x] Install required system packages: `python3-pip`, `ffmpeg`, `sox`, `libatlas-base-dev`.
- [x] Install Python modules: `numpy`, `opencv-python`, `soundfile`, `flask`.

## 2. Main Scripts
- [x] **main.py**: orchestrate global operation (capture on detection, activity schedules, calling other scripts).
- [x] **audio_capture.py**: record 8s of sound at each detection (PIR or audio threshold) and save as `.wav`.
- [x] **camera_trigger.py**: take an infrared photo during detection (PIR or audio).
- [x] **spectrogram_gen.py**: after 12 PM, convert `.wav` files to `.npy` spectrograms and delete `.wav` files if SD card exceeds 70% capacity.
- [x] **wifi_config.py**: retrieve Wi-Fi parameters sent by the mobile application and apply them.
- [x] **sync.py**: automatically send spectrograms and photos via Wi-Fi or SIM module; provide disconnected mode allowing manual copy via SD card.
- [x] **utils/energy_manager.py**: control power supply using TPL5110 so the Pi only operates from 6 PM to 10 AM.
- [x] Add unit tests for `camera_trigger.py`.

## 3. Energy Management
- [x] Implement shutdown/startup scheduling in `energy_manager.py` to limit power consumption.
- [x] Ensure spectrogram generation occurs after noon to avoid interfering with nighttime captures.

This list can be completed as the project progresses.

## 4. Additional Tasks
- [x] Document wiring and specifications in the `Hardware/` folder.
- [x] Integrate PIR sensor and audio threshold detection in `main.py`.
- [x] Create a service to receive Wi-Fi credentials from the mobile application and apply `wifi_config.py`.
- [x] Add SIM module support for data transfer when Wi-Fi is unavailable.
- [x] Write an automated installation script for Raspberry Pi (packages and configuration).
- [x] Add unit tests for `audio_capture.py` and `main.py`.
- [x] Provide a configuration file example and enable error logging.

## 5. Time Synchronization
- [x] Write a `time_config.py` script to input current time and GPS position during initial configuration.
- [x] If no position is provided, use Bern coordinates by default (46.9480 N, 7.4474 E).
- [x] Determine timezone from position using `timezonefinder` and apply it via `timedatectl`.
- [x] Install and configure `chrony` to maintain synchronized time via Wi-Fi or SIM module.
- [x] Document the procedure in `NightScanPi/README.md`.

## 6. Automatic Day/Night Cycle
- [x] Add a `sun_times.py` module calculating sunrise and sunset times based on date and GPS coordinates (e.g., via `suntime`).
- [x] Store these times in a reference file updated daily.
- [x] Adapt `energy_manager.py` and `main.py` to only activate recording from 30 min before sunset until 30 min after sunrise.
- [x] Write unit tests to verify time calculations and activity window compliance.
- [x] Update documentation to describe the solar cycle-based system configuration.

## 7. Manual Transfer Interface
- [x] Generate a mini web page accessible locally via Flask or FastAPI.
- [x] Add a **Transfer Data** button triggering file upload.
- [x] Check network connection (ping or DNS) before transfer.
- [x] Send `.npy` spectrograms and photos from `/data/exports/` folder to the VPS.
- [x] Delete local files once upload is confirmed.
- [x] Send a notification to NightScan mobile app to signal transfer completion.

## 8. Wi-Fi Wake-up by Sound Signal
- [x] Install `sounddevice`, `numpy`, `scipy` or `aubio` on the Pi.
- [x] Write a `wifi_wakeup.py` script analyzing microphone stream in real-time (FFT) and activating Wi-Fi with `sudo ifconfig wlan0 up` upon detecting 2100 Hz.
- [x] Log each detection to facilitate debugging.
- [x] Generate a one-second trigger sound `.wav` at 2100 Hz (or DTMF) and integrate it into the iOS application.
- [x] Add a **Wake NightScanPi** button in the iOS app playing this sound at full volume and displaying "Sending sound signal...".
- [x] Maintain a `wifi_awake` state in a `.status` file and automatically turn off Wi-Fi after 10 min without connection.
- [x] Log Wi-Fi activation duration.
- [ ] Test different frequencies, distance and volume needed to avoid false positives and validate operation.
- [x] Verify that sound detection doesn't consume too much energy (benchmark).