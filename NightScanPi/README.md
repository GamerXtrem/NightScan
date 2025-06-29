# NightScanPi – Autonomous Audio & Photo Trap (Raspberry Pi Zero 2 W)

This subfolder describes the embedded system used to capture wildlife activity at night.
NightScanPi runs on battery and solar power. It records audio and infrared photos
between **18:00** and **10:00**, then converts recordings to `.npy` spectrograms
for transfer over Wi‑Fi or an optional SIM module.

## First-Time Setup
When you receive the unit:

1. Insert a **64 GB** microSD card (ext4 recommended).
2. Power on the device &mdash; it starts automatically.
3. Launch the NightScan mobile app (iOS or Android) and:
   - Send your Wi‑Fi SSID and password.
   - Provide the installation GPS coordinates.
   - (Optional) Enable SIM transfer if a module and subscription are installed.

## Core Components
| Component                | Function                    |
|--------------------------|-----------------------------|
| Raspberry Pi Zero 2 W    | Main board                  |
| IR-Cut Camera (CSI)      | Night photo capture         |
| USB microphone           | Audio capture               |
| IR LEDs                  | Night vision illumination   |
| PIR sensor               | Motion detection            |
| microSD (64 GB min.)     | Data storage                |
| 18650 battery + TPL5110  | Power and timer             |
| 5 V 1 A solar panel      | Daily recharge              |
| (Optional) SIM module    | Off‑grid data transfer      |

Detailed wiring guides are found under `Hardware/`.

## Operation
**18:00–10:00** – system active
- On each PIR trigger: capture one IR photo and an 8 s audio clip.
- On audio threshold triggers: capture an IR photo and an 8 s audio clip.

**After 12:00** – processing
- Convert `.wav` files to `.npy` mel spectrograms at 22,050 Hz.
- Delete the `.wav` files if the SD card exceeds 70% capacity.

## Data Transfer
- **Wi‑Fi** (configured via the mobile app) sends spectrograms and photos
  automatically.
- **SIM module** uploads when a network is available.
- If neither is possible, you can read the files directly from the SD card.

## File Layout
```
/home/pi/nightscanpi/
├── main.py
├── audio_capture.py
├── camera_trigger.py
├── spectrogram_gen.py
├── wifi_config.py
├── sync.py
└── utils/
    └── energy_manager.py
```

## System Installation
1. Flash Raspberry Pi OS Lite (64‑bit) to the SD card.
2. Enable SSH and prepare `wifi_config.py` for connection via the mobile app.
3. Install dependencies:
   ```bash
   sudo apt update
   sudo apt install python3-pip ffmpeg sox libatlas-base-dev
   pip3 install numpy opencv-python soundfile flask
   ```

## Power Management
The TPL5110 automatically cuts power outside the active hours.
Adjust `NIGHTSCAN_START_HOUR` and `NIGHTSCAN_STOP_HOUR` before running the
scripts (`energy_manager.py`, `main.py`, and others).
Spectrogram processing is scheduled after noon to avoid interfering with
nighttime captures.

## Repository Overview
`NightScanPi/` contains the embedded side of the project. At the repository root
you will also find:
- `Audio_Training/` and `Picture_Training/` for preparing training data.
- `web/` with the Flask interface for uploads and predictions.
- `ios-app/` providing a sample mobile application.
- `wp-plugin/` with WordPress modules for uploads and stats.
- `setup_vps_infomaniak.sh` for deploying a VPS hosting the API.
- `docs/` with additional guides.

Refer to the root `README.md` for environment setup instructions.

## `Program` Directory
The `Program` folder holds the Python scripts executed on the Raspberry Pi:
- `main.py` orchestrates the nightly captures.
- `audio_capture.py` records 8 s of audio.
- `camera_trigger.py` takes an IR photo.
- `spectrogram_gen.py` converts `.wav` files to `.npy` spectrograms.
- `wifi_config.py` writes the Wi‑Fi configuration from the mobile app.
- `sync.py` uploads generated files.
- `utils/energy_manager.py` controls the active hours.
