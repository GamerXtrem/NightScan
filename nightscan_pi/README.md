NightScanPi – Autonomous Nocturnal Audio & Photo Trap (Raspberry Pi Zero 2 W)
🎯 Objective
NightScanPi is an embedded system dedicated to automated capture of sounds and images of nocturnal wildlife, running on battery and solar panel, with data transmission via Wi-Fi or SIM module. It is active between 6 PM and 10 AM, and transforms detected sounds into lighter .npy spectrograms for transfer.

🧭 First Use (Onboarding)
Upon receiving the device, the user must:

Insert a microSD card (min. 64 GB, ext4 format recommended)

Power the device (no button required, automatic startup when powered on)

Launch the NightScan mobile application (iOS / Android)

From the application:

Configure Wi-Fi by sending SSID and password

Enter the GPS position of the installation

(Optional) Enable sending via SIM module if installed and if a subscription has been activated

Configure date and time with `time_config.py` (GPS coordinates needed for timezone)

🧩 Components
Component	Function
Raspberry Pi Zero 2 W	Central unit
IR-Cut Camera (CSI)	Night photo capture
USB Microphone	Audio capture
Infrared LEDs	Night vision
PIR Detector	Motion detection
microSD Card 64 GB min.	Data storage
18650 Battery + TPL5110	Power supply and timer
Solar Panel 5V 1A	Daily recharge
(Optional) SIM Module	Transfer without Wi-Fi
### Additional Hardware Information

Detailed specifications can be found in the `Hardware/` directory:

- **Raspberry Pi Zero 2 W**: quad-core 1 GHz processor, 512 MB RAM, 2.4 GHz Wi-Fi and Bluetooth 4.2. Power consumption varies between 0.6 W and 3 W.
- **RPI IR-CUT Camera**: CSI camera module with motorized infrared filter and IR LEDs, designed for day and night vision. Maximum current around 150 mA.
- **ReSpeaker Mic Array Lite**: dual microphone board based on XMOS XU316 chipset integrating echo cancellation and noise suppression, with RGB LED.

These documents describe connection diagrams and advanced settings (camera HDR modes, microphone updates, etc.).

⏱ Operation
🕕 From 6 PM to 10 AM:

The system is active

At each detection by the PIR sensor, it captures:

1 infrared photo

1 audio recording of 8 seconds (.wav)
At each audio detection when threshold is exceeded, it captures:
1 photo 
1 audio recording of 8 seconds

🕛 Starting from 12 PM:

Audio files are transformed into .npy spectrograms
Recordings are resampled to 22,050 Hz and converted to
mel-spectrograms expressed in dB to match the processing of
`predict.py`

.wav files are automatically deleted if SD card exceeds 70% capacity

📤 Data Transfer
Via Wi-Fi configured with NightScan mobile app:

Automatic transfer of spectrograms and photos

Via SIM module:

Automatic transfer if network available and subscription active

Otherwise: user can remove SD card to access files locally

📁 File Structure
swift
Copy
Edit
/home/pi/nightscanpi/
├── main.py
├── audio_capture.py
├── camera_trigger.py
├── spectrogram_gen.py
├── wifi_config.py
├── sync.py
└── utils/
    └── energy_manager.py
🛠 System Installation
Flash Raspberry Pi OS Lite 64-bit on SD card

Enable SSH and prepare wifi_config.py scripts for connection via mobile application

Install dependencies:

bash
Copy
Edit
sudo apt update
sudo apt install python3-pip ffmpeg sox libatlas-base-dev
pip3 install numpy opencv-python soundfile sounddevice scipy flask
🔌 Energy Management
TPL5110 automatically cuts power outside useful time range

The Pi is powered only from 6 PM to 10 AM

Schedules can be adapted by defining variables
`NIGHTSCAN_START_HOUR` and `NIGHTSCAN_STOP_HOUR` before running scripts
(`energy_manager.py`, `main.py`, etc.).

Day/night cycle is enabled by default. Sunrise and sunset times
are saved in `~/sun_times.json` upon installation. This file is
automatically updated if position or date changes. To store this
information elsewhere, set `NIGHTSCAN_SUN_FILE`. The margin relative to
sunrise/sunset can be adjusted via `NIGHTSCAN_SUN_OFFSET` (in minutes).

Audio file processing (.wav → .npy) happens after 12 PM, to avoid load peaks during collection

## NightScan Repository Overview

This `NightScanPi/` folder represents the embedded part of the project. At the repository root, you'll find:
- `Audio_Training/` and `Picture_Training/` for data preparation and recognition model training.
- `web/` containing the Flask application serving as upload interface and prediction viewer.
- `ios-app/` for a mobile application example.
- `wp-plugin/` with WordPress modules dedicated to submissions from a site and statistics display.
- `setup_vps_infomaniak.sh` which automates deployment of a VPS configured to host the API.
- `docs/` where additional guides are located.

The `README.md` located at the root details these directories and explains how to install the test environment.

## `Program` Folder
This directory contains Python scripts executed on the Raspberry Pi:

- `main.py` orchestrates nighttime captures.
- `audio_capture.py` records 8s of audio.
- `camera_trigger.py` takes an infrared photo.
- `spectrogram_gen.py` converts `.wav` files to `.npy` spectrograms.
- `wifi_config.py` writes Wi-Fi configuration received via mobile application.
- `sync.py` automatically sends generated files.
- `utils/energy_manager.py` manages activity time range.
- `time_config.py` sets time and timezone at installation start.
- `manual_transfer.py` launches a small Flask interface to manually trigger data sending.

### Time Synchronization
To set time and timezone, run:
```bash
python time_config.py "2024-01-01 12:00:00" --lat 46.9 --lon 7.4
```
If no coordinates are provided, Bern coordinates are used.

To maintain accurate time during use, install and enable
`chrony` which will automatically synchronize the clock as soon as a connection
(Wi-Fi or SIM module) is available:

```bash
sudo apt install -y chrony
sudo systemctl enable --now chrony
sudo timedatectl set-ntp true
```