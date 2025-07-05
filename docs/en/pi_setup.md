# Raspberry Pi Zero 2 W Quick Setup

This guide explains how to prepare a microSD card and install the packages required to run **NightScanPi**.

## 1. Flash Raspberry Pi OS Lite
1. Download the latest [Raspberry Pi Imager](https://www.raspberrypi.com/software/).
2. Insert a microSD card (64 GB recommended).
3. Choose **Raspberry Pi OS Lite (64‑bit)** and flash the card.

## 2. Enable SSH on first boot
After flashing, create an empty file named `ssh` on the boot partition. This enables the SSH service when the Pi boots.

## 3. Install system packages
Boot the Pi and connect via SSH, then run:

```bash
sudo apt update
sudo apt install -y python3-pip ffmpeg sox libatlas-base-dev
```

## 4. Install Python modules
Create and activate a virtual environment and install the required modules:

```bash
python3 -m venv env
source env/bin/activate
pip install numpy opencv-python soundfile flask pyaudio
pip install sounddevice scipy
```

Alternatively you can run the `NightScanPi/setup_pi.sh` script which performs the steps above automatically.

## 5. Configure Wi‑Fi
Use `wifi_config.py` from the `NightScanPi/Program` directory to apply your network credentials:

```bash
python wifi_config.py <SSID> <password>
```

The Pi is now ready to run the NightScanPi scripts.
