#!/bin/bash
set -e

# Automated setup for NightScanPi on Raspberry Pi

# Ensure we can run apt commands
if [[ "$EUID" -ne 0 ]]; then
    if command -v sudo >/dev/null 2>&1; then
        SUDO="sudo"
    else
        echo "This script requires root privileges for apt commands." >&2
        exit 1
    fi
else
    SUDO=""
fi

# Enable SSH on first boot
if [ ! -e /boot/ssh ]; then
    $SUDO touch /boot/ssh
fi

# Install required system packages
PACKAGES=(python3-pip ffmpeg sox libatlas-base-dev chrony)
MISSING=()
for pkg in "${PACKAGES[@]}"; do
    dpkg -s "$pkg" >/dev/null 2>&1 || MISSING+=("$pkg")
done

if [ ${#MISSING[@]} -gt 0 ]; then
    $SUDO apt update
    $SUDO apt install -y "${MISSING[@]}"
fi

# Enable and start chrony to keep the clock in sync
if command -v systemctl >/dev/null 2>&1; then
    $SUDO systemctl enable --now chrony >/dev/null 2>&1 || true
    $SUDO timedatectl set-ntp true >/dev/null 2>&1 || true
fi

# Setup Python environment
python3 -m venv env
source env/bin/activate
pip install --upgrade pip
pip install numpy opencv-python soundfile sounddevice scipy flask pyaudio

# Install repository requirements if present
if [ -f requirements.txt ]; then
    pip install -r requirements.txt
fi

# Generate default sunrise/sunset file
python NightScanPi/Program/utils/sun_times.py "$HOME/sun_times.json"

echo "NightScanPi installation complete. Activate the environment with 'source env/bin/activate'."
