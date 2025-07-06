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
PACKAGES=(python3-pip python3-picamera2 libcamera-apps ffmpeg sox libatlas-base-dev chrony unzip wget)
MISSING=()
for pkg in "${PACKAGES[@]}"; do
    dpkg -s "$pkg" >/dev/null 2>&1 || MISSING+=("$pkg")
done

if [ ${#MISSING[@]} -gt 0 ]; then
    echo "🔄 Updating system packages..."
    $SUDO apt update
    echo "📦 Installing missing packages: ${MISSING[*]}"
    $SUDO apt install -y "${MISSING[@]}"
fi

# Enable and start chrony to keep the clock in sync
if command -v systemctl >/dev/null 2>&1; then
    $SUDO systemctl enable --now chrony >/dev/null 2>&1 || true
    $SUDO timedatectl set-ntp true >/dev/null 2>&1 || true
fi

# Setup Python environment
echo "🐍 Setting up Python environment..."
python3 -m venv env
source env/bin/activate
pip install --upgrade pip
pip install numpy opencv-python soundfile sounddevice scipy flask pyaudio picamera2

# Install repository requirements if present
if [ -f requirements.txt ]; then
    echo "📋 Installing project requirements..."
    pip install -r requirements.txt
fi

# Add user to video group for camera access
$SUDO usermod -a -G video "$USER" 2>/dev/null || true
echo "👤 Added user $USER to video group for camera access"

# Configure camera if this is a Raspberry Pi
if [ -f /proc/device-tree/model ]; then
    echo ""
    echo "🔍 Raspberry Pi detected - configuring camera..."
    
    # Make camera configuration script executable
    chmod +x NightScanPi/Hardware/configure_camera_boot.sh
    
    # Ask user if they want to configure camera now
    read -p "📸 Configure IR-CUT camera now? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "🚀 Running camera configuration..."
        ./NightScanPi/Hardware/configure_camera_boot.sh
        
        echo ""
        echo "⚠️  IMPORTANT: A system reboot is required for camera changes"
        read -p "🔄 Reboot now? (y/n): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            echo "🔄 Rebooting in 5 seconds... Press Ctrl+C to cancel"
            sleep 5
            $SUDO reboot
        else
            echo "⏰ Remember to reboot before using the camera: sudo reboot"
        fi
    else
        echo "⏭️  Camera configuration skipped. Run manually later:"
        echo "   ./NightScanPi/Hardware/configure_camera_boot.sh"
    fi
else
    echo "💻 Not running on Raspberry Pi - skipping camera configuration"
fi

# Generate default sunrise/sunset file
echo "🌅 Generating sun times data..."
python NightScanPi/Program/utils/sun_times.py "$HOME/sun_times.json"

echo ""
echo "🎉 NightScanPi installation complete!"
echo ""
echo "📋 Next steps:"
echo "1. Activate Python environment: source env/bin/activate"
if [ -f /proc/device-tree/model ]; then
    echo "2. Test camera: python NightScanPi/Program/camera_test.py --all"
    echo "3. Run main program: python NightScanPi/Program/main.py"
else
    echo "2. Transfer to Raspberry Pi and run camera configuration"
    echo "3. Test on Pi: python NightScanPi/Program/camera_test.py --all"
fi
