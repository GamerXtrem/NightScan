#!/bin/bash
set -e

# ReSpeaker Lite Audio Configuration Script for NightScanPi
# Configures USB Audio Class 2.0 support and optimizes for ReSpeaker Lite

echo "🎤 ReSpeaker Lite Audio Configuration"
echo "===================================="

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Check sudo
if [[ "$EUID" -ne 0 ]]; then
    if command -v sudo >/dev/null 2>&1; then
        SUDO="sudo"
    else
        log_error "This script requires root privileges"
        exit 1
    fi
else
    SUDO=""
fi

# Install USB Audio support packages
install_usb_audio_packages() {
    log "📦 Installing USB Audio Class 2.0 support packages..."
    
    local packages=(
        "alsa-utils"           # ALSA utilities
        "alsa-tools"           # Additional ALSA tools
        "libasound2-dev"       # ALSA development headers
        "pulseaudio"           # PulseAudio sound server
        "pulseaudio-utils"     # PulseAudio utilities
        "portaudio19-dev"      # PortAudio development
        "python3-usb"          # Python USB support
        "libusb-1.0-0-dev"     # USB library development
        "usbutils"             # USB utilities (lsusb)
        "dfu-util"             # Device Firmware Upgrade utility
    )
    
    # Update package list
    $SUDO apt update
    
    # Install packages
    for pkg in "${packages[@]}"; do
        if $SUDO apt install -y "$pkg"; then
            log_success "Installed: $pkg"
        else
            log_warning "Failed to install: $pkg"
        fi
    done
}

# Configure ALSA for ReSpeaker Lite
configure_alsa() {
    log "🔧 Configuring ALSA for ReSpeaker Lite..."
    
    # Create ALSA configuration for ReSpeaker Lite
    local alsa_conf="/etc/asound.conf"
    
    log "Creating ALSA configuration: $alsa_conf"
    
    $SUDO tee "$alsa_conf" > /dev/null << 'EOF'
# ALSA Configuration for ReSpeaker Lite
# Optimized for USB Audio Class 2.0

# Default PCM device
pcm.!default {
    type asym
    playback.pcm "playback"
    capture.pcm "capture"
}

# Default control device
ctl.!default {
    type hw
    card "ReSpeakerLite"
}

# Capture configuration for ReSpeaker Lite
pcm.capture {
    type plug
    slave {
        pcm "hw:ReSpeakerLite,0"
        format S16_LE
        rate 16000
        channels 2
    }
    # Convert stereo to mono if needed
    route_policy "duplicate"
}

# Playback configuration (fallback to default)
pcm.playback {
    type plug
    slave {
        pcm "hw:0,0"
    }
}

# ReSpeaker Lite specific configuration
pcm.respeaker {
    type hw
    card "ReSpeakerLite"
    device 0
    format S16_LE
    rate 16000
    channels 2
}

# Control interface for ReSpeaker Lite
ctl.respeaker {
    type hw
    card "ReSpeakerLite"
}

# Mono capture from ReSpeaker Lite (for compatibility)
pcm.respeaker_mono {
    type route
    slave {
        pcm "respeaker"
        channels 2
    }
    ttable.0.0 0.5
    ttable.0.1 0.5
}
EOF

    log_success "ALSA configuration created"
}

# Configure PulseAudio for ReSpeaker Lite
configure_pulseaudio() {
    log "🔊 Configuring PulseAudio for ReSpeaker Lite..."
    
    # Create PulseAudio user configuration directory
    local pulse_dir="$HOME/.config/pulse"
    mkdir -p "$pulse_dir"
    
    # Create PulseAudio configuration
    local pulse_conf="$pulse_dir/default.pa"
    
    log "Creating PulseAudio configuration: $pulse_conf"
    
    cat > "$pulse_conf" << 'EOF'
# PulseAudio Configuration for ReSpeaker Lite
# Load default configuration first
.include /etc/pulse/default.pa

# ReSpeaker Lite specific configuration
# Load USB audio module with specific parameters
load-module module-alsa-source device=hw:ReSpeakerLite,0 source_name=respeaker_source rate=16000 channels=2
load-module module-alsa-sink device=hw:0,0 sink_name=default_sink

# Set ReSpeaker as default source
set-default-source respeaker_source

# Enable echo cancellation and noise reduction
load-module module-echo-cancel source_name=respeaker_echo_cancel source_master=respeaker_source aec_method=webrtc
EOF

    log_success "PulseAudio configuration created"
    
    # Restart PulseAudio
    if pgrep pulseaudio > /dev/null; then
        log "Restarting PulseAudio..."
        pulseaudio -k || true
        sleep 1
        pulseaudio --start || true
        log_success "PulseAudio restarted"
    fi
}

# Create udev rules for ReSpeaker Lite
create_udev_rules() {
    log "📋 Creating udev rules for ReSpeaker Lite..."
    
    local udev_file="/etc/udev/rules.d/99-respeaker-lite.rules"
    
    $SUDO tee "$udev_file" > /dev/null << 'EOF'
# udev rules for ReSpeaker Lite USB Audio Device
# Vendor ID: 2886, Product ID: 0019 (USB mode)

# ReSpeaker Lite USB mode
SUBSYSTEM=="usb", ATTR{idVendor}=="2886", ATTR{idProduct}=="0019", MODE="0666", GROUP="audio", SYMLINK+="respeaker_lite"

# ALSA sound card assignment
SUBSYSTEM=="sound", KERNEL=="card*", ATTR{id}=="ReSpeakerLite", ATTR{number}=="*", ENV{SOUND_DESCRIPTION}="ReSpeaker Lite USB Audio"

# Set permissions for audio group
KERNEL=="controlC[0-9]*", ATTR{id}=="ReSpeakerLite", GROUP="audio", MODE="0664"
EOF

    log_success "udev rules created: $udev_file"
    
    # Reload udev rules
    $SUDO udevadm control --reload-rules
    $SUDO udevadm trigger
    log_success "udev rules reloaded"
}

# Configure kernel modules for USB Audio
configure_kernel_modules() {
    log "🔧 Configuring kernel modules for USB Audio..."
    
    # Ensure USB audio modules are loaded
    local modules_file="/etc/modules-load.d/respeaker.conf"
    
    $SUDO tee "$modules_file" > /dev/null << 'EOF'
# Kernel modules for ReSpeaker Lite USB Audio
snd-usb-audio
snd-usbmidi-lib
usbhid
EOF

    log_success "Kernel modules configuration created: $modules_file"
    
    # Load modules immediately
    for module in snd-usb-audio snd-usbmidi-lib; do
        if $SUDO modprobe "$module" 2>/dev/null; then
            log_success "Loaded module: $module"
        else
            log_warning "Failed to load module: $module"
        fi
    done
}

# Test ReSpeaker Lite detection
test_respeaker_detection() {
    log "🧪 Testing ReSpeaker Lite detection..."
    
    # Check USB device
    if lsusb | grep -i "2886:0019" > /dev/null; then
        log_success "ReSpeaker Lite USB device detected"
    else
        log_warning "ReSpeaker Lite USB device not found"
        log "Please check USB connection and power"
    fi
    
    # Check ALSA devices
    if aplay -l 2>/dev/null | grep -i "respeaker" > /dev/null; then
        log_success "ReSpeaker Lite ALSA device detected"
    else
        log_warning "ReSpeaker Lite ALSA device not found"
    fi
    
    # Check audio capture capability
    if arecord -l 2>/dev/null | grep -i "respeaker" > /dev/null; then
        log_success "ReSpeaker Lite audio capture available"
    else
        log_warning "ReSpeaker Lite audio capture not available"
    fi
    
    # Test firmware version if dfu-util available
    if command -v dfu-util >/dev/null 2>&1; then
        log "Checking firmware version..."
        if dfu-util -l 2>/dev/null | grep "2886:0019" > /dev/null; then
            local fw_info=$(dfu-util -l 2>/dev/null | grep "2886:0019" | head -1)
            log "Firmware info: $fw_info"
        fi
    fi
}

# Create Python test script
create_test_script() {
    log "📝 Creating Python test script..."
    
    local test_script="/tmp/test_respeaker.py"
    
    cat > "$test_script" << 'EOF'
#!/usr/bin/env python3
"""Test script for ReSpeaker Lite detection and recording."""

import sys
import time
import logging

try:
    sys.path.append('/home/pi/NightScan/NightScanPi/Program')
    from respeaker_detector import detect_respeaker, get_respeaker_config
    import pyaudio
    import wave
    
    logging.basicConfig(level=logging.INFO)
    
    print("🎤 Testing ReSpeaker Lite...")
    
    # Test detection
    respeaker_info = detect_respeaker()
    if respeaker_info:
        print(f"✅ ReSpeaker detected: {respeaker_info.device_name}")
        print(f"   Device ID: {respeaker_info.device_id}")
        print(f"   Channels: {respeaker_info.channels}")
        print(f"   Sample rates: {respeaker_info.sample_rates}")
    else:
        print("❌ ReSpeaker Lite not detected")
        sys.exit(1)
    
    # Test configuration
    config = get_respeaker_config()
    if config:
        print(f"📋 Optimal config: {config}")
        
        # Test recording
        print("🎙️ Testing 3-second recording...")
        
        p = pyaudio.PyAudio()
        stream = p.open(
            format=pyaudio.paInt16,
            channels=config['channels'],
            rate=config['sample_rate'],
            input=True,
            input_device_index=config['device_id'],
            frames_per_buffer=1024
        )
        
        frames = []
        for i in range(int(config['sample_rate'] / 1024 * 3)):  # 3 seconds
            data = stream.read(1024, exception_on_overflow=False)
            frames.append(data)
        
        stream.stop_stream()
        stream.close()
        p.terminate()
        
        # Save test recording
        with wave.open('/tmp/respeaker_test.wav', 'wb') as wf:
            wf.setnchannels(config['channels'])
            wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
            wf.setframerate(config['sample_rate'])
            wf.writeframes(b''.join(frames))
        
        print("✅ Test recording saved to /tmp/respeaker_test.wav")
        print("🎉 ReSpeaker Lite test completed successfully!")
    else:
        print("❌ Could not get ReSpeaker configuration")

except ImportError as e:
    print(f"❌ Missing dependency: {e}")
    print("Please install pyaudio: pip install pyaudio")
except Exception as e:
    print(f"❌ Test failed: {e}")
EOF

    chmod +x "$test_script"
    log_success "Test script created: $test_script"
    
    # Run test if ReSpeaker modules are available
    if python3 -c "import pyaudio" 2>/dev/null; then
        log "Running ReSpeaker test..."
        python3 "$test_script" || log_warning "Test failed - check connections and configuration"
    else
        log_warning "PyAudio not available - skipping test"
    fi
}

# Main configuration function
main() {
    log "Starting ReSpeaker Lite configuration..."
    
    # Install required packages
    install_usb_audio_packages
    
    # Configure audio systems
    configure_kernel_modules
    create_udev_rules
    configure_alsa
    configure_pulseaudio
    
    # Test the configuration
    test_respeaker_detection
    create_test_script
    
    log_success "🎉 ReSpeaker Lite configuration completed!"
    log ""
    log "📋 Next steps:"
    log "   1. Reboot the system to apply all changes"
    log "   2. Connect your ReSpeaker Lite via USB"
    log "   3. Run: python3 /tmp/test_respeaker.py"
    log "   4. Check audio with: arecord -D respeaker -f S16_LE -r 16000 -c 2 test.wav"
    log ""
    log "⚠️  Remember: ReSpeaker Lite maximum sample rate is 16kHz"
}

# Run main function
main