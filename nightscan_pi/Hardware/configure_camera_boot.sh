#!/bin/bash
"""
NightScanPi Camera Boot Configuration Script
Configures /boot/firmware/config.txt for IR-CUT camera support on Raspberry Pi.
"""

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running on Raspberry Pi
check_raspberry_pi() {
    if [ ! -f /proc/device-tree/model ]; then
        log_error "This script must be run on a Raspberry Pi"
        exit 1
    fi
    
    local model=$(cat /proc/device-tree/model 2>/dev/null | tr -d '\0')
    log "Detected Raspberry Pi: $model"
    
    # Check for Pi Zero specifically
    if [[ "$model" == *"Pi Zero"* ]]; then
        log "✅ Pi Zero detected - optimizing configuration"
        export PI_ZERO=true
    else
        export PI_ZERO=false
    fi
}

# Determine config.txt location based on OS version
get_config_path() {
    local config_paths=(
        "/boot/firmware/config.txt"  # Raspberry Pi OS Bookworm
        "/boot/config.txt"           # Raspberry Pi OS Bullseye and earlier
    )
    
    for path in "${config_paths[@]}"; do
        if [ -f "$path" ]; then
            echo "$path"
            return 0
        fi
    done
    
    log_error "Could not find config.txt file"
    exit 1
}

# Backup config.txt
backup_config() {
    local config_path="$1"
    local backup_path="${config_path}.nightscan.backup.$(date +%Y%m%d_%H%M%S)"
    
    log "Creating backup: $backup_path"
    sudo cp "$config_path" "$backup_path"
    log_success "Backup created successfully"
}

# Detect camera sensor
detect_camera_sensor() {
    log "🔍 Detecting camera sensor..."
    
    # Try to detect via libcamera first
    if command -v libcamera-hello >/dev/null 2>&1 || command -v rpicam-hello >/dev/null 2>&1; then
        local cam_cmd="libcamera-hello"
        if command -v rpicam-hello >/dev/null 2>&1; then
            cam_cmd="rpicam-hello"
        fi
        
        # Run brief camera test to detect sensor
        local sensor_info
        if sensor_info=$($cam_cmd -t 1 --nopreview 2>&1 | grep -i "using camera" || true); then
            if [[ "$sensor_info" == *"imx219"* ]]; then
                echo "imx219"
                return 0
            elif [[ "$sensor_info" == *"ov5647"* ]]; then
                echo "ov5647"
                return 0
            elif [[ "$sensor_info" == *"imx477"* ]]; then
                echo "imx477"
                return 0
            elif [[ "$sensor_info" == *"imx290"* ]]; then
                echo "imx290"
                return 0
            elif [[ "$sensor_info" == *"imx327"* ]]; then
                echo "imx327"
                return 0
            fi
        fi
    fi
    
    # Fallback detection methods
    if [ -d /proc/device-tree/cam* ] 2>/dev/null; then
        log "Camera device tree entries found"
    fi
    
    # Default to most common sensor for IR-CUT cameras
    log_warning "Could not auto-detect sensor, defaulting to IMX219"
    echo "imx219"
}

# Configure camera settings in config.txt
configure_camera() {
    local config_path="$1"
    local sensor_type="$2"
    
    log "📋 Configuring camera settings for sensor: $sensor_type"
    
    # Remove existing camera configurations
    sudo sed -i '/# NightScanPi Camera Configuration/,/# End NightScanPi Camera Configuration/d' "$config_path"
    
    # Camera configuration block
    local camera_config="
# NightScanPi Camera Configuration
# Generated on $(date)

# Disable automatic camera detection
camera_auto_detect=0

# Enable camera interface
start_x=1

# GPU memory split (important for camera)
gpu_mem=128"

    # Pi Zero specific optimizations
    if [ "$PI_ZERO" = true ]; then
        camera_config="$camera_config

# Pi Zero 2W optimizations
gpu_mem=64
disable_overscan=1"
    fi

    # Add sensor-specific configuration
    case "$sensor_type" in
        "imx219")
            camera_config="$camera_config

# IMX219 sensor configuration
dtoverlay=imx219"
            ;;
        "ov5647")
            camera_config="$camera_config

# OV5647 sensor configuration  
dtoverlay=ov5647"
            ;;
        "imx477")
            camera_config="$camera_config

# IMX477 sensor configuration
dtoverlay=imx477"
            ;;
        "imx290")
            camera_config="$camera_config

# IMX290 sensor configuration
dtoverlay=imx290,clock-frequency=37125000"
            ;;
        "imx327")
            camera_config="$camera_config

# IMX327 sensor configuration (same as IMX290)
dtoverlay=imx290,clock-frequency=37125000"
            ;;
        *)
            log_warning "Unknown sensor type: $sensor_type, using generic configuration"
            camera_config="$camera_config

# Generic camera configuration
dtoverlay=$sensor_type"
            ;;
    esac

    camera_config="$camera_config

# IR-CUT camera optimizations
# Enable hardware-accelerated video
gpu_mem_256=128
gpu_mem_512=64
gpu_mem_1024=128

# Disable rainbow splash screen to save boot time
disable_splash=1

# End NightScanPi Camera Configuration"

    # Add configuration to config.txt
    echo "$camera_config" | sudo tee -a "$config_path" >/dev/null
    log_success "Camera configuration added to $config_path"
}

# Configure additional camera-specific files
configure_camera_files() {
    local sensor_type="$1"
    
    # Create camera info file
    local info_file="/opt/nightscan/camera_info.json"
    sudo mkdir -p /opt/nightscan
    
    local camera_info="{
  \"sensor_type\": \"$sensor_type\",
  \"configured_date\": \"$(date -Iseconds)\",
  \"pi_model\": \"$(cat /proc/device-tree/model | tr -d '\0')\",
  \"config_version\": \"1.0\",
  \"features\": {
    \"ir_cut\": true,
    \"night_vision\": true,
    \"auto_exposure\": true,
    \"auto_white_balance\": true
  }
}"

    echo "$camera_info" | sudo tee "$info_file" >/dev/null
    log_success "Camera info saved to $info_file"
    
    # Set up camera permissions
    sudo usermod -a -G video "$USER" 2>/dev/null || true
    log "Added user $USER to video group"
}

# Install IMX290 JSON file if needed (Pi 5 requirement)
install_imx290_json() {
    local sensor_type="$1"
    
    if [[ "$sensor_type" == "imx290" || "$sensor_type" == "imx327" ]]; then
        local pi_model=$(cat /proc/device-tree/model | tr -d '\0')
        if [[ "$pi_model" == *"Pi 5"* ]]; then
            log "📦 Installing IMX290 JSON file for Pi 5..."
            
            local json_dir="/usr/share/libcamera/ipa/rpi/pisp"
            local json_url="https://www.waveshare.net/w/upload/7/7a/Imx290.zip"
            local temp_dir=$(mktemp -d)
            
            cd "$temp_dir"
            
            if command -v wget >/dev/null 2>&1; then
                sudo wget -q "$json_url" -O imx290.zip
            elif command -v curl >/dev/null 2>&1; then
                sudo curl -s -L "$json_url" -o imx290.zip
            else
                log_warning "wget or curl not found, skipping IMX290 JSON installation"
                return
            fi
            
            if command -v unzip >/dev/null 2>&1; then
                sudo unzip -q imx290.zip
                sudo mkdir -p "$json_dir"
                sudo cp imx290.json "$json_dir/"
                log_success "IMX290 JSON file installed"
            else
                log_warning "unzip not found, please install manually"
            fi
            
            rm -rf "$temp_dir"
        fi
    fi
}

# Validate configuration
validate_configuration() {
    local config_path="$1"
    
    log "🔍 Validating configuration..."
    
    # Check if our configuration exists
    if grep -q "NightScanPi Camera Configuration" "$config_path"; then
        log_success "Configuration found in $config_path"
    else
        log_error "Configuration not found in $config_path"
        return 1
    fi
    
    # Check for common conflicts
    local conflicts=()
    
    if grep -q "^camera_auto_detect=1" "$config_path"; then
        conflicts+=("camera_auto_detect is still enabled")
    fi
    
    if [ ${#conflicts[@]} -gt 0 ]; then
        log_warning "Configuration conflicts detected:"
        for conflict in "${conflicts[@]}"; do
            log_warning "  - $conflict"
        done
    fi
    
    log_success "Configuration validation complete"
}

# Generate test script
generate_test_script() {
    local test_script="/home/$USER/test_nightscan_camera.sh"
    
    cat > "$test_script" << 'EOF'
#!/bin/bash
# NightScanPi Camera Test Script

echo "🔍 Testing NightScanPi Camera Configuration"
echo "=========================================="

# Test libcamera
echo "📸 Testing libcamera..."
if command -v rpicam-hello >/dev/null 2>&1; then
    echo "✅ rpicam-hello available"
    echo "Running 5-second test..."
    rpicam-hello -t 5000 --nopreview
elif command -v libcamera-hello >/dev/null 2>&1; then
    echo "✅ libcamera-hello available" 
    echo "Running 5-second test..."
    libcamera-hello -t 5000 --nopreview
else
    echo "❌ No libcamera commands found"
fi

# Test Python camera
echo ""
echo "🐍 Testing Python camera integration..."
cd /path/to/NightScanPi/Program
python3 camera_test.py --status

echo ""
echo "Test complete. Check output above for any errors."
EOF

    chmod +x "$test_script"
    log_success "Test script created: $test_script"
}

# Main configuration function
main() {
    log "🚀 Starting NightScanPi Camera Boot Configuration"
    log "================================================="
    
    # Check prerequisites
    if [[ "$EUID" -eq 0 ]]; then
        log_error "Please run this script as a regular user (not root)"
        log "Use: ./configure_camera_boot.sh"
        exit 1
    fi
    
    # Check system
    check_raspberry_pi
    
    # Get config path
    local config_path
    config_path=$(get_config_path)
    log "Using config file: $config_path"
    
    # Backup existing config
    backup_config "$config_path"
    
    # Detect camera sensor
    local sensor_type
    sensor_type=$(detect_camera_sensor)
    log_success "Detected sensor: $sensor_type"
    
    # Configure camera
    configure_camera "$config_path" "$sensor_type"
    configure_camera_files "$sensor_type"
    
    # Install special files if needed
    install_imx290_json "$sensor_type"
    
    # Validate configuration
    validate_configuration "$config_path"
    
    # Generate test script
    generate_test_script
    
    log_success "🎉 Camera configuration complete!"
    log ""
    log "📋 Next steps:"
    log "1. Reboot your Raspberry Pi: sudo reboot"
    log "2. After reboot, run the test script: ~/test_nightscan_camera.sh"
    log "3. Verify camera works with: python3 NightScanPi/Program/camera_test.py --all"
    log ""
    log "⚠️  IMPORTANT: A reboot is required for changes to take effect"
}

# Show help
show_help() {
    echo "NightScanPi Camera Boot Configuration Script"
    echo ""
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  -h, --help     Show this help message"
    echo "  -s, --sensor   Specify sensor type (imx219, ov5647, imx477, imx290, imx327)"
    echo ""
    echo "Examples:"
    echo "  $0                    # Auto-detect sensor and configure"
    echo "  $0 --sensor imx219    # Force IMX219 configuration"
    echo ""
    echo "This script configures /boot/firmware/config.txt for IR-CUT camera support."
}

# Parse command line arguments
SENSOR_TYPE=""
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -s|--sensor)
            SENSOR_TYPE="$2"
            shift 2
            ;;
        *)
            log_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Override detection if sensor specified
if [ -n "$SENSOR_TYPE" ]; then
    detect_camera_sensor() {
        echo "$SENSOR_TYPE"
    }
    log "Using specified sensor: $SENSOR_TYPE"
fi

# Run main function
main "$@"