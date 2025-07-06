#!/bin/bash
set -e

# NightScanPi Automated Setup Script
# Comprehensive installation for Raspberry Pi with modern libcamera support
echo "🚀 NightScanPi Setup Script v2.0"
echo "================================="

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

# Ensure we can run apt commands
if [[ "$EUID" -ne 0 ]]; then
    if command -v sudo >/dev/null 2>&1; then
        SUDO="sudo"
    else
        log_error "This script requires root privileges for apt commands."
        exit 1
    fi
else
    SUDO=""
fi

# Check if running on Raspberry Pi
check_raspberry_pi() {
    if [ -f /proc/device-tree/model ]; then
        PI_MODEL=$(cat /proc/device-tree/model 2>/dev/null | tr -d '\0')
        log_success "Raspberry Pi detected: $PI_MODEL"
        export IS_RASPBERRY_PI=true
        
        # Check for Pi Zero specifically
        if [[ "$PI_MODEL" == *"Pi Zero"* ]]; then
            export IS_PI_ZERO=true
            log "Pi Zero detected - will apply memory optimizations"
        else
            export IS_PI_ZERO=false
        fi
    else
        log_warning "Not running on Raspberry Pi - some features will be limited"
        export IS_RASPBERRY_PI=false
        export IS_PI_ZERO=false
    fi
}

# Check OS version and capabilities
check_os_version() {
    if [ -f /etc/os-release ]; then
        source /etc/os-release
        log "Detected OS: $PRETTY_NAME"
        
        # Check for Raspberry Pi OS Bookworm vs Bullseye
        if [[ "$VERSION_CODENAME" == "bookworm" ]]; then
            export OS_VERSION="bookworm"
            export LIBCAMERA_CMD="rpicam"
            log "Using modern rpicam commands (Bookworm)"
        elif [[ "$VERSION_CODENAME" == "bullseye" ]]; then
            export OS_VERSION="bullseye"
            export LIBCAMERA_CMD="libcamera"
            log "Using libcamera commands (Bullseye)"
        else
            export OS_VERSION="unknown"
            export LIBCAMERA_CMD="libcamera"
            log_warning "Unknown OS version, using libcamera commands"
        fi
    else
        log_warning "Cannot detect OS version"
        export OS_VERSION="unknown"
        export LIBCAMERA_CMD="libcamera"
    fi
}

# Run system checks
check_raspberry_pi
check_os_version

# Enable SSH on first boot
enable_ssh() {
    local ssh_paths=("/boot/ssh" "/boot/firmware/ssh")
    for ssh_path in "${ssh_paths[@]}"; do
        if [ -d "$(dirname "$ssh_path")" ] && [ ! -e "$ssh_path" ]; then
            $SUDO touch "$ssh_path"
            log_success "SSH enabled at $ssh_path"
            break
        fi
    done
}

if [ "$IS_RASPBERRY_PI" = true ]; then
    enable_ssh
fi

# Define package lists based on system capabilities
get_package_list() {
    local base_packages=(
        "python3-pip"
        "python3-venv"
        "ffmpeg"
        "sox"
        "libatlas-base-dev"
        "chrony"
        "unzip"
        "wget"
        "curl"
        "git"
    )
    
    local camera_packages=()
    local audio_packages=(
        "alsa-utils"
        "pulseaudio"
        "portaudio19-dev"
    )
    
    # Camera packages based on OS version and Pi detection
    if [ "$IS_RASPBERRY_PI" = true ]; then
        if [ "$OS_VERSION" = "bookworm" ]; then
            camera_packages+=(
                "python3-picamera2"
                "libcamera-apps"
                "rpicam-apps"
            )
        elif [ "$OS_VERSION" = "bullseye" ]; then
            camera_packages+=(
                "python3-picamera2"
                "libcamera-apps"
                "libcamera-tools"
            )
        else
            # Fallback for older systems
            camera_packages+=(
                "python3-picamera"
                "python3-picamera2"
                "libcamera-apps"
            )
        fi
        
        # Pi Zero specific packages
        if [ "$IS_PI_ZERO" = true ]; then
            camera_packages+=(
                "libraspberrypi-bin"
                "libraspberrypi-dev"
            )
            
            # Additional Pi Zero optimizations
            base_packages+=(
                "zram-tools"      # Compressed memory for Pi Zero
                "dphys-swapfile"  # Swap management
            )
        fi
    else
        log_warning "Not on Raspberry Pi - skipping camera packages"
    fi
    
    # Combine all packages
    local all_packages=("${base_packages[@]}" "${camera_packages[@]}" "${audio_packages[@]}")
    echo "${all_packages[@]}"
}

# Install system packages with smart detection
install_system_packages() {
    log "🔄 Updating package lists..."
    $SUDO apt update
    
    # Get the appropriate package list
    local packages=($(get_package_list))
    local missing=()
    local available=()
    local unavailable=()
    
    log "📦 Checking package availability..."
    
    # Check which packages are available and missing
    for pkg in "${packages[@]}"; do
        if apt-cache show "$pkg" >/dev/null 2>&1; then
            available+=("$pkg")
            if ! dpkg -s "$pkg" >/dev/null 2>&1; then
                missing+=("$pkg")
            fi
        else
            unavailable+=("$pkg")
        fi
    done
    
    # Report package status
    if [ ${#unavailable[@]} -gt 0 ]; then
        log_warning "Unavailable packages: ${unavailable[*]}"
    fi
    
    if [ ${#missing[@]} -gt 0 ]; then
        log "Installing missing packages: ${missing[*]}"
        
        # Install with error handling
        for pkg in "${missing[@]}"; do
            if $SUDO apt install -y "$pkg"; then
                log_success "Installed: $pkg"
            else
                log_error "Failed to install: $pkg"
            fi
        done
    else
        log_success "All required packages are already installed"
    fi
}

install_system_packages

# Enable and start chrony to keep the clock in sync
configure_system_services() {
    if command -v systemctl >/dev/null 2>&1; then
        log "⏰ Configuring system services..."
        $SUDO systemctl enable --now chrony >/dev/null 2>&1 || true
        $SUDO timedatectl set-ntp true >/dev/null 2>&1 || true
        log_success "Time synchronization configured"
        
        # Enable camera on Pi
        if [ "$IS_RASPBERRY_PI" = true ]; then
            $SUDO raspi-config nonint do_camera 0 >/dev/null 2>&1 || true
            log_success "Camera interface enabled"
        fi
        
        # Pi Zero specific optimizations
        if [ "$IS_PI_ZERO" = true ]; then
            log "🔧 Applying Pi Zero 2W memory optimizations..."
            
            # Configure zRAM for compressed memory
            if command -v zramctl >/dev/null 2>&1; then
                $SUDO systemctl enable zramswap >/dev/null 2>&1 || true
                log_success "zRAM compressed memory enabled"
            fi
            
            # Configure swap file for Pi Zero
            if [ -f /etc/dphys-swapfile ]; then
                $SUDO sed -i 's/#CONF_SWAPSIZE=/CONF_SWAPSIZE=256/' /etc/dphys-swapfile 2>/dev/null || true
                $SUDO systemctl restart dphys-swapfile >/dev/null 2>&1 || true
                log_success "256MB swap file configured"
            fi
        fi
    fi
}

configure_system_services

# Setup Python environment with enhanced error handling
setup_python_environment() {
    log "🐍 Setting up Python environment..."
    
    # Check Python version
    if ! python3 --version >/dev/null 2>&1; then
        log_error "Python 3 not found"
        exit 1
    fi
    
    local python_version=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1-2)
    log "Python version: $python_version"
    
    # Create virtual environment
    if [ ! -d "env" ]; then
        log "Creating Python virtual environment..."
        python3 -m venv env
    else
        log "Virtual environment already exists"
    fi
    
    # Activate environment
    source env/bin/activate
    
    # Upgrade pip
    log "Upgrading pip..."
    pip install --upgrade pip wheel setuptools
    
    # Define Python packages based on system
    local base_python_packages=(
        "numpy"
        "soundfile"
        "sounddevice"
        "scipy"
        "flask"
        "flask-limiter"
        "flask-socketio"
        "requests"
        "Pillow"
    )
    
    local camera_python_packages=()
    local audio_python_packages=(
        "pyaudio"
        "pydub"
    )
    
    # Camera packages based on availability
    if [ "$IS_RASPBERRY_PI" = true ]; then
        # Try picamera2 first (modern)
        if pip install picamera2 >/dev/null 2>&1; then
            camera_python_packages+=("picamera2")
            log_success "Installed picamera2 (modern libcamera)"
        else
            log_warning "picamera2 not available, trying picamera (legacy)"
            camera_python_packages+=("picamera")
        fi
        
        # OpenCV optimized for Pi
        if [ "$IS_PI_ZERO" = true ]; then
            log "Installing lightweight OpenCV for Pi Zero..."
            camera_python_packages+=("opencv-python-headless")
            
            # Pi Zero specific optimizations
            camera_python_packages+=("psutil")  # Memory monitoring
        else
            camera_python_packages+=("opencv-python")
        fi
    else
        # Development environment
        camera_python_packages+=("opencv-python")
        log_warning "Development environment - camera packages may not work"
    fi
    
    # Install packages with error handling
    local all_python_packages=("${base_python_packages[@]}" "${camera_python_packages[@]}" "${audio_python_packages[@]}")
    
    for pkg in "${all_python_packages[@]}"; do
        log "Installing Python package: $pkg"
        if pip install "$pkg"; then
            log_success "Installed: $pkg"
        else
            log_warning "Failed to install: $pkg (continuing...)"
        fi
    done
}

setup_python_environment

# Install repository requirements if present
install_project_requirements() {
    if [ -f requirements.txt ]; then
        log "📋 Installing project requirements..."
        if pip install -r requirements.txt; then
            log_success "Project requirements installed"
        else
            log_warning "Some project requirements failed to install"
        fi
    else
        log "No requirements.txt found, skipping project requirements"
    fi
}

install_project_requirements

# Configure user permissions
configure_user_permissions() {
    log "👤 Configuring user permissions..."
    
    # Add user to required groups
    local groups=("video" "audio" "gpio" "spi" "i2c")
    for group in "${groups[@]}"; do
        if getent group "$group" >/dev/null; then
            $SUDO usermod -a -G "$group" "$USER" 2>/dev/null || true
            log_success "Added user $USER to group: $group"
        fi
    done
}

configure_user_permissions

# Test camera functionality
test_camera_installation() {
    if [ "$IS_RASPBERRY_PI" = true ]; then
        log "📸 Testing camera installation..."
        
        # Test libcamera commands
        if command -v "${LIBCAMERA_CMD}-hello" >/dev/null 2>&1; then
            log_success "${LIBCAMERA_CMD}-hello command available"
        else
            log_warning "${LIBCAMERA_CMD}-hello command not found"
        fi
        
        # Test Python camera modules
        if python3 -c "import picamera2; print('picamera2 available')" 2>/dev/null; then
            log_success "picamera2 Python module working"
        elif python3 -c "import picamera; print('picamera available')" 2>/dev/null; then
            log_success "picamera Python module working (legacy)"
        else
            log_warning "No working camera Python module found"
        fi
    fi
}

test_camera_installation

# Configure camera if this is a Raspberry Pi
configure_camera() {
    if [ "$IS_RASPBERRY_PI" = true ]; then
        log "🔍 Configuring camera for Raspberry Pi..."
        
        # Make camera configuration script executable
        if [ -f "NightScanPi/Hardware/configure_camera_boot.sh" ]; then
            chmod +x NightScanPi/Hardware/configure_camera_boot.sh
            
            # Check if running interactively
            if [ -t 0 ]; then
                # Interactive mode
                echo ""
                read -p "📸 Configure IR-CUT camera now? (y/n): " -n 1 -r
                echo
                if [[ $REPLY =~ ^[Yy]$ ]]; then
                    log "🚀 Running camera configuration..."
                    if ./NightScanPi/Hardware/configure_camera_boot.sh; then
                        log_success "Camera configuration completed"
                        
                        echo ""
                        log_warning "A system reboot is required for camera changes"
                        read -p "🔄 Reboot now? (y/n): " -n 1 -r
                        echo
                        if [[ $REPLY =~ ^[Yy]$ ]]; then
                            log "🔄 Rebooting in 5 seconds... Press Ctrl+C to cancel"
                            sleep 5
                            $SUDO reboot
                        else
                            log "⏰ Remember to reboot before using the camera: sudo reboot"
                        fi
                    else
                        log_error "Camera configuration failed"
                    fi
                else
                    log "⏭️  Camera configuration skipped"
                    log "Run manually later: ./NightScanPi/Hardware/configure_camera_boot.sh"
                fi
            else
                # Non-interactive mode - skip camera configuration
                log "Non-interactive mode - skipping camera configuration"
                log "Run manually: ./NightScanPi/Hardware/configure_camera_boot.sh"
            fi
        else
            log_error "Camera configuration script not found"
        fi
    else
        log "💻 Not on Raspberry Pi - skipping camera configuration"
    fi
}

configure_camera

# Generate utilities and final setup
finalize_installation() {
    log "🌅 Generating configuration files..."
    
    # Generate sun times data
    if python3 NightScanPi/Program/utils/sun_times.py "$HOME/sun_times.json" 2>/dev/null; then
        log_success "Sun times data generated"
    else
        log_warning "Failed to generate sun times data"
    fi
    
    # Create activation script
    cat > activate_nightscan.sh << 'EOF'
#!/bin/bash
# NightScanPi Environment Activation Script
echo "🚀 Activating NightScanPi environment..."
source env/bin/activate
echo "✅ Environment activated. You can now run:"
echo "  python NightScanPi/Program/camera_test.py --all"
echo "  python NightScanPi/Program/main.py"
EOF
    chmod +x activate_nightscan.sh
    log_success "Created activation script: ./activate_nightscan.sh"
}

finalize_installation

# Validate installation
validate_installation() {
    log "🔍 Validating installation..."
    
    local issues=()
    local successes=()
    
    # Check Python environment
    if [ -d "env" ] && [ -f "env/bin/activate" ]; then
        successes+=("Python virtual environment")
    else
        issues+=("Python virtual environment missing")
    fi
    
    # Check Python packages
    source env/bin/activate 2>/dev/null || true
    if python3 -c "import numpy, soundfile, flask" 2>/dev/null; then
        successes+=("Core Python packages")
    else
        issues+=("Core Python packages missing")
    fi
    
    # Check camera on Pi
    if [ "$IS_RASPBERRY_PI" = true ]; then
        if python3 -c "import picamera2" 2>/dev/null || python3 -c "import picamera" 2>/dev/null; then
            successes+=("Camera Python modules")
        else
            issues+=("Camera Python modules")
        fi
        
        if command -v "${LIBCAMERA_CMD}-hello" >/dev/null 2>&1; then
            successes+=("libcamera commands")
        else
            issues+=("libcamera commands")
        fi
    fi
    
    # Report results
    if [ ${#successes[@]} -gt 0 ]; then
        log_success "Installation validation - working components:"
        for item in "${successes[@]}"; do
            echo "  ✅ $item"
        done
    fi
    
    if [ ${#issues[@]} -gt 0 ]; then
        log_warning "Installation validation - potential issues:"
        for item in "${issues[@]}"; do
            echo "  ⚠️ $item"
        done
    else
        log_success "All components validated successfully!"
    fi
}

validate_installation

# Final summary and next steps
show_completion_summary() {
    echo ""
    log_success "🎉 NightScanPi installation complete!"
    echo ""
    
    # System information
    echo "📊 Installation Summary:"
    echo "  • System: $PI_MODEL"
    echo "  • OS: $OS_VERSION ($LIBCAMERA_CMD commands)"
    echo "  • Python: $(python3 --version | cut -d' ' -f2)"
    echo "  • Camera: $([ "$IS_RASPBERRY_PI" = true ] && echo "Configured" || echo "Development mode")"
    echo ""
    
    # Next steps
    echo "📋 Next Steps:"
    echo "1. Activate environment:"
    echo "   source env/bin/activate"
    echo "   # OR use: ./activate_nightscan.sh"
    echo ""
    
    if [ "$IS_RASPBERRY_PI" = true ]; then
        echo "2. Test camera functionality:"
        echo "   python NightScanPi/Program/camera_test.py --all"
        echo ""
        echo "3. Test sensor detection:"
        echo "   python NightScanPi/Program/camera_test.py --detect-sensor"
        echo ""
        echo "4. Run main program:"
        echo "   python NightScanPi/Program/main.py"
        echo ""
        
        if [ ! -f "/opt/nightscan/camera_info.json" ]; then
            echo "⚠️  Camera boot configuration recommended:"
            echo "   ./NightScanPi/Hardware/configure_camera_boot.sh"
            echo "   sudo reboot"
            echo ""
        fi
    else
        echo "2. Transfer to Raspberry Pi:"
        echo "   scp -r . pi@your-pi-ip:~/NightScan/"
        echo ""
        echo "3. Run setup on Pi:"
        echo "   ./NightScanPi/setup_pi.sh"
        echo ""
    fi
    
    echo "📖 Documentation:"
    echo "  • Hardware Guide: NightScanPi/Hardware/README.md"
    echo "  • Camera Config: NightScanPi/Hardware/CAMERA_CONFIGURATION_GUIDE.md"
    echo "  • Troubleshooting: Run camera_test.py for diagnostics"
    echo ""
    
    log_success "Setup completed successfully! 🚀"
}

show_completion_summary
