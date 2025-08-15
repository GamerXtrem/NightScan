#!/bin/bash
# NightScanPi Production Configuration Script
# This script helps configure a NightScanPi device for production deployment

set -e

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

echo "🚀 NightScanPi Production Configuration"
echo "======================================"

# Check if running as root
if [[ "$EUID" -eq 0 ]]; then
    log_error "Do not run this script as root. Use a regular user account."
    exit 1
fi

# Check if .env.template exists
if [[ ! -f ".env.template" ]]; then
    log_error ".env.template not found. Please run from the nightscan_pi directory."
    exit 1
fi

# Create .env file if it doesn't exist
if [[ ! -f ".env" ]]; then
    log "Creating .env file from template..."
    cp .env.template .env
    log_success ".env file created"
else
    log_warning ".env file already exists - will update existing configuration"
fi

# Interactive configuration
echo ""
log "Starting interactive configuration..."

# API Configuration
echo ""
echo "📡 API Configuration"
echo "==================="
read -p "Enter your NightScan API URL (e.g., https://api.yourdomain.com/upload): " api_url
if [[ -n "$api_url" ]]; then
    sed -i "s|NIGHTSCAN_API_URL=.*|NIGHTSCAN_API_URL=$api_url|" .env
    log_success "API URL configured: $api_url"
fi

read -p "Enter your API token (optional, press Enter to skip): " api_token
if [[ -n "$api_token" ]]; then
    sed -i "s|NIGHTSCAN_API_TOKEN=.*|NIGHTSCAN_API_TOKEN=$api_token|" .env
    log_success "API token configured"
fi

# Location Configuration
echo ""
echo "📍 Location Configuration"
echo "========================"
read -p "Enter GPS coordinates (format: lat,lon, e.g., 46.9,7.4): " gps_coords
if [[ -n "$gps_coords" ]]; then
    sed -i "s|NIGHTSCAN_GPS_COORDS=.*|NIGHTSCAN_GPS_COORDS=$gps_coords|" .env
    log_success "GPS coordinates configured: $gps_coords"
fi

read -p "Enter timezone (e.g., Europe/Zurich, press Enter for auto-detection): " timezone
if [[ -n "$timezone" ]]; then
    sed -i "s|NIGHTSCAN_TIMEZONE=.*|NIGHTSCAN_TIMEZONE=$timezone|" .env
    log_success "Timezone configured: $timezone"
fi

# Hardware Configuration
echo ""
echo "🔧 Hardware Configuration"
echo "========================="

# Check if this is a Pi Zero
if [[ -f "/proc/device-tree/model" ]]; then
    pi_model=$(cat /proc/device-tree/model | tr -d '\0')
    if [[ "$pi_model" == *"Pi Zero"* ]]; then
        log "Pi Zero detected - enabling optimizations"
        sed -i "s|NIGHTSCAN_FORCE_PI_ZERO=.*|NIGHTSCAN_FORCE_PI_ZERO=true|" .env
        
        # Set Pi Zero optimized values
        sed -i "s|NIGHTSCAN_CAMERA_RESOLUTION=.*|NIGHTSCAN_CAMERA_RESOLUTION=1280,720|" .env
        sed -i "s|NIGHTSCAN_THREADS=.*|NIGHTSCAN_THREADS=1|" .env
        sed -i "s|NIGHTSCAN_MAX_MEMORY_PERCENT=.*|NIGHTSCAN_MAX_MEMORY_PERCENT=80|" .env
        
        log_success "Pi Zero optimizations applied"
    fi
fi

# SIM Module Configuration
echo ""
read -p "Do you have a SIM module for cellular connectivity? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    read -p "Enter SIM device path (e.g., /dev/ttyUSB0): " sim_device
    if [[ -n "$sim_device" ]]; then
        sed -i "s|NIGHTSCAN_SIM_DEVICE=.*|NIGHTSCAN_SIM_DEVICE=$sim_device|" .env
        log_success "SIM device configured: $sim_device"
    fi
fi

# Audio Configuration
echo ""
echo "🎵 Audio Configuration"
echo "====================="
read -p "Enter audio detection threshold (0.0-1.0, default 0.5): " audio_threshold
if [[ -n "$audio_threshold" ]]; then
    sed -i "s|NIGHTSCAN_AUDIO_THRESHOLD=.*|NIGHTSCAN_AUDIO_THRESHOLD=$audio_threshold|" .env
    log_success "Audio threshold configured: $audio_threshold"
fi

# Create data directories
echo ""
log "Creating data directories..."
mkdir -p "$HOME/nightscan_data/audio"
mkdir -p "$HOME/nightscan_data/images"
mkdir -p "$HOME/nightscan_data/spectrograms"
log_success "Data directories created"

# Update data directory in .env
sed -i "s|NIGHTSCAN_DATA_DIR=.*|NIGHTSCAN_DATA_DIR=$HOME/nightscan_data|" .env
sed -i "s|NIGHTSCAN_LOG=.*|NIGHTSCAN_LOG=$HOME/nightscan.log|" .env

# Set file permissions
chmod 600 .env
log_success "Environment file permissions secured"

# Generate initial sun times
if [[ -n "$gps_coords" ]]; then
    log "Generating initial sun times data..."
    if python3 -c "
import sys
sys.path.append('Program')
from Program.utils.sun_times import generate_sun_times
coords = '$gps_coords'.split(',')
lat, lon = float(coords[0]), float(coords[1])
generate_sun_times(lat, lon, '$HOME/sun_times.json')
print('Sun times generated successfully')
" 2>/dev/null; then
        sed -i "s|NIGHTSCAN_SUN_FILE=.*|NIGHTSCAN_SUN_FILE=$HOME/sun_times.json|" .env
        log_success "Sun times data generated"
    else
        log_warning "Could not generate sun times data"
    fi
fi

# Test configuration
echo ""
log "Testing configuration..."

# Test API connectivity
if [[ -n "$api_url" ]]; then
    if curl -s --head "$api_url" | head -n 1 | grep -q "200\|404"; then
        log_success "API endpoint is reachable"
    else
        log_warning "API endpoint may not be reachable - check URL and network"
    fi
fi

# Create systemd service file
echo ""
read -p "Create systemd service for automatic startup? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    log "Creating systemd service..."
    
    cat > nightscan.service << EOF
[Unit]
Description=NightScan Wildlife Detection System
After=network.target
Wants=network.target

[Service]
Type=simple
User=$(whoami)
WorkingDirectory=$(pwd)
Environment=PATH=/home/$(whoami)/.local/bin:/usr/local/bin:/usr/bin:/bin
ExecStart=$(pwd)/env/bin/python Program/main.py
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF
    
    log_success "Service file created: nightscan.service"
    echo ""
    echo "To install the service, run:"
    echo "  sudo cp nightscan.service /etc/systemd/system/"
    echo "  sudo systemctl enable nightscan"
    echo "  sudo systemctl start nightscan"
fi

# Create backup script
cat > backup_config.sh << 'EOF'
#!/bin/bash
# NightScanPi Configuration Backup Script
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="$HOME/nightscan_backups"
mkdir -p "$BACKUP_DIR"

# Backup configuration files
tar -czf "$BACKUP_DIR/nightscan_config_$DATE.tar.gz" \
    .env \
    nightscan.service \
    Program/nightscan_config.db \
    "$HOME/sun_times.json" \
    2>/dev/null

echo "Configuration backed up to: $BACKUP_DIR/nightscan_config_$DATE.tar.gz"

# Keep only last 5 backups
ls -t "$BACKUP_DIR"/nightscan_config_*.tar.gz | tail -n +6 | xargs -r rm
EOF

chmod +x backup_config.sh
log_success "Backup script created: backup_config.sh"

# Final summary
echo ""
echo "🎉 Configuration Complete!"
echo "========================="
echo ""
echo "Configuration summary:"
echo "  • Environment file: .env (configured)"
echo "  • Data directory: $HOME/nightscan_data"
echo "  • Log file: $HOME/nightscan.log"
if [[ -n "$api_url" ]]; then
    echo "  • API URL: $api_url"
fi
if [[ -n "$gps_coords" ]]; then
    echo "  • GPS coordinates: $gps_coords"
fi
echo ""
echo "Next steps:"
echo "1. Review and adjust .env file if needed"
echo "2. Test the system: python Program/main.py"
echo "3. Install systemd service for automatic startup"
echo "4. Monitor logs: tail -f $HOME/nightscan.log"
echo ""
echo "For troubleshooting, run: python Program/camera_test.py --all"
echo ""
log_success "NightScanPi is ready for production! 🚀"