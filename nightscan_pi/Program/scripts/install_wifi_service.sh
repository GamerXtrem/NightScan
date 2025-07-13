#!/bin/bash
# 
# Install NightScan WiFi Service
# Sets up systemd service for automatic WiFi management
#

set -e

echo "🔧 Installing NightScan WiFi Service..."

# Check if running as root
if [[ $EUID -ne 0 ]]; then
   echo "This script must be run as root (use sudo)" 
   exit 1
fi

# Variables
NIGHTSCAN_DIR="/opt/nightscan"
PROGRAM_DIR="$NIGHTSCAN_DIR/NightScanPi/Program"
SERVICE_FILE="/etc/systemd/system/nightscan-wifi.service"

# Create directories
echo "📁 Creating directories..."
mkdir -p "$NIGHTSCAN_DIR/logs"
mkdir -p "$NIGHTSCAN_DIR/config"

# Copy program files if they don't exist
if [ ! -d "$PROGRAM_DIR" ]; then
    echo "📋 Creating program directory structure..."
    mkdir -p "$PROGRAM_DIR"
    
    # Note: In production, you would copy the actual program files here
    echo "⚠️  Please copy the NightScan program files to $PROGRAM_DIR"
fi

# Create systemd service file
echo "📝 Creating systemd service..."
cat > "$SERVICE_FILE" << 'EOF'
[Unit]
Description=NightScan WiFi Management Service
After=network.target
Wants=network.target

[Service]
Type=simple
User=root
Group=root
WorkingDirectory=/opt/nightscan
Environment=PYTHONPATH=/opt/nightscan/NightScanPi/Program
ExecStart=/usr/bin/python3 /opt/nightscan/NightScanPi/Program/scripts/wifi_startup.py --daemon
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

# Security settings
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ReadWritePaths=/opt/nightscan /etc/wpa_supplicant /etc/hostapd /etc/dnsmasq.conf
ProtectHome=true

[Install]
WantedBy=multi-user.target
EOF

# Set permissions
echo "🔒 Setting permissions..."
chmod 644 "$SERVICE_FILE"
chown -R root:root "$NIGHTSCAN_DIR"
chmod 755 "$NIGHTSCAN_DIR"
chmod 755 "$NIGHTSCAN_DIR/logs"
chmod 755 "$NIGHTSCAN_DIR/config"

# Create log rotation configuration
echo "📝 Setting up log rotation..."
cat > /etc/logrotate.d/nightscan-wifi << 'EOF'
/opt/nightscan/logs/*.log {
    daily
    rotate 7
    compress
    delaycompress
    missingok
    notifempty
    create 644 root root
    postrotate
        /bin/systemctl reload nightscan-wifi > /dev/null 2>&1 || true
    endscript
}
EOF

# Create configuration template
echo "📋 Creating configuration template..."
cat > "$NIGHTSCAN_DIR/config/wifi_networks.json.template" << 'EOF'
{
  "example_network": {
    "ssid": "example_network",
    "password": "your_password_here",
    "security": "wpa2_psk",
    "priority": 50,
    "auto_connect": true,
    "hidden": false,
    "notes": "Example network configuration"
  }
}
EOF

# Create hotspot configuration template
cat > "$NIGHTSCAN_DIR/config/hotspot_config.json.template" << 'EOF'
{
  "ssid": "NightScan-Setup",
  "password": "nightscan2024",
  "channel": 6,
  "hidden": false,
  "max_clients": 10,
  "ip_range": "192.168.4.0/24",
  "gateway": "192.168.4.1",
  "dhcp_start": "192.168.4.100",
  "dhcp_end": "192.168.4.200"
}
EOF

# Create helper script for network management
echo "🔧 Creating helper script..."
cat > "$NIGHTSCAN_DIR/nightscan-wifi" << 'EOF'
#!/bin/bash
# NightScan WiFi Helper Script

PYTHON_CMD="/usr/bin/python3"
WIFI_MANAGER="$NIGHTSCAN_DIR/NightScanPi/Program/wifi_manager.py"
export PYTHONPATH="$NIGHTSCAN_DIR/NightScanPi/Program"

case "$1" in
    "status")
        $PYTHON_CMD "$WIFI_MANAGER" --status
        ;;
    "scan")
        $PYTHON_CMD "$WIFI_MANAGER" --scan
        ;;
    "connect")
        if [ -z "$2" ]; then
            echo "Usage: $0 connect <ssid>"
            exit 1
        fi
        $PYTHON_CMD "$WIFI_MANAGER" --connect "$2"
        ;;
    "hotspot")
        $PYTHON_CMD "$WIFI_MANAGER" --hotspot
        ;;
    "add")
        if [ -z "$2" ] || [ -z "$3" ]; then
            echo "Usage: $0 add <ssid> <password>"
            exit 1
        fi
        $PYTHON_CMD "$WIFI_MANAGER" --add-network "$2" "$3"
        ;;
    "service")
        case "$2" in
            "start")
                systemctl start nightscan-wifi
                ;;
            "stop")
                systemctl stop nightscan-wifi
                ;;
            "restart")
                systemctl restart nightscan-wifi
                ;;
            "status")
                systemctl status nightscan-wifi
                ;;
            "logs")
                journalctl -u nightscan-wifi -f
                ;;
            *)
                echo "Usage: $0 service {start|stop|restart|status|logs}"
                exit 1
                ;;
        esac
        ;;
    *)
        echo "NightScan WiFi Management"
        echo "Usage: $0 {status|scan|connect|hotspot|add|service}"
        echo ""
        echo "Commands:"
        echo "  status              - Show current WiFi status"
        echo "  scan                - Scan for available networks"
        echo "  connect <ssid>      - Connect to a network"
        echo "  hotspot             - Start hotspot mode"
        echo "  add <ssid> <pass>   - Add a new network"
        echo "  service <action>    - Manage the systemd service"
        echo ""
        echo "Examples:"
        echo "  $0 scan"
        echo "  $0 add MyWiFi password123"
        echo "  $0 connect MyWiFi"
        echo "  $0 service start"
        exit 1
        ;;
esac
EOF

chmod +x "$NIGHTSCAN_DIR/nightscan-wifi"

# Create symlink for easy access
ln -sf "$NIGHTSCAN_DIR/nightscan-wifi" /usr/local/bin/nightscan-wifi

# Reload systemd
echo "🔄 Reloading systemd..."
systemctl daemon-reload

# Enable service
echo "✅ Enabling service..."
systemctl enable nightscan-wifi

echo "✅ NightScan WiFi service installed successfully!"
echo ""
echo "🎯 Next steps:"
echo "1. Configure your WiFi networks:"
echo "   sudo nightscan-wifi add 'YourNetwork' 'YourPassword'"
echo ""
echo "2. Start the service:"
echo "   sudo nightscan-wifi service start"
echo ""
echo "3. Check service status:"
echo "   sudo nightscan-wifi service status"
echo ""
echo "4. View logs:"
echo "   sudo nightscan-wifi service logs"
echo ""
echo "🌐 Available commands:"
echo "   sudo nightscan-wifi status    - Check WiFi status"
echo "   sudo nightscan-wifi scan      - Scan for networks"
echo "   sudo nightscan-wifi hotspot   - Start hotspot mode"
echo ""
echo "📁 Configuration files:"
echo "   $NIGHTSCAN_DIR/config/wifi_networks.json"
echo "   $NIGHTSCAN_DIR/config/hotspot_config.json"
echo "   $NIGHTSCAN_DIR/logs/wifi_startup.log"