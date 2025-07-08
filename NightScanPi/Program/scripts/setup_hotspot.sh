#!/bin/bash
# 
# NightScan Hotspot Setup Script
# Configures the system for hotspot mode functionality
#

set -e

echo "🚀 Setting up NightScan Hotspot capabilities..."

# Check if running as root
if [[ $EUID -ne 0 ]]; then
   echo "This script must be run as root (use sudo)" 
   exit 1
fi

# Install required packages
echo "📦 Installing required packages..."
apt-get update
apt-get install -y hostapd dnsmasq iptables-persistent

# Stop services initially
echo "🛑 Stopping services..."
systemctl stop hostapd
systemctl stop dnsmasq

# Disable services initially (will be managed by WiFiManager)
echo "🔧 Configuring services..."
systemctl disable hostapd
systemctl disable dnsmasq

# Create backup of original configurations
echo "💾 Creating configuration backups..."
if [ -f /etc/hostapd/hostapd.conf ]; then
    cp /etc/hostapd/hostapd.conf /etc/hostapd/hostapd.conf.backup
fi

if [ -f /etc/dnsmasq.conf ]; then
    cp /etc/dnsmasq.conf /etc/dnsmasq.conf.backup
fi

# Create configuration directories
echo "📁 Creating configuration directories..."
mkdir -p /opt/nightscan/config
mkdir -p /opt/nightscan/logs

# Create systemd service for NightScan WiFi Manager
echo "🔧 Creating systemd service..."
cat > /etc/systemd/system/nightscan-wifi.service << EOF
[Unit]
Description=NightScan WiFi Manager
After=network.target
Wants=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt/nightscan
ExecStart=/usr/bin/python3 /opt/nightscan/NightScanPi/Program/wifi_manager.py --daemon
Restart=always
RestartSec=10
Environment=PYTHONPATH=/opt/nightscan/NightScanPi/Program

[Install]
WantedBy=multi-user.target
EOF

# Create logrotate configuration
echo "📝 Configuring log rotation..."
cat > /etc/logrotate.d/nightscan-wifi << EOF
/opt/nightscan/logs/wifi_manager.log {
    daily
    rotate 7
    compress
    delaycompress
    missingok
    notifempty
    create 644 root root
}
EOF

# Set up IP forwarding
echo "🌐 Configuring IP forwarding..."
echo 'net.ipv4.ip_forward=1' >> /etc/sysctl.conf

# Configure iptables rules for NAT
echo "🔥 Configuring iptables..."
iptables -t nat -A POSTROUTING -o eth0 -j MASQUERADE
iptables -A FORWARD -i eth0 -o wlan0 -m state --state RELATED,ESTABLISHED -j ACCEPT
iptables -A FORWARD -i wlan0 -o eth0 -j ACCEPT

# Save iptables rules
iptables-save > /etc/iptables/rules.v4

# Create network interface configuration
echo "🔧 Configuring network interface..."
cat >> /etc/dhcpcd.conf << EOF

# NightScan WiFi Manager configuration
# Static IP for hotspot mode (will be managed dynamically)
EOF

# Set permissions
echo "🔒 Setting permissions..."
chown -R root:root /opt/nightscan/config
chmod 755 /opt/nightscan/config

# Enable and start the service
echo "🚀 Enabling NightScan WiFi service..."
systemctl daemon-reload
systemctl enable nightscan-wifi

echo "✅ NightScan Hotspot setup complete!"
echo ""
echo "🎯 Next steps:"
echo "1. Copy the WiFi manager to /opt/nightscan/NightScanPi/Program/"
echo "2. Start the service: sudo systemctl start nightscan-wifi"
echo "3. Check status: sudo systemctl status nightscan-wifi"
echo "4. View logs: sudo journalctl -u nightscan-wifi -f"
echo ""
echo "🌐 Default hotspot configuration:"
echo "   SSID: NightScan-Setup"
echo "   Password: nightscan2024"
echo "   IP: 192.168.4.1"
echo "   Web interface: http://192.168.4.1:5000"