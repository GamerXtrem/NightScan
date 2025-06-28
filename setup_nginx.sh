#!/bin/bash
set -e

# Install Nginx and create a basic configuration for NightScan.
# Usage: sudo bash setup_nginx.sh example.com
# Replace example.com with your actual domain name.

if [[ "$EUID" -ne 0 ]]; then
    if command -v sudo >/dev/null 2>&1; then
        exec sudo "$0" "$@"
    else
        echo "This script must run with root privileges." >&2
        exit 1
    fi
fi

DOMAIN="$1"
if [ -z "$DOMAIN" ]; then
    echo "Usage: $0 domain-name" >&2
    exit 1
fi

apt-get update
apt-get install -y nginx

CONFIG_PATH="/etc/nginx/sites-available/nightscan"
cat > "$CONFIG_PATH" <<EOF2
server {
    listen 80;
    server_name $DOMAIN;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
    }

    location /api/ {
        proxy_pass http://127.0.0.1:8001/;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
    }
}
EOF2

ln -sf "$CONFIG_PATH" /etc/nginx/sites-enabled/nightscan

nginx -t
systemctl restart nginx

echo "Nginx is configured. Consider enabling HTTPS with certbot."
