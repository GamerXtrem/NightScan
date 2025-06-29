# Nginx Setup

This document shows how to install Nginx and configure it as a reverse proxy for the NightScan web application and prediction API.

## Automatic setup

Run the helper script from the repository root, replacing `example.com` with your domain name:

```bash
sudo bash setup_nginx.sh example.com
```

The script installs Nginx if needed, creates `/etc/nginx/sites-available/nightscan` and enables it. The configuration forwards `/` to the Flask app on port 8000 and `/api/` to the prediction API on port 8001.

If you want to configure HTTPS automatically, use the helper `setup_nginx_tls.sh` script instead. It runs the steps above and invokes `certbot` to obtain a Let's Encrypt certificate:

```bash
sudo bash setup_nginx_tls.sh example.com
```

Both scripts configure Nginx and reload it. When using `setup_nginx_tls.sh` the service is immediately reachable at `https://example.com` with HTTP traffic redirected to the secure endpoint.

## Manual configuration

If you prefer to configure Nginx manually, create `/etc/nginx/sites-available/nightscan` with the following contents:

```nginx
server {
    listen 80;
    server_name example.com;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    location /api/ {
        proxy_pass http://127.0.0.1:8001/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

Enable the site with:

```bash
sudo ln -s /etc/nginx/sites-available/nightscan /etc/nginx/sites-enabled/nightscan
sudo nginx -t
sudo systemctl restart nginx
```

Once configured, the proxy handles all incoming requests and forwards them to the Gunicorn workers. Remember to secure the connection with HTTPS when the service is exposed on the public internet.

