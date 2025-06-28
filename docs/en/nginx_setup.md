# Nginx Setup

This document shows how to install Nginx and configure it as a reverse proxy for the NightScan web application and prediction API.

## Automatic setup

Run the helper script from the repository root, replacing `example.com` with your domain name:

```bash
sudo bash setup_nginx.sh example.com
```

The script installs Nginx if needed, creates `/etc/nginx/sites-available/nightscan` and enables it. The configuration forwards `/` to the Flask app on port 8000 and `/api/` to the prediction API on port 8001.

After running the script, verify that the service is reachable at `http://example.com`. For production deployments you should obtain HTTPS certificates, for instance with `certbot`:

```bash
sudo apt install certbot python3-certbot-nginx
sudo certbot --nginx -d example.com
```

This enables HTTPS for the site and redirects HTTP traffic to the secure endpoint.

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

