# NightScan API Gateway

## Overview

Unified API Gateway for NightScan using Kong, providing:
- Single entry point for all API services
- JWT-based authentication
- Rate limiting and quota management
- CORS handling
- Request routing and load balancing
- Health checks and monitoring

## Quick Start

### 1. Start the Gateway

```bash
# Start Kong and dependencies
docker-compose -f docker-compose.gateway.yml up -d

# Configure Kong (routes, plugins, etc.)
./gateway/setup_kong.sh
```

### 2. Test the Gateway

```bash
# Health check
curl http://localhost:8080/health

# Login (get JWT token)
curl -X POST http://localhost:8080/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"your-username","password":"your-password"}'

# Use token for API calls
curl http://localhost:8080/api/v1/detections \
  -H "Authorization: Bearer <your-jwt-token>"
```

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐
│   Clients   │────▶│  Kong Gateway │────▶│ Backend Services│
│(Web/iOS/API)│     │   (Port 8080) │     │  (8000-8012)   │
└─────────────┘     └──────────────┘     └─────────────────┘
                           │
                           ▼
                    ┌──────────────┐
                    │   Plugins    │
                    │ JWT, CORS,   │
                    │ Rate Limit   │
                    └──────────────┘
```

## Services Configuration

### Routed Services

| Service | Internal Port | Gateway Path | Authentication |
|---------|--------------|--------------|----------------|
| Web App | 8000 | `/auth/*`, `/login`, `/register` | Optional |
| API v1 | 8000 | `/api/v1/*` | JWT Required |
| Prediction | 8002 | `/api/predict` | JWT Required |
| Analytics | 8008 | `/api/analytics/*` | JWT Required |
| WebSocket | 8012 | `/ws`, `/socket.io` | JWT Optional |

### Direct Services (No Gateway)

| Service | Port | Description |
|---------|------|-------------|
| Pi Camera | 192.168.4.1:5000 | Direct connection for local network |
| Pi Location | 192.168.4.1:5001 | Direct connection for local network |

## Authentication

### JWT Token Flow

1. **Login**: `POST /api/auth/login`
   ```json
   {
     "username": "user",
     "password": "pass"
   }
   ```
   Response:
   ```json
   {
     "access_token": "eyJ...",
     "refresh_token": "eyJ...",
     "expires_in": 3600
   }
   ```

2. **Use Token**: Add to Authorization header
   ```
   Authorization: Bearer eyJ...
   ```

3. **Refresh**: `POST /api/auth/refresh`
   ```json
   {
     "refresh_token": "eyJ..."
   }
   ```

### Token Management

- Access tokens expire in 1 hour
- Refresh tokens expire in 30 days
- Tokens are automatically refreshed by clients
- Revoked tokens are blacklisted

## Rate Limiting

### Default Limits

| Tier | Per Minute | Per Hour | Per Day |
|------|------------|----------|---------|
| Free | 60 | 600 | 5,000 |
| Premium | 600 | 6,000 | 50,000 |
| Enterprise | 6,000 | 60,000 | 500,000 |

### Custom Limits

Configure in `kong.yml` or via Admin API:
```bash
curl -X PATCH http://localhost:8081/consumers/{consumer}/plugins/rate-limiting \
  -d "config.minute=100"
```

## Monitoring

### Endpoints

- `/metrics` - Prometheus metrics
- `/health` - Basic health check
- `/health/detailed` - Component status

### Admin UI

- Kong Admin API: http://localhost:8081
- Konga UI: http://localhost:1337

### Logs

```bash
# View Gateway logs
docker-compose -f docker-compose.gateway.yml logs -f kong

# View specific service logs
docker logs nightscan-gateway
```

## Configuration

### Environment Variables

```bash
# JWT Secret (generate with: openssl rand -hex 32)
JWT_SECRET_KEY=your-secret-key

# Kong Admin URL
KONG_ADMIN_URL=http://localhost:8081

# Service URLs (for Docker networking)
WEB_SERVICE_URL=http://web:8000
PREDICTION_SERVICE_URL=http://prediction:8002
```

### Modifying Routes

Edit `kong.yml` and restart:
```bash
docker-compose -f docker-compose.gateway.yml restart kong
```

Or use Admin API:
```bash
curl -X POST http://localhost:8081/services/{service}/routes \
  -d "paths[]=/new/path"
```

## Troubleshooting

### Common Issues

1. **"No Route Matched"**
   - Check route configuration: `curl http://localhost:8081/routes`
   - Verify path matching in `kong.yml`

2. **"Unauthorized"**
   - Verify JWT token is valid
   - Check token expiration
   - Ensure Authorization header format

3. **"Service Unavailable"**
   - Check backend service health
   - Verify Docker networking
   - Check service URLs in configuration

### Debug Mode

Enable debug logging:
```bash
KONG_LOG_LEVEL=debug docker-compose up
```

## Development

### Adding New Service

1. Add to `kong.yml`:
   ```yaml
   services:
     - name: new-service
       url: http://new-service:8080
       routes:
         - paths: ["/api/new"]
   ```

2. Apply configuration:
   ```bash
   ./setup_kong.sh
   ```

### Testing Locally

```bash
# Direct service access (bypass gateway)
curl http://localhost:8000/health

# Through gateway
curl http://localhost:8080/health
```

## Production Deployment

1. Use HTTPS (port 8443)
2. Set strong JWT secret
3. Configure proper CORS origins
4. Enable distributed rate limiting
5. Set up monitoring and alerting
6. Use Kong database mode for HA

## Security

- JWT secrets are rotated monthly
- All traffic should use HTTPS in production
- Rate limiting prevents abuse
- CORS configured per environment
- Request/response logging for audit