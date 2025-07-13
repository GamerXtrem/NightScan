# Session Configuration Guide

This guide explains how to configure sessions in NightScan for different deployment scenarios.

## Overview

NightScan supports three session storage backends:

1. **Filesystem** (default) - Stores sessions as JSON files
2. **Redis** (recommended for production) - Centralized session storage
3. **Memory** (not recommended) - In-process storage, breaks with multiple instances

## Configuration Options

Session configuration is managed through the `SecurityConfig` in `config.py`:

```python
@dataclass
class SecurityConfig:
    # Session storage backend
    session_backend: str = "filesystem"  # Options: redis, filesystem, memory
    session_lifetime: int = 3600  # Session lifetime in seconds (1 hour)
    session_cookie_secure: bool = True  # HTTPS only cookies
    session_cookie_httponly: bool = True  # No JavaScript access
    session_cookie_samesite: str = "Lax"  # CSRF protection
```

## Environment Configuration

### Development (Single Instance)

```bash
# .env.development
NIGHTSCAN_SECURITY__SESSION_BACKEND=filesystem
NIGHTSCAN_SECURITY__SESSION_LIFETIME=7200  # 2 hours for development
NIGHTSCAN_SECURITY__SESSION_COOKIE_SECURE=false  # Allow HTTP in dev
```

### Staging (Multi-Instance)

```bash
# .env.staging
NIGHTSCAN_SECURITY__SESSION_BACKEND=redis
NIGHTSCAN_REDIS__URL=redis://redis-staging:6379/1
NIGHTSCAN_SECURITY__SESSION_LIFETIME=3600
NIGHTSCAN_SECURITY__SESSION_COOKIE_SECURE=true
```

### Production (Multi-Instance)

```bash
# .env.production
NIGHTSCAN_SECURITY__SESSION_BACKEND=redis
NIGHTSCAN_REDIS__URL=redis://redis-prod:6379/1
NIGHTSCAN_SECURITY__SESSION_LIFETIME=1800  # 30 minutes for security
NIGHTSCAN_SECURITY__SESSION_COOKIE_SECURE=true
NIGHTSCAN_SECURITY__SESSION_COOKIE_SAMESITE=Strict  # Stricter CSRF protection
```

## Backend Details

### Filesystem Backend

- **Pros**: No additional dependencies, works out of the box
- **Cons**: Requires shared storage (NFS) for multi-instance deployments
- **Storage Location**: `{temp_dir}/sessions/`
- **File Format**: JSON files named `{session_id}.json`

```python
# Example session file: /tmp/nightscan/sessions/abc123.json
{
    "session_id": "abc123...",
    "user_id": "user123",
    "created_at": "2024-01-15T10:00:00",
    "last_activity": "2024-01-15T10:30:00",
    "ip_address": "192.168.1.100",
    "data": {}
}
```

### Redis Backend

- **Pros**: Centralized storage, high performance, supports clustering
- **Cons**: Requires Redis server
- **Key Format**: `session:{session_id}`
- **Serialization**: Pickle (for Python object support)

```bash
# Check Redis sessions
redis-cli
> KEYS session:*
> TTL session:abc123
> GET session:abc123
```

### Memory Backend

- **Warning**: Only use for single-instance deployments or testing
- **Problem**: Sessions are lost on restart and not shared between instances
- **Use Case**: Development, testing, or single-server deployments

## Multi-Instance Deployment

For deployments with multiple application instances (load balancing, high availability):

### Option 1: Redis (Recommended)

```yaml
# docker-compose.yml
services:
  redis:
    image: redis:7-alpine
    volumes:
      - redis_data:/data
    command: redis-server --appendonly yes
    
  web:
    environment:
      - NIGHTSCAN_SECURITY__SESSION_BACKEND=redis
      - NIGHTSCAN_REDIS__URL=redis://redis:6379/1
```

### Option 2: Shared Filesystem

```yaml
# docker-compose.yml
services:
  web:
    volumes:
      - sessions_data:/tmp/nightscan/sessions
    environment:
      - NIGHTSCAN_SECURITY__SESSION_BACKEND=filesystem
      
volumes:
  sessions_data:
    driver: local
    driver_opts:
      type: nfs
      o: addr=nfs-server,rw
      device: ":/shared/sessions"
```

## Session Security Best Practices

1. **Use HTTPS in Production**
   ```bash
   NIGHTSCAN_SECURITY__SESSION_COOKIE_SECURE=true
   NIGHTSCAN_SECURITY__FORCE_HTTPS=true
   ```

2. **Short Session Lifetime**
   ```bash
   # Production: 30 minutes
   NIGHTSCAN_SECURITY__SESSION_LIFETIME=1800
   ```

3. **Strict SameSite for CSRF Protection**
   ```bash
   NIGHTSCAN_SECURITY__SESSION_COOKIE_SAMESITE=Strict
   ```

4. **Regular Session Cleanup**
   ```python
   # Add to cron or scheduled task
   from security import UnifiedSecurity
   security = UnifiedSecurity()
   cleaned = security.session.cleanup_expired_sessions()
   print(f"Cleaned {cleaned} expired sessions")
   ```

## Monitoring Sessions

### Active Session Count
```python
from security import UnifiedSecurity
security = UnifiedSecurity()
count = security.session.get_active_sessions_count()
print(f"Active sessions: {count}")
```

### Session Information
```python
# Get session details (sanitized)
info = security.session.get_session_info(session_id)
print(f"User: {info['user_id']}")
print(f"Created: {info['created_at']}")
print(f"Last active: {info['last_activity']}")
print(f"Expired: {info['is_expired']}")
```

### Destroy User Sessions
```python
# Logout user from all devices
count = security.session.destroy_all_user_sessions(user_id)
print(f"Destroyed {count} sessions for user {user_id}")
```

## Troubleshooting

### Sessions Not Persisting

1. Check backend configuration:
   ```bash
   echo $NIGHTSCAN_SECURITY__SESSION_BACKEND
   ```

2. Verify Redis connection (if using Redis):
   ```bash
   redis-cli -h redis-host ping
   ```

3. Check filesystem permissions (if using filesystem):
   ```bash
   ls -la /tmp/nightscan/sessions/
   ```

### Session Errors in Logs

1. **"Failed to connect to Redis"**: Redis server is down or unreachable
2. **"Permission denied"**: Check directory permissions for filesystem backend
3. **"Session expired"**: Normal behavior, sessions expire after `session_lifetime`

### Multi-Instance Issues

If using filesystem backend with multiple instances:
1. Ensure shared storage is mounted on all instances
2. Check file locking isn't causing delays
3. Consider switching to Redis for better performance

## Migration Guide

### From Memory to Redis

1. Install Redis:
   ```bash
   docker run -d --name redis -p 6379:6379 redis:7-alpine
   ```

2. Update configuration:
   ```bash
   export NIGHTSCAN_SECURITY__SESSION_BACKEND=redis
   export NIGHTSCAN_REDIS__URL=redis://localhost:6379/1
   ```

3. Restart application (existing sessions will be lost)

### From Filesystem to Redis

1. Set up Redis as above
2. Optionally migrate existing sessions:
   ```python
   # Migration script
   import json
   from pathlib import Path
   import redis
   import pickle
   
   # Connect to Redis
   r = redis.Redis.from_url("redis://localhost:6379/1")
   
   # Read filesystem sessions
   session_dir = Path("/tmp/nightscan/sessions")
   for session_file in session_dir.glob("*.json"):
       with open(session_file) as f:
           data = json.load(f)
       
       # Store in Redis
       session_id = session_file.stem
       r.setex(
           f"session:{session_id}",
           3600,  # TTL
           pickle.dumps(data)
       )
   ```

3. Update configuration and restart

## Performance Considerations

- **Redis**: ~0.5ms per session operation
- **Filesystem**: ~2-5ms per operation (depends on disk)
- **Memory**: ~0.1ms per operation (but not distributed)

For high-traffic applications, Redis is strongly recommended.