# NightScan Rate Limiting and Authentication Security Guide

## Overview

NightScan implements a **multi-layered rate limiting system** to protect against brute force attacks, DDoS attempts, and API abuse. This guide documents all rate limiting mechanisms and authentication security features.

## Rate Limiting Architecture

### 1. Flask-Limiter (Application Level)

**Configuration:** `config.py`
```python
@dataclass
class RateLimitConfig:
    enabled: bool = True
    default_limit: str = "1000 per day"
    login_limit: str = "5 per minute"
    prediction_limit: str = "10 per minute"
    upload_limit: str = "10 per minute"
```

**Implementation:** Applied via decorators in `web/app.py`
```python
@app.route("/login", methods=["GET", "POST"])
@limiter.limit(config.rate_limit.login_limit)  # 5 per minute
def login():
    # ...

@app.route("/register", methods=["GET", "POST"])
@limiter.limit(config.rate_limit.login_limit)  # 5 per minute
def register():
    # ...
```

### 2. Custom IP Lockout System

**Location:** `web/app.py`

**Features:**
- Tracks failed login attempts per IP
- Persistent storage in `failed_logins.json`
- Automatic lockout after 5 failed attempts
- 30-minute lockout duration
- Automatic cleanup of old entries

**Configuration:**
```python
LOCKOUT_THRESHOLD = 5  # Failed attempts before lockout
LOCKOUT_WINDOW = 1800  # 30 minutes
CLEANUP_INTERVAL = 3600  # Clean old entries hourly
```

### 3. Advanced RateLimiter Module

**Location:** `security/rate_limiting.py`

**Endpoint-Specific Limits:**
```python
'auth.login': (10, 300),        # 10 requests per 5 minutes
'auth.register': (5, 300),      # 5 requests per 5 minutes  
'auth.reset_password': (3, 300), # 3 requests per 5 minutes
'api.predict': (100, 60),       # 100 requests per minute
'api.upload': (50, 60),         # 50 uploads per minute
```

**Advanced Features:**
- Burst limit protection (2x normal limit)
- Automatic IP blocking after 10 violations
- 1-hour block duration for repeated violators
- HTTP rate limit headers in responses
- Per-endpoint customization

**Response Headers:**
```
X-RateLimit-Limit: 10
X-RateLimit-Remaining: 7
X-RateLimit-Reset: 1699123456
Retry-After: 300
```

### 4. SecureAuth Rate Limiting

**Location:** `secure_auth.py`

**User-Level Protection:**
- Max 5 attempts per user identifier
- 15-minute lockout period
- Separate from IP-based limiting
- Protects against distributed attacks

## Authentication Security Features

### Password Requirements

**Validation Pattern:**
```python
PASSWORD_RE = re.compile(r"^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[\W_]).{10,}$")
```

**Requirements:**
- Minimum 10 characters (configurable)
- At least one uppercase letter
- At least one lowercase letter
- At least one digit
- At least one special character

### Additional Security Measures

1. **CAPTCHA Protection**
   - Mathematical CAPTCHA on login page
   - Prevents automated attacks
   - Session-based validation

2. **Username Validation**
   - Alphanumeric + underscore only
   - 3-30 characters
   - Prevents injection attacks

3. **Password Hashing**
   - bcrypt with 12 rounds
   - Secure salt generation
   - Timing attack resistant

4. **Session Security**
   - Secure session cookies
   - CSRF protection enabled
   - HTTPOnly and SameSite flags

5. **Security Headers** (via Talisman)
   - Force HTTPS
   - X-Frame-Options: DENY
   - Strict CSP policy
   - Referrer policy

## API Rate Limiting

### REST API Endpoints

All API endpoints inherit the default limits plus specific overrides:

```python
# Default for all API calls
@limiter.limit("1000 per day")

# Specific endpoints
@limiter.limit("10 per minute")  # Prediction API
@limiter.limit("50 per minute")  # Upload API
```

### WebSocket Rate Limiting

WebSocket connections have separate rate limiting:
- Connection limit per IP
- Message rate limiting
- Automatic disconnection on abuse

## Password Reset Security

**New Endpoints:** (Implemented in `web/password_reset.py`)

1. **`/forgot-password`** (GET/POST)
   - Rate limit: 3 per 5 minutes
   - Email enumeration protection
   - Secure token generation

2. **`/reset-password/<token>`** (GET/POST)
   - Rate limit: 5 per 10 minutes
   - Time-limited tokens (1 hour)
   - Single-use tokens
   - Strong password validation

3. **API Endpoints:**
   - `/api/password-reset/request` - 3 per 5 minutes
   - `/api/password-reset/verify` - 10 per 10 minutes
   - `/api/password-reset/reset` - 5 per 10 minutes

## Configuration

### Environment Variables

```bash
# Rate limiting
RATE_LIMIT_ENABLED=true
RATE_LIMIT_DEFAULT="1000 per day"
RATE_LIMIT_LOGIN="5 per minute"

# Security
LOCKOUT_THRESHOLD=5
LOCKOUT_WINDOW=1800
PASSWORD_MIN_LENGTH=10

# Email (for password reset)
MAIL_SERVER=smtp.gmail.com
MAIL_PORT=587
MAIL_USE_TLS=true
MAIL_USERNAME=your-email@gmail.com
MAIL_PASSWORD=your-app-password
```

### Disabling Rate Limiting (Development Only)

```python
# In config.py
rate_limit = RateLimitConfig(enabled=False)
```

## Monitoring and Metrics

### Rate Limit Statistics

Access rate limiter stats:
```python
from security.rate_limiting import rate_limiter
stats = rate_limiter.get_stats()
```

Returns:
```json
{
    "enabled": true,
    "total_tracked_keys": 150,
    "blocked_ips": 3,
    "endpoint_stats": {
        "auth.login": 45,
        "auth.register": 12
    }
}
```

### Failed Login Monitoring

Monitor failed logins:
```python
# Check specific IP
failed_attempts = get_failed_login_count('192.168.1.1')

# View all blocked IPs
blocked_ips = get_blocked_ips()
```

## Best Practices

1. **Never disable rate limiting in production**
2. **Monitor rate limit metrics regularly**
3. **Adjust limits based on legitimate usage patterns**
4. **Use different limits for different user tiers**
5. **Implement gradual back-off for repeated violations**
6. **Log all security events for audit trails**

## Troubleshooting

### User Locked Out

```python
# Manually unblock IP
from security.rate_limiting import rate_limiter
rate_limiter.unblock_ip('192.168.1.1')

# Clear failed attempts
clear_failed_logins('192.168.1.1')
```

### Testing Rate Limits

```bash
# Test login rate limit
for i in {1..10}; do
    curl -X POST http://localhost:8000/login \
         -d "username=test&password=test&captcha=wrong"
    sleep 1
done
```

### Common Issues

1. **"Too many requests" errors**
   - Check if IP is blocked
   - Verify rate limit configuration
   - Check for legitimate high usage

2. **Rate limits not working**
   - Ensure `RATE_LIMIT_ENABLED=true`
   - Check Redis connection (if using Redis backend)
   - Verify decorator placement

3. **Email not sending (password reset)**
   - Check SMTP configuration
   - Verify email credentials
   - Check spam folder

## Security Recommendations

1. **Use Redis for production** - Better performance and persistence
2. **Implement IP whitelisting** for trusted sources
3. **Add geographic rate limiting** for suspicious regions
4. **Enable comprehensive logging** of all auth events
5. **Regular security audits** of rate limit effectiveness
6. **Consider CAPTCHA escalation** after multiple failures
7. **Implement account lockout** in addition to IP lockout

## Conclusion

NightScan's rate limiting is **NOT weak** - it implements multiple layers of protection:
- ✅ Application-level rate limiting (Flask-Limiter)
- ✅ IP-based lockout system
- ✅ Advanced per-endpoint limits
- ✅ User-level rate limiting
- ✅ CAPTCHA protection
- ✅ Comprehensive security headers

The only missing component was the password reset endpoint, which has now been implemented with appropriate rate limiting.