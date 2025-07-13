# NightScan CSP Security Migration Guide

## Overview

This guide explains how to migrate from unsafe inline scripts to a secure Content Security Policy (CSP) implementation using nonces.

## Current Security Issue

**Problem**: The current CSP configuration allows `unsafe-inline` for scripts and styles, which significantly weakens XSS protection.

```python
# Current vulnerable configuration
"script-src": "'self' 'unsafe-inline'",  # XSS risk!
"style-src": "'self' 'unsafe-inline'",   # XSS risk!
```

## Solution Implementation

### 1. CSP Nonce Manager

Created `security/csp_nonce.py` which provides:
- Automatic nonce generation for each request
- CSP header management with dynamic nonces
- Environment-specific policies (dev vs production)
- Helper functions for templates

### 2. External JavaScript and CSS

Moved all inline scripts and styles to external files:
- `/web/static/js/main.js` - All JavaScript functionality
- `/web/static/css/main.css` - All CSS styles

### 3. Secure HTML Template

Created `index_secure.html` without any inline scripts or styles.

## Migration Steps

### Step 1: Update Flask Application

1. **Import the CSP manager**:
```python
from security.csp_nonce import CSPNonceManager
```

2. **Remove old CSP configuration** (lines 67-80 in web/app.py)

3. **Initialize CSP manager** after creating Flask app:
```python
app = Flask(__name__)
# ... other configurations ...
csp_manager = CSPNonceManager(app)
```

4. **Update Talisman** to not handle CSP:
```python
Talisman(app, 
    force_https=True, 
    frame_options="DENY",
    content_security_policy=None,  # Let CSP manager handle this
    # ... other settings ...
)
```

5. **Add decorator** to routes that render templates:
```python
from security.csp_nonce import csp_nonce_required

@app.route("/")
@login_required
@csp_nonce_required  # Add this
def index():
    return render_template("index_secure.html", ...)
```

### Step 2: Update Templates

1. **Replace** `index.html` with `index_secure.html`
2. **For any remaining inline scripts** that absolutely cannot be moved:
```html
<!-- Use nonce attribute -->
<script nonce="{{ csp_nonce }}">
    // Your critical inline code here
</script>
```

### Step 3: Update Nginx Configuration

1. **Replace** `nginx.production.conf` with `nginx.secure.conf`
2. **Remove** CSP header from Nginx (let Flask handle it)
3. **Ensure** static files are served with security headers

### Step 4: Testing

1. **Check browser console** for CSP violations
2. **Verify** all functionality works without inline scripts
3. **Test** WebSocket connections
4. **Monitor** for any blocked resources

## Benefits

1. **Strong XSS Protection**: Inline scripts are blocked by default
2. **Dynamic Nonces**: Each request gets a unique nonce
3. **Environment Awareness**: Stricter policies in production
4. **Future Proof**: Easy to add new external resources

## Rollback Plan

If issues arise:
1. Revert to original `index.html`
2. Re-enable `unsafe-inline` in CSP configuration
3. Keep external JS/CSS files for gradual migration

## Security Improvements

### Before:
- CSP allows any inline script (XSS vulnerable)
- No protection against injected scripts
- Mixed inline and external scripts

### After:
- Only scripts with valid nonce can execute
- Strong protection against XSS attacks
- Clean separation of concerns
- Compliant with security best practices

## Additional Considerations

### For Development

The CSP manager automatically relaxes policies in development:
- Allows `unsafe-eval` for hot reloading
- Still requires nonces for inline scripts

### For Production

Strict CSP is enforced:
- No `unsafe-inline` or `unsafe-eval`
- Only nonced or external scripts allowed
- Full XSS protection enabled

## Monitoring

1. **CSP Report-Only Mode** (optional testing phase):
```python
# In csp_nonce.py, add report-only mode
response.headers['Content-Security-Policy-Report-Only'] = csp_policy
```

2. **CSP Violation Reports**:
```python
# Add to CSP policy
"report-uri": "/api/csp-report"
```

3. **Log violations** for analysis:
```python
@app.route('/api/csp-report', methods=['POST'])
def csp_report():
    # Log CSP violations
    logger.warning(f"CSP Violation: {request.get_json()}")
    return '', 204
```

## Conclusion

This migration eliminates the XSS vulnerability while maintaining full functionality. The implementation is production-ready and follows security best practices.