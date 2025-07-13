"""
CSP Update for web/app.py

This file contains the modifications needed to implement secure CSP with nonces.
Replace the existing CSP configuration in web/app.py with this code.
"""

# Add this import at the top of web/app.py
from security.csp_nonce import CSPNonceManager

# Replace the existing CSP configuration (lines 67-80) with:
# Remove the old CSP dictionary and Talisman configuration

# After creating the Flask app, add:
csp_manager = CSPNonceManager(app)

# Update Talisman configuration to not include CSP (we'll handle it separately)
Talisman(app, 
    force_https=True, 
    frame_options="DENY",
    content_security_policy=None,  # We'll handle CSP with nonces separately
    referrer_policy="strict-origin-when-cross-origin",
    feature_policy={
        "microphone": "'self'",
        "camera": "'self'",
        "geolocation": "'self'",
        "payment": "'none'",
        "usb": "'none'"
    }
)

# For any route that needs to render templates with inline scripts, 
# you can use the @csp_nonce_required decorator:
from security.csp_nonce import csp_nonce_required

# Example update for the index route:
@app.route("/", methods=["GET", "POST"])
@login_required
@limiter.limit(config.rate_limit.upload_limit if limiter else None)
@track_request_metrics
@csp_nonce_required  # Add this decorator
def index():
    # ... existing code ...
    
    # When rendering templates, the csp_nonce will be automatically available
    return render_template("index_secure.html", 
                         predictions=predictions, 
                         remaining_bytes=remaining_bytes)