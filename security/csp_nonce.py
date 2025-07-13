"""
Content Security Policy Nonce Management for NightScan.

This module provides secure CSP nonce generation and management
to allow specific inline scripts while maintaining security.
"""

import secrets
import base64
from flask import g, current_app
from functools import wraps
from typing import Dict, Optional


class CSPNonceManager:
    """Manages CSP nonces for secure inline script execution."""
    
    def __init__(self, app=None):
        self.app = app
        if app is not None:
            self.init_app(app)
    
    def init_app(self, app):
        """Initialize CSP nonce management with Flask app."""
        self.app = app
        
        # Register before_request handler to generate nonce
        @app.before_request
        def generate_csp_nonce():
            """Generate a unique nonce for each request."""
            # Generate a cryptographically secure random nonce
            nonce_bytes = secrets.token_bytes(16)
            g.csp_nonce = base64.b64encode(nonce_bytes).decode('utf-8')
        
        # Add template global for easy access in templates
        @app.context_processor
        def inject_csp_nonce():
            """Inject CSP nonce into all templates."""
            return dict(csp_nonce=getattr(g, 'csp_nonce', ''))
        
        # Register after_request handler to add CSP header
        @app.after_request
        def add_csp_header(response):
            """Add CSP header with nonce to response."""
            # Skip if CSP is already set by another handler
            if 'Content-Security-Policy' in response.headers:
                return response
            
            # Get current nonce
            nonce = getattr(g, 'csp_nonce', None)
            if not nonce:
                return response
            
            # Build CSP policy
            csp_policy = self.build_csp_policy(nonce, app.config.get('ENV', 'production'))
            response.headers['Content-Security-Policy'] = csp_policy
            
            return response
    
    def build_csp_policy(self, nonce: str, environment: str = 'production') -> str:
        """
        Build CSP policy string with nonce.
        
        Args:
            nonce: The CSP nonce for this request
            environment: The application environment
            
        Returns:
            CSP policy string
        """
        # Base CSP directives
        directives = {
            "default-src": "'self'",
            "img-src": "'self' data: https:",
            "font-src": "'self' https://cdnjs.cloudflare.com",
            "connect-src": "'self' wss: ws:",
            "media-src": "'self'",
            "object-src": "'none'",
            "base-uri": "'self'",
            "form-action": "'self'",
            "frame-ancestors": "'none'",
            "upgrade-insecure-requests": "",
            "report-uri": "/api/csp-report"
        }
        
        # Script and style directives with nonce
        if environment == 'development':
            # More permissive in development for easier debugging
            directives["script-src"] = f"'self' 'nonce-{nonce}' 'unsafe-eval' https://cdnjs.cloudflare.com"
            directives["style-src"] = f"'self' 'nonce-{nonce}' https://cdnjs.cloudflare.com"
        else:
            # Strict in production
            directives["script-src"] = f"'self' 'nonce-{nonce}' https://cdnjs.cloudflare.com"
            directives["style-src"] = f"'self' 'nonce-{nonce}' https://cdnjs.cloudflare.com"
        
        # Build policy string
        policy_parts = []
        for directive, value in directives.items():
            if value:
                policy_parts.append(f"{directive} {value}")
            else:
                policy_parts.append(directive)
        
        return "; ".join(policy_parts)
    
    def generate_hash(self, content: str) -> str:
        """
        Generate SHA-256 hash for inline script/style content.
        
        Args:
            content: The script or style content
            
        Returns:
            Base64-encoded SHA-256 hash
        """
        import hashlib
        hash_bytes = hashlib.sha256(content.encode('utf-8')).digest()
        return base64.b64encode(hash_bytes).decode('utf-8')


def csp_nonce_required(f):
    """Decorator to ensure CSP nonce is available for a route."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not hasattr(g, 'csp_nonce'):
            # Generate nonce if not already present
            nonce_bytes = secrets.token_bytes(16)
            g.csp_nonce = base64.b64encode(nonce_bytes).decode('utf-8')
        return f(*args, **kwargs)
    return decorated_function


def get_script_hashes() -> Dict[str, str]:
    """
    Get pre-calculated hashes for allowed inline scripts.
    
    This is useful for small, static inline scripts that can't be moved
    to external files (e.g., critical performance monitoring).
    
    Returns:
        Dictionary of script name to SHA-256 hash
    """
    # Add hashes for any critical inline scripts here
    # Example: Google Analytics, critical error handling, etc.
    return {
        # "google_analytics": "sha256-xxxxx",
        # "error_handler": "sha256-yyyyy",
    }


def create_csp_compatible_script(script_content: str, nonce: Optional[str] = None) -> str:
    """
    Create a CSP-compatible script tag.
    
    Args:
        script_content: The JavaScript content
        nonce: Optional nonce (will use g.csp_nonce if not provided)
        
    Returns:
        HTML script tag with nonce attribute
    """
    if nonce is None:
        nonce = getattr(g, 'csp_nonce', '')
    
    if not nonce:
        raise ValueError("No CSP nonce available")
    
    return f'<script nonce="{nonce}">{script_content}</script>'


def create_csp_compatible_style(style_content: str, nonce: Optional[str] = None) -> str:
    """
    Create a CSP-compatible style tag.
    
    Args:
        style_content: The CSS content
        nonce: Optional nonce (will use g.csp_nonce if not provided)
        
    Returns:
        HTML style tag with nonce attribute
    """
    if nonce is None:
        nonce = getattr(g, 'csp_nonce', '')
    
    if not nonce:
        raise ValueError("No CSP nonce available")
    
    return f'<style nonce="{nonce}">{style_content}</style>'


# Example usage in templates:
# <script nonce="{{ csp_nonce }}">
#     // Your JavaScript code here
# </script>
#
# <style nonce="{{ csp_nonce }}">
#     /* Your CSS code here */
# </style>