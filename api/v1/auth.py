"""API v1 - Authentication endpoints with versioning."""

import logging

from flask import Blueprint, jsonify

from api_versioning.decorators import api_version, version_required
from api_versioning.registry import register_endpoint
from auth.auth_routes import login as original_login
from auth.auth_routes import logout as original_logout
from auth.auth_routes import refresh_token as original_refresh
from auth.auth_routes import register as original_register
from auth.auth_routes import verify_token as original_verify

logger = logging.getLogger(__name__)


def create_auth_blueprint() -> Blueprint:
    """Create auth blueprint for API v1."""
    auth_bp = Blueprint("auth_v1", __name__)

    # Register RESTful endpoints in the registry
    endpoints = [
        {
            "path": "/api/v1/sessions",
            "methods": ["POST"],
            "description": "Create new user session (login)",
            "tags": ["auth", "sessions"],
        },
        {
            "path": "/api/v1/sessions",
            "methods": ["DELETE"],
            "description": "Destroy user session (logout)",
            "tags": ["auth", "sessions"],
        },
        {
            "path": "/api/v1/users",
            "methods": ["POST"],
            "description": "Create new user account (register)",
            "tags": ["auth", "users"],
        },
        {
            "path": "/api/v1/tokens",
            "methods": ["POST"],
            "description": "Create new access token (refresh)",
            "tags": ["auth", "tokens"],
        },
        {
            "path": "/api/v1/tokens/current",
            "methods": ["GET"],
            "description": "Get current token information (verify)",
            "tags": ["auth", "tokens"],
        },
        # Legacy endpoints for backward compatibility
        {
            "path": "/api/v1/auth/login",
            "methods": ["POST"],
            "description": "[DEPRECATED] Use POST /api/v1/sessions instead",
            "tags": ["auth", "deprecated"],
        },
        {
            "path": "/api/v1/auth/logout",
            "methods": ["POST"],
            "description": "[DEPRECATED] Use DELETE /api/v1/sessions instead",
            "tags": ["auth", "deprecated"],
        },
        {
            "path": "/api/v1/auth/register",
            "methods": ["POST"],
            "description": "[DEPRECATED] Use POST /api/v1/users instead",
            "tags": ["auth", "deprecated"],
        },
        {
            "path": "/api/v1/auth/refresh",
            "methods": ["POST"],
            "description": "[DEPRECATED] Use POST /api/v1/tokens instead",
            "tags": ["auth", "deprecated"],
        },
        {
            "path": "/api/v1/auth/verify",
            "methods": ["GET"],
            "description": "[DEPRECATED] Use GET /api/v1/tokens/current instead",
            "tags": ["auth", "deprecated"],
        },
    ]

    for endpoint in endpoints:
        register_endpoint(
            path=endpoint["path"],
            methods=endpoint["methods"],
            version="v1",
            description=endpoint["description"],
            tags=endpoint["tags"],
            requires_auth=endpoint["path"] not in ["/api/v1/sessions", "/api/v1/users", "/api/v1/auth/login", "/api/v1/auth/register"],
        )

    # RESTful routes with proper HTTP methods

    @auth_bp.route("/sessions", methods=["POST"])
    @api_version("v1", description="Create user session", tags=["auth", "sessions"])
    def create_session():
        """Create new user session (RESTful login)."""
        # Use the original login logic
        return original_login()

    @auth_bp.route("/sessions", methods=["DELETE"])
    @api_version("v1", description="Destroy user session", tags=["auth", "sessions"])
    @version_required(min_version="v1")
    def destroy_session():
        """Destroy user session (RESTful logout)."""
        # Use the original logout logic
        return original_logout()

    @auth_bp.route("/users", methods=["POST"])
    @api_version("v1", description="Create user account", tags=["auth", "users"])
    def create_user():
        """Create new user account (RESTful register)."""
        # Use the original register logic
        return original_register()

    @auth_bp.route("/tokens", methods=["POST"])
    @api_version("v1", description="Create access token", tags=["auth", "tokens"])
    def create_token():
        """Create new access token (RESTful refresh)."""
        # Use the original refresh logic
        return original_refresh()

    @auth_bp.route("/tokens/current", methods=["GET"])
    @api_version("v1", description="Get current token info", tags=["auth", "tokens"])
    def get_current_token():
        """Get current token information (RESTful verify)."""
        # Use the original verify logic
        return original_verify()

    # Legacy routes for backward compatibility (deprecated)

    @auth_bp.route("/login", methods=["POST"])
    @api_version("v1", description="[DEPRECATED] Use POST /sessions", tags=["auth", "deprecated"])
    def login_v1():
        """Login endpoint for API v1 (deprecated)."""
        # Add deprecation header and use the original login logic
        response = original_login()
        if hasattr(response, 'headers'):
            response.headers['X-API-Deprecation-Warning'] = 'This endpoint is deprecated. Use POST /api/v1/sessions instead.'
            response.headers['X-API-Deprecated-Endpoint'] = '/api/v1/auth/login'
            response.headers['X-API-Replacement-Endpoint'] = '/api/v1/sessions'
        return response

    @auth_bp.route("/register", methods=["POST"])
    @api_version("v1", description="[DEPRECATED] Use POST /users", tags=["auth", "deprecated"])
    def register_v1():
        """Register endpoint for API v1 (deprecated)."""
        response = original_register()
        if hasattr(response, 'headers'):
            response.headers['X-API-Deprecation-Warning'] = 'This endpoint is deprecated. Use POST /api/v1/users instead.'
            response.headers['X-API-Deprecated-Endpoint'] = '/api/v1/auth/register'
            response.headers['X-API-Replacement-Endpoint'] = '/api/v1/users'
        return response

    @auth_bp.route("/refresh", methods=["POST"])
    @api_version("v1", description="[DEPRECATED] Use POST /tokens", tags=["auth", "deprecated"])
    def refresh_v1():
        """Refresh token endpoint for API v1 (deprecated)."""
        response = original_refresh()
        if hasattr(response, 'headers'):
            response.headers['X-API-Deprecation-Warning'] = 'This endpoint is deprecated. Use POST /api/v1/tokens instead.'
            response.headers['X-API-Deprecated-Endpoint'] = '/api/v1/auth/refresh'
            response.headers['X-API-Replacement-Endpoint'] = '/api/v1/tokens'
        return response

    @auth_bp.route("/logout", methods=["POST"])
    @api_version("v1", description="[DEPRECATED] Use DELETE /sessions", tags=["auth", "deprecated"])
    @version_required(min_version="v1")
    def logout_v1():
        """Logout endpoint for API v1 (deprecated)."""
        response = original_logout()
        if hasattr(response, 'headers'):
            response.headers['X-API-Deprecation-Warning'] = 'This endpoint is deprecated. Use DELETE /api/v1/sessions instead.'
            response.headers['X-API-Deprecated-Endpoint'] = '/api/v1/auth/logout'
            response.headers['X-API-Replacement-Endpoint'] = '/api/v1/sessions'
        return response

    @auth_bp.route("/verify", methods=["GET"])
    @api_version("v1", description="[DEPRECATED] Use GET /tokens/current", tags=["auth", "deprecated"])
    def verify_v1():
        """Verify token endpoint for API v1 (deprecated)."""
        response = original_verify()
        if hasattr(response, 'headers'):
            response.headers['X-API-Deprecation-Warning'] = 'This endpoint is deprecated. Use GET /api/v1/tokens/current instead.'
            response.headers['X-API-Deprecated-Endpoint'] = '/api/v1/auth/verify'
            response.headers['X-API-Replacement-Endpoint'] = '/api/v1/tokens/current'
        return response

    # Additional v1-specific auth endpoints

    @auth_bp.route("/change-password", methods=["POST"])
    @api_version("v1", description="Change user password", tags=["auth", "password"])
    @version_required(min_version="v1")
    def change_password_v1():
        """Change password endpoint for API v1."""
        # This would typically be implemented with proper auth checks
        return jsonify({"error": "Not implemented", "message": "Password change endpoint coming soon"}), 501

    @auth_bp.route("/sessions", methods=["GET"])
    @api_version("v1", description="Get active sessions", tags=["auth", "sessions"])
    @version_required(min_version="v1", features=["jwt_auth"])
    def get_sessions_v1():
        """Get user sessions endpoint for API v1."""
        # Placeholder for session management
        return jsonify({"sessions": [], "message": "Session management coming soon"}), 200

    return auth_bp
