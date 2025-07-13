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

    # Register endpoints in the registry
    endpoints = [
        {
            "path": "/api/v1/auth/login",
            "methods": ["POST"],
            "description": "Authenticate user and return JWT tokens",
            "tags": ["auth", "authentication"],
        },
        {
            "path": "/api/v1/auth/register",
            "methods": ["POST"],
            "description": "Register new user and return JWT tokens",
            "tags": ["auth", "registration"],
        },
        {
            "path": "/api/v1/auth/refresh",
            "methods": ["POST"],
            "description": "Refresh access token using refresh token",
            "tags": ["auth", "token"],
        },
        {
            "path": "/api/v1/auth/logout",
            "methods": ["POST"],
            "description": "Logout user and revoke tokens",
            "tags": ["auth", "logout"],
        },
        {
            "path": "/api/v1/auth/verify",
            "methods": ["GET"],
            "description": "Verify if current token is valid",
            "tags": ["auth", "token"],
        },
    ]

    for endpoint in endpoints:
        register_endpoint(
            path=endpoint["path"],
            methods=endpoint["methods"],
            version="v1",
            description=endpoint["description"],
            tags=endpoint["tags"],
            requires_auth=endpoint["path"] not in ["/api/v1/auth/login", "/api/v1/auth/register"],
        )

    # Routes with versioning decorators

    @auth_bp.route("/login", methods=["POST"])
    @api_version("v1", description="User login", tags=["auth"])
    def login_v1():
        """Login endpoint for API v1."""
        # Use the original login logic
        return original_login()

    @auth_bp.route("/register", methods=["POST"])
    @api_version("v1", description="User registration", tags=["auth"])
    def register_v1():
        """Register endpoint for API v1."""
        # Use the original register logic
        return original_register()

    @auth_bp.route("/refresh", methods=["POST"])
    @api_version("v1", description="Token refresh", tags=["auth"])
    def refresh_v1():
        """Refresh token endpoint for API v1."""
        # Use the original refresh logic
        return original_refresh()

    @auth_bp.route("/logout", methods=["POST"])
    @api_version("v1", description="User logout", tags=["auth"])
    @version_required(min_version="v1")
    def logout_v1():
        """Logout endpoint for API v1."""
        # Use the original logout logic
        return original_logout()

    @auth_bp.route("/verify", methods=["GET"])
    @api_version("v1", description="Token verification", tags=["auth"])
    def verify_v1():
        """Verify token endpoint for API v1."""
        # Use the original verify logic
        return original_verify()

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
