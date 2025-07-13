"""API Versioning Middleware for backward compatibility."""

import logging
from datetime import datetime
from functools import wraps

from flask import Flask, Response, g, jsonify, request

from .config import get_api_config

logger = logging.getLogger(__name__)


class APIVersioningMiddleware:
    """Middleware to handle API versioning, deprecation, and backward compatibility."""

    def __init__(self, app: Flask = None):
        self.app = app
        self.api_config = get_api_config()
        self.deprecation_warnings = {}

        if app:
            self.init_app(app)

    def init_app(self, app: Flask):
        """Initialize the middleware with a Flask app."""
        self.app = app

        # Register before_request handler
        app.before_request(self.handle_version_routing)

        # Register after_request handler
        app.after_request(self.add_version_headers)

        # Add error handler for version-related errors
        app.register_error_handler(APIVersionError, self.handle_version_error)

        logger.info("API Versioning Middleware initialized")

    def handle_version_routing(self):
        """Handle incoming requests for versioning and legacy route mapping."""
        path = request.path

        # Extract version from path
        version = self.api_config.get_version_from_path(path)

        # Store version in g for later use
        g.api_version = version

        # Check if this is a legacy route that needs mapping
        if not version:
            mapped_path = self.api_config.get_legacy_mapping(path)
            if mapped_path:
                # Log the usage of deprecated endpoint
                self._log_deprecated_usage(path, mapped_path)

                # Store deprecation info for response headers
                g.deprecation_warning = {"old_path": path, "new_path": mapped_path, "deprecated": True}

                # Redirect internally to the new path
                # Note: This maintains the original request method and data
                request.environ["PATH_INFO"] = mapped_path
                request.path = mapped_path

                # Update version after mapping
                g.api_version = self.api_config.get_version_from_path(mapped_path)

        # Check if version is deprecated
        if version and self.api_config.is_version_deprecated(version):
            deprecation_info = self.api_config.get_deprecation_info(version)
            g.version_deprecation = deprecation_info

    def add_version_headers(self, response: Response) -> Response:
        """Add version-related headers to the response."""
        # Add current API version header
        if hasattr(g, "api_version") and g.api_version:
            response.headers["X-API-Version"] = g.api_version

        # Add supported versions header
        active_versions = self.api_config.get_active_versions()
        response.headers["X-API-Versions-Supported"] = ", ".join(active_versions)

        # Add deprecation warning for legacy routes
        if hasattr(g, "deprecation_warning"):
            warning = g.deprecation_warning
            response.headers["X-API-Deprecation-Warning"] = (
                f"This endpoint is deprecated. Use {warning['new_path']} instead."
            )
            response.headers["X-API-Deprecated-Endpoint"] = warning["old_path"]
            response.headers["X-API-Replacement-Endpoint"] = warning["new_path"]

        # Add version deprecation warning
        if hasattr(g, "version_deprecation"):
            info = g.version_deprecation
            warning_msg = f"API version {g.api_version} is deprecated."

            if info.get("sunset_date"):
                warning_msg += f" It will be removed on {info['sunset_date']}."

            if info.get("replacement_version"):
                warning_msg += f" Please migrate to {info['replacement_version']}."

            response.headers["X-API-Version-Deprecation"] = warning_msg

            if info.get("migration_guide"):
                response.headers["X-API-Migration-Guide"] = info["migration_guide"]

        # Add rate limit headers based on version
        if hasattr(g, "api_version") and g.api_version:
            if self.api_config.is_feature_enabled(g.api_version, "rate_limiting"):
                # These would be set by rate limiting middleware
                # Just placeholder for version-specific limits
                if not response.headers.get("X-RateLimit-Limit"):
                    limits = self._get_version_rate_limits(g.api_version)
                    response.headers["X-RateLimit-Limit"] = str(limits.get("limit", 100))
                    response.headers["X-RateLimit-Window"] = str(limits.get("window", 3600))

        return response

    def handle_version_error(self, error):
        """Handle version-related errors."""
        return (
            jsonify(
                {
                    "error": str(error),
                    "type": "APIVersionError",
                    "supported_versions": self.api_config.get_active_versions(),
                    "latest_stable": self.api_config.get_latest_stable_version(),
                }
            ),
            error.status_code,
        )

    def _log_deprecated_usage(self, old_path: str, new_path: str):
        """Log usage of deprecated endpoints."""
        # Track usage counts
        if old_path not in self.deprecation_warnings:
            self.deprecation_warnings[old_path] = {
                "count": 0,
                "first_seen": datetime.utcnow(),
                "last_seen": None,
                "new_path": new_path,
            }

        self.deprecation_warnings[old_path]["count"] += 1
        self.deprecation_warnings[old_path]["last_seen"] = datetime.utcnow()

        # Log warning
        logger.warning(
            f"Deprecated API endpoint used: {old_path} -> {new_path} "
            f"(count: {self.deprecation_warnings[old_path]['count']})"
        )

        # Log client info for tracking
        logger.info(
            f"Deprecated endpoint client info - "
            f"IP: {request.remote_addr}, "
            f"User-Agent: {request.headers.get('User-Agent', 'Unknown')}, "
            f"Auth: {bool(request.headers.get('Authorization', ''))}"
        )

    def _get_version_rate_limits(self, version: str) -> dict:
        """Get rate limits for a specific API version."""
        # Version-specific rate limits
        limits = {
            "v1": {"limit": 100, "window": 3600},  # 100 requests per hour
            "v2": {"limit": 200, "window": 3600},  # 200 requests per hour
        }
        return limits.get(version, {"limit": 50, "window": 3600})

    def get_deprecation_report(self) -> dict:
        """Get a report of deprecated endpoint usage."""
        return {
            "deprecated_endpoints": self.deprecation_warnings,
            "total_deprecated_calls": sum(info["count"] for info in self.deprecation_warnings.values()),
            "unique_deprecated_endpoints": len(self.deprecation_warnings),
        }


class APIVersionError(Exception):
    """Custom exception for API version errors."""

    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.status_code = status_code


def add_versioning_middleware(app: Flask) -> APIVersioningMiddleware:
    """Convenience function to add versioning middleware to a Flask app."""
    middleware = APIVersioningMiddleware(app)
    return middleware


def version_required(min_version: str = None, max_version: str = None):
    """Decorator to enforce version requirements on routes."""

    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if not hasattr(g, "api_version") or not g.api_version:
                raise APIVersionError("API version not specified in request", status_code=400)

            version = g.api_version
            api_config = get_api_config()

            # Check minimum version
            if min_version:
                if _compare_versions(version, min_version) < 0:
                    raise APIVersionError(
                        f"This endpoint requires API version {min_version} or higher", status_code=400
                    )

            # Check maximum version
            if max_version:
                if _compare_versions(version, max_version) > 0:
                    raise APIVersionError(f"This endpoint is not available in API version {version}", status_code=400)

            # Check if version is deprecated
            if api_config.is_version_deprecated(version):
                # Still allow the request but with warning
                logger.warning(f"Request to deprecated API version: {version}")

            return f(*args, **kwargs)

        return decorated_function

    return decorator


def _compare_versions(v1: str, v2: str) -> int:
    """Compare two version strings (e.g., 'v1' and 'v2')."""
    # Simple comparison for v1, v2, etc.
    v1_num = int(v1[1:]) if v1.startswith("v") else 0
    v2_num = int(v2[1:]) if v2.startswith("v") else 0

    if v1_num < v2_num:
        return -1
    elif v1_num > v2_num:
        return 1
    else:
        return 0
