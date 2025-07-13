"""Decorators for API versioning functionality."""

import logging
from datetime import datetime
from functools import wraps
from typing import Callable, List, Optional

from flask import g, jsonify, request

from .config import get_api_config
from .registry import get_api_registry, register_endpoint

logger = logging.getLogger(__name__)


def api_version(version: str, **endpoint_kwargs):
    """Decorator to mark a route as belonging to a specific API version.

    Args:
        version: The API version (e.g., 'v1', 'v2')
        **endpoint_kwargs: Additional endpoint configuration
            - description: Endpoint description
            - tags: List of tags for categorization
            - requires_auth: Whether authentication is required
            - deprecated: Whether the endpoint is deprecated
            - replacement: Replacement endpoint path if deprecated

    """

    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            # Set version in context
            g.api_version = version

            # Check if version is supported
            api_config = get_api_config()
            if version not in api_config.versions:
                return (
                    jsonify(
                        {
                            "error": f"API version {version} is not supported",
                            "supported_versions": api_config.get_active_versions(),
                        }
                    ),
                    400,
                )

            # Check if version is deprecated
            if api_config.is_version_deprecated(version):
                deprecation_info = api_config.get_deprecation_info(version)
                g.version_deprecation = deprecation_info

            # Execute the route handler
            return f(*args, **kwargs)

        # Register endpoint in registry
        if hasattr(f, "_endpoint_path"):
            registry = get_api_registry()
            endpoint = register_endpoint(
                path=f._endpoint_path,
                methods=f._endpoint_methods or ["GET"],
                version=version,
                handler=f,
                **endpoint_kwargs,
            )

            # Store endpoint info on function for introspection
            f._api_endpoint = endpoint

        decorated_function._api_version = version
        return decorated_function

    return decorator


def deprecated_route(
    deprecated_date: Optional[datetime] = None,
    sunset_date: Optional[datetime] = None,
    replacement: Optional[str] = None,
    message: Optional[str] = None,
):
    """Decorator to mark a route as deprecated.

    Args:
        deprecated_date: When the route was deprecated
        sunset_date: When the route will be removed
        replacement: The replacement endpoint path
        message: Custom deprecation message

    """

    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            # Log deprecation warning
            endpoint_path = request.path
            logger.warning(f"Deprecated endpoint called: {endpoint_path}")

            # Set deprecation info in context
            g.deprecation_warning = {
                "deprecated": True,
                "deprecated_date": deprecated_date.isoformat() if deprecated_date else None,
                "sunset_date": sunset_date.isoformat() if sunset_date else None,
                "replacement": replacement,
                "message": message or "This endpoint is deprecated",
            }

            # Add deprecation headers will be handled by middleware

            # Execute the route handler
            result = f(*args, **kwargs)

            # If result is a Response object, add deprecation info to response
            if hasattr(result, "headers"):
                if sunset_date:
                    result.headers["Sunset"] = sunset_date.strftime("%a, %d %b %Y %H:%M:%S GMT")
                if replacement:
                    result.headers["Link"] = f'<{replacement}>; rel="successor-version"'

            return result

        # Mark function as deprecated
        decorated_function._deprecated = True
        decorated_function._deprecation_info = {
            "deprecated_date": deprecated_date,
            "sunset_date": sunset_date,
            "replacement": replacement,
            "message": message,
        }

        return decorated_function

    return decorator


def version_required(
    min_version: Optional[str] = None, max_version: Optional[str] = None, features: Optional[List[str]] = None
):
    """Decorator to enforce version requirements on routes.

    Args:
        min_version: Minimum API version required
        max_version: Maximum API version supported
        features: List of features required for this endpoint

    """

    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            # Get current version from context
            version = getattr(g, "api_version", None)

            if not version:
                # Try to extract from path
                api_config = get_api_config()
                version = api_config.get_version_from_path(request.path)

            if not version:
                return (
                    jsonify(
                        {"error": "API version not specified", "message": "Please specify API version in the URL path"}
                    ),
                    400,
                )

            api_config = get_api_config()

            # Check version requirements
            if min_version and _compare_versions(version, min_version) < 0:
                return (
                    jsonify(
                        {
                            "error": "API version too old",
                            "message": f"This endpoint requires API version {min_version} or higher",
                            "current_version": version,
                            "required_version": min_version,
                        }
                    ),
                    400,
                )

            if max_version and _compare_versions(version, max_version) > 0:
                return (
                    jsonify(
                        {
                            "error": "API version too new",
                            "message": f"This endpoint is not available in API version {version}",
                            "current_version": version,
                            "max_version": max_version,
                        }
                    ),
                    400,
                )

            # Check required features
            if features:
                missing_features = []
                for feature in features:
                    if not api_config.is_feature_enabled(version, feature):
                        missing_features.append(feature)

                if missing_features:
                    return (
                        jsonify(
                            {
                                "error": "Required features not available",
                                "message": f"This endpoint requires features not available in {version}",
                                "missing_features": missing_features,
                                "current_version": version,
                            }
                        ),
                        400,
                    )

            # Execute the route handler
            return f(*args, **kwargs)

        # Store requirements on function
        decorated_function._version_requirements = {
            "min_version": min_version,
            "max_version": max_version,
            "features": features,
        }

        return decorated_function

    return decorator


def rate_limit_by_version(default_limit: int = 100, default_window: int = 3600, version_limits: Optional[dict] = None):
    """Decorator to apply version-specific rate limiting.

    Args:
        default_limit: Default request limit
        default_window: Default time window in seconds
        version_limits: Dict of version-specific limits
            e.g., {'v1': {'limit': 100, 'window': 3600}}

    """

    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            # Get current version
            version = getattr(g, "api_version", "v1")

            # Determine rate limit for this version
            if version_limits and version in version_limits:
                limit = version_limits[version].get("limit", default_limit)
                window = version_limits[version].get("window", default_window)
            else:
                limit = default_limit
                window = default_window

            # Store rate limit info in context for middleware
            g.rate_limit = {"limit": limit, "window": window, "version": version}

            # Note: Actual rate limiting logic would be handled by
            # a separate rate limiting middleware

            return f(*args, **kwargs)

        decorated_function._rate_limits = {
            "default": {"limit": default_limit, "window": default_window},
            "versions": version_limits or {},
        }

        return decorated_function

    return decorator


def require_feature(feature: str, fallback: Optional[Callable] = None):
    """Decorator to require a specific feature for the endpoint.

    Args:
        feature: The required feature name
        fallback: Optional fallback function if feature is not available

    """

    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            version = getattr(g, "api_version", "v1")
            api_config = get_api_config()

            if not api_config.is_feature_enabled(version, feature):
                if fallback:
                    logger.info(f"Feature '{feature}' not available in {version}, using fallback")
                    return fallback(*args, **kwargs)
                else:
                    return (
                        jsonify(
                            {
                                "error": "Feature not available",
                                "message": f"Feature '{feature}' is not available in API version {version}",
                                "feature": feature,
                                "version": version,
                            }
                        ),
                        501,
                    )  # Not Implemented

            return f(*args, **kwargs)

        decorated_function._required_feature = feature
        return decorated_function

    return decorator


def transform_response_by_version(transformers: dict):
    """Decorator to transform response based on API version.

    Args:
        transformers: Dict of version-specific transform functions
            e.g., {'v1': transform_v1, 'v2': transform_v2}

    """

    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            # Execute the route handler
            result = f(*args, **kwargs)

            # Get current version
            version = getattr(g, "api_version", "v1")

            # Apply version-specific transformation
            if version in transformers:
                transformer = transformers[version]
                result = transformer(result)

            return result

        decorated_function._response_transformers = transformers
        return decorated_function

    return decorator


def _compare_versions(v1: str, v2: str) -> int:
    """Compare two version strings."""

    # Extract numeric part from version strings like 'v1', 'v2'
    def extract_number(v):
        if v.startswith("v"):
            try:
                return int(v[1:])
            except ValueError:
                return 0
        return 0

    n1 = extract_number(v1)
    n2 = extract_number(v2)

    if n1 < n2:
        return -1
    elif n1 > n2:
        return 1
    else:
        return 0
