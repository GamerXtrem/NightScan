"""API v1 - Main blueprint and sub-module registration."""

import logging

from flask import Blueprint

logger = logging.getLogger(__name__)


def create_v1_blueprint() -> Blueprint:
    """Create and configure the main v1 API blueprint."""
    # Create main v1 blueprint
    v1_bp = Blueprint("api_v1", __name__, url_prefix="/api/v1")

    # Import and register sub-blueprints
    try:
        # Auth API
        from .auth import create_auth_blueprint

        auth_bp = create_auth_blueprint()
        v1_bp.register_blueprint(auth_bp, url_prefix="/auth")
        logger.info("Registered v1 auth blueprint")
    except ImportError as e:
        logger.warning(f"Could not import auth blueprint: {e}")

    try:
        # Analytics API
        from .analytics import create_analytics_blueprint

        analytics_bp = create_analytics_blueprint()
        v1_bp.register_blueprint(analytics_bp, url_prefix="/analytics")
        logger.info("Registered v1 analytics blueprint")
    except ImportError as e:
        logger.warning(f"Could not import analytics blueprint: {e}")

    try:
        # Cache API
        from .cache import create_cache_blueprint

        cache_bp = create_cache_blueprint()
        v1_bp.register_blueprint(cache_bp, url_prefix="/cache")
        logger.info("Registered v1 cache blueprint")
    except ImportError as e:
        logger.warning(f"Could not import cache blueprint: {e}")

    try:
        # Location API
        from .location import create_location_blueprint

        location_bp = create_location_blueprint()
        v1_bp.register_blueprint(location_bp, url_prefix="/location")
        logger.info("Registered v1 location blueprint")
    except ImportError as e:
        logger.warning(f"Could not import location blueprint: {e}")

    try:
        # Files API
        from .files import create_files_blueprint

        files_bp = create_files_blueprint()
        v1_bp.register_blueprint(files_bp, url_prefix="/files")
        logger.info("Registered v1 files blueprint")
    except ImportError as e:
        logger.warning(f"Could not import files blueprint: {e}")

    try:
        # Predictions API
        from .predictions import create_predictions_blueprint

        predictions_bp = create_predictions_blueprint()
        v1_bp.register_blueprint(predictions_bp, url_prefix="/predictions")
        logger.info("Registered v1 predictions blueprint")
    except ImportError as e:
        logger.warning(f"Could not import predictions blueprint: {e}")

    try:
        # Password Reset API
        from .password_reset import create_password_reset_blueprint

        password_bp = create_password_reset_blueprint()
        v1_bp.register_blueprint(password_bp, url_prefix="/password-reset")
        logger.info("Registered v1 password reset blueprint")
    except ImportError as e:
        logger.warning(f"Could not import password reset blueprint: {e}")

    # Register existing v1 routes (detection, quota, retention)
    # These are already in api_v1.py and will be mounted at the blueprint level

    return v1_bp


__all__ = ["create_v1_blueprint"]
