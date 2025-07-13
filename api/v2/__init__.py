"""API v2 - Next generation API with unified prediction system."""

import logging

from flask import Blueprint

logger = logging.getLogger(__name__)


def create_v2_blueprint() -> Blueprint:
    """Create and configure the main v2 API blueprint."""
    # Create main v2 blueprint
    v2_bp = Blueprint("api_v2", __name__, url_prefix="/api/v2")

    # Import and register sub-blueprints
    try:
        # Unified Prediction API
        from .predict import create_predict_blueprint

        predict_bp = create_predict_blueprint()
        v2_bp.register_blueprint(predict_bp, url_prefix="/predict")
        logger.info("Registered v2 predict blueprint")
    except ImportError as e:
        logger.warning(f"Could not import predict blueprint: {e}")

    try:
        # Models Management API
        from .models import create_models_blueprint

        models_bp = create_models_blueprint()
        v2_bp.register_blueprint(models_bp, url_prefix="/models")
        logger.info("Registered v2 models blueprint")
    except ImportError as e:
        logger.warning(f"Could not import models blueprint: {e}")

    # Add v2-specific root endpoint
    @v2_bp.route("/")
    def v2_info():
        """Information about API v2."""
        return {
            "version": "v2",
            "status": "beta",
            "features": [
                "unified-prediction",
                "batch-processing",
                "real-time-streaming",
                "model-selection",
                "async-processing",
            ],
            "documentation": "/api/v2/docs",
        }

    return v2_bp


__all__ = ["create_v2_blueprint"]
