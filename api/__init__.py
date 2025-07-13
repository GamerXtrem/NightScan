"""API module for NightScan - Centralized versioned API management."""

from flask import Flask

from api_versioning import add_versioning_middleware, get_api_registry


def register_all_apis(app: Flask) -> None:
    """Register all API versions with the Flask application."""
    # Add versioning middleware first
    add_versioning_middleware(app)

    # Import and register v1 APIs
    from .v1 import create_v1_blueprint

    v1_bp = create_v1_blueprint()
    app.register_blueprint(v1_bp)

    # Import and register v2 APIs
    from .v2 import create_v2_blueprint

    v2_bp = create_v2_blueprint()
    app.register_blueprint(v2_bp)

    # Register blueprints in registry for tracking
    registry = get_api_registry()
    registry.register_blueprint("v1", "main", v1_bp)
    registry.register_blueprint("v2", "main", v2_bp)

    # Log registration
    app.logger.info("Registered all API versions")

    # Generate and log endpoint report
    report = registry.generate_endpoint_report()
    app.logger.info(f"Total endpoints registered: {report['total_endpoints']}")
    for version, stats in report["by_version"].items():
        app.logger.info(f"  {version}: {stats['count']} endpoints")


__all__ = ["register_all_apis"]
