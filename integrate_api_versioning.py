#!/usr/bin/env python3
"""Integration script to add API versioning to the existing Flask application.

This script shows how to integrate the new versioning system with minimal
changes to the existing codebase.
"""

import logging

from flask import Flask

# Import the new API versioning system
from api import register_all_apis
from api_versioning import add_versioning_middleware, get_api_config, get_api_registry
from api_versioning.utils import generate_version_migration_report, validate_version_consistency

logger = logging.getLogger(__name__)


def integrate_versioning_with_app(app: Flask) -> None:
    """Integrate API versioning with an existing Flask application.

    This function should be called during app initialization, after
    the app is created but before registering the original blueprints.
    """
    logger.info("Starting API versioning integration")

    # Step 1: Add versioning middleware
    # This handles legacy route redirects and adds version headers
    add_versioning_middleware(app)
    logger.info("Added versioning middleware")

    # Step 2: Register all versioned APIs
    # This creates /api/v1/* and /api/v2/* endpoints
    register_all_apis(app)
    logger.info("Registered versioned APIs")

    # Step 3: Configure legacy route handling
    # The middleware will automatically redirect old routes to new ones
    api_config = get_api_config()
    legacy_routes = len(api_config.legacy_mappings)
    logger.info(f"Configured {legacy_routes} legacy route mappings")

    # Step 4: Add version information endpoint
    @app.route("/api/versions")
    def api_versions_info():
        """Endpoint to get API version information."""
        config = get_api_config()
        return {
            "current_versions": config.get_active_versions(),
            "latest_stable": config.get_latest_stable_version(),
            "deprecated_versions": [v for v, c in config.versions.items() if c.status in ["deprecated", "sunset"]],
            "version_details": {
                v: {"status": c.status, "features": c.features, "release_date": c.release_date.isoformat()}
                for v, c in config.versions.items()
            },
        }

    # Step 5: Add migration status endpoint (for monitoring)
    @app.route("/api/migration-status")
    def migration_status():
        """Endpoint to check API migration status."""
        registry = get_api_registry()
        report = registry.generate_endpoint_report()

        # Get deprecation usage from middleware
        middleware = getattr(app, "_versioning_middleware", None)
        deprecation_report = middleware.get_deprecation_report() if middleware else {}

        return {
            "total_endpoints": report["total_endpoints"],
            "versioned_endpoints": report["by_version"],
            "deprecated_endpoint_usage": deprecation_report,
            "migration_progress": calculate_migration_progress(report),
        }

    # Step 6: Validate consistency
    issues = validate_version_consistency(app)
    if issues:
        logger.warning(f"Version consistency issues found: {len(issues)}")
        for issue in issues[:5]:  # Log first 5 issues
            logger.warning(f"  - {issue}")
    else:
        logger.info("Version consistency check passed")

    # Step 7: Generate initial migration report
    try:
        report = generate_version_migration_report()
        logger.info(f"Generated migration report with {len(report['migration_paths'])} paths")
    except Exception as e:
        logger.error(f"Failed to generate migration report: {e}")

    logger.info("API versioning integration completed")


def calculate_migration_progress(report: dict) -> dict:
    """Calculate the migration progress percentage."""
    total = report["total_endpoints"]
    if total == 0:
        return {"percentage": 100, "status": "no_endpoints"}

    # Count v1 and v2 endpoints
    v1_count = report["by_version"].get("v1", {}).get("count", 0)
    v2_count = report["by_version"].get("v2", {}).get("count", 0)
    versioned = v1_count + v2_count

    percentage = (versioned / total) * 100 if total > 0 else 0

    return {
        "percentage": round(percentage, 2),
        "versioned": versioned,
        "total": total,
        "status": "complete" if percentage == 100 else "in_progress",
    }


def modify_existing_app_factory(original_create_app):
    """Decorator to modify an existing create_app function to include versioning.

    Usage:
        @modify_existing_app_factory
        def create_app(config=None):
            app = Flask(__name__)
            # ... existing app setup ...
            return app
    """

    def wrapped_create_app(*args, **kwargs):
        # Call original create_app
        app = original_create_app(*args, **kwargs)

        # Integrate versioning
        integrate_versioning_with_app(app)

        # Store middleware reference for monitoring
        from api_versioning.middleware import APIVersioningMiddleware

        for handler in app.before_request_funcs.get(None, []):
            if hasattr(handler, "__self__") and isinstance(handler.__self__, APIVersioningMiddleware):
                app._versioning_middleware = handler.__self__
                break

        return app

    return wrapped_create_app


# Example: How to modify web/app.py
def example_integration():
    """Example showing how to integrate versioning into web/app.py.

    In web/app.py, add these lines after creating the Flask app:

    ```python
    # At the top of the file
    from integrate_api_versioning import integrate_versioning_with_app

    # After creating the app
    app = Flask(__name__)

    # Add versioning before registering other blueprints
    integrate_versioning_with_app(app)

    # Continue with existing setup...
    ```
    """
    pass


if __name__ == "__main__":
    # Example standalone test
    logging.basicConfig(level=logging.INFO)

    # Create a test app
    app = Flask(__name__)

    # Add some test configuration
    app.config["SECRET_KEY"] = "test-secret-key"

    # Integrate versioning
    integrate_versioning_with_app(app)

    # Print summary
    with app.app_context():
        registry = get_api_registry()
        report = registry.generate_endpoint_report()

        print("\n" + "=" * 50)
        print("API Versioning Integration Summary")
        print("=" * 50)
        print(f"Total endpoints: {report['total_endpoints']}")
        for version, stats in report["by_version"].items():
            print(f"{version}: {stats['count']} endpoints")
        print("=" * 50)
