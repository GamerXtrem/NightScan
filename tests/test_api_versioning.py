"""Tests for API versioning system."""

from datetime import datetime, timedelta

import pytest
from flask import Flask, g

from api_versioning import (
    add_versioning_middleware,
    api_version,
    deprecated_route,
    get_api_config,
    get_api_registry,
    version_required,
)


@pytest.fixture
def app():
    """Create test Flask app with versioning."""
    app = Flask(__name__)
    app.config["TESTING"] = True

    # Add versioning middleware
    add_versioning_middleware(app)

    # Add some test routes
    @app.route("/api/auth/login", methods=["POST"])
    def old_login():
        return {"message": "old login route"}

    @app.route("/api/v1/test")
    @api_version("v1")
    def test_v1():
        return {"message": "test v1", "version": g.get("api_version")}

    @app.route("/api/v2/test")
    @api_version("v2")
    def test_v2():
        return {"message": "test v2", "version": g.get("api_version")}

    return app


@pytest.fixture
def client(app):
    """Create test client."""
    return app.test_client()


class TestAPIVersioning:
    """Test API versioning functionality."""

    def test_version_extraction(self, client):
        """Test that version is correctly extracted from path."""
        # Test v1
        response = client.get("/api/v1/test")
        assert response.status_code == 200
        data = response.get_json()
        assert data["version"] == "v1"

        # Test v2
        response = client.get("/api/v2/test")
        assert response.status_code == 200
        data = response.get_json()
        assert data["version"] == "v2"

    def test_version_headers(self, client):
        """Test that version headers are added to responses."""
        response = client.get("/api/v1/test")

        # Check version headers
        assert response.headers.get("X-API-Version") == "v1"
        assert "v1" in response.headers.get("X-API-Versions-Supported", "")

    def test_legacy_route_redirect(self, client):
        """Test that legacy routes are redirected with deprecation warning."""
        response = client.post("/api/auth/login", json={"username": "test", "password": "test"})

        # Should still work
        assert response.status_code == 200

        # Should have deprecation headers
        assert "X-API-Deprecation-Warning" in response.headers
        assert "/api/v1/auth/login" in response.headers.get("X-API-Deprecation-Warning", "")
        assert response.headers.get("X-API-Deprecated-Endpoint") == "/api/auth/login"
        assert response.headers.get("X-API-Replacement-Endpoint") == "/api/v1/auth/login"

    def test_api_config(self):
        """Test API configuration management."""
        config = get_api_config()

        # Test version configs exist
        assert "v1" in config.versions
        assert "v2" in config.versions

        # Test v1 is stable
        v1_config = config.get_version_config("v1")
        assert v1_config.status == "stable"
        assert "auth" in v1_config.features

        # Test v2 is beta
        v2_config = config.get_version_config("v2")
        assert v2_config.status == "beta"
        assert "unified-prediction" in v2_config.features

        # Test latest stable version
        assert config.get_latest_stable_version() == "v1"

    def test_api_registry(self):
        """Test API endpoint registry."""
        registry = get_api_registry()

        # Register test endpoint
        from api_versioning.registry import register_endpoint

        endpoint = register_endpoint(
            path="/api/v1/test/endpoint",
            methods=["GET", "POST"],
            version="v1",
            description="Test endpoint",
            tags=["test"],
            requires_auth=True,
        )

        # Verify registration
        v1_endpoints = registry.get_endpoints_by_version("v1")
        assert any(e.path == "/api/v1/test/endpoint" for e in v1_endpoints)

        # Test tag filtering
        test_endpoints = registry.get_endpoints_by_tag("test")
        assert len(test_endpoints) > 0
        assert all("test" in e.tags for e in test_endpoints)

    def test_version_required_decorator(self, app):
        """Test version requirement decorator."""
        with app.test_request_context():
            # Test function with version requirement
            @version_required(min_version="v2")
            def requires_v2():
                return {"message": "requires v2"}

            # Test with v1 - should fail
            g.api_version = "v1"
            response = requires_v2()
            assert isinstance(response, tuple)
            assert response[1] == 400  # Bad request

            # Test with v2 - should work
            g.api_version = "v2"
            result = requires_v2()
            assert result == {"message": "requires v2"}

    def test_deprecated_route_decorator(self, app):
        """Test deprecated route decorator."""
        with app.test_request_context():
            sunset_date = datetime.utcnow() + timedelta(days=90)

            @deprecated_route(
                deprecated_date=datetime.utcnow(), sunset_date=sunset_date, replacement="/api/v2/new-endpoint"
            )
            def old_endpoint():
                return {"message": "deprecated endpoint"}, 200

            # Call the deprecated endpoint
            response, status = old_endpoint()

            # Check deprecation info in context
            assert hasattr(g, "deprecation_warning")
            assert g.deprecation_warning["deprecated"] is True
            assert g.deprecation_warning["replacement"] == "/api/v2/new-endpoint"

    def test_feature_flags(self):
        """Test version-specific feature flags."""
        config = get_api_config()

        # Test v1 features
        assert config.is_feature_enabled("v1", "jwt_auth") is True
        assert config.is_feature_enabled("v1", "batch_processing") is False

        # Test v2 features
        assert config.is_feature_enabled("v2", "batch_processing") is True
        assert config.is_feature_enabled("v2", "streaming") is True

    def test_migration_effort_calculation(self):
        """Test migration effort calculation utility."""
        from api_versioning.utils import calculate_migration_effort

        effort = calculate_migration_effort("v1", "v2")

        assert "complexity" in effort
        assert "estimated_hours" in effort
        assert "factors" in effort
        assert effort["from_version"] == "v1"
        assert effort["to_version"] == "v2"

    def test_client_migration_guide_generation(self):
        """Test client migration guide generation."""
        from api_versioning.utils import generate_client_migration_guide

        guide = generate_client_migration_guide("v1", "v2", format="markdown")

        assert "# API Migration Guide" in guide
        assert "v1 to v2" in guide
        assert "## Breaking Changes" in guide
        assert "## Migration Examples" in guide


class TestAPIVersionIntegration:
    """Test full API version integration."""

    @pytest.fixture
    def integrated_app(self):
        """Create app with full API integration."""
        app = Flask(__name__)
        app.config["TESTING"] = True
        app.config["SECRET_KEY"] = "test-secret"

        # Use the full integration function
        from integrate_api_versioning import integrate_versioning_with_app

        integrate_versioning_with_app(app)

        return app

    @pytest.fixture
    def integrated_client(self, integrated_app):
        """Create client for integrated app."""
        return integrated_app.test_client()

    def test_v1_auth_endpoints(self, integrated_client):
        """Test v1 auth endpoints are accessible."""
        # Test login endpoint
        response = integrated_client.post("/api/v1/auth/login", json={"username": "test", "password": "test123"})
        # Should return 400, 401, or 500 (missing dependencies) but not 404
        assert response.status_code in [400, 401, 500]

        # Test register endpoint
        response = integrated_client.post(
            "/api/v1/auth/register", json={"username": "newuser", "password": "pass123", "email": "test@test.com"}
        )
        # Should return 400 or success but not 404
        assert response.status_code != 404

    def test_v2_predict_endpoints(self, integrated_client):
        """Test v2 prediction endpoints are accessible."""
        # Test analyze endpoint
        response = integrated_client.post("/api/v2/predict/analyze")
        # Should return 400 (no file) but not 404
        assert response.status_code == 400
        data = response.get_json()
        assert "error" in data

        # Test batch endpoint
        response = integrated_client.post("/api/v2/predict/batch")
        assert response.status_code == 400

    def test_version_info_endpoint(self, integrated_client):
        """Test version information endpoint."""
        response = integrated_client.get("/api/versions")
        assert response.status_code == 200

        data = response.get_json()
        assert "current_versions" in data
        assert "v1" in data["current_versions"]
        assert "v2" in data["current_versions"]
        assert data["latest_stable"] == "v1"

    def test_migration_status_endpoint(self, integrated_client):
        """Test migration status monitoring endpoint."""
        response = integrated_client.get("/api/migration-status")
        assert response.status_code == 200

        data = response.get_json()
        assert "total_endpoints" in data
        assert "versioned_endpoints" in data
        assert "migration_progress" in data
        assert data["migration_progress"]["percentage"] >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
