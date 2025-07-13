"""API Version Configuration Management."""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class APIVersionConfig:
    """Configuration for a specific API version."""

    version: str
    status: str  # 'development', 'beta', 'stable', 'deprecated', 'sunset'
    release_date: datetime
    deprecated_date: Optional[datetime] = None
    sunset_date: Optional[datetime] = None
    min_client_version: Optional[str] = None
    features: List[str] = field(default_factory=list)
    endpoints: List[str] = field(default_factory=list)
    description: str = ""
    breaking_changes: List[str] = field(default_factory=list)
    migration_guide_url: Optional[str] = None


# Central API version registry
API_VERSIONS: Dict[str, APIVersionConfig] = {
    "v1": APIVersionConfig(
        version="v1",
        status="stable",
        release_date=datetime(2024, 1, 1),
        min_client_version="1.0.0",
        features=["auth", "analytics", "cache", "detections", "quota", "retention", "location", "files"],
        endpoints=[
            "/api/v1/auth/*",
            "/api/v1/analytics/*",
            "/api/v1/cache/*",
            "/api/v1/detections",
            "/api/v1/predictions",
            "/api/v1/quota/*",
            "/api/v1/retention/*",
            "/api/v1/location/*",
            "/api/v1/files/*",
            "/api/v1/filename/*",
        ],
        description="Initial stable API version with core functionality",
    ),
    "v2": APIVersionConfig(
        version="v2",
        status="beta",
        release_date=datetime(2024, 6, 1),
        min_client_version="2.0.0",
        features=["unified-prediction", "batch-processing", "real-time-streaming", "advanced-analytics"],
        endpoints=["/api/v2/predict/*", "/api/v2/models/*", "/api/v2/batch/*", "/api/v2/stream/*"],
        description="Next generation API with unified prediction system",
        breaking_changes=[
            "Unified prediction endpoint replaces separate audio/image endpoints",
            "New response format for predictions",
            "Batch processing requires different request structure",
        ],
        migration_guide_url="/docs/api/v2/migration",
    ),
}


# Legacy route mappings for backward compatibility
LEGACY_ROUTE_MAPPINGS = {
    # Authentication routes - old legacy mappings
    "/api/auth/login": "/api/v1/auth/login",
    "/api/auth/register": "/api/v1/auth/register", 
    "/api/auth/refresh": "/api/v1/auth/refresh",
    "/api/auth/logout": "/api/v1/auth/logout",
    "/api/auth/verify": "/api/v1/auth/verify",
    # Analytics routes
    "/analytics/dashboard": "/api/v1/analytics/dashboard",
    "/analytics/api/metrics": "/api/v1/analytics/metrics",
    "/analytics/api/species": "/api/v1/analytics/species",
    "/analytics/api/zones": "/api/v1/analytics/zones",
    "/analytics/export/csv": "/api/v1/analytics/export/csv",
    "/analytics/export/pdf": "/api/v1/analytics/export/pdf",
    # Cache routes
    "/api/cache/metrics": "/api/v1/cache/metrics",
    "/api/cache/health": "/api/v1/cache/health",
    "/api/cache/clear": "/api/v1/cache/clear",
    # Password reset routes
    "/api/password-reset/request": "/api/v1/password-reset/request",
    "/api/password-reset/verify": "/api/v1/password-reset/verify",
    "/api/password-reset/reset": "/api/v1/password-reset/reset",
    # Location routes
    "/api/location": "/api/v1/location",
    "/api/location/phone": "/api/v1/location/phone",
    "/api/location/history": "/api/v1/location/history",
    "/api/location/coordinates": "/api/v1/location/coordinates",
    "/api/location/status": "/api/v1/location/status",
    # File management routes
    "/api/filename/parse": "/api/v1/filename/parse",
    "/api/filename/generate": "/api/v1/filename/generate",
    "/api/files/statistics": "/api/v1/files/statistics",
    # Prediction routes (will be v2)
    "/predict/upload": "/api/v2/predict/upload",
    "/predict/file": "/api/v2/predict/file",
    "/predict/batch": "/api/v2/predict/batch",
    "/models/status": "/api/v2/models/status",
    "/models/preload": "/api/v2/models/preload",
}


# RESTful route redirections for better API design
# These handle redirections from non-RESTful endpoints to RESTful ones
RESTFUL_REDIRECTIONS = {
    # Authentication - redirect verbs to RESTful resources
    "/api/v1/auth/login": ("/api/v1/sessions", "POST"),
    "/api/v1/auth/logout": ("/api/v1/sessions", "DELETE"), 
    "/api/v1/auth/register": ("/api/v1/users", "POST"),
    "/api/v1/auth/refresh": ("/api/v1/tokens", "POST"),
    "/api/v1/auth/verify": ("/api/v1/tokens/current", "GET"),
    # Analytics - redirect to proper exports with query params
    "/api/v1/analytics/export/csv": ("/api/v1/analytics/exports?format=csv", "GET"),
    "/api/v1/analytics/export/pdf": ("/api/v1/analytics/exports?format=pdf", "GET"),
    # Cache - redirect clear verb to DELETE method
    "/api/v1/cache/clear": ("/api/v1/cache", "DELETE"),
    # Files - redirect upload verb to POST resource
    "/api/v1/files/upload": ("/api/v1/files", "POST"),
    # Predictions - redirect analyze verb to POST resource
    "/api/v1/predictions/analyze": ("/api/v1/predictions", "POST"),
}


# Feature flags per version
VERSION_FEATURES = {
    "v1": {
        "rate_limiting": True,
        "jwt_auth": True,
        "quota_management": True,
        "data_retention": True,
        "websocket_support": False,
        "batch_processing": False,
        "streaming": False,
    },
    "v2": {
        "rate_limiting": True,
        "jwt_auth": True,
        "quota_management": True,
        "data_retention": True,
        "websocket_support": True,
        "batch_processing": True,
        "streaming": True,
        "ml_model_selection": True,
        "async_processing": True,
    },
}


class APIVersionManager:
    """Manages API version configuration and provides utility methods."""

    def __init__(self):
        self.versions = API_VERSIONS
        self.legacy_mappings = LEGACY_ROUTE_MAPPINGS
        self.restful_redirections = RESTFUL_REDIRECTIONS
        self.feature_flags = VERSION_FEATURES

    def get_version_config(self, version: str) -> Optional[APIVersionConfig]:
        """Get configuration for a specific API version."""
        return self.versions.get(version)

    def get_active_versions(self) -> List[str]:
        """Get list of active (non-sunset) API versions."""
        active = []
        for version, config in self.versions.items():
            if config.status not in ["sunset", "deprecated"]:
                active.append(version)
        return active

    def get_latest_stable_version(self) -> Optional[str]:
        """Get the latest stable API version."""
        stable_versions = [(v, config) for v, config in self.versions.items() if config.status == "stable"]
        if not stable_versions:
            return None

        # Sort by release date and return the latest
        stable_versions.sort(key=lambda x: x[1].release_date, reverse=True)
        return stable_versions[0][0]

    def is_version_deprecated(self, version: str) -> bool:
        """Check if a version is deprecated."""
        config = self.get_version_config(version)
        return config and config.status in ["deprecated", "sunset"]

    def get_deprecation_info(self, version: str) -> Optional[Dict[str, Any]]:
        """Get deprecation information for a version."""
        config = self.get_version_config(version)
        if not config or not self.is_version_deprecated(version):
            return None

        return {
            "deprecated": True,
            "deprecated_date": config.deprecated_date.isoformat() if config.deprecated_date else None,
            "sunset_date": config.sunset_date.isoformat() if config.sunset_date else None,
            "migration_guide": config.migration_guide_url,
            "replacement_version": self.get_latest_stable_version(),
        }

    def get_legacy_mapping(self, path: str) -> Optional[str]:
        """Get the versioned route for a legacy path."""
        # Direct mapping
        if path in self.legacy_mappings:
            return self.legacy_mappings[path]

        # Try prefix matching for parameterized routes
        for legacy, versioned in self.legacy_mappings.items():
            if legacy.endswith("*") and path.startswith(legacy[:-1]):
                # Replace the prefix
                return path.replace(legacy[:-1], versioned[:-1])

        return None

    def get_restful_redirection(self, path: str, method: str) -> Optional[tuple]:
        """Get RESTful redirection for a non-RESTful endpoint.
        
        Returns:
            Tuple of (new_path, new_method) if redirection exists, None otherwise
        """
        if path in self.restful_redirections:
            target_path, target_method = self.restful_redirections[path]
            # Only redirect if the method matches or if target method is different
            if target_method != method:
                return (target_path, target_method)
        return None

    def is_feature_enabled(self, version: str, feature: str) -> bool:
        """Check if a feature is enabled for a specific version."""
        version_features = self.feature_flags.get(version, {})
        return version_features.get(feature, False)

    def get_version_from_path(self, path: str) -> Optional[str]:
        """Extract API version from request path."""
        if path.startswith("/api/v"):
            parts = path.split("/")
            if len(parts) >= 3:
                version = parts[2]  # e.g., 'v1' from '/api/v1/...'
                if version in self.versions:
                    return version
        return None


# Singleton instance
_api_version_manager = None


def get_api_config() -> APIVersionManager:
    """Get the global API version manager instance."""
    global _api_version_manager
    if _api_version_manager is None:
        _api_version_manager = APIVersionManager()
    return _api_version_manager
