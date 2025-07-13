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
        features=["auth", "analytics", "files", "predictions"],
        endpoints=[
            "/api/v1/sessions",
            "/api/v1/users", 
            "/api/v1/tokens",
            "/api/v1/analytics/*",
            "/api/v1/files",
            "/api/v1/predictions",
        ],
        description="RESTful API with clean resource-based endpoints",
    ),
    "v2": APIVersionConfig(
        version="v2",
        status="beta",
        release_date=datetime(2024, 6, 1),
        min_client_version="2.0.0",
        features=["unified_prediction", "batch_processing", "real_time_streaming", "advanced_analytics"],
        endpoints=["/api/v2/predictions", "/api/v2/models", "/api/v2/batch", "/api/v2/stream"],
        description="Next generation API with unified prediction system",
        breaking_changes=[
            "Unified prediction endpoint replaces separate audio/image endpoints",
            "New response format for predictions",
            "Batch processing requires different request structure",
        ],
        migration_guide_url="/docs/api/v2/migration",
    ),
}






# Feature flags per version
VERSION_FEATURES = {
    "v1": {
        "rate_limiting": True,
        "jwt_auth": True,
        "restful_design": True,
        "file_management": True,
        "ml_predictions": True,
    },
    "v2": {
        "rate_limiting": True,
        "jwt_auth": True,
        "restful_design": True,
        "file_management": True,
        "ml_predictions": True,
        "batch_processing": True,
        "streaming": True,
        "unified_prediction": True,
        "async_processing": True,
    },
}


class APIVersionManager:
    """Manages API version configuration and provides utility methods."""

    def __init__(self):
        self.versions = API_VERSIONS
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
