"""API Registry for tracking and managing versioned endpoints."""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from flask import Blueprint

logger = logging.getLogger(__name__)


@dataclass
class APIEndpoint:
    """Represents a single API endpoint with its metadata."""

    path: str
    methods: List[str]
    version: str
    handler: Optional[Callable] = None
    description: str = ""
    deprecated: bool = False
    deprecated_date: Optional[datetime] = None
    replacement_endpoint: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    requires_auth: bool = True
    rate_limit: Optional[Dict[str, int]] = None


class APIRegistry:
    """Central registry for all API endpoints across versions."""

    def __init__(self):
        # Store endpoints by version
        self._endpoints: Dict[str, List[APIEndpoint]] = {}

        # Store blueprints by version
        self._blueprints: Dict[str, Dict[str, Blueprint]] = {}

        # Track endpoint migrations
        self._migrations: Dict[str, str] = {}  # old_path -> new_path

        # Usage statistics
        self._usage_stats: Dict[str, Dict[str, Any]] = {}

    def register_endpoint(self, endpoint: APIEndpoint):
        """Register a new API endpoint."""
        version = endpoint.version

        if version not in self._endpoints:
            self._endpoints[version] = []

        # Check for duplicates
        for existing in self._endpoints[version]:
            if existing.path == endpoint.path and set(existing.methods) == set(endpoint.methods):
                logger.warning(f"Duplicate endpoint registration: {endpoint.path} ({version})")
                return

        self._endpoints[version].append(endpoint)
        logger.debug(f"Registered endpoint: {endpoint.path} ({version})")

        # Track migrations if this is a replacement
        if endpoint.replacement_endpoint:
            self._migrations[endpoint.path] = endpoint.replacement_endpoint

    def register_blueprint(self, version: str, name: str, blueprint: Blueprint):
        """Register a Flask blueprint for a specific version."""
        if version not in self._blueprints:
            self._blueprints[version] = {}

        if name in self._blueprints[version]:
            logger.warning(f"Blueprint '{name}' already registered for version {version}")
            return

        self._blueprints[version][name] = blueprint
        logger.info(f"Registered blueprint '{name}' for API version {version}")

    def get_endpoints_by_version(self, version: str) -> List[APIEndpoint]:
        """Get all endpoints for a specific version."""
        return self._endpoints.get(version, [])

    def get_endpoints_by_tag(self, tag: str, version: Optional[str] = None) -> List[APIEndpoint]:
        """Get all endpoints with a specific tag."""
        endpoints = []

        versions = [version] if version else self._endpoints.keys()

        for v in versions:
            for endpoint in self._endpoints.get(v, []):
                if tag in endpoint.tags:
                    endpoints.append(endpoint)

        return endpoints

    def get_deprecated_endpoints(self, version: Optional[str] = None) -> List[APIEndpoint]:
        """Get all deprecated endpoints."""
        deprecated = []

        versions = [version] if version else self._endpoints.keys()

        for v in versions:
            for endpoint in self._endpoints.get(v, []):
                if endpoint.deprecated:
                    deprecated.append(endpoint)

        return deprecated

    def get_endpoint_migration(self, old_path: str) -> Optional[str]:
        """Get the new path for a migrated endpoint."""
        return self._migrations.get(old_path)

    def get_blueprints(self, version: str) -> Dict[str, Blueprint]:
        """Get all blueprints for a specific version."""
        return self._blueprints.get(version, {})

    def record_usage(self, path: str, version: str, method: str, response_time: float, status_code: int):
        """Record usage statistics for an endpoint."""
        key = f"{version}:{method}:{path}"

        if key not in self._usage_stats:
            self._usage_stats[key] = {
                "count": 0,
                "total_time": 0,
                "avg_time": 0,
                "status_codes": {},
                "first_used": datetime.utcnow(),
                "last_used": None,
            }

        stats = self._usage_stats[key]
        stats["count"] += 1
        stats["total_time"] += response_time
        stats["avg_time"] = stats["total_time"] / stats["count"]
        stats["last_used"] = datetime.utcnow()

        # Track status codes
        status_key = str(status_code)
        stats["status_codes"][status_key] = stats["status_codes"].get(status_key, 0) + 1

    def get_usage_stats(self, version: Optional[str] = None) -> Dict[str, Any]:
        """Get usage statistics for endpoints."""
        if version:
            # Filter stats for specific version
            filtered_stats = {}
            prefix = f"{version}:"

            for key, stats in self._usage_stats.items():
                if key.startswith(prefix):
                    filtered_stats[key] = stats

            return filtered_stats

        return self._usage_stats.copy()

    def generate_endpoint_report(self) -> Dict[str, Any]:
        """Generate a comprehensive report of all registered endpoints."""
        report = {
            "total_endpoints": 0,
            "by_version": {},
            "deprecated_count": 0,
            "auth_required_count": 0,
            "by_tag": {},
            "migrations": self._migrations.copy(),
        }

        for version, endpoints in self._endpoints.items():
            version_stats = {
                "count": len(endpoints),
                "deprecated": 0,
                "auth_required": 0,
                "by_method": {},
                "by_tag": {},
            }

            for endpoint in endpoints:
                report["total_endpoints"] += 1

                # Count deprecated
                if endpoint.deprecated:
                    report["deprecated_count"] += 1
                    version_stats["deprecated"] += 1

                # Count auth required
                if endpoint.requires_auth:
                    report["auth_required_count"] += 1
                    version_stats["auth_required"] += 1

                # Count by method
                for method in endpoint.methods:
                    version_stats["by_method"][method] = version_stats["by_method"].get(method, 0) + 1

                # Count by tag
                for tag in endpoint.tags:
                    # Version-specific
                    version_stats["by_tag"][tag] = version_stats["by_tag"].get(tag, 0) + 1

                    # Global
                    report["by_tag"][tag] = report["by_tag"].get(tag, 0) + 1

            report["by_version"][version] = version_stats

        return report

    def validate_endpoints(self) -> List[str]:
        """Validate all registered endpoints and return any issues."""
        issues = []

        # Check for missing replacements for deprecated endpoints
        for version, endpoints in self._endpoints.items():
            for endpoint in endpoints:
                if endpoint.deprecated and not endpoint.replacement_endpoint:
                    issues.append(f"Deprecated endpoint {endpoint.path} ({version}) " f"has no replacement specified")

        # Check for circular migrations
        for old_path, new_path in self._migrations.items():
            if new_path in self._migrations and self._migrations[new_path] == old_path:
                issues.append(f"Circular migration detected: {old_path} <-> {new_path}")

        # Check for endpoints without handlers
        for version, endpoints in self._endpoints.items():
            for endpoint in endpoints:
                if not endpoint.handler and not endpoint.deprecated:
                    issues.append(f"Endpoint {endpoint.path} ({version}) has no handler")

        return issues


# Global registry instance
_api_registry = None


def get_api_registry() -> APIRegistry:
    """Get the global API registry instance."""
    global _api_registry
    if _api_registry is None:
        _api_registry = APIRegistry()
    return _api_registry


def register_endpoint(
    path: str,
    methods: List[str],
    version: str,
    handler: Optional[Callable] = None,
    description: str = "",
    tags: List[str] = None,
    requires_auth: bool = True,
    deprecated: bool = False,
    replacement: Optional[str] = None,
) -> APIEndpoint:
    """Convenience function to register an endpoint."""
    endpoint = APIEndpoint(
        path=path,
        methods=methods,
        version=version,
        handler=handler,
        description=description,
        tags=tags or [],
        requires_auth=requires_auth,
        deprecated=deprecated,
        replacement_endpoint=replacement,
    )

    registry = get_api_registry()
    registry.register_endpoint(endpoint)

    return endpoint
