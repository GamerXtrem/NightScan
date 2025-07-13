"""Utility functions for API versioning."""

import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

from flask import Flask

from .config import get_api_config
from .registry import get_api_registry

logger = logging.getLogger(__name__)


def generate_version_migration_report() -> Dict[str, Any]:
    """Generate a comprehensive migration report for all API versions."""
    api_config = get_api_config()
    registry = get_api_registry()

    report = {
        "generated_at": datetime.utcnow().isoformat(),
        "active_versions": api_config.get_active_versions(),
        "latest_stable": api_config.get_latest_stable_version(),
        "deprecation_timeline": [],
        "migration_paths": {},
        "endpoint_changes": {},
        "breaking_changes": {},
        "usage_statistics": {},
    }

    # Build deprecation timeline
    for version, config in api_config.versions.items():
        if config.deprecated_date or config.sunset_date:
            report["deprecation_timeline"].append(
                {
                    "version": version,
                    "status": config.status,
                    "deprecated_date": config.deprecated_date.isoformat() if config.deprecated_date else None,
                    "sunset_date": config.sunset_date.isoformat() if config.sunset_date else None,
                    "days_until_sunset": (
                        (config.sunset_date - datetime.utcnow()).days
                        if config.sunset_date and config.sunset_date > datetime.utcnow()
                        else None
                    ),
                }
            )

    # Sort timeline by date
    report["deprecation_timeline"].sort(key=lambda x: x["deprecated_date"] or x["sunset_date"] or "9999")

    # Build migration paths
    for old_path, new_path in api_config.legacy_mappings.items():
        report["migration_paths"][old_path] = {"new_path": new_path, "version": extract_version_from_path(new_path)}

    # Analyze endpoint changes between versions
    all_versions = sorted(api_config.versions.keys())
    for i in range(len(all_versions) - 1):
        current_version = all_versions[i]
        next_version = all_versions[i + 1]

        current_endpoints = {e.path for e in registry.get_endpoints_by_version(current_version)}
        next_endpoints = {e.path for e in registry.get_endpoints_by_version(next_version)}

        report["endpoint_changes"][f"{current_version}_to_{next_version}"] = {
            "added": list(next_endpoints - current_endpoints),
            "removed": list(current_endpoints - next_endpoints),
            "total_current": len(current_endpoints),
            "total_next": len(next_endpoints),
        }

    # Extract breaking changes
    for version, config in api_config.versions.items():
        if config.breaking_changes:
            report["breaking_changes"][version] = config.breaking_changes

    # Include usage statistics
    report["usage_statistics"] = registry.get_usage_stats()

    return report


def extract_version_from_path(path: str) -> Optional[str]:
    """Extract API version from a URL path."""
    # Match patterns like /api/v1/..., /api/v2/...
    match = re.match(r"^/api/(v\d+)/", path)
    if match:
        return match.group(1)
    return None


def validate_version_consistency(app: Flask) -> List[str]:
    """Validate version consistency across the application."""
    issues = []
    api_config = get_api_config()
    registry = get_api_registry()

    # Check all registered routes
    for rule in app.url_map.iter_rules():
        path = rule.rule
        version = extract_version_from_path(path)

        if version:
            # Check if version is registered
            if version not in api_config.versions:
                issues.append(f"Route {path} uses unregistered version {version}")

            # Check if endpoint is in registry
            endpoints = registry.get_endpoints_by_version(version)
            endpoint_paths = [e.path for e in endpoints]

            if path not in endpoint_paths:
                # Check if it's a parameterized route
                normalized_path = normalize_path_pattern(path)
                if not any(normalize_path_pattern(ep) == normalized_path for ep in endpoint_paths):
                    issues.append(f"Route {path} is not registered in endpoint registry")

    # Validate registry endpoints
    registry_issues = registry.validate_endpoints()
    issues.extend(registry_issues)

    # Check for version conflicts in blueprints
    for version, blueprints in registry._blueprints.items():
        for bp_name, blueprint in blueprints.items():
            if hasattr(blueprint, "url_prefix"):
                expected_prefix = f"/api/{version}"
                if blueprint.url_prefix and not blueprint.url_prefix.startswith(expected_prefix):
                    issues.append(
                        f"Blueprint '{bp_name}' for version {version} has "
                        f"inconsistent prefix: {blueprint.url_prefix}"
                    )

    return issues


def normalize_path_pattern(path: str) -> str:
    """Normalize a path pattern by replacing parameters with placeholders."""
    # Replace Flask path parameters like <int:id> or <string:name> with *
    return re.sub(r"<[^>]+>", "*", path)


def generate_client_migration_guide(from_version: str, to_version: str, format: str = "markdown") -> str:
    """Generate a migration guide for client developers."""
    api_config = get_api_config()
    registry = get_api_registry()

    if format == "markdown":
        guide = f"# API Migration Guide: {from_version} to {to_version}\n\n"
        guide += f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}\n\n"

        # Version information
        from_config = api_config.get_version_config(from_version)
        to_config = api_config.get_version_config(to_version)

        if not from_config or not to_config:
            return "Error: Invalid version specified"

        guide += "## Version Information\n\n"
        guide += f"- **From**: {from_version} ({from_config.status})\n"
        guide += f"- **To**: {to_version} ({to_config.status})\n"
        guide += f"- **Minimum Client Version**: {to_config.min_client_version}\n\n"

        # Breaking changes
        if to_config.breaking_changes:
            guide += "## Breaking Changes\n\n"
            for change in to_config.breaking_changes:
                guide += f"- {change}\n"
            guide += "\n"

        # Endpoint changes
        from_endpoints = registry.get_endpoints_by_version(from_version)
        to_endpoints = registry.get_endpoints_by_version(to_version)

        from_paths = {e.path: e for e in from_endpoints}
        to_paths = {e.path: e for e in to_endpoints}

        # Removed endpoints
        removed = set(from_paths.keys()) - set(to_paths.keys())
        if removed:
            guide += "## Removed Endpoints\n\n"
            guide += "| Old Endpoint | Replacement |\n"
            guide += "|--------------|-------------|\n"
            for path in sorted(removed):
                endpoint = from_paths[path]
                replacement = endpoint.replacement_endpoint or "No direct replacement"
                guide += f"| {path} | {replacement} |\n"
            guide += "\n"

        # Changed endpoints
        guide += "## Endpoint Mappings\n\n"
        guide += "| " + from_version + " | " + to_version + " | Notes |\n"
        guide += "|----------|----------|-------|\n"

        for old_path, new_path in api_config.legacy_mappings.items():
            old_version = extract_version_from_path(old_path) or "legacy"
            new_version = extract_version_from_path(new_path)

            if old_version == from_version and new_version == to_version:
                guide += f"| {old_path} | {new_path} | |\n"

        guide += "\n"

        # New features
        new_features = set(to_config.features) - set(from_config.features)
        if new_features:
            guide += "## New Features\n\n"
            for feature in sorted(new_features):
                guide += f"- **{feature}**: "
                guide += get_feature_description(feature) + "\n"
            guide += "\n"

        # Code examples
        guide += "## Migration Examples\n\n"
        guide += "### Authentication\n\n"
        guide += "```javascript\n"
        guide += f"// {from_version}\n"
        guide += "const response = await fetch('/api/auth/login', {\n"
        guide += "  method: 'POST',\n"
        guide += "  body: JSON.stringify({username, password})\n"
        guide += "});\n\n"
        guide += f"// {to_version}\n"
        guide += f"const response = await fetch('/api/{to_version}/auth/login', {{\n"
        guide += "  method: 'POST',\n"
        guide += "  headers: {'X-API-Version': '" + to_version + "'},\n"
        guide += "  body: JSON.stringify({username, password})\n"
        guide += "});\n"
        guide += "```\n\n"

        # Additional resources
        if to_config.migration_guide_url:
            guide += "## Additional Resources\n\n"
            guide += f"- [Full Migration Guide]({to_config.migration_guide_url})\n"
            guide += "- [API Documentation](/api/" + to_version + "/docs)\n"
            guide += "- [Support](/support)\n"

        return guide

    else:
        return "Unsupported format. Use 'markdown'."


def get_feature_description(feature: str) -> str:
    """Get a human-readable description for a feature."""
    descriptions = {
        "unified-prediction": "Single endpoint for both audio and image predictions",
        "batch-processing": "Process multiple files in a single request",
        "real-time-streaming": "WebSocket support for real-time updates",
        "advanced-analytics": "Enhanced analytics with custom date ranges and filters",
        "jwt_auth": "JSON Web Token based authentication",
        "quota_management": "Request quota tracking and limits",
        "data_retention": "Automatic data retention policies",
        "websocket_support": "Real-time bidirectional communication",
        "ml_model_selection": "Choose specific ML models for predictions",
        "async_processing": "Asynchronous task processing with status tracking",
    }
    return descriptions.get(feature, feature.replace("-", " ").title())


def calculate_migration_effort(from_version: str, to_version: str) -> Dict[str, Any]:
    """Calculate estimated effort for migrating between versions."""
    api_config = get_api_config()
    registry = get_api_registry()

    effort = {
        "from_version": from_version,
        "to_version": to_version,
        "complexity": "low",  # low, medium, high
        "estimated_hours": 0,
        "factors": [],
    }

    from_config = api_config.get_version_config(from_version)
    to_config = api_config.get_version_config(to_version)

    if not from_config or not to_config:
        effort["complexity"] = "unknown"
        return effort

    # Factor: Breaking changes
    if to_config.breaking_changes:
        effort["factors"].append(
            {
                "name": "Breaking changes",
                "count": len(to_config.breaking_changes),
                "impact": "high",
                "hours": len(to_config.breaking_changes) * 2,
            }
        )
        effort["estimated_hours"] += len(to_config.breaking_changes) * 2

    # Factor: Endpoint changes
    from_endpoints = registry.get_endpoints_by_version(from_version)
    to_endpoints = registry.get_endpoints_by_version(to_version)

    endpoint_changes = len(set(e.path for e in from_endpoints) - set(e.path for e in to_endpoints))
    if endpoint_changes > 0:
        effort["factors"].append(
            {"name": "Endpoint changes", "count": endpoint_changes, "impact": "medium", "hours": endpoint_changes * 0.5}
        )
        effort["estimated_hours"] += endpoint_changes * 0.5

    # Factor: New features to implement
    new_features = set(to_config.features) - set(from_config.features)
    if new_features:
        effort["factors"].append(
            {"name": "New features", "count": len(new_features), "impact": "medium", "hours": len(new_features) * 1}
        )
        effort["estimated_hours"] += len(new_features) * 1

    # Determine complexity
    if effort["estimated_hours"] < 4:
        effort["complexity"] = "low"
    elif effort["estimated_hours"] < 16:
        effort["complexity"] = "medium"
    else:
        effort["complexity"] = "high"

    return effort
