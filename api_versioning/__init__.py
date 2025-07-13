"""API Versioning System for NightScan.

This module provides centralized API versioning management including:
- Version configuration and registry
- Middleware for backward compatibility
- Automatic route deprecation handling
- Version-specific feature flags
"""

from .config import API_VERSIONS, APIVersionConfig, get_api_config
from .decorators import api_version, deprecated_route, version_required
from .middleware import APIVersioningMiddleware, add_versioning_middleware
from .registry import APIRegistry, get_api_registry

__all__ = [
    "APIVersionConfig",
    "get_api_config",
    "API_VERSIONS",
    "APIVersioningMiddleware",
    "add_versioning_middleware",
    "APIRegistry",
    "get_api_registry",
    "deprecated_route",
    "version_required",
    "api_version",
]

__version__ = "1.0.0"
