"""API v1 - Analytics endpoints with versioning."""

import logging

from flask import Blueprint, jsonify, request

from api_versioning.decorators import api_version, rate_limit_by_version, version_required
from api_versioning.registry import register_endpoint

logger = logging.getLogger(__name__)


def create_analytics_blueprint() -> Blueprint:
    """Create analytics blueprint for API v1."""
    analytics_bp = Blueprint("analytics_v1", __name__)

    # Import original analytics functions if available
    try:
        from analytics_dashboard import (
            export_csv,
            export_pdf,
            get_detection_metrics,
            get_species_statistics,
            get_zone_analytics,
        )

        analytics_available = True
    except ImportError:
        logger.warning("Analytics dashboard module not available, using placeholder endpoints")
        analytics_available = False

    # Register RESTful endpoints in the registry
    endpoints = [
        # RESTful endpoints
        {
            "path": "/api/v1/analytics/metrics",
            "methods": ["GET"],
            "description": "Get detection metrics and statistics",
            "tags": ["analytics", "metrics"],
        },
        {
            "path": "/api/v1/analytics/species",
            "methods": ["GET"],
            "description": "Get species-specific analytics",
            "tags": ["analytics", "species"],
        },
        {
            "path": "/api/v1/analytics/species/<species>",
            "methods": ["GET"],
            "description": "Get analytics for specific species",
            "tags": ["analytics", "species"],
        },
        {
            "path": "/api/v1/analytics/zones",
            "methods": ["GET"],
            "description": "Get zone-based analytics",
            "tags": ["analytics", "zones"],
        },
        {
            "path": "/api/v1/analytics/exports",
            "methods": ["GET"],
            "description": "Export analytics data (use ?format=csv or ?format=pdf)",
            "tags": ["analytics", "exports"],
        },
        {
            "path": "/api/v1/analytics/dashboard",
            "methods": ["GET"],
            "description": "Get dashboard summary data",
            "tags": ["analytics", "dashboard"],
        },
    ]

    for endpoint in endpoints:
        register_endpoint(
            path=endpoint["path"],
            methods=endpoint["methods"],
            version="v1",
            description=endpoint["description"],
            tags=endpoint["tags"],
            requires_auth=True,
        )

    # Routes with versioning decorators

    @analytics_bp.route("/metrics", methods=["GET"])
    @api_version("v1", description="Detection metrics", tags=["analytics"])
    @rate_limit_by_version(default_limit=100, version_limits={"v1": {"limit": 100, "window": 3600}})
    def get_metrics_v1():
        """Get detection metrics for API v1."""
        # Parse query parameters
        days = request.args.get("days", 30, type=int)
        start_date = request.args.get("start_date")
        end_date = request.args.get("end_date")

        if analytics_available:
            # Use real analytics
            try:
                metrics = get_detection_metrics(days=days, start_date=start_date, end_date=end_date)
                return jsonify(metrics), 200
            except Exception as e:
                logger.error(f"Error getting metrics: {e}")
                return jsonify({"error": "Failed to retrieve metrics"}), 500
        else:
            # Return placeholder data
            return (
                jsonify(
                    {
                        "total_detections": 1234,
                        "unique_species": 15,
                        "active_zones": 8,
                        "detection_trend": "increasing",
                        "period": f"{days} days",
                        "top_species": [
                            {"name": "Deer", "count": 234},
                            {"name": "Fox", "count": 189},
                            {"name": "Raccoon", "count": 156},
                        ],
                    }
                ),
                200,
            )

    @analytics_bp.route("/species", methods=["GET"])
    @api_version("v1", description="Species statistics", tags=["analytics"])
    def get_species_stats_v1():
        """Get species statistics for API v1."""
        if analytics_available:
            try:
                stats = get_species_statistics()
                return jsonify(stats), 200
            except Exception as e:
                logger.error(f"Error getting species stats: {e}")
                return jsonify({"error": "Failed to retrieve species statistics"}), 500
        else:
            return (
                jsonify(
                    {
                        "species": [
                            {"name": "Deer", "total": 234, "percentage": 18.9},
                            {"name": "Fox", "total": 189, "percentage": 15.3},
                            {"name": "Raccoon", "total": 156, "percentage": 12.6},
                            {"name": "Owl", "total": 123, "percentage": 10.0},
                        ],
                        "total_species": 15,
                    }
                ),
                200,
            )

    @analytics_bp.route("/species/<string:species>", methods=["GET"])
    @api_version("v1", description="Specific species analytics", tags=["analytics"])
    def get_species_detail_v1(species: str):
        """Get detailed analytics for a specific species."""
        days = request.args.get("days", 30, type=int)

        if analytics_available:
            try:
                details = get_species_statistics(species=species, days=days)
                return jsonify(details), 200
            except Exception as e:
                logger.error(f"Error getting species details: {e}")
                return jsonify({"error": f"Failed to retrieve data for {species}"}), 500
        else:
            return (
                jsonify(
                    {
                        "species": species,
                        "total_detections": 234,
                        "detection_rate": 7.8,  # per day
                        "peak_activity_time": "21:00-23:00",
                        "preferred_zones": ["Zone A", "Zone C"],
                        "trend": "stable",
                        "period": f"{days} days",
                    }
                ),
                200,
            )

    @analytics_bp.route("/zones", methods=["GET"])
    @api_version("v1", description="Zone analytics", tags=["analytics"])
    def get_zones_v1():
        """Get zone-based analytics for API v1."""
        if analytics_available:
            try:
                zones = get_zone_analytics()
                return jsonify(zones), 200
            except Exception as e:
                logger.error(f"Error getting zone analytics: {e}")
                return jsonify({"error": "Failed to retrieve zone data"}), 500
        else:
            return (
                jsonify(
                    {
                        "zones": [
                            {"name": "Zone A", "detections": 456, "species_count": 8},
                            {"name": "Zone B", "detections": 324, "species_count": 6},
                            {"name": "Zone C", "detections": 287, "species_count": 7},
                        ],
                        "total_zones": 8,
                        "most_active": "Zone A",
                    }
                ),
                200,
            )

    # RESTful export endpoint 
    @analytics_bp.route("/exports", methods=["GET"])
    @api_version("v1", description="Export analytics data", tags=["analytics", "exports"])
    @version_required(min_version="v1", features=["data_export"])
    def get_exports_v1():
        """Export analytics data in various formats (RESTful)."""
        format_type = request.args.get("format", "json").lower()
        
        if format_type == "csv":
            if analytics_available:
                try:
                    return export_csv()
                except Exception as e:
                    logger.error(f"Error exporting CSV: {e}")
                    return jsonify({"error": "Failed to export CSV"}), 500
            else:
                # Return sample CSV
                csv_data = "Date,Species,Zone,Count\n"
                csv_data += "2024-01-01,Deer,Zone A,5\n"
                csv_data += "2024-01-01,Fox,Zone B,3\n"
                return (
                    csv_data,
                    200,
                    {"Content-Type": "text/csv", "Content-Disposition": "attachment; filename=analytics_export.csv"},
                )
        
        elif format_type == "pdf":
            if analytics_available:
                try:
                    return export_pdf()
                except Exception as e:
                    logger.error(f"Error exporting PDF: {e}")
                    return jsonify({"error": "Failed to export PDF"}), 500
            else:
                return (
                    jsonify(
                        {"error": "PDF export not available", "message": "PDF export functionality is being implemented"}
                    ),
                    501,
                )
        
        elif format_type == "json":
            # Return analytics data as JSON
            if analytics_available:
                try:
                    metrics = get_detection_metrics()
                    species = get_species_statistics()
                    zones = get_zone_analytics()
                    
                    return jsonify({
                        "export_format": "json",
                        "timestamp": "2024-01-15T10:30:00Z",
                        "metrics": metrics,
                        "species": species,
                        "zones": zones
                    }), 200
                except Exception as e:
                    logger.error(f"Error exporting JSON: {e}")
                    return jsonify({"error": "Failed to export data"}), 500
            else:
                return jsonify({
                    "export_format": "json",
                    "timestamp": "2024-01-15T10:30:00Z",
                    "metrics": {"total_detections": 1234, "unique_species": 15},
                    "species": [{"name": "Deer", "count": 234}],
                    "zones": [{"name": "Zone A", "detections": 456}]
                }), 200
        
        else:
            return jsonify({
                "error": "Invalid format",
                "message": "Supported formats: csv, pdf, json",
                "supported_formats": ["csv", "pdf", "json"]
            }), 400


    @analytics_bp.route("/dashboard", methods=["GET"])
    @api_version("v1", description="Dashboard summary", tags=["analytics", "dashboard"])
    def get_dashboard_v1():
        """Get dashboard summary data."""
        return (
            jsonify(
                {
                    "summary": {
                        "total_detections_today": 45,
                        "total_detections_week": 312,
                        "total_detections_month": 1234,
                        "active_cameras": 5,
                        "storage_used_gb": 234.5,
                        "ai_accuracy": 94.2,
                    },
                    "recent_detections": [
                        {
                            "id": 1,
                            "species": "Deer",
                            "confidence": 0.95,
                            "timestamp": "2024-01-15T08:30:00Z",
                            "zone": "Zone A",
                        }
                    ],
                    "alerts": [],
                }
            ),
            200,
        )

    return analytics_bp
