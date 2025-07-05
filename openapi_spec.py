"""OpenAPI 3.0 specification generator for NightScan API."""

from apispec import APISpec
from apispec.ext.marshmallow import MarshmallowPlugin
from apispec_webframeworks.flask import FlaskPlugin
from flask import Flask, jsonify
import json


def create_api_spec() -> APISpec:
    """Create OpenAPI specification for NightScan API."""
    
    spec = APISpec(
        title="NightScan API",
        version="1.0.0",
        openapi_version="3.0.2",
        info={
            "description": """
NightScan Wildlife Detection API

This API provides endpoints for:
- **Audio Analysis**: Upload WAV files to detect and classify wildlife sounds
- **Detection Management**: Retrieve and filter wildlife detection records  
- **Health Monitoring**: Check service status and readiness

## Authentication
Most endpoints require authentication via session cookies. Use the web interface login or authenticate via the main application.

## Rate Limiting
- Prediction endpoint: 10 requests per minute
- Other endpoints: Various limits apply

## Caching
Prediction results are cached for 1 hour by default to improve performance for identical audio files.
            """,
            "contact": {
                "name": "NightScan Support",
                "url": "https://github.com/GamerXtrem/NightScan"
            },
            "license": {
                "name": "MIT",
                "url": "https://opensource.org/licenses/MIT"
            }
        },
        servers=[
            {
                "url": "https://api.nightscan.example.com",
                "description": "Production server"
            },
            {
                "url": "http://localhost:8001",
                "description": "Development server (API)"
            },
            {
                "url": "http://localhost:8000", 
                "description": "Development server (Web)"
            }
        ],
        plugins=[MarshmallowPlugin(), FlaskPlugin()],
        tags=[
            {
                "name": "Health",
                "description": "Service health and readiness checks"
            },
            {
                "name": "Prediction", 
                "description": "Audio analysis and species prediction"
            },
            {
                "name": "Detections",
                "description": "Wildlife detection records management"
            }
        ]
    )
    
    # Add security schemes
    spec.components.security_scheme(
        "cookieAuth",
        {
            "type": "apiKey",
            "in": "cookie",
            "name": "session",
            "description": "Session cookie authentication"
        }
    )
    
    return spec


def generate_openapi_json(app: Flask) -> dict:
    """Generate OpenAPI JSON specification from Flask app."""
    
    spec = create_api_spec()
    
    # Import schemas to register them
    from api_v1 import (
        PredictionResultSchema, SegmentPredictionSchema, PredictionResponseSchema,
        DetectionSchema, DetectionsResponseSchema, PaginationSchema,
        ErrorSchema, HealthCheckSchema, ReadinessCheckSchema
    )
    
    # Register schemas
    spec.components.schema("PredictionResult", schema=PredictionResultSchema)
    spec.components.schema("SegmentPrediction", schema=SegmentPredictionSchema) 
    spec.components.schema("PredictionResponse", schema=PredictionResponseSchema)
    spec.components.schema("Detection", schema=DetectionSchema)
    spec.components.schema("DetectionsResponse", schema=DetectionsResponseSchema)
    spec.components.schema("Pagination", schema=PaginationSchema)
    spec.components.schema("Error", schema=ErrorSchema)
    spec.components.schema("HealthCheck", schema=HealthCheckSchema)
    spec.components.schema("ReadinessCheck", schema=ReadinessCheckSchema)
    
    # Add paths from Flask app
    with app.app_context():
        # Add API v1 paths
        from api_v1 import api_v1
        
        # Health endpoints
        spec.path(
            path="/api/v1/health",
            operations={
                "get": {
                    "tags": ["Health"],
                    "summary": "Basic health check",
                    "description": "Returns basic health status of the API service",
                    "responses": {
                        "200": {
                            "description": "Service is healthy",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/HealthCheck"}
                                }
                            }
                        }
                    }
                }
            }
        )
        
        spec.path(
            path="/api/v1/ready",
            operations={
                "get": {
                    "tags": ["Health"],
                    "summary": "Comprehensive readiness check", 
                    "description": "Returns detailed readiness status including dependencies",
                    "responses": {
                        "200": {
                            "description": "Service is ready",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/ReadinessCheck"}
                                }
                            }
                        },
                        "503": {
                            "description": "Service is not ready",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/ReadinessCheck"}
                                }
                            }
                        }
                    }
                }
            }
        )
        
        # Prediction endpoint
        spec.path(
            path="/api/v1/predict",
            operations={
                "post": {
                    "tags": ["Prediction"],
                    "summary": "Predict species from audio file",
                    "description": "Upload a WAV audio file and get species predictions",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "multipart/form-data": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "file": {
                                            "type": "string",
                                            "format": "binary",
                                            "description": "WAV audio file to analyze (max 100MB, max 10 minutes)"
                                        }
                                    },
                                    "required": ["file"]
                                }
                            }
                        }
                    },
                    "responses": {
                        "200": {
                            "description": "Prediction successful",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/PredictionResponse"}
                                }
                            }
                        },
                        "400": {
                            "description": "Invalid request (bad file format, too large, etc.)",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/Error"}
                                }
                            }
                        },
                        "429": {
                            "description": "Rate limit exceeded",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/Error"}
                                }
                            }
                        },
                        "500": {
                            "description": "Internal server error",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/Error"}
                                }
                            }
                        }
                    }
                }
            }
        )
        
        # Detections endpoint
        spec.path(
            path="/api/v1/detections",
            operations={
                "get": {
                    "tags": ["Detections"],
                    "summary": "Retrieve paginated list of wildlife detections",
                    "description": "Get a paginated list of wildlife detections with optional filtering",
                    "security": [{"cookieAuth": []}],
                    "parameters": [
                        {
                            "name": "page",
                            "in": "query",
                            "schema": {"type": "integer", "minimum": 1, "default": 1},
                            "description": "Page number"
                        },
                        {
                            "name": "per_page", 
                            "in": "query",
                            "schema": {"type": "integer", "minimum": 1, "maximum": 100, "default": 50},
                            "description": "Number of items per page"
                        },
                        {
                            "name": "species",
                            "in": "query", 
                            "schema": {"type": "string"},
                            "description": "Filter by species name (partial match)"
                        }
                    ],
                    "responses": {
                        "200": {
                            "description": "List of detections",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/DetectionsResponse"}
                                }
                            }
                        },
                        "400": {
                            "description": "Invalid request parameters",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/Error"}
                                }
                            }
                        },
                        "401": {
                            "description": "Authentication required",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/Error"}
                                }
                            }
                        }
                    }
                }
            }
        )
    
    return spec.to_dict()


def create_openapi_endpoint(app: Flask):
    """Create OpenAPI documentation endpoint."""
    
    @app.route("/api/v1/openapi.json")
    def openapi_spec():
        """Return OpenAPI 3.0 specification as JSON."""
        return jsonify(generate_openapi_json(app))
    
    @app.route("/api/v1/docs")
    def api_docs():
        """Serve Swagger UI for API documentation."""
        return f"""
<!DOCTYPE html>
<html>
<head>
    <title>NightScan API Documentation</title>
    <link rel="stylesheet" type="text/css" href="https://unpkg.com/swagger-ui-dist@4.15.5/swagger-ui.css" />
    <style>
        html {{ box-sizing: border-box; overflow: -moz-scrollbars-vertical; overflow-y: scroll; }}
        *, *:before, *:after {{ box-sizing: inherit; }}
        body {{ margin:0; background: #fafafa; }}
    </style>
</head>
<body>
    <div id="swagger-ui"></div>
    <script src="https://unpkg.com/swagger-ui-dist@4.15.5/swagger-ui-bundle.js"></script>
    <script src="https://unpkg.com/swagger-ui-dist@4.15.5/swagger-ui-standalone-preset.js"></script>
    <script>
        window.onload = function() {{
            const ui = SwaggerUIBundle({{
                url: '/api/v1/openapi.json',
                dom_id: '#swagger-ui',
                deepLinking: true,
                presets: [
                    SwaggerUIBundle.presets.apis,
                    SwaggerUIStandalonePreset
                ],
                plugins: [
                    SwaggerUIBundle.plugins.DownloadUrl
                ],
                layout: "StandaloneLayout",
                tryItOutEnabled: true,
                supportedSubmitMethods: ['get', 'post', 'put', 'delete', 'patch'],
                onComplete: function() {{
                    console.log("NightScan API documentation loaded");
                }}
            }});
        }};
    </script>
</body>
</html>
        """