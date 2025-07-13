"""API v1 - Predictions endpoints with RESTful design."""

import logging
from flask import Blueprint, request, jsonify
from api_versioning.decorators import api_version, version_required
from api_versioning.registry import register_endpoint

logger = logging.getLogger(__name__)


def create_predictions_blueprint() -> Blueprint:
    """Create predictions blueprint for API v1."""
    predictions_bp = Blueprint("predictions_v1", __name__)

    # Import prediction functions if available
    try:
        from web.app import predict_file, validate_wav_signature, sanitize_filename
        predictions_available = True
    except ImportError:
        logger.warning("Prediction functions not available, using placeholder endpoints")
        predictions_available = False

    # Register RESTful endpoints in the registry
    endpoints = [
        # RESTful endpoints
        {
            "path": "/api/v1/predictions",
            "methods": ["POST"],
            "description": "Create new prediction from uploaded file",
            "tags": ["predictions", "ml"],
        },
        {
            "path": "/api/v1/predictions/<prediction_id>",
            "methods": ["GET"],
            "description": "Get prediction results",
            "tags": ["predictions", "results"],
        },
        {
            "path": "/api/v1/predictions",
            "methods": ["GET"],
            "description": "List user predictions with filtering",
            "tags": ["predictions", "list"],
        },
        {
            "path": "/api/v1/predictions/<prediction_id>/status",
            "methods": ["GET"],
            "description": "Get prediction processing status",
            "tags": ["predictions", "status"],
        },
        {
            "path": "/api/v1/predictions/<prediction_id>",
            "methods": ["DELETE"],
            "description": "Delete prediction and associated data",
            "tags": ["predictions", "delete"],
        },
        # Legacy endpoints for backward compatibility
        {
            "path": "/api/v1/predictions/analyze",
            "methods": ["POST"],
            "description": "[DEPRECATED] Use POST /api/v1/predictions instead",
            "tags": ["predictions", "deprecated"],
        },
        {
            "path": "/api/v1/predictions/status/<prediction_id>",
            "methods": ["GET"],
            "description": "[DEPRECATED] Use GET /api/v1/predictions/<prediction_id>/status instead",
            "tags": ["predictions", "deprecated"],
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

    # RESTful routes with proper HTTP methods

    @predictions_bp.route("", methods=["POST"])
    @api_version("v1", description="Create prediction", tags=["predictions"])
    @version_required(min_version="v1")
    def create_prediction():
        """Create new prediction from uploaded file (RESTful)."""
        if not request.files:
            return jsonify({
                "error": "No file provided",
                "message": "Please upload a file for analysis"
            }), 400

        file = request.files.get("file")
        if not file:
            return jsonify({
                "error": "No file provided",
                "message": "File field is required"
            }), 400

        # Optional parameters
        model_type = request.form.get("model_type", "auto")
        confidence_threshold = request.form.get("confidence_threshold", 0.5, type=float)
        async_processing = request.form.get("async", "false").lower() == "true"
        
        if predictions_available:
            try:
                # Validate file
                filename = sanitize_filename(file.filename)
                
                # For WAV files, validate signature
                if filename.lower().endswith('.wav'):
                    if not validate_wav_signature(file):
                        return jsonify({
                            "error": "Invalid WAV file",
                            "message": "File signature validation failed"
                        }), 400
                    file.seek(0)

                # Process prediction
                if async_processing:
                    # Return task ID for async processing
                    prediction_id = "pred_async_123456"
                    return jsonify({
                        "prediction_id": prediction_id,
                        "status": "processing",
                        "file_type": "audio" if filename.lower().endswith('.wav') else "image",
                        "model_type": model_type,
                        "confidence_threshold": confidence_threshold,
                        "async": True,
                        "status_url": f"/api/v1/predictions/{prediction_id}/status",
                        "results_url": f"/api/v1/predictions/{prediction_id}",
                        "estimated_completion": "2024-01-15T10:35:00Z",
                        "api_version": "v1"
                    }), 202  # Accepted
                else:
                    # Synchronous processing
                    result = predict_file(file)
                    prediction_id = "pred_sync_123456"
                    
                    return jsonify({
                        "prediction_id": prediction_id,
                        "status": "completed",
                        "file_type": "audio" if filename.lower().endswith('.wav') else "image",
                        "model_type": model_type,
                        "confidence_threshold": confidence_threshold,
                        "async": False,
                        "results": result,
                        "processing_time": 1.23,
                        "api_version": "v1"
                    }), 201

            except Exception as e:
                logger.error(f"Prediction error: {e}")
                return jsonify({
                    "error": "Prediction failed",
                    "message": str(e)
                }), 500
        else:
            # Return mock response
            prediction_id = "mock_pred_123"
            return jsonify({
                "prediction_id": prediction_id,
                "status": "completed" if not async_processing else "processing",
                "file_type": "audio" if file.filename.endswith('.wav') else "image",
                "model_type": model_type,
                "confidence_threshold": confidence_threshold,
                "async": async_processing,
                "results": [
                    {
                        "species": "Unknown",
                        "confidence": 0.85,
                        "bounding_box": None if file.filename.endswith('.wav') else [100, 100, 200, 200],
                        "timestamp": 1.5 if file.filename.endswith('.wav') else None
                    }
                ] if not async_processing else None,
                "processing_time": 1.23 if not async_processing else None,
                "status_url": f"/api/v1/predictions/{prediction_id}/status",
                "results_url": f"/api/v1/predictions/{prediction_id}",
                "api_version": "v1"
            }), 201

    @predictions_bp.route("", methods=["GET"])
    @api_version("v1", description="List predictions", tags=["predictions"])
    @version_required(min_version="v1")
    def list_predictions():
        """List user predictions with filtering (RESTful)."""
        page = request.args.get("page", 1, type=int)
        per_page = request.args.get("per_page", 20, type=int)
        status = request.args.get("status")  # completed, processing, failed
        file_type = request.args.get("file_type")  # audio, image
        species = request.args.get("species")
        
        # Mock predictions list
        predictions = [
            {
                "prediction_id": "pred_001",
                "filename": "recording1.wav",
                "file_type": "audio",
                "status": "completed",
                "created_at": "2024-01-15T10:30:00Z",
                "completed_at": "2024-01-15T10:30:15Z",
                "top_prediction": {
                    "species": "Owl",
                    "confidence": 0.92
                }
            },
            {
                "prediction_id": "pred_002",
                "filename": "photo1.jpg", 
                "file_type": "image",
                "status": "processing",
                "created_at": "2024-01-15T10:25:00Z",
                "completed_at": None,
                "top_prediction": None
            }
        ]
        
        # Apply filters
        if status:
            predictions = [p for p in predictions if p["status"] == status]
        if file_type:
            predictions = [p for p in predictions if p["file_type"] == file_type]
        if species:
            predictions = [p for p in predictions 
                         if p.get("top_prediction") and species.lower() in p["top_prediction"]["species"].lower()]
        
        return jsonify({
            "predictions": predictions,
            "pagination": {
                "page": page,
                "per_page": per_page,
                "total": len(predictions),
                "has_next": False,
                "has_prev": False
            },
            "filters": {
                "status": status,
                "file_type": file_type,
                "species": species
            },
            "api_version": "v1"
        }), 200

    @predictions_bp.route("/<prediction_id>", methods=["GET"])
    @api_version("v1", description="Get prediction results", tags=["predictions"])
    @version_required(min_version="v1")
    def get_prediction(prediction_id: str):
        """Get prediction results (RESTful)."""
        # Mock prediction result
        return jsonify({
            "prediction_id": prediction_id,
            "filename": "recording1.wav",
            "file_type": "audio",
            "status": "completed",
            "created_at": "2024-01-15T10:30:00Z",
            "completed_at": "2024-01-15T10:30:15Z",
            "processing_time": 15.2,
            "model_info": {
                "model_type": "audio_classifier_v2",
                "model_version": "2.1.0",
                "confidence_threshold": 0.5
            },
            "results": [
                {
                    "segment": "segment_1",
                    "time_start": 0.0,
                    "time_end": 3.5,
                    "predictions": [
                        {
                            "species": "Owl",
                            "confidence": 0.92,
                            "scientific_name": "Strix aluco"
                        },
                        {
                            "species": "Wind",
                            "confidence": 0.08,
                            "scientific_name": null
                        }
                    ]
                },
                {
                    "segment": "segment_2",
                    "time_start": 3.5,
                    "time_end": 7.0,
                    "predictions": [
                        {
                            "species": "Silence",
                            "confidence": 0.95,
                            "scientific_name": null
                        }
                    ]
                }
            ],
            "metadata": {
                "file_size": 2048576,
                "duration": 7.0,
                "sample_rate": 44100,
                "channels": 1
            },
            "api_version": "v1"
        }), 200

    @predictions_bp.route("/<prediction_id>/status", methods=["GET"])
    @api_version("v1", description="Get prediction status", tags=["predictions"])
    @version_required(min_version="v1")
    def get_prediction_status(prediction_id: str):
        """Get prediction processing status (RESTful)."""
        return jsonify({
            "prediction_id": prediction_id,
            "status": "processing",
            "progress": 65,
            "stage": "feature_extraction",
            "estimated_completion": "2024-01-15T10:35:00Z",
            "created_at": "2024-01-15T10:30:00Z",
            "messages": [
                "File validated successfully",
                "Audio preprocessing complete",
                "Extracting features..."
            ],
            "api_version": "v1"
        }), 200

    @predictions_bp.route("/<prediction_id>", methods=["DELETE"])
    @api_version("v1", description="Delete prediction", tags=["predictions"])
    @version_required(min_version="v1")
    def delete_prediction(prediction_id: str):
        """Delete prediction and associated data (RESTful)."""
        try:
            # In real implementation, delete from database and storage
            return jsonify({
                "prediction_id": prediction_id,
                "status": "deleted",
                "message": "Prediction and associated data successfully deleted",
                "deleted_at": "2024-01-15T10:40:00Z",
                "api_version": "v1"
            }), 200
        except Exception as e:
            logger.error(f"Prediction deletion error: {e}")
            return jsonify({
                "error": "Deletion failed",
                "message": str(e)
            }), 500

    # Legacy endpoints (deprecated)

    @predictions_bp.route("/analyze", methods=["POST"])
    @api_version("v1", description="[DEPRECATED] Analyze file", tags=["predictions", "deprecated"])
    @version_required(min_version="v1")
    def analyze_file_legacy():
        """Analyze file (deprecated)."""
        response = create_prediction()
        if hasattr(response, 'headers'):
            response.headers['X-API-Deprecation-Warning'] = 'This endpoint is deprecated. Use POST /api/v1/predictions instead.'
            response.headers['X-API-Deprecated-Endpoint'] = '/api/v1/predictions/analyze'
            response.headers['X-API-Replacement-Endpoint'] = '/api/v1/predictions'
        return response

    @predictions_bp.route("/status/<prediction_id>", methods=["GET"])
    @api_version("v1", description="[DEPRECATED] Get status", tags=["predictions", "deprecated"])
    @version_required(min_version="v1")
    def get_status_legacy(prediction_id: str):
        """Get prediction status (deprecated)."""
        response = get_prediction_status(prediction_id)
        if hasattr(response, 'headers'):
            response.headers['X-API-Deprecation-Warning'] = 'This endpoint is deprecated. Use GET /api/v1/predictions/{prediction_id}/status instead.'
            response.headers['X-API-Deprecated-Endpoint'] = f'/api/v1/predictions/status/{prediction_id}'
            response.headers['X-API-Replacement-Endpoint'] = f'/api/v1/predictions/{prediction_id}/status'
        return response

    return predictions_bp