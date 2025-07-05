"""NightScan API v1 with OpenAPI documentation."""

from flask import Blueprint, request, jsonify, current_app
from flask_login import login_required, current_user
from marshmallow import Schema, fields, ValidationError, validate, post_load
from marshmallow_dataclass import dataclass
from typing import List, Optional, Dict, Any
import time
import logging
from datetime import datetime

from metrics import track_request_metrics, record_prediction_metrics
from log_utils import log_prediction
from cache_utils import get_cache
from websocket_service import get_websocket_manager
from push_notifications import get_push_service

# Create API v1 blueprint
api_v1 = Blueprint('api_v1', __name__, url_prefix='/api/v1')

logger = logging.getLogger(__name__)


# ===== SCHEMAS =====

class PredictionResultSchema(Schema):
    """Schema for a single prediction result."""
    label = fields.Str(required=True, description="Predicted species/class")
    probability = fields.Float(required=True, validate=validate.Range(min=0, max=1), 
                              description="Confidence score (0-1)")


class SegmentPredictionSchema(Schema):
    """Schema for prediction results of an audio segment."""
    segment = fields.Str(required=True, description="Segment identifier")
    time = fields.Float(required=True, description="Time offset in seconds")
    predictions = fields.List(fields.Nested(PredictionResultSchema), required=True,
                             description="Top 3 predictions for this segment")


class PredictionResponseSchema(Schema):
    """Schema for the complete prediction API response."""
    results = fields.List(fields.Nested(SegmentPredictionSchema), required=True)
    processing_time = fields.Float(required=True, description="Processing duration in seconds")
    file_info = fields.Dict(description="Information about the processed file")


class DetectionSchema(Schema):
    """Schema for wildlife detection."""
    id = fields.Int(required=True, description="Detection ID")
    species = fields.Str(required=True, description="Detected species")
    time = fields.DateTime(required=True, description="Detection timestamp")
    latitude = fields.Float(allow_none=True, description="GPS latitude")
    longitude = fields.Float(allow_none=True, description="GPS longitude") 
    zone = fields.Str(allow_none=True, description="Detection zone/location")
    image = fields.Str(allow_none=True, description="Associated image URL")


class PaginationSchema(Schema):
    """Schema for pagination metadata."""
    page = fields.Int(required=True, description="Current page number")
    per_page = fields.Int(required=True, description="Items per page")
    total = fields.Int(required=True, description="Total number of items")
    pages = fields.Int(required=True, description="Total number of pages")
    has_next = fields.Bool(required=True, description="Whether there is a next page")
    has_prev = fields.Bool(required=True, description="Whether there is a previous page")


class DetectionsResponseSchema(Schema):
    """Schema for detections list API response."""
    detections = fields.List(fields.Nested(DetectionSchema), required=True)
    pagination = fields.Nested(PaginationSchema, required=True)


class ErrorSchema(Schema):
    """Schema for API error responses."""
    error = fields.Str(required=True, description="Error message")
    code = fields.Str(allow_none=True, description="Error code")
    details = fields.Dict(allow_none=True, description="Additional error details")


class HealthCheckSchema(Schema):
    """Schema for health check response."""
    status = fields.Str(required=True, validate=validate.OneOf(['healthy', 'unhealthy']))
    timestamp = fields.DateTime(required=True)
    version = fields.Str(required=True)
    service = fields.Str(allow_none=True)


class ReadinessCheckSchema(Schema):
    """Schema for readiness check response."""
    status = fields.Str(required=True, validate=validate.OneOf(['ready', 'not_ready']))
    checks = fields.Dict(required=True, description="Individual health checks")
    cache = fields.Dict(allow_none=True, description="Cache status information")
    timestamp = fields.DateTime(required=True)


# ===== ROUTES =====

@api_v1.route('/health', methods=['GET'])
@track_request_metrics
def health_check():
    """
    Health Check Endpoint
    ---
    tags:
      - Health
    summary: Basic health check
    description: Returns basic health status of the API service
    responses:
      200:
        description: Service is healthy
        content:
          application/json:
            schema: HealthCheckSchema
    """
    response_data = {
        "status": "healthy",
        "timestamp": datetime.utcnow(),
        "version": "1.0.0",
        "service": "nightscan-api"
    }
    
    # Validate response
    schema = HealthCheckSchema()
    try:
        result = schema.dump(response_data)
        return jsonify(result), 200
    except ValidationError as err:
        return jsonify({"error": "Response validation failed", "details": err.messages}), 500


@api_v1.route('/ready', methods=['GET'])
@track_request_metrics  
def readiness_check():
    """
    Readiness Check Endpoint
    ---
    tags:
      - Health
    summary: Comprehensive readiness check
    description: Returns detailed readiness status including dependencies
    responses:
      200:
        description: Service is ready
        content:
          application/json:
            schema: ReadinessCheckSchema
      503:
        description: Service is not ready
        content:
          application/json:
            schema: ReadinessCheckSchema
    """
    checks = {
        "model_loaded": False,
        "cache": False
    }
    
    # Check if model is loaded (for prediction API)
    try:
        model = current_app.config.get("MODEL")
        labels = current_app.config.get("LABELS")
        checks["model_loaded"] = model is not None and labels is not None
    except Exception:
        checks["model_loaded"] = False
    
    # Check cache status
    from cache_utils import cache_health_check
    cache_status = cache_health_check()
    checks["cache"] = cache_status["status"] in ["healthy", "disabled"]
    
    is_ready = all(checks.values())
    status_code = 200 if is_ready else 503
    
    response_data = {
        "status": "ready" if is_ready else "not_ready",
        "checks": checks,
        "cache": cache_status,
        "timestamp": datetime.utcnow()
    }
    
    # Validate response
    schema = ReadinessCheckSchema()
    try:
        result = schema.dump(response_data)
        return jsonify(result), status_code
    except ValidationError as err:
        return jsonify({"error": "Response validation failed", "details": err.messages}), 500


@api_v1.route('/predict', methods=['POST'])
@track_request_metrics
def predict_audio():
    """
    Audio Prediction Endpoint
    ---
    tags:
      - Prediction
    summary: Predict species from audio file
    description: Upload a WAV audio file and get species predictions
    requestBody:
      required: true
      content:
        multipart/form-data:
          schema:
            type: object
            properties:
              file:
                type: string
                format: binary
                description: WAV audio file to analyze
    responses:
      200:
        description: Prediction successful
        content:
          application/json:
            schema: PredictionResponseSchema
      400:
        description: Invalid request
        content:
          application/json:
            schema: ErrorSchema
      500:
        description: Internal server error
        content:
          application/json:
            schema: ErrorSchema
    """
    # Import here to avoid circular imports
    from Audio_Training.scripts.api_server import (
        sanitize_filename, validate_wav_signature, predict_file, MAX_FILE_SIZE
    )
    import tempfile
    from pathlib import Path
    from pydub import AudioSegment
    from pydub.exceptions import CouldntDecodeError
    
    start_time = time.time()
    
    # Validate file upload
    file = request.files.get("file")
    if not file or not file.filename:
        return jsonify({"error": "No file uploaded", "code": "NO_FILE"}), 400
    
    # Sanitize filename
    sanitized_filename = sanitize_filename(file.filename)
    
    if not sanitized_filename.lower().endswith(".wav"):
        return jsonify({"error": "WAV file required", "code": "INVALID_FORMAT"}), 400
    
    if file.mimetype not in ("audio/wav", "audio/x-wav"):
        return jsonify({"error": "Invalid content type", "code": "INVALID_MIME"}), 400
    
    if request.content_length is not None and request.content_length > MAX_FILE_SIZE:
        return jsonify({"error": "File exceeds 100 MB limit", "code": "FILE_TOO_LARGE"}), 400
    
    with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
        total = 0
        chunk_size = 64 * 1024
        while True:
            chunk = file.stream.read(chunk_size)
            if not chunk:
                break
            tmp.write(chunk)
            total += len(chunk)
            if total > MAX_FILE_SIZE:
                return jsonify({"error": "File exceeds 100 MB limit", "code": "FILE_TOO_LARGE"}), 400
        
        tmp.flush()
        
        # Read file data for caching
        with open(tmp.name, 'rb') as f:
            audio_data = f.read()
        
        # Check cache first
        cache = get_cache()
        cached_result = cache.get_prediction(audio_data)
        if cached_result is not None:
            processing_time = time.time() - start_time
            response_data = {
                "results": cached_result,
                "processing_time": processing_time,
                "file_info": {
                    "filename": sanitized_filename,
                    "size_bytes": total,
                    "cached": True
                }
            }
            
            # Validate response
            schema = PredictionResponseSchema()
            try:
                result = schema.dump(response_data)
                return jsonify(result), 200
            except ValidationError as err:
                return jsonify({"error": "Response validation failed", "details": err.messages}), 500
        
        # Validate WAV signature
        with open(tmp.name, 'rb') as wav_file:
            if not validate_wav_signature(wav_file):
                return jsonify({"error": "Invalid WAV file format", "code": "INVALID_WAV"}), 400
        
        try:
            # Additional validation with pydub
            audio = AudioSegment.from_file(tmp.name)
            if len(audio) > 600000:  # 10 minutes
                return jsonify({"error": "Audio file too long (max 10 minutes)", "code": "AUDIO_TOO_LONG"}), 400
            
            audio_duration = len(audio) / 1000.0
            
        except CouldntDecodeError:
            return jsonify({"error": "Invalid WAV file", "code": "DECODE_ERROR"}), 400
        except Exception as exc:
            return jsonify({"error": "Failed to process file", "code": "PROCESSING_ERROR"}), 500
        
        # Run prediction
        try:
            results = predict_file(
                Path(tmp.name),
                model=current_app.config["MODEL"],
                labels=current_app.config["LABELS"],
                device=current_app.config["DEVICE"],
                batch_size=current_app.config["BATCH_SIZE"],
            )
            
            processing_time = time.time() - start_time
            
            # Log metrics
            log_prediction(
                filename=sanitized_filename,
                duration=processing_time,
                result_count=len(results),
                file_size=total,
                audio_duration=audio_duration
            )
            
            record_prediction_metrics(
                duration=processing_time,
                success=True,
                file_size=total,
                audio_duration=audio_duration
            )
            
            # Cache the result
            cache.cache_prediction(audio_data, results)
            
            # Send real-time notifications
            try:
                websocket_manager = get_websocket_manager()
                push_service = get_push_service()
                
                notification_data = {
                    "filename": sanitized_filename,
                    "results": results[:3] if results else [],  # Top 3 results
                    "processing_time": processing_time,
                    "file_size": total,
                    "timestamp": datetime.utcnow().isoformat()
                }
                
                # WebSocket notification (sync version)
                websocket_manager.notify_prediction_complete_sync(notification_data)
                
                # Push notification (if user is authenticated)
                # In a real implementation, you'd get user_id from the request context
                # push_service.send_prediction_complete_notification_sync(user_id, notification_data)
                
            except Exception as e:
                # Don't fail the request if notifications fail
                logger.warning(f"Failed to send notifications: {e}")
            
            response_data = {
                "results": results,
                "processing_time": processing_time,
                "file_info": {
                    "filename": sanitized_filename,
                    "size_bytes": total,
                    "duration_seconds": audio_duration,
                    "cached": False
                }
            }
            
            # Validate response
            schema = PredictionResponseSchema()
            try:
                result = schema.dump(response_data)
                return jsonify(result), 200
            except ValidationError as err:
                return jsonify({"error": "Response validation failed", "details": err.messages}), 500
            
        except Exception as exc:
            processing_time = time.time() - start_time
            record_prediction_metrics(duration=processing_time, success=False)
            return jsonify({"error": "Prediction failed", "code": "PREDICTION_ERROR"}), 500


@api_v1.route('/detections', methods=['GET'])
@login_required
@track_request_metrics
def get_detections():
    """
    Get Wildlife Detections
    ---
    tags:
      - Detections
    summary: Retrieve paginated list of wildlife detections
    description: Get a paginated list of wildlife detections with optional filtering
    security:
      - cookieAuth: []
    parameters:
      - in: query
        name: page
        schema:
          type: integer
          minimum: 1
          default: 1
        description: Page number
      - in: query
        name: per_page
        schema:
          type: integer
          minimum: 1
          maximum: 100
          default: 50
        description: Number of items per page
      - in: query
        name: species
        schema:
          type: string
        description: Filter by species name
    responses:
      200:
        description: List of detections
        content:
          application/json:
            schema: DetectionsResponseSchema
      400:
        description: Invalid request parameters
        content:
          application/json:
            schema: ErrorSchema
    """
    # Validate query parameters
    try:
        page = int(request.args.get('page', 1))
        per_page = min(int(request.args.get('per_page', 50)), 100)
        species_filter = request.args.get('species')
        
        if page < 1 or per_page < 1:
            return jsonify({"error": "Invalid pagination parameters", "code": "INVALID_PARAMS"}), 400
            
    except ValueError:
        return jsonify({"error": "Invalid parameter format", "code": "INVALID_FORMAT"}), 400
    
    # Import here to avoid circular imports
    from web.app import Detection
    
    # Build query
    query = Detection.query
    if species_filter:
        query = query.filter(Detection.species.ilike(f"%{species_filter}%"))
    
    query = query.order_by(Detection.time.desc())
    
    # Paginate
    detections = query.paginate(
        page=page, per_page=per_page, error_out=False
    )
    
    response_data = {
        "detections": [d.to_dict() for d in detections.items],
        "pagination": {
            "page": page,
            "per_page": per_page,
            "total": detections.total,
            "pages": detections.pages,
            "has_next": detections.has_next,
            "has_prev": detections.has_prev
        }
    }
    
    # Validate response
    schema = DetectionsResponseSchema()
    try:
        result = schema.dump(response_data)
        return jsonify(result), 200
    except ValidationError as err:
        return jsonify({"error": "Response validation failed", "details": err.messages}), 500


# ===== ERROR HANDLERS =====

@api_v1.errorhandler(ValidationError)
def handle_validation_error(error):
    """Handle marshmallow validation errors."""
    return jsonify({
        "error": "Validation failed",
        "code": "VALIDATION_ERROR",
        "details": error.messages
    }), 400


@api_v1.errorhandler(404)
def handle_not_found(error):
    """Handle 404 errors."""
    return jsonify({
        "error": "Endpoint not found",
        "code": "NOT_FOUND"
    }), 404


@api_v1.errorhandler(405)
def handle_method_not_allowed(error):
    """Handle 405 errors."""
    return jsonify({
        "error": "Method not allowed",
        "code": "METHOD_NOT_ALLOWED"
    }), 405


@api_v1.errorhandler(500)
def handle_internal_error(error):
    """Handle 500 errors."""
    return jsonify({
        "error": "Internal server error",
        "code": "INTERNAL_ERROR"
    }), 500