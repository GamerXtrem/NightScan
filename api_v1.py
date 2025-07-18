"""NightScan API v1 with OpenAPI documentation."""

from flask import Blueprint, request, jsonify, current_app
from flask_login import login_required, current_user
from marshmallow import Schema, fields, ValidationError, validate, post_load
from marshmallow_dataclass import dataclass
from typing import List, Optional, Dict, Any
import time
import logging
import os
from datetime import datetime

from metrics import track_request_metrics, record_prediction_metrics
from log_utils import log_prediction
from cache_utils import get_cache
from websocket_service import get_websocket_manager
from push_notifications import get_push_service
from cache_middleware import cache_for_user_data, cache_for_static_api
from exceptions import (
    DatabaseError, RecordNotFoundError, DuplicateRecordError,
    QuotaExceededError, RateLimitExceededError, PredictionError,
    ModelNotAvailableError, PredictionFailedError, ValidationError as CustomValidationError,
    ExternalServiceError, CacheServiceError, NotificationServiceError,
    convert_exception, CircuitBreakerOpenException
)

# Circuit breaker imports - using centralized configuration
from circuit_breaker_config import (
    get_database_circuit_breaker, get_cache_circuit_breaker,
    get_http_circuit_breaker, get_ml_circuit_breaker
)

# Create API v1 blueprint
api_v1 = Blueprint('api_v1', __name__, url_prefix='/api/v1')

logger = logging.getLogger(__name__)

def get_or_create_api_circuit_breakers():
    """Get circuit breakers for API operations using centralized configuration."""
    # Get circuit breakers from centralized manager
    db_circuit = get_database_circuit_breaker("api_v1")
    cache_circuit = get_cache_circuit_breaker("api_v1")
    ml_circuit = get_ml_circuit_breaker("api_v1", model_path=os.environ.get("MODEL_PATH"))
    notification_circuit = get_http_circuit_breaker("api_v1", service_type="notification")
    
    return db_circuit, cache_circuit, ml_circuit, notification_circuit


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
    except (AttributeError, KeyError) as e:
        logger.warning(f"Model configuration check failed: {e}")
        checks["model_loaded"] = False
    except Exception as e:
        logger.error(f"Unexpected error checking model configuration: {e}")
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
@login_required
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
    from audio_training.scripts.api_server import (
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
    
    # Check user quota before processing
    from quota_manager import get_quota_manager
    quota_manager = get_quota_manager()
    
    file_size_bytes = request.content_length or 0
    quota_check = quota_manager.check_quota_before_prediction(current_user.id, file_size_bytes)
    
    if not quota_check['allowed']:
        error_response = {
            "error": quota_check['message'], 
            "code": quota_check['reason'].upper(),
            "quota_status": {
                "current_usage": quota_check.get('current_usage', 0),
                "monthly_quota": quota_check.get('monthly_quota', 0),
                "plan_type": quota_check.get('plan_type', 'unknown')
            }
        }
        
        # Add upgrade information for quota exceeded
        if quota_check['reason'] == 'quota_exceeded':
            error_response["upgrade_required"] = True
            error_response["recommended_plan"] = quota_check.get('recommended_plan')
        
        return jsonify(error_response), 429  # Too Many Requests
    
    with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
        total = 0
        chunk_size = 64 * 1024
        
        # Calculate hash while streaming
        import hashlib
        hasher = hashlib.sha256()
        
        while True:
            chunk = file.stream.read(chunk_size)
            if not chunk:
                break
            tmp.write(chunk)
            hasher.update(chunk)  # Update hash while streaming
            total += len(chunk)
            if total > MAX_FILE_SIZE:
                return jsonify({"error": "File exceeds 100 MB limit", "code": "FILE_TOO_LARGE"}), 400
        
        tmp.flush()
        
        # Get file hash without reloading
        file_hash = hasher.hexdigest()
        
        # Protected cache operations with circuit breaker
        db_circuit, cache_circuit, ml_circuit, notification_circuit = get_or_create_api_circuit_breakers()
        
        cached_result = None
        try:
            def get_cached_prediction():
                cache = get_cache()
                return cache.get_prediction_by_hash(file_hash)
            
            cached_result = cache_circuit.call(get_cached_prediction)
            logger.debug(f"Cache lookup successful for file hash {file_hash[:8]}...")
            
        except CircuitBreakerOpenException as e:
            logger.warning(f"Cache circuit breaker open for prediction cache lookup: {e}")
            cached_result = None
        except CacheServiceError as e:
            logger.warning(f"Cache service error for prediction lookup: {e}")
            cached_result = None
        except Exception as e:
            logger.error(f"Unexpected cache error for prediction lookup: {e}")
            cached_result = None
            
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
            
            # Protected quota consumption with circuit breaker
            try:
                def consume_quota_for_cached():
                    return quota_manager.consume_quota(current_user.id, total)
                
                quota_result = db_circuit.call(consume_quota_for_cached)
                if quota_result.get('allowed'):
                    response_data["quota_status"] = {
                        "current_usage": quota_result.get('current_usage'),
                        "monthly_quota": quota_result.get('monthly_quota'),
                        "remaining": quota_result.get('remaining'),
                        "plan_type": quota_result.get('plan_type')
                    }
                    logger.debug(f"Quota consumed successfully for cached result - user {current_user.id}")
                    
            except CircuitBreakerOpenException as e:
                logger.warning(f"Database circuit breaker open for quota consumption - user {current_user.id}: {e}")
                # Continue with cached result even if quota check fails
            except QuotaExceededError as e:
                logger.warning(f"Quota exceeded for cached result - user {current_user.id}: {e}")
                return jsonify(e.to_dict()), 429
            except DatabaseError as e:
                logger.error(f"Database error consuming quota for cached result - user {current_user.id}: {e}")
                # Continue with cached result even if quota check fails
            except Exception as e:
                logger.error(f"Unexpected error consuming quota for cached result - user {current_user.id}: {e}")
                # Continue with cached result even if quota check fails
            
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
        except (OSError, IOError, PermissionError) as e:
            logger.error(f"File system error processing audio file: {e}")
            return jsonify({"error": "Unable to read audio file", "code": "FILE_READ_ERROR"}), 400
        except Exception as exc:
            logger.error(f"Unexpected error processing audio file: {exc}")
            error = convert_exception(exc, operation='audio_processing', file_name=filename)
            return jsonify(error.to_dict()), 500
        
        # Check if async mode is enabled
        async_mode = current_app.config.get("ASYNC_PREDICTIONS", True)
        
        if async_mode:
            # Queue the prediction task
            from prediction_queue import get_prediction_queue
            from web.tasks import process_prediction
            
            queue = get_prediction_queue()
            
            # Create task in queue
            task_id = queue.create_task(
                file_path=tmp.name,
                file_hash=file_hash,
                filename=sanitized_filename,
                user_id=getattr(current_user, 'id', None) if 'current_user' in globals() else None,
                priority=0  # Default priority
            )
            
            # Dispatch to Celery worker
            process_prediction.delay(task_id)
            
            # Return immediately with task ID
            return jsonify({
                "task_id": task_id,
                "status": "processing",
                "status_url": f"/api/v1/status/{task_id}",
                "result_url": f"/api/v1/result/{task_id}",
                "message": "Prediction queued for processing"
            }), 202  # 202 Accepted
            
        else:
            # Protected ML prediction with circuit breaker (synchronous mode)
            try:
                def run_ml_prediction_api():
                    from audio_training.scripts.api_server import predict_file
                    return predict_file(
                        Path(tmp.name),
                        model=current_app.config["MODEL"],
                        labels=current_app.config["LABELS"],
                        device=current_app.config["DEVICE"],
                        batch_size=current_app.config["BATCH_SIZE"],
                    )
                
                results = ml_circuit.call(run_ml_prediction_api)
                logger.info(f"ML prediction successful for {sanitized_filename}")
                
            except CircuitBreakerOpenException as e:
                logger.error(f"ML circuit breaker open for {sanitized_filename}: {e}")
                # Return fallback response
                return jsonify({
                    "error": "ML service temporarily unavailable",
                    "code": "ML_SERVICE_UNAVAILABLE",
                    "details": "The machine learning service is currently experiencing issues. Please try again later.",
                    "circuit_open": True
                }), 503
            except PredictionError as e:
                logger.error(f"ML prediction error for {sanitized_filename}: {e}")
                return jsonify({
                    "error": "Prediction processing failed",
                    "code": "PREDICTION_FAILED",
                    "details": str(e)
                }), 500
            except Exception as e:
                logger.error(f"Unexpected ML error for {sanitized_filename}: {e}")
                error = convert_exception(e, operation='ml_prediction', file_name=sanitized_filename)
                return jsonify(error.to_dict()), 500
            
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
            
            # Protected cache storage with circuit breaker
            try:
                def cache_prediction_result():
                    cache = get_cache()
                    cache.cache_prediction_by_hash(file_hash, results)
                    return True
                
                cache_circuit.call(cache_prediction_result)
                logger.debug(f"Cached prediction result for {sanitized_filename}")
                
            except CircuitBreakerOpenException as e:
                logger.warning(f"Cache circuit breaker open for storing result {sanitized_filename}: {e}")
            except CacheServiceError as e:
                logger.warning(f"Cache service error for storing result {sanitized_filename}: {e}")
            except Exception as e:
                logger.error(f"Unexpected cache error for storing result {sanitized_filename}: {e}")
            
            # Protected notification sending with circuit breaker
            try:
                def send_prediction_notifications():
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
                    return True
                
                notification_circuit.call(send_prediction_notifications)
                logger.info(f"Notifications sent successfully for {sanitized_filename}")
                
            except CircuitBreakerOpenException as e:
                # Don't fail the request if notification circuit is open
                logger.warning(f"Notification circuit breaker open for {sanitized_filename}: {e}")
            except NotificationServiceError as e:
                # Don't fail the request if notifications fail
                logger.warning(f"Notification service failed for {sanitized_filename}: {e}")
            except ExternalServiceError as e:
                # Don't fail the request if external services fail
                logger.warning(f"External service failed during notification for {sanitized_filename}: {e}")
            except Exception as e:
                # Don't fail the request if notifications fail
                logger.warning(f"Unexpected error sending notifications for {sanitized_filename}: {e}")
            
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
            
            # Protected quota consumption after successful prediction
            try:
                def consume_quota_after_prediction():
                    return quota_manager.consume_quota(current_user.id, total)
                
                quota_result = db_circuit.call(consume_quota_after_prediction)
                if quota_result.get('allowed'):
                    logger.info(f"Quota consumed for user {current_user.id}: {quota_result}")
                    # Add quota info to response
                    response_data["quota_status"] = {
                        "current_usage": quota_result.get('current_usage'),
                        "monthly_quota": quota_result.get('monthly_quota'),
                        "remaining": quota_result.get('remaining'),
                        "plan_type": quota_result.get('plan_type')
                    }
                else:
                    logger.warning(f"Quota consumption failed for user {current_user.id}: {quota_result}")
                    
            except CircuitBreakerOpenException as e:
                logger.warning(f"Database circuit breaker open for quota consumption - user {current_user.id}: {e}")
                # Continue with response even if quota check fails
            except QuotaExceededError as e:
                logger.warning(f"Quota exceeded for user {current_user.id}: {e}")
                # Prediction already completed, so don't fail the request
                response_data["quota_warning"] = e.user_message
            except DatabaseError as e:
                logger.error(f"Database error consuming quota for user {current_user.id}: {e}")
                # Continue with response even if quota update fails
            except Exception as e:
                logger.error(f"Unexpected error consuming quota for user {current_user.id}: {e}")
                # Continue with response even if quota update fails
            
            # Validate response
            schema = PredictionResponseSchema()
            try:
                result = schema.dump(response_data)
                return jsonify(result), 200
            except ValidationError as err:
                return jsonify({"error": "Response validation failed", "details": err.messages}), 500


# ===== QUOTA MANAGEMENT ENDPOINTS =====

@api_v1.route('/quota/status', methods=['GET'])
@login_required
@track_request_metrics
def get_quota_status():
    """
    Get User Quota Status
    ---
    tags:
      - Quota
    summary: Get current quota usage and limits
    description: Returns detailed quota information for the authenticated user
    security:
      - cookieAuth: []
    responses:
      200:
        description: Quota status retrieved successfully
        content:
          application/json:
            schema:
              type: object
              properties:
                user_id:
                  type: integer
                plan_type:
                  type: string
                current_usage:
                  type: integer
                monthly_quota:
                  type: integer
                remaining:
                  type: integer
                usage_percentage:
                  type: number
                reset_date:
                  type: string
                  format: date-time
                days_until_reset:
                  type: integer
      401:
        description: User not authenticated
    """
    db_circuit, cache_circuit, ml_circuit, notification_circuit = get_or_create_api_circuit_breakers()
    
    try:
        def get_quota_status():
            from quota_manager import get_quota_manager
            quota_manager = get_quota_manager()
            return quota_manager.get_user_quota_status(current_user.id)
        
        status = db_circuit.call(get_quota_status)
        
        response_data = {
            "user_id": status.user_id,
            "plan_type": status.plan_type,
            "current_usage": status.current_usage,
            "monthly_quota": status.monthly_quota,
            "remaining": status.remaining,
            "usage_percentage": round(status.usage_percentage, 2),
            "reset_date": status.reset_date.isoformat() if status.reset_date else None,
            "days_until_reset": status.days_until_reset,
            "last_prediction_at": status.last_prediction_at.isoformat() if status.last_prediction_at else None
        }
        
        logger.info(f"Quota status retrieved successfully for user {current_user.id}")
        return jsonify(response_data), 200
        
    except CircuitBreakerOpenException as e:
        logger.error(f"Database circuit breaker open for quota status - user {current_user.id}: {e}")
        return jsonify({
            "error": "Database service temporarily unavailable",
            "code": "DATABASE_SERVICE_UNAVAILABLE",
            "details": "Unable to retrieve quota information at this time. Please try again later.",
            "circuit_open": True
        }), 503
    except DatabaseError as e:
        logger.error(f"Database error getting quota status for user {current_user.id}: {e}")
        return jsonify(e.to_dict()), 500
    except Exception as e:
        logger.error(f"Unexpected error getting quota status for user {current_user.id}: {e}")
        error = convert_exception(e, operation='quota_status_retrieval', user_id=current_user.id)
        return jsonify(error.to_dict()), 500


@api_v1.route('/quota/plans', methods=['GET'])
@track_request_metrics
def get_available_plans():
    """
    Get Available Plans
    ---
    tags:
      - Quota
    summary: Get all available subscription plans
    description: Returns list of all available plans with features and pricing
    responses:
      200:
        description: Plans retrieved successfully
        content:
          application/json:
            schema:
              type: object
              properties:
                plans:
                  type: array
                  items:
                    type: object
                    properties:
                      plan_type:
                        type: string
                      plan_name:
                        type: string
                      monthly_quota:
                        type: integer
                      max_file_size_mb:
                        type: integer
                      price_monthly:
                        type: number
                      features:
                        type: object
    """
    db_circuit, cache_circuit, ml_circuit, notification_circuit = get_or_create_api_circuit_breakers()
    
    try:
        def get_available_plans():
            from quota_manager import get_quota_manager
            quota_manager = get_quota_manager()
            return quota_manager.get_available_plans()
        
        plans = db_circuit.call(get_available_plans)
        
        logger.info("Available plans retrieved successfully")
        return jsonify({"plans": plans}), 200
        
    except CircuitBreakerOpenException as e:
        logger.error(f"Database circuit breaker open for getting plans: {e}")
        return jsonify({
            "error": "Database service temporarily unavailable",
            "code": "DATABASE_SERVICE_UNAVAILABLE",
            "details": "Unable to retrieve plan information at this time. Please try again later.",
            "circuit_open": True
        }), 503
    except DatabaseError as e:
        logger.error(f"Database error getting available plans: {e}")
        return jsonify(e.to_dict()), 500
    except Exception as e:
        logger.error(f"Unexpected error getting available plans: {e}")
        error = convert_exception(e, operation='plans_retrieval')
        return jsonify(error.to_dict()), 500


@api_v1.route('/quota/upgrade', methods=['POST'])
@login_required
@track_request_metrics
def upgrade_plan():
    """
    Upgrade User Plan
    ---
    tags:
      - Quota
    summary: Upgrade user to a new plan
    description: Upgrade the authenticated user to a new subscription plan
    security:
      - cookieAuth: []
    requestBody:
      required: true
      content:
        application/json:
          schema:
            type: object
            properties:
              plan_type:
                type: string
                description: Target plan type
                enum: [premium, enterprise]
            required:
              - plan_type
    responses:
      200:
        description: Plan upgraded successfully
        content:
          application/json:
            schema:
              type: object
              properties:
                success:
                  type: boolean
                message:
                  type: string
                new_quota:
                  type: integer
                old_quota:
                  type: integer
      400:
        description: Invalid plan or upgrade request
      401:
        description: User not authenticated
    """
    try:
        data = request.get_json()
        if not data or 'plan_type' not in data:
            return jsonify({"error": "plan_type is required"}), 400
        
        plan_type = data['plan_type']
        
        from quota_manager import get_quota_manager, PlanType
        quota_manager = get_quota_manager()
        
        # Validate plan type
        valid_plans = [p.value for p in PlanType]
        if plan_type not in valid_plans:
            return jsonify({"error": f"Invalid plan type. Must be one of: {valid_plans}"}), 400
        
        result = quota_manager.upgrade_user_plan(current_user.id, plan_type)
        
        if result['success']:
            return jsonify(result), 200
        else:
            return jsonify(result), 400
            
    except Exception as e:
        logger.error(f"Error upgrading plan for user {current_user.id}: {e}")
        return jsonify({"error": "Failed to upgrade plan"}), 500


@api_v1.route('/quota/analytics', methods=['GET'])
@login_required
@track_request_metrics
def get_usage_analytics():
    """
    Get Usage Analytics
    ---
    tags:
      - Quota
    summary: Get detailed usage analytics
    description: Returns detailed usage analytics for the authenticated user
    security:
      - cookieAuth: []
    parameters:
      - in: query
        name: days
        schema:
          type: integer
          minimum: 1
          maximum: 90
          default: 30
        description: Number of days to include in analytics
    responses:
      200:
        description: Analytics retrieved successfully
        content:
          application/json:
            schema:
              type: object
              properties:
                period_days:
                  type: integer
                total_predictions:
                  type: integer
                total_file_size_mb:
                  type: number
                average_processing_time_ms:
                  type: number
                daily_breakdown:
                  type: array
                  items:
                    type: object
      401:
        description: User not authenticated
    """
    try:
        days = request.args.get('days', default=30, type=int)
        
        # Validate days parameter
        if days < 1 or days > 90:
            return jsonify({"error": "days must be between 1 and 90"}), 400
        
        from quota_manager import get_quota_manager
        quota_manager = get_quota_manager()
        
        analytics = quota_manager.get_usage_analytics(current_user.id, days)
        
        return jsonify(analytics), 200
        
    except Exception as e:
        logger.error(f"Error getting usage analytics for user {current_user.id}: {e}")
        return jsonify({"error": "Failed to retrieve analytics"}), 500


@api_v1.route('/quota/check', methods=['POST'])
@login_required
@track_request_metrics
def check_quota():
    """
    Check Quota Before Upload
    ---
    tags:
      - Quota
    summary: Check if user can make a prediction
    description: Check quota limits before uploading a file for prediction
    security:
      - cookieAuth: []
    requestBody:
      required: true
      content:
        application/json:
          schema:
            type: object
            properties:
              file_size_bytes:
                type: integer
                description: Size of the file to be uploaded
            required:
              - file_size_bytes
    responses:
      200:
        description: Quota check completed
        content:
          application/json:
            schema:
              type: object
              properties:
                allowed:
                  type: boolean
                reason:
                  type: string
                message:
                  type: string
                current_usage:
                  type: integer
                monthly_quota:
                  type: integer
                remaining:
                  type: integer
      400:
        description: Invalid request
      401:
        description: User not authenticated
    """
    try:
        data = request.get_json()
        if not data or 'file_size_bytes' not in data:
            return jsonify({"error": "file_size_bytes is required"}), 400
        
        file_size_bytes = data['file_size_bytes']
        
        if not isinstance(file_size_bytes, int) or file_size_bytes < 0:
            return jsonify({"error": "file_size_bytes must be a non-negative integer"}), 400
        
        from quota_manager import get_quota_manager
        quota_manager = get_quota_manager()
        
        quota_check = quota_manager.check_quota_before_prediction(current_user.id, file_size_bytes)
        
        return jsonify(quota_check), 200
        
    except Exception as e:
        logger.error(f"Error checking quota for user {current_user.id}: {e}")
        return jsonify({"error": "Failed to check quota"}), 500


@api_v1.route('/detections', methods=['GET'])
@login_required
@track_request_metrics
@cache_for_user_data(ttl=60)  # Cache for 1 minute
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


# ===== DATA RETENTION ENDPOINTS =====

class RetentionPolicySchema(Schema):
    """Schema for data retention policy."""
    user_id = fields.Int(required=True, description="User ID")
    plan_type = fields.Str(required=True, description="Current plan type")
    plan_name = fields.Str(required=True, description="Plan display name")
    retention_days = fields.Int(required=True, description="Data retention period in days")
    retention_description = fields.Str(required=True, description="Human-readable retention period")


class RetentionStatsSchema(Schema):
    """Schema for user retention statistics."""
    user_id = fields.Int(required=True, description="User ID")
    plan_type = fields.Str(required=True, description="Current plan type")
    retention_days = fields.Int(required=True, description="Retention period in days")
    total_predictions = fields.Int(required=True, description="Total predictions count")
    expired_predictions = fields.Int(required=True, description="Expired predictions count")
    expiring_soon = fields.Int(required=True, description="Predictions expiring soon")
    expired_size_mb = fields.Float(required=True, description="Size of expired data in MB")


class PlanRetentionSchema(Schema):
    """Schema for plan with retention information."""
    plan_type = fields.Str(required=True, description="Plan identifier")
    plan_name = fields.Str(required=True, description="Plan display name")
    monthly_quota = fields.Int(required=True, description="Monthly prediction quota")
    data_retention_days = fields.Int(required=True, description="Data retention days")
    retention_description = fields.Str(required=True, description="Human-readable retention")
    price_monthly = fields.Float(required=True, description="Monthly price")


@api_v1.route('/retention/policy', methods=['GET'])
@login_required
@track_request_metrics
def get_user_retention_policy():
    """
    Get current user's data retention policy.
    ---
    tags:
      - Data Retention
    security:
      - login_required: []
    responses:
      200:
        description: Current retention policy
        content:
          application/json:
            schema: RetentionPolicySchema
      401:
        description: Authentication required
        content:
          application/json:
            schema: ErrorSchema
    """
    try:
        from quota_manager import get_quota_manager
        
        quota_manager = get_quota_manager()
        policy = quota_manager.get_user_retention_policy(current_user.id)
        
        if policy.get('error'):
            return jsonify({
                "error": policy['error'],
                "code": "RETENTION_ERROR"
            }), 400
        
        return jsonify(policy)
        
    except Exception as e:
        logger.error(f"Error getting retention policy: {e}")
        return jsonify({
            "error": "Failed to get retention policy",
            "code": "RETENTION_ERROR"
        }), 500


@api_v1.route('/retention/stats', methods=['GET'])
@login_required
@track_request_metrics
def get_user_retention_stats():
    """
    Get user's data retention statistics.
    ---
    tags:
      - Data Retention
    security:
      - login_required: []
    responses:
      200:
        description: User retention statistics
        content:
          application/json:
            schema: RetentionStatsSchema
      401:
        description: Authentication required
        content:
          application/json:
            schema: ErrorSchema
    """
    try:
        from quota_manager import get_quota_manager
        
        quota_manager = get_quota_manager()
        stats = quota_manager.get_retention_stats_for_user(current_user.id)
        
        if stats.get('error'):
            return jsonify({
                "error": stats['error'],
                "code": "RETENTION_ERROR"
            }), 400
        
        return jsonify(stats)
        
    except Exception as e:
        logger.error(f"Error getting retention stats: {e}")
        return jsonify({
            "error": "Failed to get retention statistics",
            "code": "RETENTION_ERROR"
        }), 500


@api_v1.route('/retention/expired-count', methods=['GET'])
@login_required
@track_request_metrics
def get_expired_predictions_count():
    """
    Get count of expired predictions for current user.
    ---
    tags:
      - Data Retention
    security:
      - login_required: []
    responses:
      200:
        description: Count of expired predictions
        content:
          application/json:
            schema:
              type: object
              properties:
                user_id:
                  type: integer
                  description: User ID
                expired_predictions:
                  type: integer
                  description: Number of expired predictions
                expiring_soon:
                  type: integer
                  description: Number of predictions expiring soon
                expired_size_mb:
                  type: number
                  description: Size of expired data in MB
    """
    try:
        from quota_manager import get_quota_manager
        
        quota_manager = get_quota_manager()
        expired_info = quota_manager.get_expired_predictions_count(current_user.id)
        
        if expired_info.get('error'):
            return jsonify({
                "error": expired_info['error'],
                "code": "RETENTION_ERROR"
            }), 400
        
        return jsonify(expired_info)
        
    except Exception as e:
        logger.error(f"Error getting expired predictions count: {e}")
        return jsonify({
            "error": "Failed to get expired predictions count",
            "code": "RETENTION_ERROR"
        }), 500


@api_v1.route('/retention/plans', methods=['GET'])
@track_request_metrics
def get_plans_with_retention():
    """
    Get all available plans with retention information.
    ---
    tags:
      - Data Retention
    responses:
      200:
        description: List of plans with retention info
        content:
          application/json:
            schema:
              type: array
              items: PlanRetentionSchema
    """
    try:
        from quota_manager import get_quota_manager
        
        quota_manager = get_quota_manager()
        plans = quota_manager.get_all_plans_with_retention()
        
        return jsonify({
            "plans": plans,
            "total": len(plans)
        })
        
    except Exception as e:
        logger.error(f"Error getting plans with retention: {e}")
        return jsonify({
            "error": "Failed to get plans information",
            "code": "RETENTION_ERROR"
        }), 500


@api_v1.route('/retention/cleanup/preview', methods=['POST'])
@login_required
@track_request_metrics
def preview_retention_cleanup():
    """
    Preview what would be deleted in a retention cleanup (dry run).
    ---
    tags:
      - Data Retention
    security:
      - login_required: []
    responses:
      200:
        description: Preview of cleanup operation
        content:
          application/json:
            schema:
              type: object
              properties:
                dry_run:
                  type: boolean
                  description: Always true for preview
                deleted_count:
                  type: integer
                  description: Number of predictions that would be deleted
                total_size_deleted_mb:
                  type: number
                  description: Total size that would be freed in MB
    """
    try:
        from quota_manager import get_quota_manager
        
        quota_manager = get_quota_manager()
        preview = quota_manager.cleanup_expired_predictions(
            user_id=current_user.id,
            dry_run=True
        )
        
        return jsonify(preview)
        
    except Exception as e:
        logger.error(f"Error previewing cleanup: {e}")
        return jsonify({
            "error": "Failed to preview cleanup",
            "code": "RETENTION_ERROR"
        }), 500


@api_v1.route('/predictions', methods=['GET'])
@login_required
@track_request_metrics
def get_user_predictions():
    """
    Get user's predictions with retention filtering.
    ---
    tags:
      - Predictions
    security:
      - login_required: []
    parameters:
      - name: page
        in: query
        schema:
          type: integer
          minimum: 1
          default: 1
        description: Page number
      - name: per_page
        in: query
        schema:
          type: integer
          minimum: 1
          maximum: 100
          default: 20
        description: Items per page
      - name: include_expired
        in: query
        schema:
          type: boolean
          default: false
        description: Include expired predictions (normally hidden)
    responses:
      200:
        description: List of user's predictions
        content:
          application/json:
            schema:
              type: object
              properties:
                predictions:
                  type: array
                  items:
                    type: object
                    properties:
                      id:
                        type: integer
                      filename:
                        type: string
                      result:
                        type: string
                      file_size:
                        type: integer
                      created_at:
                        type: string
                        format: date-time
                      days_old:
                        type: integer
                      expires_in_days:
                        type: integer
                pagination:
                  $ref: '#/components/schemas/PaginationSchema'
                retention_info:
                  type: object
                  properties:
                    retention_days:
                      type: integer
                    plan_type:
                      type: string
    """
    try:
        # Validate query parameters
        page = int(request.args.get('page', 1))
        per_page = min(int(request.args.get('per_page', 20)), 100)
        include_expired = request.args.get('include_expired', 'false').lower() == 'true'
        
        if page < 1 or per_page < 1:
            return jsonify({"error": "Invalid pagination parameters"}), 400
        
        from web.app import Prediction
        from quota_manager import get_quota_manager
        
        # Get user's retention policy
        quota_manager = get_quota_manager()
        retention_policy = quota_manager.get_user_retention_policy(current_user.id)
        retention_days = retention_policy.get('retention_days', 30)
        
        # Build query
        query = Prediction.query.filter_by(user_id=current_user.id)
        
        # Filter expired predictions unless explicitly requested
        if not include_expired:
            from sqlalchemy import text
            query = query.filter(
                text(f"EXTRACT(DAY FROM (CURRENT_TIMESTAMP - created_at)) <= {retention_days}")
            )
        
        query = query.order_by(Prediction.created_at.desc())
        
        # Paginate
        predictions = query.paginate(
            page=page,
            per_page=per_page,
            error_out=False
        )
        
        # Format results
        results = []
        for pred in predictions.items:
            from datetime import datetime
            days_old = (datetime.now() - pred.created_at).days if pred.created_at else 0
            expires_in = max(0, retention_days - days_old)
            
            pred_dict = pred.to_dict()
            pred_dict.update({
                'days_old': days_old,
                'expires_in_days': expires_in,
                'is_expired': days_old > retention_days
            })
            results.append(pred_dict)
        
        return jsonify({
            "predictions": results,
            "pagination": {
                "page": predictions.page,
                "per_page": predictions.per_page,
                "total": predictions.total,
                "pages": predictions.pages,
                "has_next": predictions.has_next,
                "has_prev": predictions.has_prev
            },
            "retention_info": {
                "retention_days": retention_days,
                "plan_type": retention_policy.get('plan_type'),
                "plan_name": retention_policy.get('plan_name')
            }
        })
        
    except ValueError:
        return jsonify({"error": "Invalid parameter format"}), 400
    except Exception as e:
        logger.error(f"Error getting predictions: {e}")
        return jsonify({
            "error": "Failed to get predictions",
            "code": "PREDICTIONS_ERROR"
        }), 500


@api_v1.route("/status/<task_id>", methods=["GET"])
@track_request_metrics
def get_task_status(task_id):
    """
    Get prediction task status.
    
    Returns task status including progress and completion state.
    """
    from prediction_queue import get_prediction_queue
    
    queue = get_prediction_queue()
    status = queue.get_task_status(task_id)
    
    if not status:
        return jsonify({
            "error": "Task not found",
            "code": "TASK_NOT_FOUND"
        }), 404
    
    return jsonify(status)


@api_v1.route("/result/<task_id>", methods=["GET"])
@track_request_metrics
def get_task_result(task_id):
    """
    Get prediction task results.
    
    Returns results if task is completed, error if not ready.
    """
    from prediction_queue import get_prediction_queue
    
    queue = get_prediction_queue()
    
    # First check status
    status = queue.get_task_status(task_id)
    if not status:
        return jsonify({
            "error": "Task not found",
            "code": "TASK_NOT_FOUND"
        }), 404
    
    if status['status'] == 'pending':
        return jsonify({
            "error": "Task is still pending",
            "code": "TASK_PENDING",
            "status": status
        }), 202
    
    if status['status'] == 'processing':
        return jsonify({
            "error": "Task is still processing",
            "code": "TASK_PROCESSING",
            "status": status
        }), 202
    
    if status['status'] == 'failed':
        return jsonify({
            "error": "Task failed",
            "code": "TASK_FAILED",
            "status": status
        }), 500
    
    if status['status'] == 'cancelled':
        return jsonify({
            "error": "Task was cancelled",
            "code": "TASK_CANCELLED",
            "status": status
        }), 410  # Gone
    
    # Get results
    results = queue.get_task_result(task_id)
    if not results:
        return jsonify({
            "error": "Results not available",
            "code": "RESULTS_NOT_AVAILABLE"
        }), 404
    
    return jsonify({
        "task_id": task_id,
        "status": "completed",
        "results": results,
        "metadata": {
            "created_at": status['created_at'],
            "completed_at": status['completed_at'],
            "filename": status['filename']
        }
    })


@api_v1.route("/tasks", methods=["GET"])
@login_required
@track_request_metrics
def get_user_tasks():
    """
    Get all tasks for the current user.
    
    Query parameters:
    - include_completed: Include completed tasks (default: true)
    """
    from prediction_queue import get_prediction_queue
    
    include_completed = request.args.get('include_completed', 'true').lower() == 'true'
    
    queue = get_prediction_queue()
    tasks = queue.get_user_tasks(current_user.id, include_completed=include_completed)
    
    return jsonify({
        "tasks": tasks,
        "count": len(tasks)
    })


@api_v1.route("/cancel/<task_id>", methods=["POST"])
@login_required
@track_request_metrics
def cancel_task(task_id):
    """Cancel a pending prediction task."""
    from prediction_queue import get_prediction_queue
    
    queue = get_prediction_queue()
    
    # Verify task belongs to user
    status = queue.get_task_status(task_id)
    if not status:
        return jsonify({
            "error": "Task not found",
            "code": "TASK_NOT_FOUND"
        }), 404
    
    # Check permissions (if user tracking is implemented)
    # For now, allow cancellation
    
    success = queue.cancel_task(task_id)
    if success:
        return jsonify({
            "message": "Task cancelled successfully",
            "task_id": task_id
        })
    else:
        return jsonify({
            "error": "Cannot cancel task in current state",
            "code": "CANCEL_FAILED"
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