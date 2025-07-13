import io
import json
import os
import requests
from celery import Celery
from flask import Flask
from flask_sqlalchemy import SQLAlchemy

# Circuit breaker imports - using centralized configuration
from circuit_breaker_config import (
    get_database_circuit_breaker, get_cache_circuit_breaker,
    get_http_circuit_breaker, get_ml_circuit_breaker
)
from exceptions import (
    ExternalServiceError, DatabaseError, CacheServiceError, 
    PredictionError, CircuitBreakerOpenException
)
import logging

logger = logging.getLogger(__name__)

celery = Celery(
    __name__,
    broker=os.environ.get("CELERY_BROKER_URL", "redis://localhost:6379/0"),
    backend=os.environ.get("CELERY_RESULT_BACKEND", "redis://localhost:6379/0"),
)

celery.conf.task_always_eager = os.environ.get("CELERY_TASK_ALWAYS_EAGER") == "1"

# Lazy app creation to avoid circular import issues
_db = SQLAlchemy()

def get_or_create_circuit_breakers():
    """Get circuit breakers for task operations using centralized configuration."""
    # Get circuit breakers from centralized manager
    db_circuit = get_database_circuit_breaker("celery_workers")
    cache_circuit = get_cache_circuit_breaker("celery_workers")
    ml_api_circuit = get_http_circuit_breaker(
        "celery_workers", 
        base_url=os.environ.get("PREDICT_API_URL", "http://localhost:8001"),
        service_type="ml_api"
    )
    ml_circuit = get_ml_circuit_breaker("celery_workers", model_path=os.environ.get("MODEL_PATH"))
    notification_circuit = get_http_circuit_breaker("celery_workers", service_type="notification")
    
    return ml_api_circuit, db_circuit, cache_circuit, ml_circuit, notification_circuit


def create_celery_app() -> Flask:
    from .app import create_app

    app = create_app()
    _db.init_app(app)
    return app


@celery.task
def run_prediction(pred_id: int, filename: str, file_path: str, api_url: str) -> None:
    import asyncio
    import time
    app = create_celery_app()
    with app.app_context():
        start_time = time.time()
        ml_api_circuit, db_circuit, cache_circuit, ml_circuit, notification_circuit = get_or_create_circuit_breakers()
        
        try:
            # Protected ML API call with circuit breaker
            try:
                with open(file_path, 'rb') as f:
                    resp = ml_api_circuit.post(
                        "/predict",
                        files={"file": (filename, f, "audio/wav")},
                        timeout=30,
                    )
                resp.raise_for_status()
                result = resp.json()
                status = "completed"
                logger.info(f"ML API prediction successful for {filename}")
                
            except CircuitBreakerOpenException as e:
                logger.error(f"ML API circuit breaker open for {filename}: {e}")
                result = {"error": "ML service temporarily unavailable", "circuit_open": True}
                status = "error"
            except ExternalServiceError as e:
                logger.error(f"ML API service error for {filename}: {e}")
                result = {"error": str(e)}
                status = "error"
                
        except requests.RequestException as exc:  # pragma: no cover - network errors
            logger.error(f"Request exception for {filename}: {exc}")
            result = {"error": str(exc)}
            status = "error"
        except FileNotFoundError:
            logger.error(f"File not found: {file_path}")
            result = {"error": "Temporary file not found"}
            status = "error"
        finally:
            # Clean up temporary file
            try:
                if os.path.exists(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(f"Failed to delete temp file {file_path}: {e}")

        processing_time = time.time() - start_time

        from .app import Prediction, db  # import here to avoid circular deps

        # Protected database operations with circuit breaker
        try:
            def update_prediction():
                pred = db.session.get(Prediction, pred_id)
                if pred:
                    pred.result = json.dumps(result)
                    db.session.commit()
                    return pred
                return None
            
            pred = db_circuit.call(update_prediction)
            if pred:
                logger.info(f"Database update successful for prediction {pred_id}")
            
        except CircuitBreakerOpenException as e:
            logger.error(f"Database circuit breaker open for prediction {pred_id}: {e}")
            # Store in cache as fallback
            try:
                cache_circuit.set(f"failed_prediction:{pred_id}", {
                    'result': result,
                    'status': status,
                    'processing_time': processing_time
                }, ttl=3600)
            except Exception as cache_err:
                logger.error(f"Failed to cache prediction result: {cache_err}")
                
        except DatabaseError as e:
            logger.error(f"Database error updating prediction {pred_id}: {e}")
            # Store in cache as fallback
            try:
                cache_circuit.set(f"failed_prediction:{pred_id}", {
                    'result': result,
                    'status': status,
                    'processing_time': processing_time
                }, ttl=3600)
            except Exception as cache_err:
                logger.error(f"Failed to cache prediction result: {cache_err}")
            
            # Protected notification sending
            if pred:  # Only send notifications if database update succeeded
                try:
                    def send_notification():
                        from notification_service import get_notification_service
                        notification_service = get_notification_service(db)
                        
                        # Prepare notification data
                        notification_data = {
                            'filename': filename,
                            'status': status,
                            'processing_time': f"{processing_time:.1f} seconds",
                            'results': result
                        }
                        
                        # Send async notification
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        try:
                            loop.run_until_complete(
                                notification_service.send_prediction_complete_notification(
                                    notification_data, pred.user_id
                                )
                            )
                        finally:
                            loop.close()
                        return True
                    
                    notification_circuit.call(send_notification)
                    logger.info(f"Notification sent successfully for {filename}")
                    
                except CircuitBreakerOpenException as e:
                    logger.warning(f"Notification circuit breaker open for {filename}: {e}")
                except ExternalServiceError as e:
                    logger.warning(f"Notification service error for {filename}: {e}")
                except Exception as e:
                    logger.error(f"Unexpected notification error for {filename}: {e}")


@celery.task
def send_detection_notifications(detection_data: dict, user_ids: list = None) -> None:
    """Send notifications for new detection."""
    import asyncio
    app = create_celery_app()
    with app.app_context():
        ml_api_circuit, db_circuit, cache_circuit, ml_circuit, notification_circuit = get_or_create_circuit_breakers()
        
        try:
            def send_detection_notification():
                from notification_service import get_notification_service
                notification_service = get_notification_service(_db)
                
                # Send async notification
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    loop.run_until_complete(
                        notification_service.send_detection_notification(detection_data, user_ids)
                    )
                finally:
                    loop.close()
                return True
            
            notification_circuit.call(send_detection_notification)
            logger.info(f"Detection notifications sent successfully")
            
        except CircuitBreakerOpenException as e:
            logger.warning(f"Notification circuit breaker open for detection notification: {e}")
        except ExternalServiceError as e:
            logger.warning(f"Notification service error for detection notification: {e}")
        except Exception as e:
            logger.error(f"Unexpected error sending detection notifications: {e}")


@celery.task
def send_system_alert_notifications(alert_data: dict, user_ids: list = None) -> None:
    """Send system alert notifications."""
    import asyncio
    app = create_celery_app()
    with app.app_context():
        ml_api_circuit, db_circuit, cache_circuit, ml_circuit, notification_circuit = get_or_create_circuit_breakers()
        
        try:
            def send_system_alert():
                from notification_service import get_notification_service
                notification_service = get_notification_service(_db)
                
                # Send async notification
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    loop.run_until_complete(
                        notification_service.send_system_alert(alert_data, user_ids)
                    )
                finally:
                    loop.close()
                return True
            
            notification_circuit.call(send_system_alert)
            logger.info(f"System alert notifications sent successfully")
            
        except CircuitBreakerOpenException as e:
            logger.warning(f"Notification circuit breaker open for system alert: {e}")
        except ExternalServiceError as e:
            logger.warning(f"Notification service error for system alert: {e}")
        except Exception as e:
            logger.error(f"Unexpected error sending system alert notifications: {e}")


@celery.task(bind=True, max_retries=3)
def process_prediction(self, task_id: str) -> None:
    """
    Process a queued prediction task.
    
    This task retrieves prediction details from the queue,
    runs the ML model, and stores results.
    """
    import time
    import tempfile
    from pathlib import Path
    from prediction_queue import get_prediction_queue
    from audio_training.scripts.api_server import predict_file
    from cache_utils import get_cache
    from metrics import record_prediction_metrics
    from log_utils import log_prediction
    
    app = create_celery_app()
    with app.app_context():
        queue = get_prediction_queue()
        
        # Get task from queue
        task = queue.get_next_task()
        if not task or task.task_id != task_id:
            # Task not found or mismatch
            return
        
        try:
            # Update progress
            queue.update_task_progress(task_id, 10)
            
            # Get model configuration
            model = app.config["MODEL"]
            labels = app.config["LABELS"]
            device = app.config["DEVICE"]
            batch_size = app.config["BATCH_SIZE"]
            
            # Check if file still exists
            if not Path(task.file_path).exists():
                raise FileNotFoundError(f"Input file not found: {task.file_path}")
            
            # Update progress
            queue.update_task_progress(task_id, 20)
            
            # Protected cache operations
            cached_result = None
            try:
                def get_cached_prediction():
                    from cache_utils import get_cache
                    cache = get_cache()
                    return cache.get_prediction_by_hash(task.file_hash)
                
                cached_result = cache_circuit.call(get_cached_prediction)
                logger.debug(f"Cache lookup successful for task {task_id}")
                
            except CircuitBreakerOpenException as e:
                logger.warning(f"Cache circuit breaker open for task {task_id}: {e}")
                cached_result = None
            except CacheServiceError as e:
                logger.warning(f"Cache service error for task {task_id}: {e}")
                cached_result = None
            
            if cached_result is not None:
                # Use cached result
                results = cached_result
                processing_time = 0.1  # Cache hit
                queue.update_task_progress(task_id, 90)
                logger.info(f"Using cached result for task {task_id}")
            else:
                # Protected ML prediction
                start_time = time.time()
                
                # Update progress periodically during prediction
                queue.update_task_progress(task_id, 30)
                
                try:
                    def run_ml_prediction():
                        from audio_training.scripts.api_server import predict_file
                        return predict_file(
                            Path(task.file_path),
                            model=model,
                            labels=labels,
                            device=device,
                            batch_size=batch_size,
                        )
                    
                    results = ml_circuit.call(run_ml_prediction)
                    processing_time = time.time() - start_time
                    logger.info(f"ML prediction successful for task {task_id} in {processing_time:.2f}s")
                    
                except CircuitBreakerOpenException as e:
                    logger.error(f"ML circuit breaker open for task {task_id}: {e}")
                    # Use fallback or simplified prediction
                    results = [{"species": "unknown", "confidence": 0.0, "error": "ML service unavailable"}]
                    processing_time = time.time() - start_time
                except PredictionError as e:
                    logger.error(f"ML prediction error for task {task_id}: {e}")
                    results = [{"species": "error", "confidence": 0.0, "error": str(e)}]
                    processing_time = time.time() - start_time
                
                # Protected cache storage
                try:
                    def cache_prediction():
                        from cache_utils import get_cache
                        cache = get_cache()
                        cache.cache_prediction_by_hash(task.file_hash, results)
                        return True
                    
                    cache_circuit.call(cache_prediction)
                    logger.debug(f"Cached prediction result for task {task_id}")
                    
                except CircuitBreakerOpenException as e:
                    logger.warning(f"Cache circuit breaker open for caching task {task_id}: {e}")
                except CacheServiceError as e:
                    logger.warning(f"Cache service error for caching task {task_id}: {e}")
                
                # Update progress
                queue.update_task_progress(task_id, 90)
                
                # Log metrics
                log_prediction(
                    filename=task.filename,
                    duration=processing_time,
                    result_count=len(results),
                    file_size=0,  # Could store this in task
                    audio_duration=0  # Could calculate this
                )
                
                record_prediction_metrics(
                    duration=processing_time,
                    success=True,
                    file_size=0,
                    audio_duration=0
                )
            
            # Complete task with results
            queue.complete_task(task_id, {
                'results': results,
                'processing_time': processing_time,
                'cached': cached_result is not None
            })
            
            # Protected notification sending
            try:
                def send_notifications():
                    from notification_service import get_notification_service
                    from websocket_service import get_websocket_manager
                    
                    # WebSocket notification
                    websocket_manager = get_websocket_manager()
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        loop.run_until_complete(
                            websocket_manager.notify_prediction_complete({
                                'task_id': task_id,
                                'filename': task.filename,
                                'status': 'completed',
                                'processing_time': processing_time
                            })
                        )
                        
                        # Push notification if user_id available
                        if task.user_id:
                            notification_service = get_notification_service(_db)
                            notification_data = {
                                'task_id': task_id,
                                'filename': task.filename,
                                'status': 'completed',
                                'message': f"Prediction completed for {task.filename}"
                            }
                            
                            loop.run_until_complete(
                                notification_service.send_prediction_complete_notification(
                                    notification_data, task.user_id
                                )
                            )
                    finally:
                        loop.close()
                    return True
                
                notification_circuit.call(send_notifications)
                logger.info(f"Notifications sent successfully for task {task_id}")
                
            except CircuitBreakerOpenException as e:
                logger.warning(f"Notification circuit breaker open for task {task_id}: {e}")
            except ExternalServiceError as e:
                logger.warning(f"Notification service error for task {task_id}: {e}")
            except Exception as e:
                logger.error(f"Unexpected notification error for task {task_id}: {e}")
            
        except Exception as e:
            # Task failed
            error_msg = str(e)
            queue.fail_task(task_id, error_msg)
            
            # Log failure
            print(f"Prediction task {task_id} failed: {error_msg}")
            
            # Retry if appropriate
            if self.request.retries < self.max_retries:
                # Exponential backoff
                countdown = 2 ** self.request.retries
                raise self.retry(exc=e, countdown=countdown)
        
        finally:
            # Clean up temporary file
            try:
                if Path(task.file_path).exists():
                    os.unlink(task.file_path)
            except Exception as e:
                print(f"Failed to clean up temp file: {e}")
