import io
import json
import os
import requests
from celery import Celery
from flask import Flask
from flask_sqlalchemy import SQLAlchemy

celery = Celery(
    __name__,
    broker=os.environ.get("CELERY_BROKER_URL", "redis://localhost:6379/0"),
    backend=os.environ.get("CELERY_RESULT_BACKEND", "redis://localhost:6379/0"),
)

celery.conf.task_always_eager = os.environ.get("CELERY_TASK_ALWAYS_EAGER") == "1"

# Lazy app creation to avoid circular import issues
_db = SQLAlchemy()


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
        try:
            # Open the file from path instead of using passed data
            with open(file_path, 'rb') as f:
                resp = requests.post(
                    api_url,
                    files={"file": (filename, f, "audio/wav")},
                    timeout=30,
                )
            resp.raise_for_status()
            result = resp.json()
            status = "completed"
        except requests.RequestException as exc:  # pragma: no cover - network errors
            result = {"error": str(exc)}
            status = "error"
        except FileNotFoundError:
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

        pred = db.session.get(Prediction, pred_id)
        if pred:
            pred.result = json.dumps(result)
            db.session.commit()
            
            # Send notification
            try:
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
                loop.run_until_complete(
                    notification_service.send_prediction_complete_notification(
                        notification_data, pred.user_id
                    )
                )
                loop.close()
                
            except Exception as e:
                print(f"Failed to send notification: {e}")


@celery.task
def send_detection_notifications(detection_data: dict, user_ids: list = None) -> None:
    """Send notifications for new detection."""
    import asyncio
    app = create_celery_app()
    with app.app_context():
        try:
            from notification_service import get_notification_service
            notification_service = get_notification_service(_db)
            
            # Send async notification
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(
                notification_service.send_detection_notification(detection_data, user_ids)
            )
            loop.close()
            
        except Exception as e:
            print(f"Failed to send detection notifications: {e}")


@celery.task
def send_system_alert_notifications(alert_data: dict, user_ids: list = None) -> None:
    """Send system alert notifications."""
    import asyncio
    app = create_celery_app()
    with app.app_context():
        try:
            from notification_service import get_notification_service
            notification_service = get_notification_service(_db)
            
            # Send async notification
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(
                notification_service.send_system_alert(alert_data, user_ids)
            )
            loop.close()
            
        except Exception as e:
            print(f"Failed to send system alert notifications: {e}")


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
            
            # Check cache first
            cache = get_cache()
            cached_result = cache.get_prediction_by_hash(task.file_hash)
            
            if cached_result is not None:
                # Use cached result
                results = cached_result
                processing_time = 0.1  # Cache hit
                queue.update_task_progress(task_id, 90)
            else:
                # Run prediction
                start_time = time.time()
                
                # Update progress periodically during prediction
                queue.update_task_progress(task_id, 30)
                
                results = predict_file(
                    Path(task.file_path),
                    model=model,
                    labels=labels,
                    device=device,
                    batch_size=batch_size,
                )
                
                processing_time = time.time() - start_time
                
                # Cache the result
                cache.cache_prediction_by_hash(task.file_hash, results)
                
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
            
            # Send notifications
            try:
                from notification_service import get_notification_service
                from websocket_service import get_websocket_manager
                
                # WebSocket notification
                websocket_manager = get_websocket_manager()
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(
                    websocket_manager.notify_prediction_complete({
                        'task_id': task_id,
                        'filename': task.filename,
                        'status': 'completed',
                        'processing_time': processing_time
                    })
                )
                loop.close()
                
                # Push notification if user_id available
                if task.user_id:
                    notification_service = get_notification_service(_db)
                    notification_data = {
                        'task_id': task_id,
                        'filename': task.filename,
                        'status': 'completed',
                        'message': f"Prediction completed for {task.filename}"
                    }
                    
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    loop.run_until_complete(
                        notification_service.send_prediction_complete_notification(
                            notification_data, task.user_id
                        )
                    )
                    loop.close()
                    
            except Exception as e:
                print(f"Failed to send notifications: {e}")
            
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
