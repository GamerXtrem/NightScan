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
def run_prediction(pred_id: int, filename: str, data: bytes, api_url: str) -> None:
    import asyncio
    import time
    app = create_celery_app()
    with app.app_context():
        start_time = time.time()
        try:
            resp = requests.post(
                api_url,
                files={"file": (filename, io.BytesIO(data), "audio/wav")},
                timeout=30,
            )
            resp.raise_for_status()
            result = resp.json()
            status = "completed"
        except requests.RequestException as exc:  # pragma: no cover - network errors
            result = {"error": str(exc)}
            status = "error"

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
