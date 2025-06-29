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
    app = create_celery_app()
    with app.app_context():
        try:
            resp = requests.post(
                api_url,
                files={"file": (filename, io.BytesIO(data), "audio/wav")},
                timeout=30,
            )
            resp.raise_for_status()
            result = resp.json()
        except requests.RequestException as exc:  # pragma: no cover - network errors
            result = {"error": str(exc)}

        from .app import Prediction, db  # import here to avoid circular deps

        pred = db.session.get(Prediction, pred_id)
        if pred:
            pred.result = json.dumps(result)
            db.session.commit()
