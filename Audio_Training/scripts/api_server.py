"""Minimal Flask API serving model predictions.

If you need to call this endpoint from another domain (for example the
WordPress uploader plugin), set the ``API_CORS_ORIGINS`` environment
variable to a comma separated list of allowed origins. When this
variable is defined the server expects ``flask_cors`` to be installed and
initializes it automatically so that the responses include the correct
``Access-Control-Allow-Origin`` header::

    export API_CORS_ORIGINS="https://your-wordpress.example"
    gunicorn Audio_Training.scripts.api_server:application
"""

import argparse
import tempfile
import sys
from pathlib import Path
from typing import List, Dict, Optional
import pathlib
import os
import logging
import io
from types import SimpleNamespace
from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError
from flask import Flask, request, jsonify
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import torch
from torch.utils.data import DataLoader
from torchvision import models

sys.path.append(str(pathlib.Path(__file__).resolve().parent))

import predict

MAX_FILE_SIZE = 100 * 1024 * 1024  # 100 MB limit for uploads

app = Flask(__name__)

logger = logging.getLogger(__name__)

model: torch.nn.Module | None = None
labels: List[str] | None = None
device: torch.device | None = None


def load_model(model_path: Path, csv_dir: Path) -> None:
    """Load the model weights and class labels.

    Any error while reading the model file or the CSV will raise a
    ``RuntimeError`` so that the caller can abort startup cleanly.
    """
    global model, labels, device
    try:
        labels = predict.load_labels(csv_dir)
    except Exception as exc:  # pragma: no cover - unexpected
        logger.exception("Failed to load label CSV from %s: %s", csv_dir, exc)
        raise RuntimeError("Could not read training CSV") from exc

    num_classes = len(labels)
    model = models.resnet18()
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        state = torch.load(model_path, map_location=device)
        model.load_state_dict(state)
    except Exception as exc:  # pragma: no cover - unexpected
        logger.exception("Failed to load model from %s: %s", model_path, exc)
        raise RuntimeError("Could not load model weights") from exc

    model.to(device)
    model.eval()


def create_app(model_path: Optional[Path] = None, csv_dir: Optional[Path] = None) -> Flask:
    """Load the model and return the Flask application."""
    if model_path is None:
        mp_env = os.environ.get("MODEL_PATH")
        if not mp_env:
            raise RuntimeError("MODEL_PATH environment variable not set")
        model_path = Path(mp_env)
    if csv_dir is None:
        csv_env = os.environ.get("CSV_DIR")
        if not csv_env:
            raise RuntimeError("CSV_DIR environment variable not set")
        csv_dir = Path(csv_env)
    load_model(model_path, csv_dir)
    rate_limit = os.environ.get("API_RATE_LIMIT", "60 per minute")
    Limiter(get_remote_address, app=app, default_limits=[rate_limit])
    cors_origins = os.environ.get("API_CORS_ORIGINS")
    if cors_origins:
        origins = [o.strip() for o in cors_origins.split(",") if o.strip()]
        try:
            from flask_cors import CORS
        except ImportError as exc:
            raise RuntimeError(
                "API_CORS_ORIGINS is set but flask_cors is not installed"
            ) from exc
        CORS(app, origins=origins)
    return app


application = create_app()


def predict_file(path: Path) -> List[Dict]:
    """Return prediction results for ``path``.

    Any error while preparing the dataset or running inference will raise a
    ``RuntimeError``.
    """
    try:
        dataset = predict.AudioDataset([path])
        if len(dataset) == 0:
            return []
        loader = DataLoader(dataset, batch_size=1, shuffle=False)
    except Exception as exc:  # pragma: no cover - unexpected
        logger.exception("Failed to prepare dataset for %s: %s", path, exc)
        raise RuntimeError("Invalid audio file") from exc
    softmax = torch.nn.Softmax(dim=1)
    results: List[Dict] = []
    with torch.no_grad():
        for batch, paths in loader:
            batch = batch.to(device)
            try:
                outputs = model(batch)
                probs = softmax(outputs)
            except Exception as exc:  # pragma: no cover - unexpected
                logger.exception("Inference failed on %s: %s", path, exc)
                raise RuntimeError("Inference failed") from exc
            values, indices = torch.topk(probs, k=3, dim=1)
            for p, vals, idxs in zip(paths, values, indices):
                seg_idx = 0
                if "#" in p:
                    try:
                        seg_idx = int(p.rsplit("#", 1)[1])
                    except ValueError:
                        seg_idx = 0
                results.append(
                    {
                        "segment": p,
                        "time": seg_idx * (predict.TARGET_DURATION_MS / 1000),
                        "predictions": [
                            {"label": labels[idx.item()], "probability": float(val)}
                            for val, idx in zip(vals, idxs)
                        ],
                    }
                )
    return results


@app.post("/api/predict")
def api_predict():
    file = request.files.get("file")
    if not file or not file.filename:
        if request.mimetype in ("audio/wav", "audio/x-wav") and request.data:
            file = SimpleNamespace(
                filename="upload.wav",
                mimetype=request.mimetype,
                stream=io.BytesIO(request.get_data()),
            )
        else:
            return jsonify({"error": "No file uploaded"}), 400
    if not file.filename.lower().endswith(".wav"):
        return jsonify({"error": "WAV file required"}), 400
    if file.mimetype not in ("audio/wav", "audio/x-wav"):
        return jsonify({"error": "Invalid content type"}), 400
    if request.content_length is not None and request.content_length > MAX_FILE_SIZE:
        return jsonify({"error": "File exceeds 100 MB limit"}), 400
    with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
        total = 0
        while True:
            chunk = file.stream.read(8192)
            if not chunk:
                break
            tmp.write(chunk)
            total += len(chunk)
            if total > MAX_FILE_SIZE:
                tmp.close()
                return jsonify({"error": "File exceeds 100 MB limit"}), 400
        tmp.flush()
        if request.content_length is None and total > MAX_FILE_SIZE:
            return jsonify({"error": "File exceeds 100 MB limit"}), 400
        try:
            AudioSegment.from_file(tmp.name)
        except CouldntDecodeError:
            logger.warning("Invalid WAV file uploaded")
            return jsonify({"error": "Invalid WAV file"}), 400
        except Exception as exc:  # pragma: no cover - unexpected
            logger.exception("Error reading uploaded file: %s", exc)
            return jsonify({"error": "Failed to process file"}), 500
        try:
            results = predict_file(Path(tmp.name))
        except Exception as exc:  # pragma: no cover - prediction failed
            logger.exception("Prediction failed: %s", exc)
            return jsonify({"error": "Internal server error"}), 500
    return jsonify(results)


def main() -> None:
    parser = argparse.ArgumentParser(description="Start prediction API server")
    parser.add_argument("--model_path", type=Path, help="Path to trained model")
    parser.add_argument("--csv_dir", type=Path, help="Directory containing train.csv")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()
    logging.basicConfig(
        format="%(levelname)s:%(processName)s:%(message)s", level=logging.INFO
    )
    create_app(args.model_path, args.csv_dir).run(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
