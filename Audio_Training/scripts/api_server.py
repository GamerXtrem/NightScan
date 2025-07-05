"""Minimal Flask API serving model predictions.

If you need to call this endpoint from another domain (for example the
WordPress uploader plugin), set the ``API_CORS_ORIGINS`` environment
variable to a comma separated list of allowed origins. When this
variable is defined the server expects ``flask_cors`` to be installed and
initializes it automatically so that the responses include the correct
``Access-Control-Allow-Origin`` header::

    export API_CORS_ORIGINS="https://your-wordpress.example"
    gunicorn Audio_Training.scripts.api_server:application

To speed up inference you may also define ``API_BATCH_SIZE`` to process
multiple segments at once. Set ``PREDICT_LOG_FILE`` (or pass ``--log-file``)
to append JSON results to a file.
"""

import argparse
import tempfile
import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import pathlib
import os
import logging
import json
import struct
from datetime import datetime
import psutil
import time as time_module

from log_utils import setup_logging
from metrics import (track_request_metrics, record_prediction_metrics, 
                    get_metrics, CONTENT_TYPE_LATEST)
from cache_utils import get_cache, cache_health_check
from api_v1 import api_v1
from openapi_spec import create_openapi_endpoint
from config import get_config
import io
from types import SimpleNamespace
from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError
from flask import Flask, request, jsonify, current_app
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import torch
from torch.utils.data import DataLoader
from torchvision import models

sys.path.append(str(pathlib.Path(__file__).resolve().parent))

import predict

# Load configuration
config = get_config()
MAX_FILE_SIZE = config.upload.max_file_size

app = Flask(__name__)

logger = logging.getLogger(__name__)
setup_logging()


def validate_wav_signature(file_obj) -> bool:
    """Validate complete WAV file signature and structure."""
    try:
        pos = file_obj.tell()
        file_obj.seek(0)
        
        # Read and validate RIFF header
        riff_header = file_obj.read(12)
        if len(riff_header) < 12:
            return False
        
        # Check RIFF signature
        if riff_header[0:4] != b"RIFF":
            return False
        
        # Get file size from RIFF header
        riff_size = struct.unpack('<I', riff_header[4:8])[0]
        
        # Check WAVE signature
        if riff_header[8:12] != b"WAVE":
            return False
        
        # Read fmt chunk
        fmt_header = file_obj.read(8)
        if len(fmt_header) < 8:
            return False
        
        # Check fmt chunk signature
        if fmt_header[0:4] != b"fmt ":
            return False
        
        # Get fmt chunk size
        fmt_size = struct.unpack('<I', fmt_header[4:8])[0]
        
        # Validate fmt chunk size (should be 16 for PCM)
        if fmt_size < 16:
            return False
        
        # Read fmt chunk data
        fmt_data = file_obj.read(fmt_size)
        if len(fmt_data) < 16:
            return False
        
        # Validate audio format (1 = PCM)
        audio_format = struct.unpack('<H', fmt_data[0:2])[0]
        if audio_format != 1:  # Only allow PCM format
            return False
        
        # Validate number of channels (1 or 2)
        channels = struct.unpack('<H', fmt_data[2:4])[0]
        if channels not in [1, 2]:
            return False
        
        # Validate sample rate (common rates)
        sample_rate = struct.unpack('<I', fmt_data[4:8])[0]
        if sample_rate not in [8000, 16000, 22050, 44100, 48000, 96000]:
            return False
        
        # Validate bits per sample
        bits_per_sample = struct.unpack('<H', fmt_data[14:16])[0]
        if bits_per_sample not in [8, 16, 24, 32]:
            return False
        
        # Find data chunk
        while True:
            chunk_header = file_obj.read(8)
            if len(chunk_header) < 8:
                return False
            
            chunk_id = chunk_header[0:4]
            chunk_size = struct.unpack('<I', chunk_header[4:8])[0]
            
            if chunk_id == b"data":
                # Found data chunk, validate size
                if chunk_size == 0:
                    return False
                break
            else:
                # Skip other chunks
                file_obj.seek(chunk_size, 1)
        
        return True
        
    except (struct.error, ValueError, IOError) as e:
        logger.warning(f"WAV validation error: {e}")
        return False
    finally:
        file_obj.seek(pos)


def sanitize_filename(filename: str) -> str:
    """Sanitize filename to prevent path traversal attacks."""
    # Remove path separators and dangerous characters
    import re
    filename = re.sub(r'[<>:"/\\|?*]', '', filename)
    # Remove null bytes
    filename = filename.replace('\x00', '')
    # Limit length
    filename = filename[:255]
    # Ensure it ends with .wav
    if not filename.lower().endswith('.wav'):
        filename += '.wav'
    return filename


def load_model(model_path: Path, csv_dir: Path) -> Tuple[torch.nn.Module, List[str], torch.device]:
    """Load the model weights and class labels.

    Any error while reading the model file or the CSV will raise a
    ``RuntimeError`` so that the caller can abort startup cleanly.
    """
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
    return model, labels, device


def create_app(
    model_path: Optional[Path] = None,
    csv_dir: Optional[Path] = None,
    *,
    batch_size: Optional[int] = None,
    log_file: Optional[Path] = None,
) -> Flask:
    """Load the model and return the Flask application."""
    if model_path is None:
        model_path = Path(config.model.model_path)
    if csv_dir is None:
        csv_dir = Path(config.model.csv_dir)
    model, labels, device = load_model(model_path, csv_dir)
    app.config["MODEL"] = model
    app.config["LABELS"] = labels
    app.config["DEVICE"] = device
    if batch_size is not None:
        app.config["BATCH_SIZE"] = batch_size
    else:
        app.config["BATCH_SIZE"] = config.model.batch_size
    if log_file is None:
        lf_env = os.environ.get("PREDICT_LOG_FILE")
        if lf_env:
            log_file = Path(lf_env)
    if log_file is not None:
        app.config["LOG_FILE"] = log_file
    # Enhanced rate limiting for API from config
    if config.rate_limit.enabled:
        limiter = Limiter(
            get_remote_address, 
            app=app, 
            default_limits=[config.api.rate_limit]
        )
    else:
        limiter = None
        
    app.config["LIMITER"] = limiter
    if config.api.cors_origins:
        try:
            from flask_cors import CORS
        except ImportError as exc:
            raise RuntimeError(
                "CORS origins configured but flask_cors is not installed"
            ) from exc
        CORS(app, origins=config.api.cors_origins)
    return app

application = create_app()

# Register API v1 blueprint and OpenAPI docs
with application.app_context():
    application.register_blueprint(api_v1)
    create_openapi_endpoint(application)


@app.route("/metrics")
def metrics_endpoint():
    """Prometheus metrics endpoint for API."""
    return get_metrics(), 200, {'Content-Type': CONTENT_TYPE_LATEST}


@app.route("/health")
@track_request_metrics
def health_check():
    """Basic health check endpoint for API."""
    return jsonify({
        "status": "healthy",
        "service": "prediction-api",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0"
    })


@app.route("/ready")
@track_request_metrics
def readiness_check():
    """Comprehensive readiness check for prediction API."""
    checks = {
        "model_loaded": False,
        "disk_space": False,
        "memory": False,
        "gpu_available": False
    }
    
    # Check if model is loaded
    try:
        model = current_app.config.get("MODEL")
        labels = current_app.config.get("LABELS")
        checks["model_loaded"] = model is not None and labels is not None
    except Exception as e:
        logger.error(f"Model check failed: {e}")
    
    # Check disk space (> 2GB free for temp files)
    try:
        disk_usage = psutil.disk_usage('/')
        free_gb = disk_usage.free / (1024**3)
        checks["disk_space"] = free_gb > 2.0
    except Exception as e:
        logger.error(f"Disk space check failed: {e}")
    
    # Check memory usage (< 85% for ML workloads)
    try:
        memory = psutil.virtual_memory()
        checks["memory"] = memory.percent < 85.0
    except Exception as e:
        logger.error(f"Memory check failed: {e}")
    
    # Check GPU availability
    try:
        import torch
        checks["gpu_available"] = torch.cuda.is_available()
    except Exception:
        checks["gpu_available"] = False
    
    # API is ready if model is loaded and basic resources are available
    critical_checks = ["model_loaded", "disk_space", "memory"]
    is_ready = all(checks[check] for check in critical_checks)
    status_code = 200 if is_ready else 503
    
    # Add cache health check
    cache_status = cache_health_check()
    checks["cache"] = cache_status["status"] == "healthy"
    
    return jsonify({
        "status": "ready" if is_ready else "not_ready",
        "checks": checks,
        "cache": cache_status,
        "timestamp": datetime.utcnow().isoformat()
    }), status_code


def predict_file(
    path: Path,
    *,
    model: torch.nn.Module,
    labels: List[str],
    device: torch.device,
    batch_size: int,
) -> List[Dict]:
    """Return prediction results for ``path``.

    Any error while preparing the dataset or running inference will raise a
    ``RuntimeError``.
    """
    try:
        dataset = predict.AudioDataset([path])
        if len(dataset) == 0:
            raise RuntimeError("No audio segments found")
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
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
@limiter.limit(config.rate_limit.prediction_limit if limiter else None)
@track_request_metrics
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
    # Sanitize filename
    sanitized_filename = sanitize_filename(file.filename)
    
    if not sanitized_filename.lower().endswith(".wav"):
        return jsonify({"error": "WAV file required"}), 400
    if file.mimetype not in ("audio/wav", "audio/x-wav"):
        return jsonify({"error": "Invalid content type"}), 400
    if request.content_length is not None and request.content_length > MAX_FILE_SIZE:
        return jsonify({"error": "File exceeds 100 MB limit"}), 400
    with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
        total = 0
        chunk_size = 64 * 1024  # Optimized to 64KB chunks for better I/O performance
        while True:
            chunk = file.stream.read(chunk_size)
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
        # Read file data for caching
        with open(tmp.name, 'rb') as f:
            audio_data = f.read()
        
        # Check cache first
        cache = get_cache()
        cached_result = cache.get_prediction(audio_data)
        if cached_result is not None:
            logger.info("Returning cached prediction result")
            return jsonify(cached_result)
        
        # Validate WAV signature before processing
        with open(tmp.name, 'rb') as wav_file:
            if not validate_wav_signature(wav_file):
                logger.warning("Invalid WAV file signature uploaded")
                return jsonify({"error": "Invalid WAV file format"}), 400
        
        try:
            # Additional validation with pydub
            audio = AudioSegment.from_file(tmp.name)
            # Check duration based on config
            max_duration_ms = config.model.max_audio_duration * 1000
            if len(audio) > max_duration_ms:
                max_minutes = config.model.max_audio_duration // 60
                return jsonify({"error": f"Audio file too long (max {max_minutes} minutes)"}), 400
        except CouldntDecodeError:
            logger.warning("Invalid WAV file uploaded")
            return jsonify({"error": "Invalid WAV file"}), 400
        except Exception as exc:  # pragma: no cover - unexpected
            logger.exception("Error reading uploaded file: %s", exc)
            return jsonify({"error": "Failed to process file"}), 500
        start_time = time.time()
        try:
            results = predict_file(
                Path(tmp.name),
                model=current_app.config["MODEL"],
                labels=current_app.config["LABELS"],
                device=current_app.config["DEVICE"],
                batch_size=current_app.config["BATCH_SIZE"],
            )
            duration = time.time() - start_time
            
            # Log prediction metrics
            from log_utils import log_prediction
            log_prediction(
                filename=sanitized_filename,
                duration=duration,
                result_count=len(results),
                file_size=total,
                audio_duration=len(audio) / 1000.0 if 'audio' in locals() else None
            )
            
            # Record Prometheus metrics
            record_prediction_metrics(
                duration=duration,
                success=True,
                file_size=total,
                audio_duration=len(audio) / 1000.0 if 'audio' in locals() else None
            )
            
            # Cache the result
            cache.cache_prediction(audio_data, results)
        except RuntimeError as exc:  # pragma: no cover - prediction failed
            duration = time.time() - start_time
            record_prediction_metrics(duration=duration, success=False)
            if str(exc) == "No audio segments found":
                return jsonify({"error": "Audio file is silent"}), 400
            logger.exception("Prediction failed: %s", exc)
            return jsonify({"error": "Internal server error"}), 500
        except Exception as exc:  # pragma: no cover - unexpected
            duration = time.time() - start_time
            record_prediction_metrics(duration=duration, success=False)
            logger.exception("Prediction failed: %s", exc)
            return jsonify({"error": "Internal server error"}), 500
    log_file = current_app.config.get("LOG_FILE")
    if log_file:
        try:
            with open(log_file, "a", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False)
                f.write("\n")
        except Exception:  # pragma: no cover - unexpected
            logger.exception("Failed to save predictions to %s", log_file)
    return jsonify(results)


def main() -> None:
    parser = argparse.ArgumentParser(description="Start prediction API server")
    parser.add_argument("--model_path", type=Path, help="Path to trained model")
    parser.add_argument("--csv_dir", type=Path, help="Directory containing train.csv")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Number of segments processed in parallel",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Append JSON results to this file",
    )
    args = parser.parse_args()
    create_app(
        args.model_path,
        args.csv_dir,
        batch_size=args.batch_size,
        log_file=args.log_file,
    ).run(
        host=args.host, port=args.port
    )


if __name__ == "__main__":
    main()
