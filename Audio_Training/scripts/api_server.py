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

from log_utils import setup_logging
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

MAX_FILE_SIZE = 100 * 1024 * 1024  # 100 MB limit for uploads

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
        mp_env = os.environ.get("MODEL_PATH")
        if not mp_env:
            raise RuntimeError("MODEL_PATH environment variable not set")
        model_path = Path(mp_env)
    if csv_dir is None:
        csv_env = os.environ.get("CSV_DIR")
        if not csv_env:
            raise RuntimeError("CSV_DIR environment variable not set")
        csv_dir = Path(csv_env)
    model, labels, device = load_model(model_path, csv_dir)
    app.config["MODEL"] = model
    app.config["LABELS"] = labels
    app.config["DEVICE"] = device
    if batch_size is not None:
        app.config["BATCH_SIZE"] = batch_size
    else:
        bs_env = os.environ.get("API_BATCH_SIZE")
        if bs_env:
            try:
                app.config["BATCH_SIZE"] = int(bs_env)
            except ValueError as exc:
                raise RuntimeError("Invalid API_BATCH_SIZE value") from exc
        else:
            app.config["BATCH_SIZE"] = 1
    if log_file is None:
        lf_env = os.environ.get("PREDICT_LOG_FILE")
        if lf_env:
            log_file = Path(lf_env)
    if log_file is not None:
        app.config["LOG_FILE"] = log_file
    # Enhanced rate limiting for API
    rate_limit = os.environ.get("API_RATE_LIMIT", "60 per minute")
    limiter = Limiter(
        get_remote_address, 
        app=app, 
        default_limits=[rate_limit, "1000 per day"]
    )
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
@limiter.limit("10 per minute")  # More restrictive for prediction endpoint
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
        # Validate WAV signature before processing
        with open(tmp.name, 'rb') as wav_file:
            if not validate_wav_signature(wav_file):
                logger.warning("Invalid WAV file signature uploaded")
                return jsonify({"error": "Invalid WAV file format"}), 400
        
        try:
            # Additional validation with pydub
            audio = AudioSegment.from_file(tmp.name)
            # Check duration (reject files longer than 10 minutes)
            if len(audio) > 600000:  # 10 minutes in milliseconds
                return jsonify({"error": "Audio file too long (max 10 minutes)"}), 400
        except CouldntDecodeError:
            logger.warning("Invalid WAV file uploaded")
            return jsonify({"error": "Invalid WAV file"}), 400
        except Exception as exc:  # pragma: no cover - unexpected
            logger.exception("Error reading uploaded file: %s", exc)
            return jsonify({"error": "Failed to process file"}), 500
        try:
            results = predict_file(
                Path(tmp.name),
                model=current_app.config["MODEL"],
                labels=current_app.config["LABELS"],
                device=current_app.config["DEVICE"],
                batch_size=current_app.config["BATCH_SIZE"],
            )
        except RuntimeError as exc:  # pragma: no cover - prediction failed
            if str(exc) == "No audio segments found":
                return jsonify({"error": "Audio file is silent"}), 400
            logger.exception("Prediction failed: %s", exc)
            return jsonify({"error": "Internal server error"}), 500
        except Exception as exc:  # pragma: no cover - unexpected
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
