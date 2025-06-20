"""Minimal Flask API serving model predictions.

If you need to call this endpoint from another domain (for example the
WordPress uploader plugin), enable Cross‑Origin Resource Sharing. Install
``flask_cors`` then initialize it like so::

    from flask_cors import CORS
    CORS(app, origins=["https://your-wordpress.example"])

Replace the URL with your own site so the response includes the correct
``Access-Control-Allow-Origin`` header.
"""

import argparse
import tempfile
import sys
from pathlib import Path
from typing import List, Dict, Optional
import pathlib
import os

MAX_FILE_SIZE = 100 * 1024 * 1024  # 100 MB limit for uploads

from flask import Flask, request, jsonify
import torch
from torch.utils.data import DataLoader
from torchvision import models

sys.path.append(str(pathlib.Path(__file__).resolve().parent))

import predict

app = Flask(__name__)

model: torch.nn.Module
labels: List[str]
device: torch.device


def load_model(model_path: Path, csv_dir: Path) -> None:
    global model, labels, device
    labels = predict.load_labels(csv_dir)
    num_classes = len(labels)
    model = models.resnet18()
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
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
    return app


application = create_app()


def predict_file(path: Path) -> List[Dict]:
    dataset = predict.AudioDataset([path])
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    softmax = torch.nn.Softmax(dim=1)
    results: List[Dict] = []
    with torch.no_grad():
        for batch, paths in loader:
            batch = batch.to(device)
            outputs = model(batch)
            probs = softmax(outputs)
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
        results = predict_file(Path(tmp.name))
    return jsonify(results)


def main() -> None:
    parser = argparse.ArgumentParser(description="Start prediction API server")
    parser.add_argument("--model_path", type=Path, help="Path to trained model")
    parser.add_argument("--csv_dir", type=Path, help="Directory containing train.csv")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()
    create_app(args.model_path, args.csv_dir).run(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
