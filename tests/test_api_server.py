import io
import os
import types
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))


def create_test_app():
    path = Path(__file__).resolve().parents[1] / 'Audio_Training' / 'scripts' / 'api_server.py'
    source = path.read_text().replace('application = create_app()', '# application = create_app()')
    module = types.ModuleType('api_server_test')
    module.__file__ = str(path)
    exec(compile(source, str(path), 'exec'), module.__dict__)
    module.load_model = lambda *a, **k: None
    module.AudioSegment.from_file = lambda *a, **k: None
    module.predict_file = lambda *a, **k: []
    os.environ['MODEL_PATH'] = str(path)
    os.environ['CSV_DIR'] = str(path.parent)
    return module.create_app()


def test_invalid_file(tmp_path):
    app = create_test_app()
    client = app.test_client()
    data = {"file": (io.BytesIO(b"dummy"), "test.txt")}
    resp = client.post("/api/predict", data=data, content_type="multipart/form-data")
    assert resp.status_code == 400
    assert resp.get_json()["error"] == "WAV file required"


def test_rate_limiting(monkeypatch):
    monkeypatch.setenv("API_RATE_LIMIT", "2 per minute")
    app = create_test_app()
    client = app.test_client()
    for _ in range(2):
        data = {"file": (io.BytesIO(b"dummy"), "test.wav")}
        client.post("/api/predict", data=data, content_type="multipart/form-data")
    data = {"file": (io.BytesIO(b"dummy"), "test.wav")}
    resp = client.post("/api/predict", data=data, content_type="multipart/form-data")
    assert resp.status_code == 429
