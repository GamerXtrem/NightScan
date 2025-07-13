import io
import os
import types
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from tests.helpers import ResponseAssertions


def create_test_app(log_file=None):
    path = Path(__file__).resolve().parents[1] / 'audio_training' / 'scripts' / 'api_server.py'
    source = path.read_text().replace('application = create_app()', '# application = create_app()')
    module = types.ModuleType('api_server_test')
    module.__file__ = str(path)
    exec(compile(source, str(path), 'exec'), module.__dict__)
    module.load_model = lambda *a, **k: (None, [], None)
    module.AudioSegment.from_file = lambda *a, **k: None
    module.predict_file = lambda *a, **k: []
    os.environ['MODEL_PATH'] = str(path)
    os.environ['CSV_DIR'] = str(path.parent)
    if log_file:
        os.environ['PREDICT_LOG_FILE'] = str(log_file)
    else:
        os.environ.pop('PREDICT_LOG_FILE', None)
    return module.create_app()


def test_invalid_file(tmp_path):
    app = create_test_app()
    client = app.test_client()
    data = {"file": (io.BytesIO(b"dummy"), "test.txt")}
    resp = client.post("/api/predict", data=data, content_type="multipart/form-data")
    
    # Enhanced assertions with comprehensive validation
    ResponseAssertions.assert_error_response(resp, 400, ['wav', 'file', 'required'])
    
    # Validate specific error message and response structure
    json_data = resp.get_json()
    assert json_data["error"] == "WAV file required"
    assert "code" not in json_data or json_data.get("code") == "INVALID_FILE_TYPE"


def test_raw_audio_upload():
    app = create_test_app()
    client = app.test_client()
    resp = client.post(
        "/api/predict",
        data=b"RIFF0000WAVEfmt ",
        content_type="audio/wav",
    )
    
    # Enhanced assertions with response validation
    ResponseAssertions.assert_success_response(resp)
    
    # Validate response structure for audio processing
    json_data = resp.get_json()
    assert json_data is not None, "Response should contain JSON data"
    
    # Check for expected prediction response fields (even if mocked)
    expected_fields = ['predictions', 'processing_time']
    for field in expected_fields:
        if field in json_data:
            assert json_data[field] is not None, f"Field '{field}' should not be null"


def test_rate_limiting(monkeypatch):
    monkeypatch.setenv("API_RATE_LIMIT", "2 per minute")
    app = create_test_app()
    client = app.test_client()
    
    # Make requests up to the rate limit
    for _ in range(2):
        data = {"file": (io.BytesIO(b"dummy"), "test.wav")}
        client.post("/api/predict", data=data, content_type="multipart/form-data")
    
    # Third request should trigger rate limiting
    data = {"file": (io.BytesIO(b"dummy"), "test.wav")}
    resp = client.post("/api/predict", data=data, content_type="multipart/form-data")
    
    # Enhanced rate limiting assertions
    ResponseAssertions.assert_rate_limit_error(resp)
    
    # Validate rate limiting headers if present
    if 'X-RateLimit-Remaining' in resp.headers:
        assert int(resp.headers['X-RateLimit-Remaining']) == 0, \
            "Rate limit remaining should be 0"
    
    # Validate error message content
    json_data = resp.get_json()
    if json_data and 'error' in json_data:
        error_msg = json_data['error'].lower()
        assert any(keyword in error_msg for keyword in ['rate', 'limit', 'too many']), \
            f"Error message should indicate rate limiting: {json_data['error']}"


def test_logging_predictions(tmp_path):
    log_path = tmp_path / "log.jsonl"
    app = create_test_app(log_file=log_path)
    client = app.test_client()
    resp = client.post(
        "/api/predict",
        data=b"RIFF0000WAVEfmt ",
        content_type="audio/wav",
    )
    
    # Enhanced assertions with comprehensive validation
    ResponseAssertions.assert_success_response(resp)
    
    # Validate logging functionality
    assert log_path.exists(), "Prediction log file should be created"
    
    # Validate log content structure
    lines = log_path.read_text().splitlines()
    assert len(lines) == 1, "Should have exactly one log entry"
    
    # Validate log entry is valid JSON
    import json
    try:
        log_entry = json.loads(lines[0])
        assert isinstance(log_entry, dict), "Log entry should be a JSON object"
        
        # Check for expected log fields
        expected_log_fields = ['timestamp', 'request']
        for field in expected_log_fields:
            if field in log_entry:
                assert log_entry[field] is not None, f"Log field '{field}' should not be null"
    except json.JSONDecodeError:
        assert False, f"Log entry should be valid JSON: {lines[0]}"
