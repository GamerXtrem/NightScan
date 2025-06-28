import sys
from pathlib import Path
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from NightScanPi.Program import camera_trigger


def test_capture_image(monkeypatch, tmp_path):
    captured = []

    class DummyCamera:
        def __init__(self):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            pass

        def capture(self, out_path):
            Path(out_path).write_bytes(b"data")
            captured.append(Path(out_path))

    monkeypatch.setattr(camera_trigger, "PiCamera", DummyCamera)
    out = camera_trigger.capture_image(tmp_path)
    assert out in captured
    assert out.exists()


def test_capture_image_no_camera(monkeypatch, tmp_path):
    monkeypatch.setattr(camera_trigger, "PiCamera", None)
    with pytest.raises(RuntimeError):
        camera_trigger.capture_image(tmp_path)

