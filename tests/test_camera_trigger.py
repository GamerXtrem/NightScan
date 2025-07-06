import sys
from pathlib import Path
import pytest
from unittest.mock import Mock, patch

sys.path.append(str(Path(__file__).resolve().parents[1]))

from NightScanPi.Program import camera_trigger


class TestCameraManager:
    """Test the modern CameraManager functionality."""
    
    def test_camera_manager_picamera2_available(self, monkeypatch):
        """Test CameraManager when picamera2 is available."""
        # Mock picamera2
        mock_picamera2 = Mock()
        mock_camera_instance = Mock()
        mock_picamera2.return_value = mock_camera_instance
        
        monkeypatch.setattr(camera_trigger, "Picamera2", mock_picamera2)
        monkeypatch.setattr(camera_trigger, "PiCamera", None)
        
        manager = camera_trigger.CameraManager()
        assert manager.api_type == "picamera2"
        assert manager.is_available()
    
    def test_camera_manager_picamera_fallback(self, monkeypatch):
        """Test CameraManager fallback to legacy picamera."""
        monkeypatch.setattr(camera_trigger, "Picamera2", None)
        monkeypatch.setattr(camera_trigger, "PiCamera", Mock)
        
        manager = camera_trigger.CameraManager()
        assert manager.api_type == "picamera"
        assert manager.is_available()
    
    def test_camera_manager_no_camera(self, monkeypatch):
        """Test CameraManager when no camera API is available."""
        monkeypatch.setattr(camera_trigger, "Picamera2", None)
        monkeypatch.setattr(camera_trigger, "PiCamera", None)
        
        manager = camera_trigger.CameraManager()
        assert manager.api_type is None
        assert not manager.is_available()


def test_capture_image_modern(monkeypatch, tmp_path):
    """Test image capture with modern picamera2 API."""
    captured_files = []

    # Mock picamera2 camera
    class DummyPicamera2:
        def __init__(self):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            pass

        def create_still_configuration(self, **kwargs):
            return {"config": "test"}

        def configure(self, config):
            pass

        def set_controls(self, controls):
            pass

        def start(self):
            pass

        def stop(self):
            pass

        def capture_file(self, path):
            Path(path).write_bytes(b"mock_image_data")
            captured_files.append(Path(path))

    # Mock the CameraManager to use modern API
    mock_manager = Mock()
    mock_manager.is_available.return_value = True
    mock_manager.api_type = "picamera2"
    mock_manager.capture_image.return_value = True
    
    monkeypatch.setattr(camera_trigger, "Picamera2", DummyPicamera2)
    monkeypatch.setattr(camera_trigger, "get_camera_manager", lambda: mock_manager)
    
    # Mock the actual capture to create the file
    def mock_capture(output_path, resolution):
        Path(output_path).write_bytes(b"mock_image_data")
        captured_files.append(output_path)
        return True
    
    mock_manager.capture_image = mock_capture
    
    out = camera_trigger.capture_image(tmp_path)
    assert out in captured_files
    assert out.exists()


def test_capture_image_legacy(monkeypatch, tmp_path):
    """Test image capture with legacy picamera API."""
    captured_files = []

    class DummyPiCamera:
        def __init__(self):
            self.resolution = None

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            pass

        def capture(self, out_path):
            Path(out_path).write_bytes(b"legacy_image_data")
            captured_files.append(Path(out_path))

    # Mock the CameraManager to use legacy API
    mock_manager = Mock()
    mock_manager.is_available.return_value = True
    mock_manager.api_type = "picamera"
    
    def mock_capture(output_path, resolution):
        Path(output_path).write_bytes(b"legacy_image_data")
        captured_files.append(output_path)
        return True
    
    mock_manager.capture_image = mock_capture
    
    monkeypatch.setattr(camera_trigger, "PiCamera", DummyPiCamera)
    monkeypatch.setattr(camera_trigger, "get_camera_manager", lambda: mock_manager)
    
    out = camera_trigger.capture_image(tmp_path)
    assert out in captured_files
    assert out.exists()


def test_capture_image_no_camera(monkeypatch, tmp_path):
    """Test image capture when no camera is available."""
    # Mock the CameraManager to report no camera
    mock_manager = Mock()
    mock_manager.is_available.return_value = False
    mock_manager.api_type = None
    
    monkeypatch.setattr(camera_trigger, "get_camera_manager", lambda: mock_manager)
    
    with pytest.raises(RuntimeError, match="Camera not available"):
        camera_trigger.capture_image(tmp_path)


def test_test_camera_function(monkeypatch):
    """Test the camera test functionality."""
    # Mock successful camera test
    mock_manager = Mock()
    mock_manager.is_available.return_value = True
    mock_manager.capture_image.return_value = True
    
    monkeypatch.setattr(camera_trigger, "get_camera_manager", lambda: mock_manager)
    
    # Mock Path.exists to return True for test file
    with patch("pathlib.Path.exists", return_value=True):
        with patch("pathlib.Path.unlink"):
            result = camera_trigger.test_camera()
    
    assert result is True


def test_get_camera_info(monkeypatch):
    """Test camera information gathering."""
    # Mock camera availability
    mock_manager = Mock()
    mock_manager.api_type = "picamera2"
    mock_manager.is_available.return_value = True
    
    monkeypatch.setattr(camera_trigger, "get_camera_manager", lambda: mock_manager)
    monkeypatch.setattr(camera_trigger, "Picamera2", Mock())  # Available
    monkeypatch.setattr(camera_trigger, "PiCamera", None)     # Not available
    monkeypatch.setattr(camera_trigger, "test_camera", lambda: True)
    
    info = camera_trigger.get_camera_info()
    
    assert info["picamera2_available"] is True
    assert info["picamera_available"] is False
    assert info["active_api"] == "picamera2"
    assert info["camera_working"] is True

