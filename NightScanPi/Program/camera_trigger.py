"""Camera capture helpers for NightScanPi."""
from __future__ import annotations

import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple

# Modern picamera2 API (libcamera-based)
try:
    from picamera2 import Picamera2
    from picamera2.controls import Controls
except ImportError:  # pragma: no cover - not available in CI
    Picamera2 = None  # type: ignore
    Controls = None  # type: ignore

# Legacy picamera API (fallback for older systems)
try:
    from picamera import PiCamera
except ImportError:  # pragma: no cover - not available in CI
    PiCamera = None  # type: ignore

logger = logging.getLogger(__name__)


class CameraManager:
    """Modern camera manager with picamera2 and legacy fallback."""
    
    def __init__(self):
        self.camera = None
        self.api_type = None
        self._initialize_camera()
    
    def _initialize_camera(self) -> None:
        """Initialize camera with modern API first, fallback to legacy."""
        # Try modern picamera2 first
        if Picamera2 is not None:
            try:
                self.camera = Picamera2()
                self.api_type = "picamera2"
                logger.info("Using modern picamera2 API")
                return
            except Exception as e:
                logger.warning(f"Failed to initialize picamera2: {e}")
                self.camera = None
        
        # Fallback to legacy picamera
        if PiCamera is not None:
            try:
                self.api_type = "picamera"
                logger.warning("Using legacy picamera API - consider upgrading to picamera2")
                return
            except Exception as e:
                logger.error(f"Failed to initialize legacy picamera: {e}")
        
        # No camera available
        self.api_type = None
        logger.error("No camera API available")
    
    def is_available(self) -> bool:
        """Check if camera is available."""
        return self.api_type is not None
    
    def capture_image_modern(self, output_path: Path, resolution: Tuple[int, int] = (1920, 1080)) -> bool:
        """Capture image using modern picamera2 API."""
        if self.api_type != "picamera2" or Picamera2 is None:
            return False
        
        try:
            # Create camera instance for this capture
            with Picamera2() as camera:
                # Configure camera for still capture
                config = camera.create_still_configuration(
                    main={"size": resolution, "format": "RGB888"},
                    lores={"size": (640, 480), "format": "YUV420"},
                    display="lores"
                )
                camera.configure(config)
                
                # Apply camera controls for better image quality
                controls = {
                    "AwbEnable": True,  # Auto white balance
                    "AeEnable": True,   # Auto exposure
                }
                camera.set_controls(controls)
                
                # Start camera and allow time to adjust
                camera.start()
                time.sleep(2)  # Allow auto-exposure to settle
                
                # Capture the image
                camera.capture_file(str(output_path))
                camera.stop()
                
                logger.info(f"Image captured successfully: {output_path}")
                return True
                
        except Exception as e:
            logger.error(f"Modern camera capture failed: {e}")
            return False
    
    def capture_image_legacy(self, output_path: Path, resolution: Tuple[int, int] = (1920, 1080)) -> bool:
        """Capture image using legacy picamera API."""
        if self.api_type != "picamera" or PiCamera is None:
            return False
        
        try:
            with PiCamera() as camera:
                camera.resolution = resolution
                # Allow camera to warm up
                time.sleep(2)
                camera.capture(str(output_path))
                
                logger.info(f"Image captured successfully (legacy): {output_path}")
                return True
                
        except Exception as e:
            logger.error(f"Legacy camera capture failed: {e}")
            return False
    
    def capture_image(self, output_path: Path, resolution: Tuple[int, int] = (1920, 1080)) -> bool:
        """Capture image using available API."""
        if self.api_type == "picamera2":
            return self.capture_image_modern(output_path, resolution)
        elif self.api_type == "picamera":
            return self.capture_image_legacy(output_path, resolution)
        else:
            logger.error("No camera API available for capture")
            return False


# Global camera manager instance
_camera_manager: Optional[CameraManager] = None


def get_camera_manager() -> CameraManager:
    """Get global camera manager instance."""
    global _camera_manager
    if _camera_manager is None:
        _camera_manager = CameraManager()
    return _camera_manager


def capture_image(out_dir: Path, resolution: Tuple[int, int] = (1920, 1080)) -> Path:
    """Capture a single JPEG image and return the file path.
    
    Args:
        out_dir: Output directory for the image
        resolution: Image resolution (width, height)
        
    Returns:
        Path to the captured image
        
    Raises:
        RuntimeError: If camera is not available or capture fails
    """
    camera_manager = get_camera_manager()
    
    if not camera_manager.is_available():
        raise RuntimeError("Camera not available - ensure picamera2 or picamera is installed")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = datetime.now().strftime("%Y%m%d_%H%M%S.jpg")
    out_path = out_dir / filename

    success = camera_manager.capture_image(out_path, resolution)
    if not success:
        raise RuntimeError(f"Failed to capture image to {out_path}")

    return out_path


def test_camera() -> bool:
    """Test camera functionality and return True if working."""
    try:
        camera_manager = get_camera_manager()
        if not camera_manager.is_available():
            logger.error("Camera test failed: No camera API available")
            return False
        
        # Try capturing a test image to /tmp
        test_path = Path("/tmp/nightscan_camera_test.jpg")
        success = camera_manager.capture_image(test_path, (640, 480))
        
        if success and test_path.exists():
            test_path.unlink()  # Clean up test file
            logger.info("Camera test passed")
            return True
        else:
            logger.error("Camera test failed: Could not capture test image")
            return False
            
    except Exception as e:
        logger.error(f"Camera test failed with exception: {e}")
        return False


def get_camera_info() -> dict:
    """Get information about available camera APIs and hardware."""
    info = {
        "picamera2_available": Picamera2 is not None,
        "picamera_available": PiCamera is not None,
        "active_api": None,
        "camera_working": False
    }
    
    camera_manager = get_camera_manager()
    info["active_api"] = camera_manager.api_type
    info["camera_working"] = camera_manager.is_available() and test_camera()
    
    return info


if __name__ == "__main__":
    print(capture_image(Path("images")))
