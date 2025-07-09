"""Camera capture helpers for NightScanPi."""
from __future__ import annotations

import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple
from filename_utils import FilenameGenerator
from location_manager import location_manager

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
        """Capture image using modern picamera2 API with memory optimization."""
        if self.api_type != "picamera2" or Picamera2 is None:
            return False
        
        try:
            # Use memory-optimized context for Pi Zero
            from .utils.pi_zero_optimizer import memory_optimized_operation
            
            with memory_optimized_operation():
                # Prepare night vision system for capture
                self._prepare_night_vision()
                
                # Get sensor-specific settings
                sensor_settings = self._get_sensor_optimized_settings(resolution)
                
                # Create camera instance for this capture
                with Picamera2() as camera:
                    # Configure camera for still capture with Pi Zero optimizations
                    lores_size = (320, 240) if self._is_pi_zero() else (640, 480)
                    
                    config = camera.create_still_configuration(
                        main={"size": resolution, "format": "RGB888"},
                        lores={"size": lores_size, "format": "YUV420"},
                        display="lores"
                    )
                    camera.configure(config)
                    
                    # Apply sensor-optimized camera controls
                    camera.set_controls(sensor_settings["controls"])
                    
                    # Start camera and allow time to adjust
                    camera.start()
                    time.sleep(sensor_settings["settling_time"])
                    
                    # Capture the image
                    camera.capture_file(str(output_path))
                    camera.stop()
                    
                    logger.info(f"Image captured successfully: {output_path}")
                    return True
                
        except Exception as e:
            logger.error(f"Modern camera capture failed: {e}")
            return False
    
    def _get_sensor_optimized_settings(self, resolution: Tuple[int, int]) -> dict:
        """Get sensor-optimized camera settings with advanced tuning."""
        try:
            from .camera_sensor_detector import detect_camera_sensor, get_camera_sensor_info
            from .utils.camera_tuning import get_optimal_tuning
            
            sensor_type = detect_camera_sensor()
            sensor_info = get_camera_sensor_info(sensor_type) if sensor_type else None
            
            # Get optimal tuning parameters
            tuning = get_optimal_tuning(sensor_type)
            
            # Base settings from tuning
            settings = {
                "controls": {
                    "AwbEnable": tuning.awb_enable,
                    "AeEnable": tuning.ae_enable,
                    "Brightness": tuning.brightness,
                    "Contrast": tuning.contrast,
                    "Saturation": tuning.saturation,
                    "Sharpness": tuning.sharpness,
                    "AnalogueGain": tuning.analogue_gain,
                },
                "settling_time": 2.0
            }
            
            # Add advanced tuning controls
            if tuning.exposure_time is not None:
                settings["controls"]["ExposureTime"] = tuning.exposure_time
                settings["controls"]["AeEnable"] = False  # Manual exposure
            
            if tuning.awb_mode is not None:
                settings["controls"]["AwbMode"] = tuning.awb_mode
            
            if tuning.ae_constraint_mode is not None:
                settings["controls"]["AeConstraintMode"] = tuning.ae_constraint_mode
            
            if tuning.ae_metering_mode is not None:
                settings["controls"]["AeMeteringMode"] = tuning.ae_metering_mode
            
            if tuning.lens_position is not None:
                settings["controls"]["LensPosition"] = tuning.lens_position
            
            if tuning.noise_reduction_mode is not None:
                settings["controls"]["NoiseReductionMode"] = tuning.noise_reduction_mode
            
            logger.info(f"Using advanced tuning for {sensor_type or 'unknown'} sensor")
            
            if sensor_info:
                # Resolution-based adjustments
                max_res = sensor_info.resolution
                if resolution[0] > max_res[0] or resolution[1] > max_res[1]:
                    logger.warning(f"Requested resolution {resolution} exceeds sensor maximum {max_res}")
                
                # Night mode specific adjustments
                if tuning.night_mode_enabled:
                    settings["settling_time"] = 3.0  # Longer settling for night mode
                    logger.info("Night mode tuning applied")
            
            return settings
            
        except Exception as e:
            logger.warning(f"Failed to get sensor-optimized settings: {e}")
            # Return safe fallback settings
            return {
                "controls": {
                    "AwbEnable": True,
                    "AeEnable": True,
                    "Brightness": 0.0,
                    "Contrast": 1.0,
                    "Saturation": 1.0,
                    "Sharpness": 1.0,
                    "AnalogueGain": 2.0,
                },
                "settling_time": 2.0
            }
    
    def capture_image_legacy(self, output_path: Path, resolution: Tuple[int, int] = (1920, 1080)) -> bool:
        """Capture image using legacy picamera API."""
        if self.api_type != "picamera" or PiCamera is None:
            return False
        
        try:
            # Prepare night vision system for capture
            self._prepare_night_vision()
            
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
    
    def _is_pi_zero(self) -> bool:
        """Check if running on Pi Zero."""
        try:
            from .utils.pi_zero_optimizer import is_pi_zero
            return is_pi_zero()
        except:
            return False
    
    def _prepare_night_vision(self) -> None:
        """Prepare night vision system for capture."""
        try:
            from .utils.ir_night_vision import prepare_camera_for_capture
            prepare_camera_for_capture()
        except ImportError:
            logger.debug("Night vision module not available")
        except Exception as e:
            logger.warning(f"Night vision preparation failed: {e}")
    
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


def capture_image(out_dir: Path, resolution: Optional[Tuple[int, int]] = None) -> Path:
    """Capture a single JPEG image and return the file path.
    
    Args:
        out_dir: Output directory for the image
        resolution: Image resolution (width, height). If None, uses sensor-recommended resolution.
        
    Returns:
        Path to the captured image
        
    Raises:
        RuntimeError: If camera is not available or capture fails
    """
    camera_manager = get_camera_manager()
    
    if not camera_manager.is_available():
        raise RuntimeError("Camera not available - ensure picamera2 or picamera is installed")

    # Auto-detect optimal resolution if not specified
    if resolution is None:
        resolution = get_optimal_resolution()

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate filename with GPS metadata using new format
    filename_gen = FilenameGenerator(location_manager)
    filename = filename_gen.generate_image_filename()
    out_path = out_dir / filename

    success = camera_manager.capture_image(out_path, resolution)
    if not success:
        raise RuntimeError(f"Failed to capture image to {out_path}")

    return out_path


def get_optimal_resolution() -> Tuple[int, int]:
    """Get optimal resolution for detected camera sensor and Pi model."""
    try:
        from .camera_sensor_detector import detect_camera_sensor, get_sensor_detector
        from .utils.pi_zero_optimizer import optimize_camera_resolution
        
        detector = get_sensor_detector()
        sensor_type = detect_camera_sensor()
        
        # Get sensor-recommended resolution
        if sensor_type:
            recommended_res = detector.get_recommended_resolution(sensor_type)
            logger.info(f"Sensor-recommended resolution: {recommended_res}")
        else:
            recommended_res = (1920, 1080)
        
        # Optimize for Pi Zero 2W if needed
        optimal_res = optimize_camera_resolution(recommended_res)
        
        if optimal_res != recommended_res:
            logger.info(f"Resolution optimized for Pi Zero: {recommended_res} → {optimal_res}")
        
        return optimal_res
        
    except Exception as e:
        logger.warning(f"Failed to get optimal resolution: {e}")
    
    # Default fallback resolution (optimized for Pi Zero if applicable)
    try:
        from .utils.pi_zero_optimizer import optimize_camera_resolution
        return optimize_camera_resolution((1920, 1080))
    except:
        return (1920, 1080)


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
    from .camera_sensor_detector import detect_camera_sensor, get_camera_sensor_info, get_recommended_camera_settings
    
    info = {
        "picamera2_available": Picamera2 is not None,
        "picamera_available": PiCamera is not None,
        "active_api": None,
        "camera_working": False,
        "sensor_type": None,
        "sensor_info": None,
        "recommended_settings": None
    }
    
    camera_manager = get_camera_manager()
    info["active_api"] = camera_manager.api_type
    info["camera_working"] = camera_manager.is_available() and test_camera()
    
    # Detect camera sensor
    try:
        sensor_type = detect_camera_sensor()
        if sensor_type:
            info["sensor_type"] = sensor_type
            sensor_info = get_camera_sensor_info(sensor_type)
            if sensor_info:
                info["sensor_info"] = {
                    "name": sensor_info.name,
                    "model": sensor_info.model,
                    "resolution": sensor_info.resolution,
                    "capabilities": sensor_info.capabilities,
                    "ir_cut_support": sensor_info.ir_cut_support,
                    "night_vision": sensor_info.night_vision,
                    "dtoverlay": sensor_info.dtoverlay
                }
                info["recommended_settings"] = get_recommended_camera_settings(sensor_type)
    except Exception as e:
        logger.warning(f"Sensor detection failed: {e}")
    
    return info


if __name__ == "__main__":
    print(capture_image(Path("images")))
