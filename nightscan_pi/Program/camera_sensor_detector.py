"""
Camera Sensor Detection Module for NightScanPi
Automatically detects and identifies camera sensors on Raspberry Pi.
"""

import re
import subprocess
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class SensorInfo:
    """Information about a detected camera sensor."""
    name: str
    model: str
    resolution: tuple
    capabilities: List[str]
    dtoverlay: str
    recommended_settings: Dict[str, Any]
    ir_cut_support: bool = False
    night_vision: bool = False


class CameraSensorDetector:
    """Detects and identifies camera sensors on Raspberry Pi."""
    
    # Database of known sensors and their characteristics
    SENSOR_DATABASE = {
        "imx219": SensorInfo(
            name="IMX219",
            model="Sony IMX219 8MP",
            resolution=(3280, 2464),
            capabilities=["auto_exposure", "auto_white_balance", "ir_cut"],
            dtoverlay="imx219",
            recommended_settings={
                "gpu_mem": 128,
                "resolution_default": (1920, 1080),
                "framerate_max": 30,
                "night_mode": True
            },
            ir_cut_support=True,
            night_vision=True
        ),
        "ov5647": SensorInfo(
            name="OV5647", 
            model="OmniVision OV5647 5MP",
            resolution=(2592, 1944),
            capabilities=["auto_exposure", "auto_white_balance"],
            dtoverlay="ov5647",
            recommended_settings={
                "gpu_mem": 64,
                "resolution_default": (1640, 1232),
                "framerate_max": 30,
                "night_mode": False
            },
            ir_cut_support=False,
            night_vision=False
        ),
        "imx477": SensorInfo(
            name="IMX477",
            model="Sony IMX477 12MP HQ",
            resolution=(4056, 3040),
            capabilities=["auto_exposure", "auto_white_balance", "manual_focus", "ir_cut"],
            dtoverlay="imx477",
            recommended_settings={
                "gpu_mem": 256,
                "resolution_default": (2028, 1520),
                "framerate_max": 20,
                "night_mode": True
            },
            ir_cut_support=True,
            night_vision=True
        ),
        "imx290": SensorInfo(
            name="IMX290",
            model="Sony IMX290 2MP Low-Light",
            resolution=(1920, 1080),
            capabilities=["ultra_low_light", "auto_exposure", "ir_optimized"],
            dtoverlay="imx290,clock-frequency=37125000",
            recommended_settings={
                "gpu_mem": 64,
                "resolution_default": (1920, 1080),
                "framerate_max": 60,
                "night_mode": True
            },
            ir_cut_support=True,
            night_vision=True
        ),
        "imx327": SensorInfo(
            name="IMX327",
            model="Sony IMX327 2MP Low-Light",
            resolution=(1920, 1080),
            capabilities=["ultra_low_light", "auto_exposure", "ir_optimized"],
            dtoverlay="imx290,clock-frequency=37125000",  # Uses same overlay as IMX290
            recommended_settings={
                "gpu_mem": 64,
                "resolution_default": (1920, 1080),
                "framerate_max": 60,
                "night_mode": True
            },
            ir_cut_support=True,
            night_vision=True
        ),
        "ov9281": SensorInfo(
            name="OV9281",
            model="OmniVision OV9281 1MP Global Shutter",
            resolution=(1280, 800),
            capabilities=["global_shutter", "high_speed", "mono_color"],
            dtoverlay="ov9281",
            recommended_settings={
                "gpu_mem": 64,
                "resolution_default": (1280, 800),
                "framerate_max": 120,
                "night_mode": False
            },
            ir_cut_support=False,
            night_vision=False
        ),
        "imx378": SensorInfo(
            name="IMX378",
            model="Sony IMX378 12MP",
            resolution=(4056, 3040),
            capabilities=["auto_exposure", "auto_white_balance", "4k_video"],
            dtoverlay="imx378",
            recommended_settings={
                "gpu_mem": 256,
                "resolution_default": (2028, 1520),
                "framerate_max": 30,
                "night_mode": True
            },
            ir_cut_support=True,
            night_vision=True
        ),
        "imx519": SensorInfo(
            name="IMX519",
            model="Sony IMX519 16MP",
            resolution=(4656, 3496),
            capabilities=["auto_exposure", "auto_white_balance", "4k_video", "hdr"],
            dtoverlay="imx519",
            recommended_settings={
                "gpu_mem": 256,
                "resolution_default": (2328, 1748),
                "framerate_max": 30,
                "night_mode": True
            },
            ir_cut_support=True,
            night_vision=True
        ),
        "imx708": SensorInfo(
            name="IMX708",
            model="Sony IMX708 12MP Camera Module 3",
            resolution=(4608, 2592),
            capabilities=["auto_exposure", "auto_white_balance", "auto_focus", "hdr"],
            dtoverlay="imx708",
            recommended_settings={
                "gpu_mem": 256,
                "resolution_default": (2304, 1296),
                "framerate_max": 30,
                "night_mode": True
            },
            ir_cut_support=True,
            night_vision=True
        ),
        "imx296": SensorInfo(
            name="IMX296",
            model="Sony IMX296 1.58MP Global Shutter",
            resolution=(1456, 1088),
            capabilities=["global_shutter", "high_speed", "low_light"],
            dtoverlay="imx296",
            recommended_settings={
                "gpu_mem": 64,
                "resolution_default": (1456, 1088),
                "framerate_max": 60,
                "night_mode": True
            },
            ir_cut_support=True,
            night_vision=True
        )
    }
    
    def __init__(self):
        self.detected_sensors = []
        self.active_sensor = None
        
    def detect_via_libcamera(self) -> Optional[str]:
        """Detect camera sensor using libcamera commands."""
        try:
            # Try modern rpicam-hello first
            cmd = ["rpicam-hello", "--list-cameras"]
            if not self._command_exists("rpicam-hello"):
                # Fallback to older libcamera-hello
                cmd = ["libcamera-hello", "--list-cameras"]
                if not self._command_exists("libcamera-hello"):
                    logger.warning("No libcamera commands available")
                    return None
            
            logger.info(f"Running camera detection: {' '.join(cmd)}")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode != 0:
                logger.warning(f"Camera detection failed: {result.stderr}")
                return None
            
            output = result.stdout.lower()
            logger.debug(f"Camera detection output: {output}")
            
            # Parse output for sensor information
            for sensor_id, sensor_info in self.SENSOR_DATABASE.items():
                sensor_patterns = [
                    sensor_id,
                    sensor_info.name.lower(),
                    sensor_info.model.lower().split()[0]  # First part of model name
                ]
                
                for pattern in sensor_patterns:
                    if pattern in output:
                        logger.info(f"Detected sensor via libcamera: {sensor_info.name}")
                        return sensor_id
            
            logger.warning("Camera detected but sensor type unknown")
            return "unknown"
            
        except subprocess.TimeoutExpired:
            logger.error("Camera detection timed out")
            return None
        except Exception as e:
            logger.error(f"Camera detection error: {e}")
            return None
    
    def detect_via_dmesg(self) -> Optional[str]:
        """Detect camera sensor from kernel messages."""
        try:
            result = subprocess.run(
                ["dmesg"],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode != 0:
                return None
            
            output = result.stdout.lower()
            
            # Look for sensor initialization messages
            sensor_patterns = {
                "imx219": [r"imx219.*probe", r"imx219.*detected"],
                "ov5647": [r"ov5647.*probe", r"ov5647.*detected"],
                "imx477": [r"imx477.*probe", r"imx477.*detected"],
                "imx290": [r"imx290.*probe", r"imx290.*detected"],
                "imx327": [r"imx327.*probe", r"imx327.*detected"],
                "ov9281": [r"ov9281.*probe", r"ov9281.*detected"],
                "imx378": [r"imx378.*probe", r"imx378.*detected"],
                "imx519": [r"imx519.*probe", r"imx519.*detected"],
                "imx708": [r"imx708.*probe", r"imx708.*detected"],
                "imx296": [r"imx296.*probe", r"imx296.*detected"],
            }
            
            for sensor_id, patterns in sensor_patterns.items():
                for pattern in patterns:
                    if re.search(pattern, output):
                        logger.info(f"Detected sensor via dmesg: {sensor_id}")
                        return sensor_id
            
            return None
            
        except Exception as e:
            logger.error(f"dmesg detection error: {e}")
            return None
    
    def detect_via_device_tree(self) -> Optional[str]:
        """Detect camera sensor from device tree."""
        try:
            # Check for camera device tree entries
            dt_paths = [
                "/proc/device-tree/soc/i2c@7e804000",
                "/proc/device-tree/soc/i2c0/",
                "/sys/firmware/devicetree/base/soc/i2c*"
            ]
            
            for dt_path in dt_paths:
                path = Path(dt_path)
                if path.exists():
                    # Look for camera sensor entries
                    for item in path.rglob("*"):
                        if item.is_dir():
                            name = item.name.lower()
                            for sensor_id in self.SENSOR_DATABASE.keys():
                                if sensor_id in name:
                                    logger.info(f"Detected sensor via device tree: {sensor_id}")
                                    return sensor_id
            
            return None
            
        except Exception as e:
            logger.error(f"Device tree detection error: {e}")
            return None
    
    def detect_via_vcgencmd(self) -> Optional[bool]:
        """Check if camera is detected at hardware level."""
        try:
            result = subprocess.run(
                ["vcgencmd", "get_camera"],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode != 0:
                return None
            
            output = result.stdout.strip()
            # Parse output like "supported=1 detected=1"
            if "detected=1" in output:
                logger.info("Camera detected via vcgencmd")
                return True
            else:
                logger.warning("No camera detected via vcgencmd")
                return False
                
        except Exception as e:
            logger.error(f"vcgencmd detection error: {e}")
            return None
    
    def detect_via_config_txt(self) -> Optional[str]:
        """Check config.txt for configured camera overlay."""
        config_paths = [
            "/boot/firmware/config.txt",
            "/boot/config.txt"
        ]
        
        try:
            for config_path in config_paths:
                if not Path(config_path).exists():
                    continue
                
                with open(config_path, 'r') as f:
                    content = f.read()
                
                # Look for dtoverlay entries
                overlay_pattern = r'dtoverlay\s*=\s*([^,\s]+)'
                matches = re.findall(overlay_pattern, content, re.IGNORECASE)
                
                for match in matches:
                    sensor_id = match.lower()
                    if sensor_id in self.SENSOR_DATABASE:
                        logger.info(f"Found configured sensor in config.txt: {sensor_id}")
                        return sensor_id
                    # Handle special cases
                    elif sensor_id == "imx290" and "clock-frequency" in content:
                        # Could be IMX290 or IMX327
                        logger.info("Found IMX290/IMX327 configuration in config.txt")
                        return "imx290"  # Default to IMX290
                
                return None
                
        except Exception as e:
            logger.error(f"Config.txt detection error: {e}")
            return None
    
    def detect_sensor(self) -> Optional[str]:
        """Run comprehensive sensor detection."""
        logger.info("🔍 Running comprehensive camera sensor detection...")
        
        detection_methods = [
            ("libcamera", self.detect_via_libcamera),
            ("dmesg", self.detect_via_dmesg),
            ("config.txt", self.detect_via_config_txt),
            ("device_tree", self.detect_via_device_tree),
        ]
        
        detected_sensors = []
        
        for method_name, method in detection_methods:
            try:
                result = method()
                if result:
                    detected_sensors.append((method_name, result))
                    logger.info(f"✅ {method_name}: detected {result}")
                else:
                    logger.debug(f"❌ {method_name}: no detection")
            except Exception as e:
                logger.warning(f"⚠️ {method_name}: detection failed - {e}")
        
        # Check hardware detection
        hw_detected = self.detect_via_vcgencmd()
        if hw_detected is False:
            logger.error("❌ No camera detected at hardware level")
            return None
        elif hw_detected is True:
            logger.info("✅ Camera detected at hardware level")
        
        if not detected_sensors:
            logger.warning("No sensor detected by any method, using default IMX219")
            return "imx219"  # Most common fallback
        
        # If multiple detections, prefer libcamera result
        sensor_counts = {}
        for method, sensor in detected_sensors:
            sensor_counts[sensor] = sensor_counts.get(sensor, 0) + 1
        
        # Choose most commonly detected sensor
        best_sensor = max(sensor_counts.items(), key=lambda x: x[1])[0]
        
        if len(set(sensor_counts.keys())) > 1:
            logger.warning(f"Multiple sensors detected: {sensor_counts}, choosing {best_sensor}")
        
        self.active_sensor = best_sensor
        logger.info(f"🎯 Final detection result: {best_sensor}")
        return best_sensor
    
    def get_sensor_info(self, sensor_id: Optional[str] = None) -> Optional[SensorInfo]:
        """Get detailed information about a sensor."""
        if sensor_id is None:
            sensor_id = self.active_sensor
        
        if sensor_id and sensor_id in self.SENSOR_DATABASE:
            return self.SENSOR_DATABASE[sensor_id]
        
        return None
    
    def get_recommended_resolution(self, sensor_id: Optional[str] = None) -> tuple:
        """Get recommended resolution for a sensor."""
        sensor_info = self.get_sensor_info(sensor_id)
        if sensor_info:
            return sensor_info.recommended_settings.get("resolution_default", (1920, 1080))
        return (1920, 1080)
    
    def supports_night_vision(self, sensor_id: Optional[str] = None) -> bool:
        """Check if sensor supports night vision/IR."""
        sensor_info = self.get_sensor_info(sensor_id)
        return sensor_info.night_vision if sensor_info else False
    
    def get_all_sensor_info(self) -> Dict[str, SensorInfo]:
        """Get information about all known sensors."""
        return self.SENSOR_DATABASE.copy()
    
    def _command_exists(self, command: str) -> bool:
        """Check if a command exists on the system."""
        try:
            subprocess.run(
                ["which", command],
                capture_output=True,
                check=True
            )
            return True
        except subprocess.CalledProcessError:
            return False


# Global detector instance
_detector: Optional[CameraSensorDetector] = None


def get_sensor_detector() -> CameraSensorDetector:
    """Get global sensor detector instance."""
    global _detector
    if _detector is None:
        _detector = CameraSensorDetector()
    return _detector


def detect_camera_sensor() -> Optional[str]:
    """Detect camera sensor using all available methods."""
    detector = get_sensor_detector()
    return detector.detect_sensor()


def get_camera_sensor_info(sensor_id: Optional[str] = None) -> Optional[SensorInfo]:
    """Get information about detected or specified camera sensor."""
    detector = get_sensor_detector()
    return detector.get_sensor_info(sensor_id)


def get_recommended_camera_settings(sensor_id: Optional[str] = None) -> Dict[str, Any]:
    """Get recommended camera settings for detected sensor."""
    sensor_info = get_camera_sensor_info(sensor_id)
    if sensor_info:
        return sensor_info.recommended_settings.copy()
    
    # Default settings
    return {
        "gpu_mem": 128,
        "resolution_default": (1920, 1080),
        "framerate_max": 30,
        "night_mode": False
    }


if __name__ == "__main__":
    # Test sensor detection
    logging.basicConfig(level=logging.INFO)
    
    detector = CameraSensorDetector()
    sensor = detector.detect_sensor()
    
    if sensor:
        info = detector.get_sensor_info(sensor)
        print(f"\n🎯 Detected Sensor: {info.name if info else sensor}")
        if info:
            print(f"   Model: {info.model}")
            print(f"   Resolution: {info.resolution[0]}x{info.resolution[1]}")
            print(f"   Capabilities: {', '.join(info.capabilities)}")
            print(f"   IR-CUT Support: {'Yes' if info.ir_cut_support else 'No'}")
            print(f"   Night Vision: {'Yes' if info.night_vision else 'No'}")
            print(f"   Recommended Settings: {info.recommended_settings}")
    else:
        print("❌ No camera sensor detected")