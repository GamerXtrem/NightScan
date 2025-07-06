"""
Camera Tuning and Image Quality Optimization for NightScanPi

This module provides advanced camera tuning capabilities for optimal image quality
across different sensors, lighting conditions, and use cases.

Features:
- Sensor-specific tuning parameters
- Automatic exposure and white balance optimization
- Day/night mode specific tuning
- Custom tuning file generation
- Image quality assessment and auto-adjustment
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class TuningParameters:
    """Camera tuning parameters for optimal image quality."""
    # Automatic Exposure (AE) settings
    ae_enable: bool = True
    ae_constraint_mode: int = 0  # 0=normal, 1=highlight, 2=shadows
    ae_exposure_mode: int = 0    # 0=normal, 1=short, 2=long
    ae_metering_mode: int = 0    # 0=centre, 1=spot, 2=matrix
    
    # Automatic White Balance (AWB) settings  
    awb_enable: bool = True
    awb_mode: int = 0           # 0=auto, 1=incandescent, 2=tungsten, 3=fluorescent, 4=indoor, 5=daylight, 6=cloudy
    
    # Lens Control settings
    lens_position: Optional[float] = None  # Focus position (0.0-32.0, None=auto)
    
    # Color settings
    brightness: float = 0.0     # -1.0 to 1.0
    contrast: float = 1.0       # 0.0 to 32.0
    saturation: float = 1.0     # 0.0 to 32.0
    sharpness: float = 1.0      # 0.0 to 16.0
    
    # Gain and exposure
    analogue_gain: float = 1.0  # 1.0 to 16.0
    exposure_time: Optional[int] = None  # Microseconds, None=auto
    
    # Noise reduction
    noise_reduction_mode: int = 1  # 0=off, 1=fast, 2=high_quality
    
    # HDR and special modes
    hdr_mode: int = 0           # 0=off, 1=single-frame, 2=multi-frame
    
    # Night mode specific
    night_mode_enabled: bool = False
    night_brightness_boost: float = 0.2
    night_gain_boost: float = 2.0
    night_exposure_multiplier: float = 2.0


@dataclass
class SensorTuning:
    """Sensor-specific tuning configuration."""
    sensor_name: str
    default_params: TuningParameters
    day_params: TuningParameters
    night_params: TuningParameters
    quality_presets: Dict[str, TuningParameters]


class CameraTuningManager:
    """Advanced camera tuning and optimization manager."""
    
    def __init__(self):
        self.tuning_dir = Path(__file__).parent.parent.parent / "Hardware" / "camera_tuning"
        self.tuning_dir.mkdir(parents=True, exist_ok=True)
        
        # Load sensor tuning configurations
        self.sensor_tunings = self._initialize_sensor_tunings()
        
        # Current active tuning
        self.current_tuning: Optional[TuningParameters] = None
        self.current_sensor: Optional[str] = None
    
    def _initialize_sensor_tunings(self) -> Dict[str, SensorTuning]:
        """Initialize sensor-specific tuning configurations."""
        tunings = {}
        
        # IMX219 (Most common, 8MP, good general purpose)
        imx219_default = TuningParameters(
            brightness=0.05,
            contrast=1.1,
            saturation=1.05,
            sharpness=1.2,
            analogue_gain=2.0,
            ae_metering_mode=2,  # Matrix metering
            awb_mode=0,          # Auto
            noise_reduction_mode=1
        )
        
        imx219_day = TuningParameters(
            brightness=0.0,
            contrast=1.15,
            saturation=1.1,
            sharpness=1.3,
            analogue_gain=1.5,
            ae_constraint_mode=1,  # Highlight preservation
            awb_mode=5,            # Daylight
            noise_reduction_mode=1
        )
        
        imx219_night = TuningParameters(
            brightness=0.15,
            contrast=0.9,
            saturation=0.8,
            sharpness=0.9,
            analogue_gain=4.0,
            exposure_time=33000,   # 33ms
            ae_constraint_mode=2,  # Shadow enhancement
            awb_mode=4,            # Indoor
            noise_reduction_mode=2,  # High quality
            night_mode_enabled=True,
            night_brightness_boost=0.25,
            night_gain_boost=2.5
        )
        
        tunings["imx219"] = SensorTuning(
            sensor_name="IMX219",
            default_params=imx219_default,
            day_params=imx219_day,
            night_params=imx219_night,
            quality_presets={
                "high_quality": TuningParameters(
                    brightness=0.0, contrast=1.2, saturation=1.1, sharpness=1.5,
                    analogue_gain=1.0, noise_reduction_mode=2
                ),
                "fast_capture": TuningParameters(
                    brightness=0.0, contrast=1.0, saturation=1.0, sharpness=1.0,
                    analogue_gain=2.0, noise_reduction_mode=0
                ),
                "low_light": TuningParameters(
                    brightness=0.2, contrast=0.8, saturation=0.9, sharpness=0.8,
                    analogue_gain=8.0, exposure_time=50000, noise_reduction_mode=2
                )
            }
        )
        
        # IMX477 (HQ Camera, 12MP, excellent quality)
        imx477_default = TuningParameters(
            brightness=0.0,
            contrast=1.15,
            saturation=1.05,
            sharpness=1.4,
            analogue_gain=1.5,
            ae_metering_mode=2,
            awb_mode=0,
            noise_reduction_mode=2  # High quality for HQ camera
        )
        
        imx477_day = TuningParameters(
            brightness=-0.05,
            contrast=1.25,
            saturation=1.15,
            sharpness=1.6,
            analogue_gain=1.0,
            ae_constraint_mode=1,
            awb_mode=5,
            noise_reduction_mode=2
        )
        
        imx477_night = TuningParameters(
            brightness=0.1,
            contrast=1.0,
            saturation=0.9,
            sharpness=1.0,
            analogue_gain=6.0,
            exposure_time=40000,
            ae_constraint_mode=2,
            awb_mode=4,
            noise_reduction_mode=2,
            night_mode_enabled=True,
            night_brightness_boost=0.2,
            night_gain_boost=3.0
        )
        
        tunings["imx477"] = SensorTuning(
            sensor_name="IMX477",
            default_params=imx477_default,
            day_params=imx477_day,
            night_params=imx477_night,
            quality_presets={
                "ultra_quality": TuningParameters(
                    brightness=0.0, contrast=1.3, saturation=1.2, sharpness=2.0,
                    analogue_gain=1.0, noise_reduction_mode=2
                ),
                "portrait": TuningParameters(
                    brightness=0.05, contrast=1.1, saturation=1.05, sharpness=1.2,
                    analogue_gain=1.5, ae_metering_mode=1  # Spot metering
                )
            }
        )
        
        # IMX290/IMX327 (Ultra low-light, specialized for night vision)
        imx290_default = TuningParameters(
            brightness=0.1,
            contrast=0.9,
            saturation=0.8,
            sharpness=1.0,
            analogue_gain=4.0,
            exposure_time=33000,
            ae_metering_mode=2,
            awb_mode=4,  # Indoor by default
            noise_reduction_mode=2
        )
        
        imx290_night = TuningParameters(
            brightness=0.25,
            contrast=0.8,
            saturation=0.7,
            sharpness=0.8,
            analogue_gain=8.0,
            exposure_time=50000,
            ae_constraint_mode=2,
            awb_mode=4,
            noise_reduction_mode=2,
            night_mode_enabled=True,
            night_brightness_boost=0.35,
            night_gain_boost=4.0
        )
        
        tunings["imx290"] = tunings["imx327"] = SensorTuning(
            sensor_name="IMX290/IMX327",
            default_params=imx290_default,
            day_params=imx290_default,  # Specialized for low-light
            night_params=imx290_night,
            quality_presets={
                "ultra_low_light": TuningParameters(
                    brightness=0.3, contrast=0.7, saturation=0.6, sharpness=0.7,
                    analogue_gain=12.0, exposure_time=66000, noise_reduction_mode=2
                ),
                "security": TuningParameters(
                    brightness=0.15, contrast=0.95, saturation=0.8, sharpness=1.1,
                    analogue_gain=6.0, exposure_time=40000, noise_reduction_mode=1
                )
            }
        )
        
        # OV5647 (Original Pi camera, 5MP, legacy)
        ov5647_default = TuningParameters(
            brightness=0.05,
            contrast=1.05,
            saturation=1.0,
            sharpness=1.1,
            analogue_gain=2.5,
            ae_metering_mode=0,  # Centre weighted
            awb_mode=0,
            noise_reduction_mode=1
        )
        
        tunings["ov5647"] = SensorTuning(
            sensor_name="OV5647",
            default_params=ov5647_default,
            day_params=ov5647_default,
            night_params=TuningParameters(
                brightness=0.2, contrast=0.9, saturation=0.8, sharpness=0.9,
                analogue_gain=6.0, exposure_time=40000, awb_mode=4, noise_reduction_mode=2,
                night_mode_enabled=True
            ),
            quality_presets={
                "legacy_quality": TuningParameters(
                    brightness=0.0, contrast=1.1, saturation=1.05, sharpness=1.2,
                    analogue_gain=2.0, noise_reduction_mode=1
                )
            }
        )
        
        return tunings
    
    def get_sensor_tuning(self, sensor_type: str) -> Optional[SensorTuning]:
        """Get tuning configuration for a specific sensor."""
        return self.sensor_tunings.get(sensor_type.lower())
    
    def get_optimal_tuning(self, sensor_type: Optional[str] = None, 
                          mode: str = "auto", preset: Optional[str] = None) -> TuningParameters:
        """Get optimal tuning parameters for current conditions."""
        
        # Auto-detect sensor if not provided
        if sensor_type is None:
            try:
                from ..camera_sensor_detector import detect_camera_sensor
                sensor_type = detect_camera_sensor()
            except ImportError:
                logger.warning("Sensor detection not available, using IMX219 defaults")
                sensor_type = "imx219"
        
        if not sensor_type:
            sensor_type = "imx219"  # Safe default
        
        sensor_tuning = self.get_sensor_tuning(sensor_type)
        if not sensor_tuning:
            # Fallback to IMX219 if sensor not recognized
            logger.warning(f"Unknown sensor {sensor_type}, using IMX219 tuning")
            sensor_tuning = self.sensor_tunings["imx219"]
        
        # Return preset if specified
        if preset and preset in sensor_tuning.quality_presets:
            tuning = sensor_tuning.quality_presets[preset]
            logger.info(f"Using {preset} preset for {sensor_tuning.sensor_name}")
            return tuning
        
        # Determine mode
        if mode == "auto":
            try:
                from ..utils.ir_night_vision import is_night_mode
                is_night = is_night_mode()
                mode = "night" if is_night else "day"
            except ImportError:
                # Fallback to time-based detection
                from datetime import datetime
                hour = datetime.now().hour
                mode = "night" if 18 <= hour or hour <= 6 else "day"
        
        # Return appropriate tuning
        if mode == "night":
            tuning = sensor_tuning.night_params
            logger.info(f"Using night tuning for {sensor_tuning.sensor_name}")
        elif mode == "day":
            tuning = sensor_tuning.day_params  
            logger.info(f"Using day tuning for {sensor_tuning.sensor_name}")
        else:
            tuning = sensor_tuning.default_params
            logger.info(f"Using default tuning for {sensor_tuning.sensor_name}")
        
        self.current_tuning = tuning
        self.current_sensor = sensor_type
        return tuning
    
    def apply_tuning_to_camera(self, camera, tuning: Optional[TuningParameters] = None):
        """Apply tuning parameters to a picamera2 camera object."""
        if tuning is None:
            tuning = self.current_tuning
        
        if tuning is None:
            logger.warning("No tuning parameters available")
            return
        
        try:
            controls = {}
            
            # Automatic Exposure
            if tuning.ae_enable:
                controls["AeEnable"] = True
                if tuning.ae_constraint_mode is not None:
                    controls["AeConstraintMode"] = tuning.ae_constraint_mode
                if tuning.ae_exposure_mode is not None:
                    controls["AeExposureMode"] = tuning.ae_exposure_mode
                if tuning.ae_metering_mode is not None:
                    controls["AeMeteringMode"] = tuning.ae_metering_mode
            else:
                controls["AeEnable"] = False
            
            # Manual exposure override
            if tuning.exposure_time is not None:
                controls["ExposureTime"] = tuning.exposure_time
                controls["AeEnable"] = False  # Disable auto when manual
            
            # Automatic White Balance
            if tuning.awb_enable:
                controls["AwbEnable"] = True
                if tuning.awb_mode is not None:
                    controls["AwbMode"] = tuning.awb_mode
            else:
                controls["AwbEnable"] = False
            
            # Lens position (focus)
            if tuning.lens_position is not None:
                controls["LensPosition"] = tuning.lens_position
            
            # Color adjustments
            if tuning.brightness != 0.0:
                controls["Brightness"] = tuning.brightness
            if tuning.contrast != 1.0:
                controls["Contrast"] = tuning.contrast
            if tuning.saturation != 1.0:
                controls["Saturation"] = tuning.saturation
            if tuning.sharpness != 1.0:
                controls["Sharpness"] = tuning.sharpness
            
            # Gain
            if tuning.analogue_gain != 1.0:
                controls["AnalogueGain"] = tuning.analogue_gain
            
            # Noise reduction
            if tuning.noise_reduction_mode is not None:
                controls["NoiseReductionMode"] = tuning.noise_reduction_mode
            
            # Apply controls to camera
            camera.set_controls(controls)
            
            logger.info(f"Applied tuning: {len(controls)} controls set")
            logger.debug(f"Controls applied: {controls}")
            
        except Exception as e:
            logger.error(f"Failed to apply camera tuning: {e}")
    
    def generate_tuning_file(self, sensor_type: str, output_path: Optional[Path] = None) -> Path:
        """Generate a libcamera tuning file for the sensor."""
        sensor_tuning = self.get_sensor_tuning(sensor_type)
        if not sensor_tuning:
            raise ValueError(f"No tuning available for sensor {sensor_type}")
        
        if output_path is None:
            output_path = self.tuning_dir / f"{sensor_type}_nightscan_tuning.json"
        
        # Create libcamera-compatible tuning file
        tuning_data = {
            "version": 2.0,
            "target": "nightscanpi",
            "algorithms": [
                {
                    "rpi.black_level": {
                        "black_level": 4096
                    }
                },
                {
                    "rpi.dpc": {}
                },
                {
                    "rpi.lux": {
                        "reference_shutter_speed": 10000,
                        "reference_gain": 1.0,
                        "reference_aperture": 1.0,
                        "reference_lux": 900,
                        "reference_Y": 12744
                    }
                },
                {
                    "rpi.noise": {
                        "reference_constant": 0,
                        "reference_slope": 2.776
                    }
                },
                {
                    "rpi.geq": {
                        "offset": 204,
                        "slope": 0.01078
                    }
                },
                {
                    "rpi.sdn": {
                        "deviation": 1.6,
                        "strength": 0.85
                    }
                },
                {
                    "rpi.awb": {
                        "priors": [
                            {"lux": 0, "prior": [2000, 1.0, 3000, 1.0, 6500, 1.0, 8000, 1.0]},
                            {"lux": 800, "prior": [2000, 1.0, 3000, 1.0, 6500, 1.0, 8000, 1.0]},
                            {"lux": 1500, "prior": [2000, 1.0, 3000, 1.0, 6500, 1.0, 8000, 1.0]}
                        ],
                        "modes": {
                            "auto": {"lo": 2500, "hi": 8000},
                            "incandescent": {"lo": 2500, "hi": 3000},
                            "tungsten": {"lo": 3000, "hi": 3500},
                            "fluorescent": {"lo": 4000, "hi": 4700},
                            "indoor": {"lo": 3000, "hi": 5000},
                            "daylight": {"lo": 5500, "hi": 6500},
                            "cloudy": {"lo": 7000, "hi": 8500}
                        },
                        "bayes": 0
                    }
                },
                {
                    "rpi.agc": {
                        "metering_modes": {
                            "centre-weighted": {
                                "weights": [3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0]
                            },
                            "spot": {
                                "weights": [2, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                            },
                            "matrix": {
                                "weights": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
                            }
                        },
                        "exposure_modes": {
                            "normal": {"shutter": [100, 10000, 30000, 60000], "gain": [1.0, 2.0, 4.0, 6.0]},
                            "short": {"shutter": [100, 5000, 10000, 20000], "gain": [1.0, 2.0, 4.0, 6.0]},
                            "long": {"shutter": [100, 30000, 60000, 120000], "gain": [1.0, 2.0, 4.0, 8.0]}
                        },
                        "constraint_modes": {
                            "normal": {"bound": "CONSTRAINT_NORMAL"},
                            "highlight": {"bound": "CONSTRAINT_UPPER"},
                            "shadows": {"bound": "CONSTRAINT_LOWER"}
                        }
                    }
                },
                {
                    "rpi.alsc": {
                        "omega": 1.3,
                        "n_iter": 100,
                        "luminance_strength": 0.8
                    }
                },
                {
                    "rpi.contrast": {
                        "ce_enable": 1,
                        "gamma_curve": [
                            0, 0, 1024, 5040, 2048, 9338, 3072, 12356,
                            4096, 15312, 5120, 18051, 6144, 20790, 7168, 23193,
                            8192, 25744, 9216, 27942, 10240, 30035, 11264, 32005,
                            12288, 33975, 13312, 35815, 14336, 37600, 15360, 39168,
                            16384, 40642, 18432, 43379, 20480, 45749, 22528, 47753,
                            24576, 49621, 26624, 51253, 28672, 52698, 30720, 53925,
                            32768, 54979, 36864, 57012, 40960, 58656, 45056, 59954,
                            49152, 61183, 53248, 62355, 57344, 63419, 61440, 64476,
                            65535, 65535
                        ]
                    }
                },
                {
                    "rpi.ccm": {
                        "ccms": [
                            {
                                "ct": 2850,
                                "ccm": [2.12089, -0.52461, -0.59629, -0.85342, 2.80445, -0.95103, -0.26897, -1.14678, 2.41575]
                            },
                            {
                                "ct": 5600,
                                "ccm": [2.01027, -0.54053, -0.46973, -0.42444, 1.94154, -0.51711, 0.00243, -0.81054, 1.80811]
                            },
                            {
                                "ct": 6500,
                                "ccm": [1.98861, -0.49240, -0.49621, -0.39053, 1.87474, -0.48421, 0.02566, -0.81933, 1.79367]
                            }
                        ]
                    }
                },
                {
                    "rpi.sharpen": {
                        "threshold": 0.25,
                        "strength": sensor_tuning.default_params.sharpness,
                        "limit": 1.0
                    }
                }
            ]
        }
        
        # Save tuning file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(tuning_data, f, indent=2)
        
        logger.info(f"Generated tuning file: {output_path}")
        return output_path
    
    def save_custom_tuning(self, name: str, tuning: TuningParameters, sensor_type: Optional[str] = None) -> Path:
        """Save a custom tuning configuration."""
        if sensor_type is None:
            sensor_type = self.current_sensor or "custom"
        
        config_path = self.tuning_dir / f"{sensor_type}_{name}_tuning.json"
        
        config_data = {
            "name": name,
            "sensor_type": sensor_type,
            "created": str(Path().cwd()),
            "tuning": asdict(tuning)
        }
        
        with open(config_path, 'w') as f:
            json.dump(config_data, f, indent=2)
        
        logger.info(f"Saved custom tuning '{name}' to {config_path}")
        return config_path
    
    def load_custom_tuning(self, config_path: Path) -> TuningParameters:
        """Load a custom tuning configuration."""
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        
        tuning_dict = config_data["tuning"]
        tuning = TuningParameters(**tuning_dict)
        
        logger.info(f"Loaded custom tuning from {config_path}")
        return tuning
    
    def optimize_for_conditions(self, brightness_target: float = 0.5, 
                               contrast_target: float = 0.7) -> TuningParameters:
        """Auto-optimize tuning parameters based on target image conditions."""
        if self.current_tuning is None:
            self.get_optimal_tuning()  # Initialize with auto-detected values
        
        # Create optimized tuning based on current settings
        optimized = TuningParameters(
            **asdict(self.current_tuning)
        )
        
        # Adjust based on targets
        if brightness_target < 0.3:
            # Low light optimization
            optimized.analogue_gain = min(optimized.analogue_gain * 1.5, 8.0)
            optimized.brightness = max(optimized.brightness + 0.1, -0.5)
            optimized.noise_reduction_mode = 2
        elif brightness_target > 0.7:
            # Bright light optimization
            optimized.analogue_gain = max(optimized.analogue_gain * 0.8, 1.0)
            optimized.brightness = min(optimized.brightness - 0.05, 0.5)
            optimized.contrast = min(optimized.contrast * 1.1, 2.0)
        
        if contrast_target > 0.8:
            # High contrast optimization
            optimized.contrast = min(optimized.contrast * 1.2, 2.0)
            optimized.sharpness = min(optimized.sharpness * 1.1, 2.0)
        
        logger.info(f"Optimized tuning for brightness={brightness_target}, contrast={contrast_target}")
        return optimized


# Global tuning manager instance
_tuning_manager: Optional[CameraTuningManager] = None


def get_tuning_manager() -> CameraTuningManager:
    """Get global camera tuning manager instance."""
    global _tuning_manager
    if _tuning_manager is None:
        _tuning_manager = CameraTuningManager()
    return _tuning_manager


def get_optimal_tuning(sensor_type: Optional[str] = None, mode: str = "auto", 
                      preset: Optional[str] = None) -> TuningParameters:
    """Get optimal tuning parameters for current conditions."""
    manager = get_tuning_manager()
    return manager.get_optimal_tuning(sensor_type, mode, preset)


def apply_tuning_to_camera(camera, tuning: Optional[TuningParameters] = None):
    """Apply tuning parameters to a camera object."""
    manager = get_tuning_manager()
    manager.apply_tuning_to_camera(camera, tuning)


if __name__ == "__main__":
    # Test camera tuning system
    print("🎨 Testing Camera Tuning System")
    print("=" * 40)
    
    manager = get_tuning_manager()
    
    # Test different sensors
    sensors = ["imx219", "imx477", "imx290", "ov5647"]
    
    for sensor in sensors:
        print(f"\n📸 Testing {sensor.upper()} tuning:")
        
        # Test different modes
        for mode in ["day", "night", "auto"]:
            tuning = manager.get_optimal_tuning(sensor, mode)
            print(f"  {mode}: brightness={tuning.brightness:.2f}, gain={tuning.analogue_gain:.1f}")
        
        # Test presets
        sensor_tuning = manager.get_sensor_tuning(sensor)
        if sensor_tuning and sensor_tuning.quality_presets:
            preset_name = list(sensor_tuning.quality_presets.keys())[0]
            preset_tuning = manager.get_optimal_tuning(sensor, preset=preset_name)
            print(f"  {preset_name}: sharpness={preset_tuning.sharpness:.1f}")
    
    print("\n✅ Camera tuning test completed")