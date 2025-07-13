"""
Camera Validation and Testing Utilities for NightScanPi

This module provides comprehensive camera validation tools similar to rpicam-hello,
with additional features for IR-CUT cameras, sensor detection, and NightScan-specific testing.

Features:
- Camera hardware detection and validation
- Image quality assessment 
- Performance benchmarking
- Night vision testing
- Sensor-specific validation
- libcamera command testing
"""

import os
import sys
import time
import logging
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class CameraTestResult:
    """Result of a camera test."""
    test_name: str
    success: bool
    duration: float
    details: Dict[str, Any]
    error_message: Optional[str] = None


@dataclass
class ImageQualityMetrics:
    """Image quality assessment metrics."""
    brightness: float
    contrast: float
    sharpness: float
    noise_level: float
    file_size: int
    resolution: Tuple[int, int]
    format: str


class CameraValidator:
    """Comprehensive camera validation and testing suite."""
    
    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = Path(output_dir or "camera_validation")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.test_results: List[CameraTestResult] = []
        
        # Import camera modules if available
        try:
            from .. import camera_trigger
            from .. import camera_sensor_detector
            self.camera_module = camera_trigger
            self.sensor_module = camera_sensor_detector
        except ImportError:
            self.camera_module = None
            self.sensor_module = None
            logger.warning("Camera modules not available for testing")
    
    def run_libcamera_hello_test(self, duration: int = 5) -> CameraTestResult:
        """Test camera using libcamera-hello or rpicam-hello command."""
        start_time = time.time()
        test_name = "libcamera_hello"
        
        try:
            # Determine which command to use
            commands_to_try = ["rpicam-hello", "libcamera-hello"]
            cmd = None
            
            for command in commands_to_try:
                if subprocess.run(["which", command], capture_output=True).returncode == 0:
                    cmd = command
                    break
            
            if not cmd:
                return CameraTestResult(
                    test_name=test_name,
                    success=False,
                    duration=time.time() - start_time,
                    details={},
                    error_message="No libcamera command available (rpicam-hello or libcamera-hello)"
                )
            
            # Run camera hello test
            result = subprocess.run(
                [cmd, "--timeout", str(duration * 1000), "--nopreview"],
                capture_output=True,
                text=True,
                timeout=duration + 10
            )
            
            success = result.returncode == 0
            details = {
                "command": cmd,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "return_code": result.returncode
            }
            
            return CameraTestResult(
                test_name=test_name,
                success=success,
                duration=time.time() - start_time,
                details=details,
                error_message=result.stderr if not success else None
            )
        
        except subprocess.TimeoutExpired:
            return CameraTestResult(
                test_name=test_name,
                success=False,
                duration=time.time() - start_time,
                details={},
                error_message=f"Command timed out after {duration} seconds"
            )
        except Exception as e:
            return CameraTestResult(
                test_name=test_name,
                success=False,
                duration=time.time() - start_time,
                details={},
                error_message=str(e)
            )
    
    def test_camera_capture(self) -> CameraTestResult:
        """Test camera capture using NightScan camera module."""
        start_time = time.time()
        test_name = "camera_capture"
        
        if not self.camera_module:
            return CameraTestResult(
                test_name=test_name,
                success=False,
                duration=0,
                details={},
                error_message="Camera module not available"
            )
        
        try:
            # Test basic camera capture
            output_path = self.output_dir / f"test_capture_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            
            captured_path = self.camera_module.capture_image(self.output_dir)
            
            success = captured_path.exists() and captured_path.stat().st_size > 0
            
            details = {
                "output_path": str(captured_path),
                "file_size": captured_path.stat().st_size if success else 0,
                "exists": captured_path.exists()
            }
            
            return CameraTestResult(
                test_name=test_name,
                success=success,
                duration=time.time() - start_time,
                details=details,
                error_message=None if success else "Capture failed or empty file"
            )
        
        except Exception as e:
            return CameraTestResult(
                test_name=test_name,
                success=False,
                duration=time.time() - start_time,
                details={},
                error_message=str(e)
            )
    
    def test_sensor_detection(self) -> CameraTestResult:
        """Test camera sensor detection functionality."""
        start_time = time.time()
        test_name = "sensor_detection"
        
        if not self.sensor_module:
            return CameraTestResult(
                test_name=test_name,
                success=False,
                duration=0,
                details={},
                error_message="Sensor detection module not available"
            )
        
        try:
            detector = self.sensor_module.CameraSensorDetector()
            sensor_type = detector.detect_sensor()
            
            if sensor_type:
                sensor_info = detector.get_sensor_info(sensor_type)
                success = True
                details = {
                    "detected_sensor": sensor_type,
                    "sensor_info": {
                        "name": sensor_info.name if sensor_info else None,
                        "model": sensor_info.model if sensor_info else None,
                        "resolution": sensor_info.resolution if sensor_info else None,
                        "ir_cut_support": sensor_info.ir_cut_support if sensor_info else None,
                        "night_vision": sensor_info.night_vision if sensor_info else None
                    } if sensor_info else None
                }
                error_message = None
            else:
                success = False
                details = {}
                error_message = "No camera sensor detected"
            
            return CameraTestResult(
                test_name=test_name,
                success=success,
                duration=time.time() - start_time,
                details=details,
                error_message=error_message
            )
        
        except Exception as e:
            return CameraTestResult(
                test_name=test_name,
                success=False,
                duration=time.time() - start_time,
                details={},
                error_message=str(e)
            )
    
    def test_night_vision(self) -> CameraTestResult:
        """Test night vision IR-CUT functionality."""
        start_time = time.time()
        test_name = "night_vision"
        
        try:
            from ..utils.ir_night_vision import get_night_vision
            
            nv = get_night_vision()
            
            if not nv.gpio_initialized:
                return CameraTestResult(
                    test_name=test_name,
                    success=False,
                    duration=time.time() - start_time,
                    details={"gpio_initialized": False},
                    error_message="GPIO not initialized - night vision hardware not available"
                )
            
            # Test day mode
            day_success = nv.set_day_mode()
            time.sleep(1)
            
            # Test night mode
            night_success = nv.set_night_mode()
            time.sleep(1)
            
            # Return to auto mode
            auto_success = nv.auto_adjust_mode()
            
            success = day_success and night_success and auto_success
            
            details = {
                "gpio_initialized": nv.gpio_initialized,
                "day_mode_test": day_success,
                "night_mode_test": night_success,
                "auto_mode_test": auto_success,
                "current_mode": nv.current_mode,
                "leds_enabled": nv.leds_enabled
            }
            
            return CameraTestResult(
                test_name=test_name,
                success=success,
                duration=time.time() - start_time,
                details=details,
                error_message=None if success else "One or more night vision tests failed"
            )
        
        except ImportError:
            return CameraTestResult(
                test_name=test_name,
                success=False,
                duration=0,
                details={},
                error_message="Night vision module not available"
            )
        except Exception as e:
            return CameraTestResult(
                test_name=test_name,
                success=False,
                duration=time.time() - start_time,
                details={},
                error_message=str(e)
            )
    
    def test_multiple_resolutions(self) -> CameraTestResult:
        """Test camera capture at multiple resolutions."""
        start_time = time.time()
        test_name = "multiple_resolutions"
        
        if not self.camera_module:
            return CameraTestResult(
                test_name=test_name,
                success=False,
                duration=0,
                details={},
                error_message="Camera module not available"
            )
        
        resolutions = [
            (640, 480),    # VGA
            (1280, 720),   # 720p
            (1920, 1080),  # 1080p
        ]
        
        results = {}
        overall_success = True
        
        try:
            camera_manager = self.camera_module.get_camera_manager()
            
            for width, height in resolutions:
                resolution_name = f"{width}x{height}"
                output_path = self.output_dir / f"test_{resolution_name}_{datetime.now().strftime('%H%M%S')}.jpg"
                
                try:
                    success = camera_manager.capture_image(output_path, (width, height))
                    file_size = output_path.stat().st_size if output_path.exists() else 0
                    
                    results[resolution_name] = {
                        "success": success,
                        "file_size": file_size,
                        "path": str(output_path)
                    }
                    
                    if not success:
                        overall_success = False
                
                except Exception as e:
                    results[resolution_name] = {
                        "success": False,
                        "error": str(e)
                    }
                    overall_success = False
                
                time.sleep(0.5)  # Brief pause between captures
            
            return CameraTestResult(
                test_name=test_name,
                success=overall_success,
                duration=time.time() - start_time,
                details={"resolution_tests": results},
                error_message=None if overall_success else "One or more resolution tests failed"
            )
        
        except Exception as e:
            return CameraTestResult(
                test_name=test_name,
                success=False,
                duration=time.time() - start_time,
                details={},
                error_message=str(e)
            )
    
    def analyze_image_quality(self, image_path: Path) -> Optional[ImageQualityMetrics]:
        """Analyze image quality metrics."""
        try:
            import cv2
            import numpy as np
            
            if not image_path.exists():
                return None
            
            # Read image
            image = cv2.imread(str(image_path))
            if image is None:
                return None
            
            # Convert to grayscale for analysis
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Calculate metrics
            brightness = np.mean(gray)
            contrast = gray.std()
            
            # Sharpness using Laplacian variance
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            sharpness = laplacian_var
            
            # Noise estimation (simplified)
            noise_level = np.std(gray - cv2.GaussianBlur(gray, (5, 5), 0))
            
            return ImageQualityMetrics(
                brightness=brightness,
                contrast=contrast,
                sharpness=sharpness,
                noise_level=noise_level,
                file_size=image_path.stat().st_size,
                resolution=(image.shape[1], image.shape[0]),
                format=image_path.suffix.lower()
            )
        
        except ImportError:
            logger.warning("OpenCV not available for image quality analysis")
            return None
        except Exception as e:
            logger.error(f"Error analyzing image quality: {e}")
            return None
    
    def test_image_quality(self) -> CameraTestResult:
        """Test and analyze image quality."""
        start_time = time.time()
        test_name = "image_quality"
        
        if not self.camera_module:
            return CameraTestResult(
                test_name=test_name,
                success=False,
                duration=0,
                details={},
                error_message="Camera module not available"
            )
        
        try:
            # Capture test image
            output_path = self.output_dir / f"quality_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            captured_path = self.camera_module.capture_image(self.output_dir)
            
            if not captured_path.exists():
                return CameraTestResult(
                    test_name=test_name,
                    success=False,
                    duration=time.time() - start_time,
                    details={},
                    error_message="Failed to capture test image"
                )
            
            # Analyze quality
            quality_metrics = self.analyze_image_quality(captured_path)
            
            success = quality_metrics is not None
            details = {
                "image_path": str(captured_path),
                "quality_metrics": {
                    "brightness": quality_metrics.brightness,
                    "contrast": quality_metrics.contrast,
                    "sharpness": quality_metrics.sharpness,
                    "noise_level": quality_metrics.noise_level,
                    "file_size": quality_metrics.file_size,
                    "resolution": quality_metrics.resolution,
                    "format": quality_metrics.format
                } if quality_metrics else None
            }
            
            return CameraTestResult(
                test_name=test_name,
                success=success,
                duration=time.time() - start_time,
                details=details,
                error_message=None if success else "Image quality analysis failed"
            )
        
        except Exception as e:
            return CameraTestResult(
                test_name=test_name,
                success=False,
                duration=time.time() - start_time,
                details={},
                error_message=str(e)
            )
    
    def run_performance_benchmark(self, num_captures: int = 10) -> CameraTestResult:
        """Run camera performance benchmark."""
        start_time = time.time()
        test_name = "performance_benchmark"
        
        if not self.camera_module:
            return CameraTestResult(
                test_name=test_name,
                success=False,
                duration=0,
                details={},
                error_message="Camera module not available"
            )
        
        try:
            capture_times = []
            successful_captures = 0
            
            camera_manager = self.camera_module.get_camera_manager()
            
            for i in range(num_captures):
                capture_start = time.time()
                output_path = self.output_dir / f"benchmark_{i:03d}_{datetime.now().strftime('%H%M%S')}.jpg"
                
                try:
                    success = camera_manager.capture_image(output_path, (1280, 720))
                    capture_time = time.time() - capture_start
                    
                    if success and output_path.exists():
                        capture_times.append(capture_time)
                        successful_captures += 1
                        # Clean up immediately to save space
                        output_path.unlink()
                
                except Exception as e:
                    logger.debug(f"Benchmark capture {i} failed: {e}")
                
                time.sleep(0.1)  # Brief pause between captures
            
            if capture_times:
                avg_time = sum(capture_times) / len(capture_times)
                min_time = min(capture_times)
                max_time = max(capture_times)
                fps = 1.0 / avg_time if avg_time > 0 else 0
            else:
                avg_time = min_time = max_time = fps = 0
            
            success = successful_captures > 0
            
            details = {
                "total_captures": num_captures,
                "successful_captures": successful_captures,
                "success_rate": successful_captures / num_captures,
                "average_capture_time": avg_time,
                "min_capture_time": min_time,
                "max_capture_time": max_time,
                "estimated_fps": fps
            }
            
            return CameraTestResult(
                test_name=test_name,
                success=success,
                duration=time.time() - start_time,
                details=details,
                error_message=None if success else "No successful captures"
            )
        
        except Exception as e:
            return CameraTestResult(
                test_name=test_name,
                success=False,
                duration=time.time() - start_time,
                details={},
                error_message=str(e)
            )
    
    def run_all_tests(self) -> Dict[str, CameraTestResult]:
        """Run all camera validation tests."""
        logger.info("Starting comprehensive camera validation")
        
        tests = [
            ("libcamera_hello", lambda: self.run_libcamera_hello_test()),
            ("sensor_detection", lambda: self.test_sensor_detection()),
            ("camera_capture", lambda: self.test_camera_capture()),
            ("multiple_resolutions", lambda: self.test_multiple_resolutions()),
            ("image_quality", lambda: self.test_image_quality()),
            ("night_vision", lambda: self.test_night_vision()),
            ("performance_benchmark", lambda: self.run_performance_benchmark()),
        ]
        
        results = {}
        
        for test_name, test_func in tests:
            logger.info(f"Running test: {test_name}")
            result = test_func()
            results[test_name] = result
            self.test_results.append(result)
            
            status = "✅ PASSED" if result.success else "❌ FAILED"
            logger.info(f"Test {test_name}: {status} ({result.duration:.2f}s)")
        
        return results
    
    def generate_report(self) -> str:
        """Generate a comprehensive test report."""
        if not self.test_results:
            return "No tests have been run."
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for r in self.test_results if r.success)
        failed_tests = total_tests - passed_tests
        
        report = [
            "🔍 NightScanPi Camera Validation Report",
            "=" * 50,
            f"Total Tests: {total_tests}",
            f"Passed: {passed_tests} ✅",
            f"Failed: {failed_tests} ❌",
            f"Success Rate: {passed_tests/total_tests:.1%}",
            "",
            "Test Results:",
            "-" * 20
        ]
        
        for result in self.test_results:
            status = "✅ PASSED" if result.success else "❌ FAILED"
            report.append(f"{result.test_name}: {status} ({result.duration:.2f}s)")
            
            if result.error_message:
                report.append(f"  Error: {result.error_message}")
            
            if result.details:
                for key, value in result.details.items():
                    if isinstance(value, dict):
                        report.append(f"  {key}:")
                        for k, v in value.items():
                            report.append(f"    {k}: {v}")
                    else:
                        report.append(f"  {key}: {value}")
            
            report.append("")
        
        return "\n".join(report)


def main():
    """Main function for running camera validation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="NightScanPi Camera Validation Tool")
    parser.add_argument("--output-dir", default="camera_validation", help="Output directory for test files")
    parser.add_argument("--test", choices=[
        "libcamera", "sensor", "capture", "resolutions", "quality", "night-vision", "performance", "all"
    ], default="all", help="Specific test to run")
    parser.add_argument("--num-captures", type=int, default=10, help="Number of captures for performance test")
    
    args = parser.parse_args()
    
    validator = CameraValidator(Path(args.output_dir))
    
    if args.test == "all":
        results = validator.run_all_tests()
    elif args.test == "libcamera":
        results = {"libcamera_hello": validator.run_libcamera_hello_test()}
    elif args.test == "sensor":
        results = {"sensor_detection": validator.test_sensor_detection()}
    elif args.test == "capture":
        results = {"camera_capture": validator.test_camera_capture()}
    elif args.test == "resolutions":
        results = {"multiple_resolutions": validator.test_multiple_resolutions()}
    elif args.test == "quality":
        results = {"image_quality": validator.test_image_quality()}
    elif args.test == "night-vision":
        results = {"night_vision": validator.test_night_vision()}
    elif args.test == "performance":
        results = {"performance_benchmark": validator.run_performance_benchmark(args.num_captures)}
    
    # Print report
    print(validator.generate_report())
    
    # Return appropriate exit code
    all_passed = all(result.success for result in results.values())
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())