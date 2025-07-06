#!/usr/bin/env python3
"""
Camera Test Utility for NightScanPi
Tests camera functionality and provides diagnostic information.
"""

import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime

# Add the program directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from camera_trigger import get_camera_info, test_camera, capture_image, get_camera_manager


def print_camera_status():
    """Print detailed camera status and capabilities."""
    print("🔍 NightScanPi Camera Diagnostic Tool")
    print("=" * 50)
    
    info = get_camera_info()
    
    print(f"📋 Camera API Status:")
    print(f"  • picamera2 available: {'✅' if info['picamera2_available'] else '❌'}")
    print(f"  • picamera (legacy) available: {'✅' if info['picamera_available'] else '❌'}")
    print(f"  • Active API: {info['active_api'] or 'None'}")
    print(f"  • Camera working: {'✅' if info['camera_working'] else '❌'}")
    
    if info['active_api']:
        camera_manager = get_camera_manager()
        if camera_manager.api_type == "picamera2":
            print(f"  • Using modern libcamera stack (RECOMMENDED)")
        elif camera_manager.api_type == "picamera":
            print(f"  • Using legacy picamera (UPDATE RECOMMENDED)")
    
    # Display sensor information
    if info.get('sensor_type'):
        print(f"\n📸 Camera Sensor Information:")
        print(f"  • Detected Sensor: {info['sensor_type'].upper()}")
        
        sensor_info = info.get('sensor_info')
        if sensor_info:
            print(f"  • Model: {sensor_info['name']} ({sensor_info['model']})")
            print(f"  • Max Resolution: {sensor_info['resolution'][0]}x{sensor_info['resolution'][1]}")
            print(f"  • Capabilities: {', '.join(sensor_info['capabilities'])}")
            print(f"  • IR-CUT Support: {'✅' if sensor_info['ir_cut_support'] else '❌'}")
            print(f"  • Night Vision: {'✅' if sensor_info['night_vision'] else '❌'}")
            print(f"  • Boot Config: dtoverlay={sensor_info['dtoverlay']}")
            
        recommended = info.get('recommended_settings')
        if recommended:
            print(f"\n⚙️ Recommended Settings:")
            print(f"  • Default Resolution: {recommended['resolution_default'][0]}x{recommended['resolution_default'][1]}")
            print(f"  • Max Framerate: {recommended['framerate_max']} fps")
            print(f"  • GPU Memory: {recommended['gpu_mem']} MB")
            print(f"  • Night Mode: {'✅' if recommended['night_mode'] else '❌'}")
    else:
        print(f"\n⚠️ Camera sensor detection failed or no camera detected")
    
    # Display night vision status
    print_night_vision_status()
    
    # Display Pi Zero optimization status
    print_pi_zero_status()
    
    print()


def run_camera_test():
    """Run comprehensive camera test."""
    print("🧪 Running Camera Test...")
    
    success = test_camera()
    if success:
        print("✅ Camera test PASSED")
        return True
    else:
        print("❌ Camera test FAILED")
        return False


def capture_test_image(output_dir: str = "test_images"):
    """Capture a test image and display information."""
    print(f"📸 Capturing test image to {output_dir}/...")
    
    try:
        output_path = capture_image(Path(output_dir))
        print(f"✅ Image captured successfully: {output_path}")
        
        # Display file info
        if output_path.exists():
            file_size = output_path.stat().st_size
            print(f"  • File size: {file_size:,} bytes ({file_size/1024:.1f} KB)")
            print(f"  • Timestamp: {datetime.fromtimestamp(output_path.stat().st_mtime)}")
        
        return str(output_path)
        
    except Exception as e:
        print(f"❌ Image capture failed: {e}")
        return None


def test_sensor_detection():
    """Test comprehensive sensor detection."""
    print("🔍 Testing Camera Sensor Detection...")
    
    try:
        # Import here to avoid issues if module doesn't exist
        sys.path.insert(0, str(Path(__file__).parent))
        from camera_sensor_detector import CameraSensorDetector
        
        detector = CameraSensorDetector()
        
        print("📋 Running detection methods:")
        
        # Test individual detection methods
        methods = [
            ("libcamera", detector.detect_via_libcamera),
            ("dmesg", detector.detect_via_dmesg),
            ("config.txt", detector.detect_via_config_txt),
            ("device_tree", detector.detect_via_device_tree),
            ("vcgencmd", detector.detect_via_vcgencmd),
        ]
        
        results = {}
        for method_name, method in methods:
            try:
                result = method()
                results[method_name] = result
                if result:
                    print(f"  ✅ {method_name}: {result}")
                else:
                    print(f"  ❌ {method_name}: no detection")
            except Exception as e:
                results[method_name] = f"Error: {e}"
                print(f"  ⚠️ {method_name}: error - {e}")
        
        # Run comprehensive detection
        print(f"\n🎯 Comprehensive Detection:")
        final_sensor = detector.detect_sensor()
        if final_sensor:
            sensor_info = detector.get_sensor_info(final_sensor)
            print(f"  • Final Result: {final_sensor.upper()}")
            if sensor_info:
                print(f"  • Model: {sensor_info.model}")
                print(f"  • Resolution: {sensor_info.resolution[0]}x{sensor_info.resolution[1]}")
                print(f"  • IR-CUT: {'Yes' if sensor_info.ir_cut_support else 'No'}")
                print(f"  • Night Vision: {'Yes' if sensor_info.night_vision else 'No'}")
        else:
            print(f"  • Final Result: No sensor detected")
        
        return final_sensor is not None
        
    except ImportError as e:
        print(f"❌ Sensor detection module not available: {e}")
        return False
    except Exception as e:
        print(f"❌ Sensor detection failed: {e}")
        return False


def check_system_requirements():
    """Check system requirements for camera functionality."""
    print("🔧 Checking System Requirements...")
    
    requirements = {
        "Python 3.7+": sys.version_info >= (3, 7),
        "pathlib": True,  # Built-in since Python 3.4
    }
    
    # Check optional dependencies
    try:
        import picamera2
        requirements["picamera2"] = True
    except ImportError:
        requirements["picamera2"] = False
    
    try:
        import picamera
        requirements["picamera (legacy)"] = True
    except ImportError:
        requirements["picamera (legacy)"] = False
    
    # Check if running on Raspberry Pi
    try:
        with open("/proc/device-tree/model", "r") as f:
            model = f.read().strip()
            requirements[f"Raspberry Pi ({model[:30]}...)"] = "raspberry" in model.lower()
    except:
        requirements["Raspberry Pi detection"] = False
    
    print("📋 System Requirements:")
    for req, status in requirements.items():
        icon = "✅" if status else "❌"
        print(f"  • {req}: {icon}")
    
    return all(requirements.values())


def print_night_vision_status():
    """Print night vision system status."""
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        from utils.ir_night_vision import get_night_vision_status
        
        status = get_night_vision_status()
        
        print(f"\n🌙 Night Vision System:")
        print(f"  • GPIO Available: {'✅' if status['gpio_available'] else '❌'}")
        print(f"  • GPIO Initialized: {'✅' if status['gpio_initialized'] else '❌'}")
        print(f"  • Current Mode: {status['current_mode'] or 'Unknown'}")
        print(f"  • Night Time: {'✅' if status['is_night_time'] else '❌'}")
        print(f"  • IR LEDs: {'✅' if status['leds_enabled'] else '❌'}")
        print(f"  • Auto Mode: {'✅' if status['auto_mode'] else '❌'}")
        print(f"  • IR-CUT Pin: GPIO {status['ircut_pin']}")
        print(f"  • IR LED Pin: GPIO {status['irled_pin']}")
        print(f"  • LED Brightness: {status['led_brightness']:.1%}")
        print(f"  • LED Feature: {'✅' if status['led_feature_enabled'] else '❌'}")
        
    except ImportError:
        print(f"\n🌙 Night Vision System: ❌ Not available")
    except Exception as e:
        print(f"\n🌙 Night Vision System: ⚠️ Error - {e}")


def print_pi_zero_status():
    """Print Pi Zero optimization status."""
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        from utils.pi_zero_optimizer import get_memory_status
        
        status = get_memory_status()
        
        print(f"\n🔧 Pi Zero 2W Optimizations:")
        print(f"  • Pi Zero Detected: {'✅' if status['is_pi_zero'] else '❌'}")
        print(f"  • Optimizations Active: {'✅' if status['optimizations_active'] else '❌'}")
        
        memory = status['memory']
        print(f"  • Memory Total: {memory['total_mb']:.0f}MB")
        print(f"  • Memory Used: {memory['used_mb']:.0f}MB ({memory['percent_used']:.1f}%)")
        print(f"  • Memory Available: {memory['available_mb']:.0f}MB")
        print(f"  • Swap Usage: {memory['swap_mb']:.0f}MB")
        
        if status['optimizations_active']:
            settings = status['settings']
            print(f"  • Max Camera Res: {settings['max_camera_resolution']}")
            print(f"  • GPU Memory: {settings['gpu_memory_mb']}MB")
            print(f"  • Max Audio Buffer: {settings['max_audio_buffer']}")
            print(f"  • Memory Monitoring: {'✅' if settings['monitoring_active'] else '❌'}")
        
    except ImportError:
        print(f"\n🔧 Pi Zero 2W Optimizations: ❌ Not available")
    except Exception as e:
        print(f"\n🔧 Pi Zero 2W Optimizations: ⚠️ Error - {e}")


def test_pi_zero_optimizations():
    """Test Pi Zero memory optimization functionality."""
    print("🔧 Testing Pi Zero 2W Optimizations...")
    
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        from utils.pi_zero_optimizer import get_optimizer, optimize_camera_resolution, optimize_audio_buffer_size, cleanup_memory
        
        optimizer = get_optimizer()
        
        if not optimizer.is_pi_zero:
            print("ℹ️ Not running on Pi Zero - optimization tests limited")
        
        print("📋 Testing optimization functions:")
        
        # Test resolution optimization
        test_resolutions = [(1920, 1080), (1280, 720), (640, 480)]
        print("  • Camera resolution optimization:")
        for res in test_resolutions:
            optimized = optimize_camera_resolution(res)
            status = "→" if optimized != res else "="
            print(f"    {res} {status} {optimized}")
        
        # Test audio buffer optimization
        test_buffers = [1024, 2048, 4096]
        print("  • Audio buffer optimization:")
        for buffer in test_buffers:
            optimized = optimize_audio_buffer_size(buffer)
            status = "→" if optimized != buffer else "="
            print(f"    {buffer} {status} {optimized}")
        
        # Test memory cleanup
        print("  • Memory cleanup test...")
        initial_memory = optimizer.get_memory_info()
        cleanup_memory(force=True)
        final_memory = optimizer.get_memory_info()
        freed = initial_memory.used_mb - final_memory.used_mb
        print(f"    Freed {freed:.1f}MB of memory")
        
        print("✅ Pi Zero optimization test PASSED")
        return True
        
    except ImportError:
        print("❌ Pi Zero optimizer module not available")
        return False
    except Exception as e:
        print(f"❌ Pi Zero optimization test FAILED: {e}")
        return False


def run_camera_validation():
    """Run comprehensive camera validation tests."""
    print("🔍 Running Comprehensive Camera Validation...")
    
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        from utils.camera_validator import CameraValidator
        
        validator = CameraValidator(Path("camera_validation"))
        results = validator.run_all_tests()
        
        # Display summary
        passed = sum(1 for r in results.values() if r.success)
        total = len(results)
        
        print(f"\n📊 Validation Summary:")
        print(f"  • Total Tests: {total}")
        print(f"  • Passed: {passed} ✅")
        print(f"  • Failed: {total - passed} ❌")
        print(f"  • Success Rate: {passed/total:.1%}")
        
        if passed == total:
            print("✅ All validation tests PASSED")
        else:
            print("⚠️ Some validation tests FAILED")
            print("\nFailed tests:")
            for name, result in results.items():
                if not result.success:
                    print(f"  • {name}: {result.error_message}")
        
        return passed == total
        
    except ImportError:
        print("❌ Camera validator module not available")
        return False
    except Exception as e:
        print(f"❌ Camera validation FAILED: {e}")
        return False


def test_night_vision_control():
    """Test night vision control functionality."""
    print("🌙 Testing Night Vision Control...")
    
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        from utils.ir_night_vision import get_night_vision
        
        nv = get_night_vision()
        
        if not nv.gpio_initialized:
            print("❌ GPIO not initialized - hardware control not available")
            return False
        
        print("📋 Testing mode switching:")
        
        # Test day mode
        print("  • Setting day mode...")
        if nv.set_day_mode():
            print("  ✅ Day mode set successfully")
        else:
            print("  ❌ Failed to set day mode")
            return False
        
        time.sleep(1)
        
        # Test night mode
        print("  • Setting night mode...")
        if nv.set_night_mode():
            print("  ✅ Night mode set successfully")
        else:
            print("  ❌ Failed to set night mode")
            return False
        
        time.sleep(1)
        
        # Test auto mode
        print("  • Testing auto mode...")
        if nv.auto_adjust_mode():
            print(f"  ✅ Auto mode set to: {nv.current_mode}")
        else:
            print("  ❌ Failed to set auto mode")
            return False
        
        # Test LED control
        print("  • Testing LED brightness control...")
        original_brightness = nv.config.irled_brightness
        nv.set_led_brightness(0.5)
        time.sleep(0.5)
        nv.set_led_brightness(original_brightness)
        print("  ✅ LED brightness control working")
        
        print("✅ Night vision control test PASSED")
        return True
        
    except ImportError:
        print("❌ Night vision module not available")
        return False
    except Exception as e:
        print(f"❌ Night vision control test FAILED: {e}")
        return False


def main():
    """Main diagnostic function."""
    parser = argparse.ArgumentParser(
        description="NightScanPi Camera Diagnostic Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python camera_test.py --status              # Show camera status
  python camera_test.py --test                # Run camera test
  python camera_test.py --capture             # Capture test image
  python camera_test.py --detect-sensor       # Test sensor detection
  python camera_test.py --test-night-vision   # Test IR-CUT and LED control
  python camera_test.py --test-pi-zero        # Test Pi Zero 2W optimizations
  python camera_test.py --validate            # Run comprehensive camera validation
  python camera_test.py --all                 # Run all diagnostics
  python camera_test.py --json                # Output in JSON format
        """
    )
    
    parser.add_argument("--status", action="store_true", 
                       help="Show camera status and API information")
    parser.add_argument("--test", action="store_true",
                       help="Run camera functionality test")
    parser.add_argument("--capture", action="store_true",
                       help="Capture a test image")
    parser.add_argument("--detect-sensor", action="store_true",
                       help="Test sensor detection specifically")
    parser.add_argument("--test-night-vision", action="store_true",
                       help="Test night vision control functionality")
    parser.add_argument("--test-pi-zero", action="store_true",
                       help="Test Pi Zero 2W memory optimizations")
    parser.add_argument("--validate", action="store_true",
                       help="Run comprehensive camera validation tests")
    parser.add_argument("--all", action="store_true",
                       help="Run all diagnostics")
    parser.add_argument("--json", action="store_true",
                       help="Output results in JSON format")
    parser.add_argument("--output-dir", default="test_images",
                       help="Directory for test images (default: test_images)")
    
    args = parser.parse_args()
    
    # If no specific action, show status by default
    if not any([args.status, args.test, args.capture, args.detect_sensor, args.test_night_vision, args.test_pi_zero, args.validate, args.all]):
        args.status = True
    
    results = {}
    
    if args.json:
        # JSON output mode
        if args.status or args.all:
            results["camera_info"] = get_camera_info()
            results["system_requirements"] = {}
        
        if args.test or args.all:
            results["camera_test"] = test_camera()
        
        if args.capture or args.all:
            try:
                output_path = capture_image(Path(args.output_dir))
                results["test_image"] = {
                    "success": True,
                    "path": str(output_path),
                    "size": output_path.stat().st_size if output_path.exists() else 0
                }
            except Exception as e:
                results["test_image"] = {
                    "success": False,
                    "error": str(e)
                }
        
        if args.detect_sensor or args.all:
            results["sensor_detection"] = test_sensor_detection()
        
        if args.test_night_vision or args.all:
            results["night_vision_test"] = test_night_vision_control()
        
        if args.test_pi_zero or args.all:
            results["pi_zero_test"] = test_pi_zero_optimizations()
        
        if args.validate or args.all:
            results["validation_test"] = run_camera_validation()
        
        print(json.dumps(results, indent=2))
        
    else:
        # Human-readable output mode
        if args.status or args.all:
            print_camera_status()
            
        if args.test or args.all:
            test_success = run_camera_test()
            print()
            
        if args.capture or args.all:
            capture_test_image(args.output_dir)
            print()
            
        if args.detect_sensor or args.all:
            test_sensor_detection()
            print()
        
        if args.test_night_vision or args.all:
            test_night_vision_control()
            print()
        
        if args.test_pi_zero or args.all:
            test_pi_zero_optimizations()
            print()
        
        if args.validate or args.all:
            run_camera_validation()
            print()
        
        if args.all:
            check_system_requirements()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())