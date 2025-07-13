#!/usr/bin/env python3
"""
NightScan Camera Hello - rpicam-hello equivalent for NightScanPi

A comprehensive camera testing tool that provides similar functionality to rpicam-hello
but with additional NightScan-specific features like IR-CUT testing and sensor detection.

Usage:
    python nightscan_camera_hello.py [options]

Examples:
    python nightscan_camera_hello.py --timeout 5000    # 5 second preview
    python nightscan_camera_hello.py --capture         # Capture test image
    python nightscan_camera_hello.py --list-cameras    # List available cameras
    python nightscan_camera_hello.py --test-all        # Run all tests
"""

import sys
import time
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

# Add the program directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from utils.camera_validator import CameraValidator
from camera_trigger import get_camera_info, capture_image, get_camera_manager


def list_cameras():
    """List available cameras and their capabilities."""
    print("🔍 NightScan Camera Detection")
    print("=" * 40)
    
    # Try libcamera list
    commands = ["rpicam-hello", "libcamera-hello"]
    
    for cmd in commands:
        try:
            result = subprocess.run([cmd, "--list-cameras"], capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                print(f"📋 {cmd} --list-cameras:")
                print(result.stdout)
                break
        except (subprocess.TimeoutExpired, FileNotFoundError):
            continue
    else:
        print("❌ No libcamera commands available")
    
    # NightScan camera info
    print("\n📸 NightScan Camera Information:")
    try:
        info = get_camera_info()
        
        print(f"  • API Available: picamera2={'✅' if info['picamera2_available'] else '❌'}, picamera={'✅' if info['picamera_available'] else '❌'}")
        print(f"  • Active API: {info['active_api'] or 'None'}")
        print(f"  • Camera Working: {'✅' if info['camera_working'] else '❌'}")
        
        if info.get('sensor_type'):
            print(f"  • Detected Sensor: {info['sensor_type'].upper()}")
            sensor_info = info.get('sensor_info')
            if sensor_info:
                print(f"  • Model: {sensor_info['model']}")
                print(f"  • Resolution: {sensor_info['resolution'][0]}x{sensor_info['resolution'][1]}")
                print(f"  • IR-CUT Support: {'✅' if sensor_info['ir_cut_support'] else '❌'}")
                print(f"  • Night Vision: {'✅' if sensor_info['night_vision'] else '❌'}")
    
    except Exception as e:
        print(f"  ❌ Error getting camera info: {e}")


def run_preview(timeout_ms: int = 5000):
    """Run camera preview similar to rpicam-hello."""
    print(f"🎥 Starting {timeout_ms}ms camera preview...")
    
    # Try native libcamera command first
    commands = ["rpicam-hello", "libcamera-hello"]
    
    for cmd in commands:
        try:
            result = subprocess.run([
                cmd, 
                "--timeout", str(timeout_ms),
                "--width", "1280",
                "--height", "720"
            ], timeout=(timeout_ms / 1000) + 5)
            
            if result.returncode == 0:
                print(f"✅ Preview completed successfully using {cmd}")
                return True
            
        except (subprocess.TimeoutExpired, FileNotFoundError):
            continue
    
    print("❌ Native preview commands not available")
    
    # Fallback to NightScan capture
    try:
        print("📸 Fallback: Capturing test image with NightScan...")
        output_path = capture_image(Path("test_images"))
        print(f"✅ Test image captured: {output_path}")
        return True
    
    except Exception as e:
        print(f"❌ Capture failed: {e}")
        return False


def capture_test_image(output_path: str = None):
    """Capture a test image."""
    if output_path:
        output_dir = Path(output_path).parent
        filename = Path(output_path).name
    else:
        output_dir = Path("test_images")
        filename = f"nightscan_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    
    print(f"📸 Capturing test image to {output_dir / filename}...")
    
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        captured_path = capture_image(output_dir)
        
        # Rename to desired filename if specified
        if output_path and captured_path.name != filename:
            final_path = output_dir / filename
            captured_path.rename(final_path)
            captured_path = final_path
        
        file_size = captured_path.stat().st_size
        print(f"✅ Image captured successfully:")
        print(f"   📁 Path: {captured_path}")
        print(f"   📏 Size: {file_size:,} bytes ({file_size/1024:.1f} KB)")
        
        # Basic image info
        try:
            import cv2
            img = cv2.imread(str(captured_path))
            if img is not None:
                height, width = img.shape[:2]
                print(f"   🖼️  Resolution: {width}x{height}")
        except ImportError:
            pass
        
        return str(captured_path)
    
    except Exception as e:
        print(f"❌ Capture failed: {e}")
        return None


def run_comprehensive_tests():
    """Run comprehensive camera validation tests."""
    print("🧪 Running Comprehensive Camera Tests")
    print("=" * 50)
    
    validator = CameraValidator(Path("camera_validation"))
    results = validator.run_all_tests()
    
    print("\n" + validator.generate_report())
    
    # Summary
    passed = sum(1 for r in results.values() if r.success)
    total = len(results)
    
    if passed == total:
        print(f"\n🎉 All tests passed! ({passed}/{total})")
        return True
    else:
        print(f"\n⚠️ {total - passed} test(s) failed. ({passed}/{total} passed)")
        return False


def check_system_requirements():
    """Check system requirements and camera availability."""
    print("🔧 System Requirements Check")
    print("=" * 30)
    
    # Check Python version
    import sys
    print(f"🐍 Python: {sys.version.split()[0]} {'✅' if sys.version_info >= (3, 7) else '❌'}")
    
    # Check if on Raspberry Pi
    try:
        with open("/proc/device-tree/model", "r") as f:
            model = f.read().strip()
            print(f"🍓 Hardware: {model[:50]}...")
            is_pi = "raspberry" in model.lower()
            print(f"🎯 Raspberry Pi: {'✅' if is_pi else '❌'}")
    except:
        print("🎯 Raspberry Pi: ❌ Not detected")
    
    # Check libcamera commands
    commands = ["rpicam-hello", "libcamera-hello"]
    for cmd in commands:
        available = subprocess.run(["which", cmd], capture_output=True).returncode == 0
        print(f"📹 {cmd}: {'✅' if available else '❌'}")
    
    # Check Python camera modules
    try:
        import picamera2
        print("📚 picamera2: ✅")
    except ImportError:
        print("📚 picamera2: ❌")
    
    try:
        import picamera
        print("📚 picamera: ✅")
    except ImportError:
        print("📚 picamera: ❌")
    
    # Check OpenCV
    try:
        import cv2
        print(f"👁️ OpenCV: ✅ ({cv2.__version__})")
    except ImportError:
        print("👁️ OpenCV: ❌")
    
    print()


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(
        description="NightScan Camera Hello - Enhanced rpicam-hello equivalent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python nightscan_camera_hello.py --timeout 5000      # 5 second preview
  python nightscan_camera_hello.py --capture           # Capture test image
  python nightscan_camera_hello.py --list-cameras      # List cameras
  python nightscan_camera_hello.py --test-all          # Run all validation tests
  python nightscan_camera_hello.py --check-system      # Check system requirements
  
  # rpicam-hello compatibility:
  python nightscan_camera_hello.py --timeout 10000 --width 1920 --height 1080
        """
    )
    
    # Main actions (mutually exclusive)
    action_group = parser.add_mutually_exclusive_group()
    action_group.add_argument("--list-cameras", action="store_true",
                             help="List available cameras and their capabilities")
    action_group.add_argument("--capture", action="store_true",
                             help="Capture a test image")
    action_group.add_argument("--test-all", action="store_true",
                             help="Run comprehensive camera validation tests")
    action_group.add_argument("--check-system", action="store_true",
                             help="Check system requirements and camera availability")
    
    # Preview options (default action)
    parser.add_argument("--timeout", type=int, default=5000,
                       help="Time in milliseconds for preview (default: 5000)")
    parser.add_argument("--width", type=int, default=1280,
                       help="Preview width (default: 1280)")
    parser.add_argument("--height", type=int, default=720,
                       help="Preview height (default: 720)")
    
    # Capture options
    parser.add_argument("--output", "-o", type=str,
                       help="Output filename for captured image")
    
    # Compatibility options
    parser.add_argument("--nopreview", action="store_true",
                       help="Skip preview (rpicam-hello compatibility)")
    
    args = parser.parse_args()
    
    try:
        if args.list_cameras:
            list_cameras()
            return 0
        
        elif args.capture:
            success = capture_test_image(args.output)
            return 0 if success else 1
        
        elif args.test_all:
            success = run_comprehensive_tests()
            return 0 if success else 1
        
        elif args.check_system:
            check_system_requirements()
            return 0
        
        else:
            # Default action: run preview
            if args.nopreview:
                print("ℹ️ Preview skipped (--nopreview)")
                return 0
            else:
                success = run_preview(args.timeout)
                return 0 if success else 1
    
    except KeyboardInterrupt:
        print("\n⏹️ Interrupted by user")
        return 130
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())