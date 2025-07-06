#!/usr/bin/env python3
"""
Camera Test Utility for NightScanPi
Tests camera functionality and provides diagnostic information.
"""

import sys
import json
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


def main():
    """Main diagnostic function."""
    parser = argparse.ArgumentParser(
        description="NightScanPi Camera Diagnostic Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python camera_test.py --status          # Show camera status
  python camera_test.py --test            # Run camera test
  python camera_test.py --capture         # Capture test image
  python camera_test.py --all             # Run all diagnostics
  python camera_test.py --json            # Output in JSON format
        """
    )
    
    parser.add_argument("--status", action="store_true", 
                       help="Show camera status and API information")
    parser.add_argument("--test", action="store_true",
                       help="Run camera functionality test")
    parser.add_argument("--capture", action="store_true",
                       help="Capture a test image")
    parser.add_argument("--all", action="store_true",
                       help="Run all diagnostics")
    parser.add_argument("--json", action="store_true",
                       help="Output results in JSON format")
    parser.add_argument("--output-dir", default="test_images",
                       help="Directory for test images (default: test_images)")
    
    args = parser.parse_args()
    
    # If no specific action, show status by default
    if not any([args.status, args.test, args.capture, args.all]):
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
        
        if args.all:
            check_system_requirements()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())