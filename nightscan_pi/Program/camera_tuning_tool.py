#!/usr/bin/env python3
"""
NightScan Camera Tuning Tool

Interactive tool for optimizing camera settings and generating custom tuning configurations.
Provides real-time preview of tuning adjustments and image quality metrics.

Usage:
    python camera_tuning_tool.py [options]

Examples:
    python camera_tuning_tool.py --auto-tune           # Auto-optimize for current conditions
    python camera_tuning_tool.py --preset high_quality # Apply quality preset
    python camera_tuning_tool.py --generate-tuning     # Generate libcamera tuning file
    python camera_tuning_tool.py --interactive         # Interactive tuning mode
"""

import sys
import time
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

# Add the program directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from .utils.camera_tuning import (
    get_tuning_manager, TuningParameters, get_optimal_tuning
)
from .camera_trigger import get_camera_manager, capture_image
from .camera_sensor_detector import detect_camera_sensor


class CameraTuningTool:
    """Interactive camera tuning and optimization tool."""
    
    def __init__(self):
        self.tuning_manager = get_tuning_manager()
        self.camera_manager = None
        self.current_sensor = None
        self.current_tuning = None
        
        # Initialize camera system
        self._initialize_camera()
    
    def _initialize_camera(self):
        """Initialize camera and detect sensor."""
        try:
            self.camera_manager = get_camera_manager()
            if not self.camera_manager.is_available():
                print("❌ No camera available")
                return False
            
            self.current_sensor = detect_camera_sensor()
            if self.current_sensor:
                print(f"📸 Detected sensor: {self.current_sensor.upper()}")
            else:
                print("⚠️ Sensor detection failed, using IMX219 defaults")
                self.current_sensor = "imx219"
            
            return True
            
        except Exception as e:
            print(f"❌ Camera initialization failed: {e}")
            return False
    
    def show_current_tuning(self):
        """Display current tuning parameters."""
        if self.current_tuning is None:
            self.current_tuning = get_optimal_tuning(self.current_sensor)
        
        print("\n🎨 Current Tuning Parameters:")
        print("=" * 40)
        print(f"Sensor: {self.current_sensor.upper()}")
        print(f"Brightness: {self.current_tuning.brightness:.2f}")
        print(f"Contrast: {self.current_tuning.contrast:.2f}")
        print(f"Saturation: {self.current_tuning.saturation:.2f}")
        print(f"Sharpness: {self.current_tuning.sharpness:.2f}")
        print(f"Analogue Gain: {self.current_tuning.analogue_gain:.1f}")
        print(f"Auto Exposure: {'Enabled' if self.current_tuning.ae_enable else 'Disabled'}")
        print(f"Auto White Balance: {'Enabled' if self.current_tuning.awb_enable else 'Disabled'}")
        
        if self.current_tuning.exposure_time:
            print(f"Manual Exposure: {self.current_tuning.exposure_time} µs")
        if self.current_tuning.awb_mode is not None:
            awb_modes = ["Auto", "Incandescent", "Tungsten", "Fluorescent", "Indoor", "Daylight", "Cloudy"]
            mode_name = awb_modes[self.current_tuning.awb_mode] if self.current_tuning.awb_mode < len(awb_modes) else "Unknown"
            print(f"AWB Mode: {mode_name}")
        
        print(f"Night Mode: {'Enabled' if self.current_tuning.night_mode_enabled else 'Disabled'}")
        print()
    
    def capture_test_image(self, suffix: str = "") -> Optional[Path]:
        """Capture a test image with current tuning."""
        if not self.camera_manager or not self.camera_manager.is_available():
            print("❌ Camera not available")
            return None
        
        try:
            output_dir = Path("tuning_test_images")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            print(f"📸 Capturing test image{suffix}...")
            captured_path = capture_image(output_dir)
            
            # Rename with descriptive suffix
            if suffix:
                new_name = f"tuning_{timestamp}_{suffix}.jpg"
                new_path = captured_path.parent / new_name
                captured_path.rename(new_path)
                captured_path = new_path
            
            print(f"✅ Image saved: {captured_path}")
            return captured_path
            
        except Exception as e:
            print(f"❌ Capture failed: {e}")
            return None
    
    def analyze_image_quality(self, image_path: Path) -> Dict[str, float]:
        """Analyze image quality metrics."""
        try:
            import cv2
            import numpy as np
            
            image = cv2.imread(str(image_path))
            if image is None:
                return {}
            
            # Convert to grayscale for analysis
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Calculate metrics
            brightness = np.mean(gray) / 255.0
            contrast = gray.std() / 255.0
            
            # Sharpness using Laplacian variance
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            sharpness = laplacian_var / 10000.0  # Normalize
            
            # Noise estimation
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            noise = np.std(gray - blur) / 255.0
            
            return {
                "brightness": brightness,
                "contrast": contrast,
                "sharpness": sharpness,
                "noise": noise
            }
            
        except ImportError:
            print("⚠️ OpenCV not available for image analysis")
            return {}
        except Exception as e:
            print(f"⚠️ Image analysis failed: {e}")
            return {}
    
    def auto_tune(self) -> bool:
        """Automatically optimize tuning based on test images."""
        print("🔄 Starting automatic tuning optimization...")
        
        # Capture baseline image
        baseline_path = self.capture_test_image("baseline")
        if not baseline_path:
            return False
        
        baseline_metrics = self.analyze_image_quality(baseline_path)
        if not baseline_metrics:
            print("⚠️ Cannot analyze image quality, continuing with standard optimization")
            self.current_tuning = get_optimal_tuning(self.current_sensor, "auto")
            return True
        
        print(f"📊 Baseline metrics:")
        for metric, value in baseline_metrics.items():
            print(f"  {metric}: {value:.3f}")
        
        # Optimize based on metrics
        optimized_tuning = self.tuning_manager.optimize_for_conditions(
            brightness_target=baseline_metrics.get("brightness", 0.5),
            contrast_target=baseline_metrics.get("contrast", 0.7)
        )
        
        self.current_tuning = optimized_tuning
        
        # Capture optimized image
        optimized_path = self.capture_test_image("optimized")
        if optimized_path:
            optimized_metrics = self.analyze_image_quality(optimized_path)
            
            if optimized_metrics:
                print(f"📊 Optimized metrics:")
                for metric, value in optimized_metrics.items():
                    baseline_val = baseline_metrics.get(metric, 0)
                    change = value - baseline_val
                    direction = "↑" if change > 0.01 else "↓" if change < -0.01 else "="
                    print(f"  {metric}: {value:.3f} ({direction} {change:+.3f})")
        
        print("✅ Auto-tuning completed")
        return True
    
    def apply_preset(self, preset_name: str) -> bool:
        """Apply a quality preset."""
        sensor_tuning = self.tuning_manager.get_sensor_tuning(self.current_sensor)
        if not sensor_tuning:
            print(f"❌ No tuning available for sensor {self.current_sensor}")
            return False
        
        if preset_name not in sensor_tuning.quality_presets:
            available = list(sensor_tuning.quality_presets.keys())
            print(f"❌ Preset '{preset_name}' not available")
            print(f"Available presets: {', '.join(available)}")
            return False
        
        self.current_tuning = sensor_tuning.quality_presets[preset_name]
        print(f"✅ Applied preset: {preset_name}")
        return True
    
    def list_presets(self):
        """List available quality presets for current sensor."""
        sensor_tuning = self.tuning_manager.get_sensor_tuning(self.current_sensor)
        if not sensor_tuning:
            print(f"❌ No tuning available for sensor {self.current_sensor}")
            return
        
        print(f"\n📋 Available presets for {self.current_sensor.upper()}:")
        for preset_name in sensor_tuning.quality_presets.keys():
            print(f"  • {preset_name}")
        print()
    
    def generate_tuning_file(self, output_path: Optional[Path] = None) -> bool:
        """Generate libcamera tuning file."""
        try:
            if output_path is None:
                output_path = Path(f"{self.current_sensor}_nightscan_tuning.json")
            
            generated_path = self.tuning_manager.generate_tuning_file(
                self.current_sensor, output_path
            )
            
            print(f"✅ Generated tuning file: {generated_path}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to generate tuning file: {e}")
            return False
    
    def save_custom_tuning(self, name: str) -> bool:
        """Save current tuning as custom configuration."""
        if self.current_tuning is None:
            print("❌ No current tuning to save")
            return False
        
        try:
            saved_path = self.tuning_manager.save_custom_tuning(
                name, self.current_tuning, self.current_sensor
            )
            print(f"✅ Saved custom tuning '{name}' to {saved_path}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to save tuning: {e}")
            return False
    
    def interactive_mode(self):
        """Interactive tuning adjustment mode."""
        print("\n🎛️ Interactive Tuning Mode")
        print("=" * 30)
        print("Commands:")
        print("  show    - Show current tuning")
        print("  capture - Capture test image")
        print("  auto    - Auto-optimize tuning")
        print("  preset <name> - Apply preset")
        print("  list    - List available presets")
        print("  save <name> - Save current tuning")
        print("  generate - Generate tuning file")
        print("  quit    - Exit interactive mode")
        print()
        
        while True:
            try:
                command = input("tuning> ").strip().split()
                if not command:
                    continue
                
                cmd = command[0].lower()
                
                if cmd in ['quit', 'exit', 'q']:
                    break
                elif cmd == 'show':
                    self.show_current_tuning()
                elif cmd == 'capture':
                    self.capture_test_image("interactive")
                elif cmd == 'auto':
                    self.auto_tune()
                elif cmd == 'preset' and len(command) > 1:
                    self.apply_preset(command[1])
                elif cmd == 'list':
                    self.list_presets()
                elif cmd == 'save' and len(command) > 1:
                    self.save_custom_tuning(command[1])
                elif cmd == 'generate':
                    self.generate_tuning_file()
                else:
                    print("❌ Unknown command. Type 'quit' to exit.")
                    
            except KeyboardInterrupt:
                print("\n👋 Exiting interactive mode...")
                break
            except Exception as e:
                print(f"❌ Error: {e}")


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(
        description="NightScan Camera Tuning Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python camera_tuning_tool.py --auto-tune           # Auto-optimize settings
  python camera_tuning_tool.py --preset high_quality # Apply quality preset
  python camera_tuning_tool.py --list-presets        # Show available presets
  python camera_tuning_tool.py --generate-tuning     # Generate tuning file
  python camera_tuning_tool.py --interactive         # Interactive mode
        """
    )
    
    parser.add_argument("--auto-tune", action="store_true",
                       help="Automatically optimize tuning based on current conditions")
    parser.add_argument("--preset", type=str,
                       help="Apply a specific quality preset")
    parser.add_argument("--list-presets", action="store_true",
                       help="List available quality presets")
    parser.add_argument("--generate-tuning", action="store_true",
                       help="Generate libcamera tuning file")
    parser.add_argument("--save-tuning", type=str,
                       help="Save current tuning with given name")
    parser.add_argument("--output", "-o", type=str,
                       help="Output file path for generated tuning")
    parser.add_argument("--interactive", action="store_true",
                       help="Enter interactive tuning mode")
    parser.add_argument("--show", action="store_true",
                       help="Show current tuning parameters")
    parser.add_argument("--capture", action="store_true",
                       help="Capture test image with current tuning")
    
    args = parser.parse_args()
    
    # Create tuning tool instance
    tool = CameraTuningTool()
    
    # If no specific action, show current tuning
    if not any([args.auto_tune, args.preset, args.list_presets, args.generate_tuning,
                args.save_tuning, args.interactive, args.show, args.capture]):
        args.show = True
    
    try:
        success = True
        
        if args.show:
            tool.show_current_tuning()
        
        if args.list_presets:
            tool.list_presets()
        
        if args.preset:
            success &= tool.apply_preset(args.preset)
        
        if args.auto_tune:
            success &= tool.auto_tune()
        
        if args.capture:
            result = tool.capture_test_image("manual")
            success &= result is not None
        
        if args.generate_tuning:
            output_path = Path(args.output) if args.output else None
            success &= tool.generate_tuning_file(output_path)
        
        if args.save_tuning:
            success &= tool.save_custom_tuning(args.save_tuning)
        
        if args.interactive:
            tool.interactive_mode()
        
        return 0 if success else 1
        
    except KeyboardInterrupt:
        print("\n👋 Interrupted by user")
        return 130
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())