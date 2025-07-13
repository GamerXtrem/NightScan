# 📸 Camera Validation and Testing Guide for NightScanPi

Comprehensive guide for testing and validating camera functionality on Raspberry Pi with IR-CUT cameras.

## 🚀 Quick Start

### Basic Camera Test
```bash
# Test camera with 5-second preview
./nightscan-camera-hello

# Capture a test image
./nightscan-camera-hello --capture

# List available cameras
./nightscan-camera-hello --list-cameras

# Run comprehensive validation tests
./nightscan-camera-hello --test-all
```

### Using Python Scripts Directly
```bash
# Activate NightScan environment
source env/bin/activate

# Basic camera diagnostics
python NightScanPi/Program/camera_test.py --status

# Run comprehensive validation
python NightScanPi/Program/camera_test.py --validate

# Test specific functionality
python NightScanPi/Program/camera_test.py --test-night-vision
python NightScanPi/Program/camera_test.py --test-pi-zero
```

## 🔧 Validation Tools Overview

### 1. **nightscan-camera-hello** (Shell Script)
Simple wrapper providing rpicam-hello equivalent functionality:
- **Preview testing** with configurable timeout
- **Image capture** with quality analysis
- **Camera detection** and listing
- **System requirement** checks

### 2. **camera_test.py** (Diagnostic Tool)
Core diagnostic and testing utility:
- **Camera API status** (picamera2/picamera)
- **Sensor detection** and information
- **Night vision testing** (IR-CUT, LEDs)
- **Pi Zero optimizations** validation
- **Comprehensive validation** suite

### 3. **camera_validator.py** (Validation Engine)
Comprehensive testing framework:
- **Hardware validation** (libcamera commands)
- **Performance benchmarking** (capture speed, FPS)
- **Image quality analysis** (brightness, contrast, sharpness)
- **Multiple resolution testing**
- **Night vision functionality**

## 📋 Available Tests

### **Hardware Tests**
| Test | Description | Tool |
|------|-------------|------|
| **libcamera_hello** | Native libcamera command test | camera_validator |
| **sensor_detection** | Camera sensor identification | camera_test/validator |
| **camera_capture** | Basic image capture test | All tools |
| **multiple_resolutions** | Test various capture resolutions | camera_validator |

### **Functionality Tests**
| Test | Description | Tool |
|------|-------------|------|
| **night_vision** | IR-CUT filter and LED control | camera_test/validator |
| **image_quality** | Brightness, contrast, sharpness analysis | camera_validator |
| **performance_benchmark** | Capture speed and FPS testing | camera_validator |

### **System Tests**
| Test | Description | Tool |
|------|-------------|------|
| **pi_zero_optimizations** | Memory and resolution optimization | camera_test |
| **system_requirements** | Python modules and dependencies | camera_test |

## 🎯 Test Commands Reference

### Basic Diagnostics
```bash
# Show camera status and capabilities
python camera_test.py --status

# Test camera capture functionality
python camera_test.py --test

# Capture and analyze test image
python camera_test.py --capture
```

### Sensor and Detection
```bash
# Test sensor detection methods
python camera_test.py --detect-sensor

# List cameras with rpicam-hello equivalent
./nightscan-camera-hello --list-cameras
```

### Night Vision Testing
```bash
# Test IR-CUT and LED control
python camera_test.py --test-night-vision

# Manual night vision preview
./nightscan-camera-hello --timeout 10000  # During night hours
```

### Performance and Optimization
```bash
# Test Pi Zero memory optimizations
python camera_test.py --test-pi-zero

# Run performance benchmark
python camera_validator.py --test performance --num-captures 20
```

### Comprehensive Validation
```bash
# All tests in camera_test.py
python camera_test.py --all

# Comprehensive validation suite
python camera_test.py --validate

# Full validation with nightscan-camera-hello
./nightscan-camera-hello --test-all
```

## 🔍 Understanding Test Results

### **Camera Status Output**
```
🔍 NightScanPi Camera Diagnostic Tool
==================================================
📋 Camera API Status:
  • picamera2 available: ✅
  • picamera (legacy) available: ❌
  • Active API: picamera2
  • Camera working: ✅
  • Using modern libcamera stack (RECOMMENDED)

📸 Camera Sensor Information:
  • Detected Sensor: IMX219
  • Model: IMX219 (Sony IMX219 8MP)
  • Max Resolution: 3280x2464
  • Capabilities: auto_exposure, auto_white_balance, ir_cut
  • IR-CUT Support: ✅
  • Night Vision: ✅
  • Boot Config: dtoverlay=imx219

🌙 Night Vision System:
  • GPIO Available: ✅
  • GPIO Initialized: ✅
  • Current Mode: day
  • Night Time: ❌
  • IR LEDs: ❌
  • Auto Mode: ✅
  • IR-CUT Pin: GPIO 18
  • IR LED Pin: GPIO 19
  • LED Brightness: 80.0%
  • LED Feature: ✅

🔧 Pi Zero 2W Optimizations:
  • Pi Zero Detected: ✅
  • Optimizations Active: ✅
  • Memory Total: 512MB
  • Memory Used: 245MB (47.8%)
  • Memory Available: 267MB
  • Swap Usage: 0MB
  • Max Camera Res: (1280, 720)
  • GPU Memory: 64MB
  • Max Audio Buffer: 512
  • Memory Monitoring: ✅
```

### **Validation Test Results**
```
📊 Validation Summary:
  • Total Tests: 7
  • Passed: 6 ✅
  • Failed: 1 ❌
  • Success Rate: 85.7%

Test Results:
libcamera_hello: ✅ PASSED (2.15s)
sensor_detection: ✅ PASSED (0.89s)
camera_capture: ✅ PASSED (3.42s)
multiple_resolutions: ✅ PASSED (8.73s)
image_quality: ✅ PASSED (4.21s)
night_vision: ❌ FAILED (1.05s)
  Error: GPIO not initialized - night vision hardware not available
performance_benchmark: ✅ PASSED (12.34s)
```

## 🛠️ Troubleshooting Guide

### **Common Issues and Solutions**

#### **Camera Not Detected**
```
❌ Camera test FAILED
```
**Solutions:**
1. Check camera connection (CSI cable)
2. Verify camera enabled in config.txt: `camera_auto_detect=1`
3. Run: `./nightscan-camera-hello --check-system`
4. Test with: `rpicam-hello --list-cameras`

#### **picamera2 Not Available**
```
❌ picamera2 available: ❌
```
**Solutions:**
1. Install picamera2: `pip install picamera2`
2. Update system: `sudo apt update && sudo apt install python3-picamera2`
3. Use Pi Zero requirements: `pip install -r requirements-pi-zero.txt`

#### **Night Vision Tests Failing**
```
❌ Night vision control test FAILED: GPIO not initialized
```
**Solutions:**
1. Check GPIO connections (GPIO 18, 19)
2. Run as root: `sudo python camera_test.py --test-night-vision`
3. Install RPi.GPIO: `pip install RPi.GPIO`
4. Verify hardware wiring per IR_CUT_WIRING_GUIDE.md

#### **Memory Issues on Pi Zero**
```
⚠️ High memory usage: 90.5% - triggering cleanup
```
**Solutions:**
1. Enable Pi Zero optimizations (automatic on detection)
2. Reduce capture resolution: use 720p instead of 1080p
3. Enable swap: `sudo dphys-swapfile setup && sudo dphys-swapfile swapon`
4. Use lightweight requirements: `requirements-pi-zero.txt`

#### **Low Image Quality**
```
Image quality metrics: brightness=45.2, contrast=12.8, sharpness=234.5
```
**Solutions:**
1. Check lens focus and cleanliness
2. Verify adequate lighting for current mode (day/night)
3. Test with different resolutions
4. Adjust camera position and angle

## 📊 Performance Benchmarks

### **Expected Performance (Pi Zero 2W)**
| Metric | 480p | 720p | 1080p |
|--------|------|------|-------|
| **Capture Time** | 1.2s | 1.8s | 2.5s |
| **File Size** | 45KB | 95KB | 180KB |
| **Memory Usage** | +15MB | +25MB | +40MB |
| **FPS Estimate** | 0.8 | 0.6 | 0.4 |

### **Expected Performance (Pi 4)**
| Metric | 720p | 1080p | 4K |
|--------|------|-------|-----|
| **Capture Time** | 0.8s | 1.1s | 2.8s |
| **File Size** | 95KB | 180KB | 850KB |
| **Memory Usage** | +20MB | +35MB | +120MB |
| **FPS Estimate** | 1.2 | 0.9 | 0.4 |

## 🔧 Advanced Testing

### **Custom Resolution Testing**
```bash
# Test specific resolution
python -c "
from camera_trigger import capture_image
from pathlib import Path
capture_image(Path('test'), (1640, 1232))
"
```

### **Night Vision Timing Test**
```bash
# Test mode switching timing
python -c "
from utils.ir_night_vision import get_night_vision
import time
nv = get_night_vision()
start = time.time()
nv.set_night_mode()
night_time = time.time() - start
start = time.time()
nv.set_day_mode()
day_time = time.time() - start
print(f'Night mode: {night_time:.2f}s, Day mode: {day_time:.2f}s')
"
```

### **Memory Usage Monitoring**
```bash
# Monitor memory during capture
python -c "
from utils.pi_zero_optimizer import get_optimizer
import time
opt = get_optimizer()
for i in range(5):
    mem = opt.get_memory_info()
    print(f'Memory: {mem.used_mb:.0f}MB ({mem.percent_used:.1f}%)')
    time.sleep(2)
"
```

## 📈 Validation Automation

### **CI/CD Integration**
```bash
#!/bin/bash
# ci_camera_test.sh - Automated camera validation

set -e

echo "🔍 Running automated camera validation..."

# Basic functionality test
python camera_test.py --test --json > camera_test_results.json

# Validate JSON output
if jq -e '.camera_test == true' camera_test_results.json; then
    echo "✅ Camera test passed"
else
    echo "❌ Camera test failed"
    exit 1
fi

# Sensor detection test
if python camera_test.py --detect-sensor --json | jq -e '.sensor_detection == true'; then
    echo "✅ Sensor detection passed"
else
    echo "⚠️ Sensor detection failed (may be expected in CI)"
fi

echo "🎉 Automated validation completed"
```

### **Scheduled Health Checks**
```bash
# Add to crontab: 0 */6 * * * /path/to/camera_health_check.sh
#!/bin/bash
# camera_health_check.sh - Periodic camera health check

LOG_FILE="/var/log/nightscan_camera_health.log"
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

echo "[$TIMESTAMP] Starting camera health check" >> "$LOG_FILE"

if timeout 30 python camera_test.py --test; then
    echo "[$TIMESTAMP] Camera health check PASSED" >> "$LOG_FILE"
else
    echo "[$TIMESTAMP] Camera health check FAILED" >> "$LOG_FILE"
    # Send alert (email, webhook, etc.)
fi
```

## 🎓 Best Practices

### **Regular Testing Schedule**
1. **Daily**: Basic camera functionality (`--test`)
2. **Weekly**: Comprehensive validation (`--validate`)
3. **Monthly**: Full system validation (`--all`)
4. **After changes**: Night vision and optimizations tests

### **Performance Monitoring**
1. **Capture timing** trends over time
2. **Memory usage** patterns
3. **Image quality** metrics
4. **Error rate** tracking

### **Maintenance Checklist**
- [ ] Camera lens cleaning
- [ ] CSI cable connection check
- [ ] GPIO wiring verification
- [ ] Software dependency updates
- [ ] Boot configuration validation
- [ ] Performance benchmark comparison

## 📚 Related Documentation

- **Hardware Setup**: `IR_CUT_WIRING_GUIDE.md`
- **Camera Configuration**: `CAMERA_CONFIGURATION_GUIDE.md`
- **Pi Zero Optimizations**: Memory optimization features in code
- **Sensor Detection**: `camera_sensor_detector.py` module
- **Night Vision**: `ir_night_vision.py` module

---

**Ready to validate your NightScanPi camera system!** 📸✨