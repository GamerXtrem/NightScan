# 🔌 IR-CUT Camera Wiring Guide for NightScanPi

Guide for wiring IR-CUT cameras and infrared LED arrays to enable automatic day/night vision switching on Raspberry Pi Zero 2W.

## 📋 Hardware Requirements

### IR-CUT Camera Module
- **Supported Sensors**: IMX219, IMX477, IMX290, IMX327, IMX378, IMX519, IMX708, IMX296
- **IR-CUT Control**: GPIO-controlled filter switching
- **Interface**: CSI camera connector + IR-CUT control wire

### IR LED Array
- **Wavelength**: 850nm (recommended)
- **Operating Voltage**: 3.3V
- **Current**: 300mA - 2A per LED (depending on drive current)
- **Control**: EN pad for enable/disable control

### Raspberry Pi Zero 2W
- **GPIO Pins Available**: 26 pins (40-pin header with 14 pins physical limitations)
- **Power Budget**: 3W total system (including camera and LEDs)

## 🔧 GPIO Pin Assignment

### Default Pin Configuration
```
┌─────────────────────────────────────────┐
│ Raspberry Pi Zero 2W GPIO Header       │
├─────────────────────────────────────────┤
│ Pin 1  (3.3V)     ←→  Pin 2  (5V)      │
│ Pin 3  (GPIO 2)   ←→  Pin 4  (5V)      │
│ Pin 5  (GPIO 3)   ←→  Pin 6  (GND)     │
│ Pin 7  (GPIO 4)   ←→  Pin 8  (GPIO 14) │ ← TPL5110 DONE
│ Pin 9  (GND)      ←→  Pin 10 (GPIO 15) │
│ Pin 11 (GPIO 17)  ←→  Pin 12 (GPIO 18) │ ← IR-CUT Control
│ Pin 13 (GPIO 27)  ←→  Pin 14 (GND)     │
│ Pin 15 (GPIO 22)  ←→  Pin 16 (GPIO 23) │
│ Pin 17 (3.3V)     ←→  Pin 18 (GPIO 24) │
│ Pin 19 (GPIO 10)  ←→  Pin 20 (GND)     │
│ Pin 21 (GPIO 9)   ←→  Pin 22 (GPIO 25) │
│ Pin 23 (GPIO 11)  ←→  Pin 24 (GPIO 8)  │
│ Pin 25 (GND)      ←→  Pin 26 (GPIO 7)  │
│ Pin 27 (GPIO 0)   ←→  Pin 28 (GPIO 1)  │
│ Pin 29 (GPIO 5)   ←→  Pin 30 (GND)     │
│ Pin 31 (GPIO 6)   ←→  Pin 32 (GPIO 12) │
│ Pin 33 (GPIO 13)  ←→  Pin 34 (GND)     │
│ Pin 35 (GPIO 19)  ←→  Pin 36 (GPIO 16) │ ← IR LED Control
│ Pin 37 (GPIO 26)  ←→  Pin 38 (GPIO 20) │
│ Pin 39 (GND)      ←→  Pin 40 (GPIO 21) │
└─────────────────────────────────────────┘
```

### Pin Assignments for NightScanPi
| Function | GPIO Pin | Physical Pin | Wire Color | Notes |
|----------|----------|--------------|------------|-------|
| **PIR Sensor** | GPIO 17 | Pin 11 | White | Motion detection |
| **IR-CUT Control** | GPIO 18 | Pin 12 | Yellow | Camera filter control |
| **IR LED Control** | GPIO 19 | Pin 35 | Red | LED array enable |
| **TPL5110 DONE** | GPIO 4 | Pin 7 | Blue | Power management |

## 🔗 Wiring Connections

### IR-CUT Camera Wiring
```
┌─────────────────┐    CSI Cable    ┌─────────────────┐
│   Pi Zero 2W    │◄────────────────┤   IR-CUT Camera │
│                 │                 │                 │
│ GPIO 18 (Pin 12)│◄────────────────┤ IR-CUT Control  │ (Yellow wire)
│ 3.3V (Pin 17)   │◄────────────────┤ Camera VCC      │ (Red wire)
│ GND (Pin 20)     │◄────────────────┤ Camera GND      │ (Black wire)
└─────────────────┘                 └─────────────────┘
```

### IR LED Array Wiring
```
┌─────────────────┐                 ┌─────────────────┐
│   Pi Zero 2W    │                 │   IR LED Array  │
│                 │                 │   (2x 850nm)    │
│ GPIO 19 (Pin 35)│────────────────►│ EN Control      │ (Red wire)
│ 3.3V (Pin 17)   │────────────────►│ LED VCC         │ (Red wire)
│ GND (Pin 25)     │────────────────►│ LED GND         │ (Black wire)
└─────────────────┘                 └─────────────────┘
```

## ⚙️ IR-CUT Filter Control Logic

### Day Mode (Normal Light)
- **IR-CUT Filter**: ✅ **ACTIVE** (blocks infrared light)
- **GPIO 18 State**: **HIGH** (3.3V)
- **IR LEDs**: ❌ **OFF** (no infrared illumination needed)
- **Image Quality**: Natural colors, good in bright light

### Night Mode (Low Light)
- **IR-CUT Filter**: ❌ **REMOVED** (allows infrared light)
- **GPIO 18 State**: **LOW** (0V)
- **IR LEDs**: ✅ **ON** (infrared illumination active)
- **Image Quality**: Black & white, enhanced low-light sensitivity

## 💡 IR LED Control System

### PWM Brightness Control
```python
# LED brightness control (0.0 - 1.0)
brightness_levels = {
    'off': 0.0,        # LEDs completely off
    'low': 0.3,        # 30% brightness (600mA)
    'medium': 0.6,     # 60% brightness (1.2A)
    'high': 0.8,       # 80% brightness (1.6A)
    'max': 1.0         # 100% brightness (2A)
}
```

### Power Consumption Table
| Brightness | Current per LED | Total Current (2 LEDs) | Power @ 3.3V |
|------------|-----------------|-------------------------|---------------|
| 30%        | 300mA          | 600mA                   | 2.0W         |
| 60%        | 600mA          | 1.2A                    | 4.0W         |
| 80%        | 800mA          | 1.6A                    | 5.3W         |
| 100%       | 1A             | 2A                      | 6.6W         |

## 🔌 Complete System Wiring Diagram

```
        ┌─────────────────────────────────────────────────────────┐
        │                 Raspberry Pi Zero 2W                    │
        │                                                         │
        │  ┌─────┐                                    ┌─────┐     │
        │  │ CSI │◄──── Camera Cable ────────────────►│     │     │
        │  └─────┘                                    │     │     │
        │                                             │ IR- │     │
        │  GPIO 17 (PIR) ◄──── White ────────────────►│ CUT │     │
        │  GPIO 18 (IR-CUT) ◄─ Yellow ───────────────►│Cam  │     │
        │  GPIO 19 (LED) ──── Red ─────────┐          │     │     │
        │  GPIO 4 (DONE) ◄─── Blue ────────┼─────────►└─────┘     │
        │                                  │                      │
        │  3.3V ────── Red ────────────────┼─────┐                │
        │  GND ─────── Black ──────────────┼───┐ │                │
        └──────────────────────────────────┼───┼─┼────────────────┘
                                          │   │ │
                                          │   │ └─── Camera VCC
                                          │   └───── Camera GND
                                          │
                                ┌─────────▼─────────┐
                                │   IR LED Array    │
                                │    (2x 850nm)     │
                                │                   │
                                │ EN ◄──── Red      │
                                │ VCC ◄─── Red      │
                                │ GND ◄─── Black    │
                                └───────────────────┘
```

## 📊 Configuration Variables

### Environment Variables
```bash
# GPIO Pin Configuration
export NIGHTSCAN_IRCUT_PIN=18      # IR-CUT filter control
export NIGHTSCAN_IRLED_PIN=19      # IR LED array control

# Night Vision Settings  
export NIGHTSCAN_AUTO_NIGHT_MODE=true     # Automatic day/night switching
export NIGHTSCAN_IRLED_ENABLED=true       # Enable IR LED control
export NIGHTSCAN_IRLED_BRIGHTNESS=0.8     # LED brightness (0.0-1.0)

# Power Management
export NIGHTSCAN_IRLED_POWER=2.0          # Power per LED in watts
```

### Boot Configuration (/boot/firmware/config.txt)
```ini
# Camera configuration
camera_auto_detect=0
start_x=1
gpu_mem=128

# Disable camera LED to save power
disable_camera_led=1

# IR-CUT camera overlay (adjust for your sensor)
dtoverlay=imx219

# GPIO configuration
gpio=18=op,dh    # IR-CUT control - start HIGH (day mode)
gpio=19=op,dl    # IR LED control - start LOW (LEDs off)
```

## 🧪 Testing Commands

### Test IR-CUT Control
```bash
# Test day mode (filter active)
echo "1" > /sys/class/gpio/gpio18/value

# Test night mode (filter removed) 
echo "0" > /sys/class/gpio/gpio18/value
```

### Test IR LED Control
```bash
# Turn LEDs on
echo "1" > /sys/class/gpio/gpio19/value

# Turn LEDs off
echo "0" > /sys/class/gpio/gpio19/value
```

### Test with NightScanPi
```bash
# Test night vision system
python camera_test.py --test-night-vision

# Test camera with night vision
python camera_test.py --capture --all

# Check night vision status
python camera_test.py --status
```

## ⚠️ Important Notes

### Power Considerations
- **Total System Budget**: 3W (Pi Zero 2W base consumption)
- **IR LED Impact**: +2-6W additional consumption
- **Battery Life**: Significantly reduced when IR LEDs are active
- **Recommendation**: Use PWM brightness control to optimize power

### Safety Guidelines
- **Voltage**: Never exceed 3.3V on GPIO pins
- **Current**: GPIO pins limited to 16mA - use transistor switching for LEDs
- **Heat**: IR LEDs generate heat - ensure adequate ventilation
- **Eye Safety**: 850nm IR light is invisible but powerful - avoid direct exposure

### GPIO Protection
- **ESD Protection**: Use anti-static precautions when wiring
- **Pullup/Pulldown**: Configure GPIO pins with appropriate resistors
- **Short Circuit**: Ensure proper wiring to avoid GPIO damage

## 🔧 Troubleshooting

### IR-CUT Not Switching
1. Check GPIO 18 wiring and connections
2. Verify 3.3V power supply to camera
3. Test with manual GPIO control
4. Check camera model compatibility

### IR LEDs Not Working
1. Verify GPIO 19 connection to EN pad
2. Check 3.3V power supply to LEDs
3. Test LED current consumption
4. Verify PWM signal with oscilloscope

### Poor Night Vision Quality
1. Increase IR LED brightness
2. Check IR-CUT filter removal in night mode
3. Verify camera sensor supports night vision
4. Adjust camera exposure settings

### High Power Consumption
1. Reduce IR LED brightness (PWM duty cycle)
2. Use duty cycle control (pulse LEDs only during capture)
3. Implement adaptive brightness based on ambient light
4. Consider power management optimizations

## 📖 Additional Resources

- **Camera Configuration Guide**: `NightScanPi/Hardware/CAMERA_CONFIGURATION_GUIDE.md`
- **Sensor Detection**: `camera_sensor_detector.py` 
- **Night Vision Control**: `utils/ir_night_vision.py`
- **Power Management**: `utils/energy_manager.py`
- **Testing Tools**: `camera_test.py --help`

## ✅ Verification Checklist

- [ ] IR-CUT camera connected via CSI interface
- [ ] GPIO 18 connected to IR-CUT control wire
- [ ] GPIO 19 connected to IR LED EN control
- [ ] 3.3V power connected to camera and LEDs
- [ ] GND connections properly wired
- [ ] Boot configuration updated with camera overlay
- [ ] Environment variables configured
- [ ] Night vision system tested with `camera_test.py`
- [ ] Day/night mode switching verified
- [ ] IR LED brightness control working
- [ ] Power consumption within acceptable range

**Setup complete! Your NightScanPi now has automatic day/night vision capabilities!** 🌙📸