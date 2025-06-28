# NightScanPi Hardware Wiring Guide

This document summarises how to connect the core components used by the NightScanPi system.

## Required Components
- **Raspberry Pi Zero 2 W**
- **RPi IR-CUT Camera**
- **ReSpeaker Mic Array Lite**
- **PIR motion sensor** (HC-SR501 or similar)
- **TPL5110 timer board** with a single 18650 Li-Ion cell
- **5 V solar panel** (optional for recharging)

## Wiring Overview
1. **Power**: The Pi receives 5 V from the TPL5110 timer. Connect the timer’s output to the Pi’s USB power port and its `DONE` pin to a free GPIO (default: `GPIO4`). The timer’s `VDD` is connected to the battery through its `Vin` pin.
2. **Camera**: Use the CSI cable to attach the IR-CUT camera to the Pi’s camera connector. Ensure the metal contacts on the cable face the HDMI port.
3. **Microphone Array**: Plug the ReSpeaker Lite via USB to the Pi. For minimal power draw, you can instead wire it via I2S using the pins shown in the official documentation.
4. **PIR Sensor**: Connect the sensor’s `VCC` to 5 V, `GND` to ground and `OUT` to `GPIO17`. This pin triggers photo and audio capture when motion is detected.
5. **Solar Panel (optional)**: Attach the panel to a charging circuit that feeds the battery powering the TPL5110.

The following diagram illustrates the basic connections:

```
Battery -> TPL5110 -> Raspberry Pi
                     |-- Camera (CSI)
                     |-- ReSpeaker Lite (USB/I2S)
                     '-- PIR Sensor (GPIO17)
```

## Characteristics
- The Pi operates only between **18 h and 10 h** as managed by `energy_manager.py`.
- Typical total consumption is under **3 W** during captures.
- Use a **64 GB microSD** card (ext4 recommended) to store audio and image files.

For further details about each component, refer to the individual files in this directory.
