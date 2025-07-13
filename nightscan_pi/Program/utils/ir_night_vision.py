"""
IR-CUT Camera Control and Night Vision Management for NightScanPi

This module provides comprehensive control for IR-CUT cameras and infrared LED illumination,
enabling automatic day/night mode switching and optimal night vision capture.

Hardware Requirements:
- IR-CUT camera with GPIO control pin (default: GPIO 18)
- IR LED array with enable control (default: GPIO 19)
- Compatible with Pi Zero 2W and other Raspberry Pi models

GPIO Control:
- IR-CUT Filter: HIGH = Day mode (filter active), LOW = Night mode (filter removed)
- IR LEDs: HIGH = LEDs on (infrared illumination), LOW = LEDs off
"""

import os
import time
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass

# GPIO control
try:
    import RPi.GPIO as GPIO
    GPIO_AVAILABLE = True
except ImportError:
    GPIO_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class NightVisionConfig:
    """Configuration for night vision system."""
    ircut_pin: int = 18
    irled_pin: int = 19
    auto_night_mode: bool = True
    irled_enabled: bool = True
    irled_brightness: float = 0.8  # PWM duty cycle 0.0-1.0
    night_start_hour: int = 18  # 18h
    night_end_hour: int = 10   # 10h
    led_warmup_time: float = 0.5  # seconds
    mode_switch_delay: float = 1.0  # seconds


class IRNightVision:
    """IR-CUT camera control and night vision management."""
    
    def __init__(self, config: Optional[NightVisionConfig] = None):
        self.config = config or NightVisionConfig()
        self.gpio_initialized = False
        self.current_mode = None  # 'day' or 'night'
        self.leds_enabled = False
        self.pwm_led = None
        
        # Load configuration from environment
        self._load_env_config()
        
        # Initialize GPIO if available
        if GPIO_AVAILABLE:
            self._init_gpio()
        else:
            logger.warning("RPi.GPIO not available - night vision control disabled")
    
    def _load_env_config(self):
        """Load configuration from environment variables."""
        self.config.ircut_pin = int(os.getenv('NIGHTSCAN_IRCUT_PIN', self.config.ircut_pin))
        self.config.irled_pin = int(os.getenv('NIGHTSCAN_IRLED_PIN', self.config.irled_pin))
        self.config.auto_night_mode = os.getenv('NIGHTSCAN_AUTO_NIGHT_MODE', 'true').lower() == 'true'
        self.config.irled_enabled = os.getenv('NIGHTSCAN_IRLED_ENABLED', 'true').lower() == 'true'
        
        # IR LED brightness (0.0-1.0)
        brightness = float(os.getenv('NIGHTSCAN_IRLED_BRIGHTNESS', self.config.irled_brightness))
        self.config.irled_brightness = max(0.0, min(1.0, brightness))
        
        logger.info(f"IR-CUT pin: {self.config.ircut_pin}, IR LED pin: {self.config.irled_pin}")
        logger.info(f"Auto night mode: {self.config.auto_night_mode}, IR LEDs: {self.config.irled_enabled}")
    
    def _init_gpio(self):
        """Initialize GPIO pins for IR-CUT and LED control."""
        try:
            # Set GPIO mode
            GPIO.setmode(GPIO.BCM)
            GPIO.setwarnings(False)
            
            # Setup IR-CUT control pin
            GPIO.setup(self.config.ircut_pin, GPIO.OUT)
            GPIO.output(self.config.ircut_pin, GPIO.HIGH)  # Start in day mode
            
            # Setup IR LED control pin with PWM
            GPIO.setup(self.config.irled_pin, GPIO.OUT)
            self.pwm_led = GPIO.PWM(self.config.irled_pin, 1000)  # 1kHz PWM
            self.pwm_led.start(0)  # Start with LEDs off
            
            self.gpio_initialized = True
            self.current_mode = 'day'
            self.leds_enabled = False
            
            logger.info("GPIO initialized for IR-CUT and LED control")
            
        except Exception as e:
            logger.error(f"Failed to initialize GPIO: {e}")
            self.gpio_initialized = False
    
    def is_night_time(self) -> bool:
        """Check if it's currently night time based on configuration."""
        if not self.config.auto_night_mode:
            return False
        
        try:
            # Try to use sun times data if available
            from .sun_times import get_sun_times
            sun_times = get_sun_times()
            
            if sun_times:
                now = datetime.now(timezone.utc)
                sunset = sun_times.get('sunset')
                sunrise = sun_times.get('sunrise')
                
                if sunset and sunrise:
                    # Parse sunset/sunrise times
                    sunset_time = datetime.fromisoformat(sunset.replace('Z', '+00:00'))
                    sunrise_time = datetime.fromisoformat(sunrise.replace('Z', '+00:00'))
                    
                    # Check if current time is between sunset and sunrise
                    if sunset_time <= now or now <= sunrise_time:
                        return True
                    else:
                        return False
        
        except Exception as e:
            logger.debug(f"Sun times not available, using hour-based detection: {e}")
        
        # Fallback to hour-based detection
        current_hour = datetime.now().hour
        
        if self.config.night_start_hour > self.config.night_end_hour:
            # Night period crosses midnight (e.g., 18h-10h)
            return current_hour >= self.config.night_start_hour or current_hour <= self.config.night_end_hour
        else:
            # Night period within same day
            return self.config.night_start_hour <= current_hour <= self.config.night_end_hour
    
    def set_day_mode(self):
        """Switch to day mode (IR-CUT filter active, LEDs off)."""
        if not self.gpio_initialized:
            logger.warning("GPIO not initialized - cannot set day mode")
            return False
        
        try:
            # Disable IR LEDs first
            self._set_ir_leds(False)
            
            # Wait for LEDs to turn off
            time.sleep(0.1)
            
            # Enable IR-CUT filter (day mode)
            GPIO.output(self.config.ircut_pin, GPIO.HIGH)
            
            # Allow time for filter to switch
            time.sleep(self.config.mode_switch_delay)
            
            self.current_mode = 'day'
            logger.info("Switched to day mode - IR-CUT filter active")
            return True
            
        except Exception as e:
            logger.error(f"Failed to set day mode: {e}")
            return False
    
    def set_night_mode(self):
        """Switch to night mode (IR-CUT filter removed, LEDs on if enabled)."""
        if not self.gpio_initialized:
            logger.warning("GPIO not initialized - cannot set night mode")
            return False
        
        try:
            # Disable IR-CUT filter (night mode)
            GPIO.output(self.config.ircut_pin, GPIO.LOW)
            
            # Allow time for filter to switch
            time.sleep(self.config.mode_switch_delay)
            
            # Enable IR LEDs if configured
            if self.config.irled_enabled:
                self._set_ir_leds(True)
                # Allow LEDs to warm up
                time.sleep(self.config.led_warmup_time)
            
            self.current_mode = 'night'
            logger.info("Switched to night mode - IR-CUT filter removed")
            return True
            
        except Exception as e:
            logger.error(f"Failed to set night mode: {e}")
            return False
    
    def _set_ir_leds(self, enabled: bool, brightness: Optional[float] = None):
        """Control IR LED illumination."""
        if not self.gpio_initialized or not self.pwm_led:
            return
        
        try:
            if enabled and self.config.irled_enabled:
                # Calculate PWM duty cycle
                duty_cycle = (brightness or self.config.irled_brightness) * 100
                duty_cycle = max(0, min(100, duty_cycle))
                
                self.pwm_led.ChangeDutyCycle(duty_cycle)
                self.leds_enabled = True
                logger.debug(f"IR LEDs enabled at {duty_cycle:.1f}% brightness")
            else:
                self.pwm_led.ChangeDutyCycle(0)
                self.leds_enabled = False
                logger.debug("IR LEDs disabled")
                
        except Exception as e:
            logger.error(f"Failed to control IR LEDs: {e}")
    
    def auto_adjust_mode(self) -> bool:
        """Automatically adjust IR-CUT mode based on time of day."""
        if not self.config.auto_night_mode:
            return True
        
        should_be_night = self.is_night_time()
        
        if should_be_night and self.current_mode != 'night':
            logger.info("Auto-switching to night mode")
            return self.set_night_mode()
        elif not should_be_night and self.current_mode != 'day':
            logger.info("Auto-switching to day mode")
            return self.set_day_mode()
        
        return True
    
    def prepare_for_capture(self, force_mode: Optional[str] = None) -> bool:
        """Prepare IR-CUT system for image capture.
        
        Args:
            force_mode: Force specific mode ('day' or 'night'), or None for auto
            
        Returns:
            True if preparation successful
        """
        if not self.gpio_initialized:
            return True  # No GPIO control available, continue anyway
        
        try:
            if force_mode == 'day':
                success = self.set_day_mode()
            elif force_mode == 'night':
                success = self.set_night_mode()
            else:
                success = self.auto_adjust_mode()
            
            if success:
                logger.debug(f"Camera prepared for capture in {self.current_mode} mode")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to prepare for capture: {e}")
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """Get current night vision system status."""
        return {
            'gpio_available': GPIO_AVAILABLE,
            'gpio_initialized': self.gpio_initialized,
            'current_mode': self.current_mode,
            'leds_enabled': self.leds_enabled,
            'is_night_time': self.is_night_time(),
            'auto_mode': self.config.auto_night_mode,
            'ircut_pin': self.config.ircut_pin,
            'irled_pin': self.config.irled_pin,
            'led_brightness': self.config.irled_brightness,
            'led_feature_enabled': self.config.irled_enabled
        }
    
    def set_led_brightness(self, brightness: float):
        """Set IR LED brightness (0.0-1.0)."""
        brightness = max(0.0, min(1.0, brightness))
        self.config.irled_brightness = brightness
        
        # Update current LEDs if they're on
        if self.leds_enabled:
            self._set_ir_leds(True, brightness)
        
        logger.info(f"IR LED brightness set to {brightness:.1%}")
    
    def manual_led_control(self, enabled: bool, brightness: Optional[float] = None):
        """Manual control of IR LEDs (overrides auto mode)."""
        if brightness is not None:
            self.set_led_brightness(brightness)
        
        self._set_ir_leds(enabled, brightness)
        logger.info(f"Manual LED control: {'ON' if enabled else 'OFF'}")
    
    def cleanup(self):
        """Clean up GPIO resources."""
        if self.gpio_initialized:
            try:
                # Turn off IR LEDs
                if self.pwm_led:
                    self.pwm_led.stop()
                
                # Set IR-CUT to day mode (safe default)
                GPIO.output(self.config.ircut_pin, GPIO.HIGH)
                
                # Clean up GPIO
                GPIO.cleanup([self.config.ircut_pin, self.config.irled_pin])
                
                logger.info("GPIO cleanup completed")
                
            except Exception as e:
                logger.error(f"Error during GPIO cleanup: {e}")
    
    def __del__(self):
        """Destructor to ensure GPIO cleanup."""
        self.cleanup()


# Global instance
_night_vision: Optional[IRNightVision] = None


def get_night_vision() -> IRNightVision:
    """Get global night vision controller instance."""
    global _night_vision
    if _night_vision is None:
        _night_vision = IRNightVision()
    return _night_vision


def is_night_mode() -> bool:
    """Quick check if system should be in night mode."""
    nv = get_night_vision()
    return nv.is_night_time()


def prepare_camera_for_capture(force_mode: Optional[str] = None) -> bool:
    """Prepare camera for capture with appropriate IR-CUT mode.
    
    Args:
        force_mode: Force specific mode ('day' or 'night'), or None for auto
        
    Returns:
        True if preparation successful
    """
    nv = get_night_vision()
    return nv.prepare_for_capture(force_mode)


def get_night_vision_status() -> Dict[str, Any]:
    """Get current night vision system status."""
    nv = get_night_vision()
    return nv.get_status()


if __name__ == "__main__":
    # Test night vision system
    print("🌙 Testing IR-CUT Night Vision System")
    print("=" * 40)
    
    nv = get_night_vision()
    status = nv.get_status()
    
    print(f"GPIO Available: {'✅' if status['gpio_available'] else '❌'}")
    print(f"GPIO Initialized: {'✅' if status['gpio_initialized'] else '❌'}")
    print(f"Current Mode: {status['current_mode'] or 'Unknown'}")
    print(f"Night Time: {'✅' if status['is_night_time'] else '❌'}")
    print(f"LEDs Enabled: {'✅' if status['leds_enabled'] else '❌'}")
    print(f"Auto Mode: {'✅' if status['auto_mode'] else '❌'}")
    print(f"IR-CUT Pin: GPIO {status['ircut_pin']}")
    print(f"IR LED Pin: GPIO {status['irled_pin']}")
    print(f"LED Brightness: {status['led_brightness']:.1%}")
    
    if status['gpio_initialized']:
        print("\n🧪 Testing mode switching...")
        print("Setting night mode...")
        nv.set_night_mode()
        time.sleep(2)
        
        print("Setting day mode...")
        nv.set_day_mode()
        time.sleep(2)
        
        print("Auto-adjusting mode...")
        nv.auto_adjust_mode()
        
        print(f"Final mode: {nv.current_mode}")
    
    print("\n✅ Night vision test completed")