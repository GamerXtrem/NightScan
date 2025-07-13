"""Simple energy scheduling for NightScanPi."""
from __future__ import annotations

from datetime import datetime, timedelta
import os
import subprocess
import time
from pathlib import Path

from . import sun_times

try:  # pragma: no cover - not installed in CI
    import RPi.GPIO as GPIO
except Exception:  # pragma: no cover - missing dependency
    GPIO = None


# Active hours can be customized through environment variables
START_HOUR = int(os.getenv("NIGHTSCAN_START_HOUR", "18"))
STOP_HOUR = int(os.getenv("NIGHTSCAN_STOP_HOUR", "10"))
DONE_PIN = int(os.getenv("NIGHTSCAN_DONE_PIN", "4"))

# Power consumption estimates (watts)
BASE_POWER = 3.0  # Pi Zero 2W base consumption
IR_LED_POWER = float(os.getenv("NIGHTSCAN_IRLED_POWER", "2.0"))  # Per LED at full brightness

# Mandatory sunrise/sunset scheduling
DEFAULT_SUN_FILE = Path.home() / "sun_times.json"
SUN_FILE = Path(os.getenv("NIGHTSCAN_SUN_FILE", str(DEFAULT_SUN_FILE)))
SUN_OFFSET = timedelta(minutes=int(os.getenv("NIGHTSCAN_SUN_OFFSET", "30")))


def _sun_period(now: datetime) -> tuple[datetime, datetime]:
    """Return start/stop datetimes based on sunrise and sunset."""
    day = now.date()
    sun_path = SUN_FILE
    _, sunrise_today, sunset_today = sun_times.get_or_update_sun_times(sun_path, day)
    _, sunrise_yesterday, sunset_yesterday = sun_times.get_or_update_sun_times(
        sun_path, day - timedelta(days=1)
    )
    _, sunrise_tomorrow, _ = sun_times.get_or_update_sun_times(
        sun_path, day + timedelta(days=1)
    )

    start_prev = sunset_yesterday - SUN_OFFSET
    stop_prev = sunrise_today + SUN_OFFSET
    start_curr = sunset_today - SUN_OFFSET
    stop_curr = sunrise_tomorrow + SUN_OFFSET

    if now <= stop_prev:
        return start_prev, stop_prev
    return start_curr, stop_curr


def within_active_period(now: datetime | None = None) -> bool:
    """Return ``True`` if ``now`` falls within the configured active period."""
    if now is None:
        now = datetime.now()
    start, stop = _sun_period(now)
    return start <= now < stop


def signal_done(pin: int | None = None) -> None:
    """Signal the TPL5110 DONE pin to cut power."""
    if pin is None:
        pin = DONE_PIN
    if GPIO is None:
        return
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(pin, GPIO.OUT)
    GPIO.output(pin, True)
    time.sleep(0.1)
    GPIO.cleanup(pin)

def shutdown() -> None:
    """Shut down the system."""
    subprocess.run(["sudo", "shutdown", "-h", "now"], check=False)
    signal_done()


def next_stop_time(now: datetime | None = None) -> datetime:
    """Return the next time the system should shut down."""
    if now is None:
        now = datetime.now()
    _, stop = _sun_period(now)
    if now >= stop:
        _, stop = _sun_period(now + timedelta(days=1))
    return stop


def get_power_consumption() -> dict:
    """Get current power consumption estimate including IR LEDs."""
    try:
        from .ir_night_vision import get_night_vision_status
        nv_status = get_night_vision_status()
        
        total_power = BASE_POWER
        led_power = 0.0
        
        if nv_status.get('leds_enabled', False) and nv_status.get('led_feature_enabled', False):
            # Calculate LED power based on brightness
            brightness = nv_status.get('led_brightness', 0.8)
            led_count = 2  # Assume 2 IR LEDs
            led_power = IR_LED_POWER * led_count * brightness
            total_power += led_power
        
        return {
            'base_power': BASE_POWER,
            'led_power': led_power,
            'total_power': total_power,
            'led_enabled': nv_status.get('leds_enabled', False),
            'led_brightness': nv_status.get('led_brightness', 0.0)
        }
        
    except Exception:
        # Fallback if night vision not available
        return {
            'base_power': BASE_POWER,
            'led_power': 0.0,
            'total_power': BASE_POWER,
            'led_enabled': False,
            'led_brightness': 0.0
        }


def should_reduce_power() -> bool:
    """Check if power consumption should be reduced (e.g., low battery)."""
    power_info = get_power_consumption()
    
    # If total power > 4W, consider reducing LED power
    if power_info['total_power'] > 4.0:
        return True
    
    # Could add battery level monitoring here in the future
    return False


def optimize_led_brightness_for_power() -> float:
    """Get optimal LED brightness for current power situation."""
    if should_reduce_power():
        # Reduce to 50% brightness if power consumption is too high
        return 0.5
    else:
        # Use configured brightness
        return float(os.getenv("NIGHTSCAN_IRLED_BRIGHTNESS", "0.8"))


def schedule_shutdown(now: datetime | None = None) -> None:
    """Schedule system shutdown at the next stop time."""
    stop = next_stop_time(now)
    time_str = stop.strftime("%H:%M")
    subprocess.run(["sudo", "shutdown", "-h", time_str], check=False)


if __name__ == "__main__":
    if not within_active_period():
        shutdown()
