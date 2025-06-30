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

# Optional sunrise/sunset scheduling
SUN_FILE = os.getenv("NIGHTSCAN_SUN_FILE")
SUN_OFFSET = timedelta(minutes=int(os.getenv("NIGHTSCAN_SUN_OFFSET", "30")))


def _sun_period(now: datetime) -> tuple[datetime, datetime]:
    """Return start/stop datetimes based on sunrise and sunset."""
    day = now.date()
    sun_path = Path(SUN_FILE)
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
    if SUN_FILE:
        start, stop = _sun_period(now)
        return start <= now < stop
    start = now.replace(hour=START_HOUR, minute=0, second=0, microsecond=0)
    stop = now.replace(hour=STOP_HOUR, minute=0, second=0, microsecond=0)
    if START_HOUR > STOP_HOUR:
        return now >= start or now < stop
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
    if SUN_FILE:
        _, stop = _sun_period(now)
        if now >= stop:
            _, stop = _sun_period(now + timedelta(days=1))
        return stop
    stop = now.replace(hour=STOP_HOUR, minute=0, second=0, microsecond=0)
    if now >= stop:
        stop += timedelta(days=1)
    return stop


def schedule_shutdown(now: datetime | None = None) -> None:
    """Schedule system shutdown at the next stop time."""
    stop = next_stop_time(now)
    time_str = stop.strftime("%H:%M")
    subprocess.run(["sudo", "shutdown", "-h", time_str], check=False)


if __name__ == "__main__":
    if not within_active_period():
        shutdown()
