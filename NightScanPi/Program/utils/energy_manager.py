"""Simple energy scheduling for NightScanPi."""
from __future__ import annotations

from datetime import datetime, timedelta
import os
import subprocess
import time

try:  # pragma: no cover - not installed in CI
    import RPi.GPIO as GPIO
except Exception:  # pragma: no cover - missing dependency
    GPIO = None


# Active hours can be customized through environment variables
START_HOUR = int(os.getenv("NIGHTSCAN_START_HOUR", "18"))
STOP_HOUR = int(os.getenv("NIGHTSCAN_STOP_HOUR", "10"))
DONE_PIN = int(os.getenv("NIGHTSCAN_DONE_PIN", "4"))


def within_active_period(now: datetime | None = None) -> bool:
    """Return ``True`` if ``now`` is between START_HOUR and STOP_HOUR."""
    if now is None:
        now = datetime.now()
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
