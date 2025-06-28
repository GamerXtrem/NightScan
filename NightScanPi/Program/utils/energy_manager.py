"""Simple energy scheduling for NightScanPi."""
from __future__ import annotations

from datetime import datetime, timedelta
import os
import subprocess


# Active hours can be customized through environment variables
START_HOUR = int(os.getenv("NIGHTSCAN_START_HOUR", "18"))
STOP_HOUR = int(os.getenv("NIGHTSCAN_STOP_HOUR", "10"))


def within_active_period(now: datetime | None = None) -> bool:
    """Return ``True`` if ``now`` is between START_HOUR and STOP_HOUR."""
    if now is None:
        now = datetime.now()
    start = now.replace(hour=START_HOUR, minute=0, second=0, microsecond=0)
    stop = now.replace(hour=STOP_HOUR, minute=0, second=0, microsecond=0)
    if START_HOUR > STOP_HOUR:
        return now >= start or now < stop
    return start <= now < stop


def shutdown() -> None:
    """Shut down the system."""
    subprocess.run(["sudo", "shutdown", "-h", "now"], check=False)


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
