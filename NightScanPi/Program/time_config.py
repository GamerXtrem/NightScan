"""Initial time and timezone setup for NightScanPi."""
from __future__ import annotations

import argparse
import subprocess

try:  # pragma: no cover - optional dependency
    from timezonefinder import TimezoneFinder
except Exception:  # pragma: no cover - not installed
    TimezoneFinder = None

DEFAULT_LAT = 46.9480
DEFAULT_LON = 7.4474
DEFAULT_TZ = "Europe/Zurich"


def guess_timezone(lat: float, lon: float) -> str:
    """Return timezone name for the coordinates or ``DEFAULT_TZ``."""
    if TimezoneFinder is None:
        return DEFAULT_TZ
    tz = TimezoneFinder().timezone_at(lat=lat, lng=lon)
    return tz or DEFAULT_TZ


def set_timezone(tz: str) -> None:
    """Apply timezone using ``timedatectl``."""
    subprocess.run(["sudo", "timedatectl", "set-timezone", tz], check=False)


def set_time(dt_str: str) -> None:
    """Set the system time using ``timedatectl``."""
    subprocess.run(["sudo", "timedatectl", "set-time", dt_str], check=False)


def configure(dt_str: str, lat: float | None = None, lon: float | None = None) -> str:
    """Configure time and timezone and return the applied timezone."""
    if lat is None or lon is None:
        lat, lon = DEFAULT_LAT, DEFAULT_LON
    tz = guess_timezone(lat, lon)
    set_timezone(tz)
    set_time(dt_str)
    return tz


def main() -> None:
    parser = argparse.ArgumentParser(description="Initial time configuration")
    parser.add_argument("datetime", help="Current date/time 'YYYY-MM-DD HH:MM:SS'")
    parser.add_argument("--lat", type=float, help="Latitude")
    parser.add_argument("--lon", type=float, help="Longitude")
    args = parser.parse_args()
    tz = configure(args.datetime, args.lat, args.lon)
    print(f"Timezone set to {tz}")


if __name__ == "__main__":
    main()
