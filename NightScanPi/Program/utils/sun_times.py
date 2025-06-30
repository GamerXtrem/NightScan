"""Sunrise and sunset calculations for NightScanPi."""
from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
import json

from .. import time_config

try:  # pragma: no cover - optional dependency
    from suntime import Sun
except Exception:  # pragma: no cover - not installed
    Sun = None  # type: ignore


def get_sun_times(
    day: date, lat: float | None = None, lon: float | None = None
) -> tuple[datetime, datetime]:
    """Return local sunrise and sunset ``datetime`` objects for ``day``."""
    if lat is None or lon is None:
        lat, lon = time_config.DEFAULT_LAT, time_config.DEFAULT_LON
    if Sun is None:
        raise RuntimeError("suntime is not available")
    sun = Sun(lat, lon)
    sunrise = sun.get_local_sunrise_time(day)
    sunset = sun.get_local_sunset_time(day)
    return sunrise, sunset


def save_sun_times(
    file_path: Path, day: date, lat: float | None = None, lon: float | None = None
) -> tuple[datetime, datetime]:
    """Compute sunrise/sunset and save them in ``file_path`` as JSON."""
    sunrise, sunset = get_sun_times(day, lat, lon)
    out = {
        "date": day.isoformat(),
        "sunrise": sunrise.isoformat(),
        "sunset": sunset.isoformat(),
    }
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(json.dumps(out))
    return sunrise, sunset


def load_sun_times(file_path: Path) -> tuple[date, datetime, datetime]:
    """Load sun times from ``file_path`` and return day, sunrise and sunset."""
    data = json.loads(Path(file_path).read_text())
    day = date.fromisoformat(data["date"])
    sunrise = datetime.fromisoformat(data["sunrise"])
    sunset = datetime.fromisoformat(data["sunset"])
    return day, sunrise, sunset


def get_or_update_sun_times(
    file_path: Path,
    day: date | None = None,
    lat: float | None = None,
    lon: float | None = None,
) -> tuple[date, datetime, datetime]:
    """Return sun times from ``file_path`` updating it if necessary."""
    if day is None:
        day = date.today()
    if file_path.exists():
        try:
            saved_day, sunrise, sunset = load_sun_times(file_path)
            if saved_day == day:
                return saved_day, sunrise, sunset
        except Exception:
            pass
    sunrise, sunset = save_sun_times(file_path, day, lat, lon)
    return day, sunrise, sunset


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Save today's sun times")
    parser.add_argument("file", nargs="?", default=str(time_config.DEFAULT_SUN_FILE))
    parser.add_argument("--lat", type=float, help="Latitude")
    parser.add_argument("--lon", type=float, help="Longitude")
    args = parser.parse_args()
    save_sun_times(Path(args.file), date.today(), args.lat, args.lon)


if __name__ == "__main__":
    main()
