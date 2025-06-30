import sys
import types
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from NightScanPi.Program import time_config


def test_configure_defaults(monkeypatch):
    tf = types.SimpleNamespace(timezone_at=lambda lat, lng: "Europe/Zurich")
    monkeypatch.setattr(time_config, "TimezoneFinder", lambda: tf)
    calls = []

    def fake_run(cmd, check=False):
        calls.append(cmd)

    monkeypatch.setattr(time_config.subprocess, "run", fake_run)

    saved = []

    def fake_save(path, day, lat=None, lon=None):
        saved.append(path)

    monkeypatch.setattr(time_config.sun_times, "save_sun_times", fake_save)

    tz = time_config.configure("2024-01-01 12:00:00")
    assert tz == "Europe/Zurich"
    assert calls == [
        ["sudo", "timedatectl", "set-timezone", "Europe/Zurich"],
        ["sudo", "timedatectl", "set-time", "2024-01-01 12:00:00"],
    ]
    assert saved and saved[0] == time_config.DEFAULT_SUN_FILE


def test_guess_timezone_fallback(monkeypatch):
    monkeypatch.setattr(time_config, "TimezoneFinder", None)
    tz = time_config.guess_timezone(0, 0)
    assert tz == time_config.DEFAULT_TZ
