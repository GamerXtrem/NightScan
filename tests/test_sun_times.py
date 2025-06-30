import pytest
import sys
import importlib
from pathlib import Path
from datetime import date, datetime

sys.path.append(str(Path(__file__).resolve().parents[1]))
MODULE_PATH = 'NightScanPi.Program.utils.sun_times'


def reload_module(sun_class):
    if MODULE_PATH in sys.modules:
        del sys.modules[MODULE_PATH]
    mod = importlib.import_module(MODULE_PATH)
    mod.Sun = sun_class
    return mod


def test_get_sun_times(monkeypatch):
    class DummySun:
        def __init__(self, lat, lon):
            self.lat = lat
            self.lon = lon

        def get_local_sunrise_time(self, d):
            assert isinstance(d, date)
            return datetime(2024, 1, 1, 7, 0, 0)

        def get_local_sunset_time(self, d):
            assert isinstance(d, date)
            return datetime(2024, 1, 1, 17, 0, 0)

    mod = reload_module(DummySun)
    sunrise, sunset = mod.get_sun_times(date(2024, 1, 1), 1.0, 2.0)
    assert sunrise.hour == 7
    assert sunset.hour == 17


def test_save_and_load(tmp_path):
    class DummySun:
        def __init__(self, lat, lon):
            pass

        def get_local_sunrise_time(self, d):
            return datetime(2024, 1, 2, 6, 30, 0)

        def get_local_sunset_time(self, d):
            return datetime(2024, 1, 2, 18, 30, 0)

    mod = reload_module(DummySun)
    file_path = tmp_path / 'sun.json'
    sunrise, sunset = mod.save_sun_times(file_path, date(2024, 1, 2))
    assert file_path.exists()
    day, sr, ss = mod.load_sun_times(file_path)
    assert day == date(2024, 1, 2)
    assert sr == sunrise
    assert ss == sunset


def test_missing_library(monkeypatch):
    mod = reload_module(None)
    with pytest.raises(RuntimeError):
        mod.get_sun_times(date.today())


def test_get_or_update(tmp_path):
    class DummySun:
        def __init__(self, lat, lon):
            pass

        def get_local_sunrise_time(self, d):
            return datetime(2024, 1, 3, 6, 0, 0)

        def get_local_sunset_time(self, d):
            return datetime(2024, 1, 3, 18, 0, 0)

    mod = reload_module(DummySun)
    file_path = tmp_path / "sun.json"
    day, sr, ss = mod.get_or_update_sun_times(file_path, date(2024, 1, 3))
    assert file_path.exists()
    assert day == date(2024, 1, 3)
    # second call should read the same values without recomputing
    day2, sr2, ss2 = mod.get_or_update_sun_times(file_path, date(2024, 1, 3))
    assert (day2, sr2, ss2) == (day, sr, ss)
    # new date should trigger an update
    day3, sr3, ss3 = mod.get_or_update_sun_times(file_path, date(2024, 1, 4))
    assert day3 == date(2024, 1, 4)

