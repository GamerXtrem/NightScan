import sys
import importlib
from pathlib import Path
from datetime import datetime
import os

sys.path.append(str(Path(__file__).resolve().parents[1]))

MODULE_PATH = 'NightScanPi.Program.utils.energy_manager'


def reload_energy_manager(start=None, stop=None, pin=None, sun_file=None, offset=None):
    if start is not None:
        os.environ['NIGHTSCAN_START_HOUR'] = str(start)
    else:
        os.environ.pop('NIGHTSCAN_START_HOUR', None)
    if stop is not None:
        os.environ['NIGHTSCAN_STOP_HOUR'] = str(stop)
    else:
        os.environ.pop('NIGHTSCAN_STOP_HOUR', None)
    if pin is not None:
        os.environ['NIGHTSCAN_DONE_PIN'] = str(pin)
    else:
        os.environ.pop('NIGHTSCAN_DONE_PIN', None)
    if sun_file is not None:
        os.environ['NIGHTSCAN_SUN_FILE'] = str(sun_file)
    else:
        os.environ.pop('NIGHTSCAN_SUN_FILE', None)
    if offset is not None:
        os.environ['NIGHTSCAN_SUN_OFFSET'] = str(offset)
    else:
        os.environ.pop('NIGHTSCAN_SUN_OFFSET', None)
    if MODULE_PATH in sys.modules:
        del sys.modules[MODULE_PATH]
    return importlib.import_module(MODULE_PATH)


def test_within_active_period_defaults(monkeypatch):
    mod = reload_energy_manager()
    t_active = datetime(2022, 1, 1, 19, 0, 0)
    t_inactive = datetime(2022, 1, 1, 11, 0, 0)
    assert mod.START_HOUR == 18
    assert mod.STOP_HOUR == 10
    assert mod.within_active_period(t_active)
    assert not mod.within_active_period(t_inactive)


def test_custom_hours(monkeypatch):
    mod = reload_energy_manager(start=6, stop=18)
    t_active = datetime(2022, 1, 1, 7, 0, 0)
    t_inactive = datetime(2022, 1, 1, 19, 0, 0)
    assert mod.START_HOUR == 6
    assert mod.STOP_HOUR == 18
    assert mod.within_active_period(t_active)
    assert not mod.within_active_period(t_inactive)


def test_cross_midnight(monkeypatch):
    mod = reload_energy_manager(start=20, stop=5)
    assert mod.within_active_period(datetime(2022, 1, 1, 21, 0, 0))
    assert mod.within_active_period(datetime(2022, 1, 1, 4, 0, 0))
    assert not mod.within_active_period(datetime(2022, 1, 1, 12, 0, 0))


def test_next_stop_time(monkeypatch):
    mod = reload_energy_manager(stop=10)
    now = datetime(2022, 1, 1, 9, 0, 0)
    assert mod.next_stop_time(now) == datetime(2022, 1, 1, 10, 0, 0)

    now_after = datetime(2022, 1, 1, 11, 0, 0)
    assert mod.next_stop_time(now_after) == datetime(2022, 1, 2, 10, 0, 0)


def test_schedule_shutdown(monkeypatch):
    run_args = []

    def fake_run(cmd, check=False):
        run_args.append(cmd)

    mod = reload_energy_manager(stop=10)
    monkeypatch.setattr(mod.subprocess, "run", fake_run)
    mod.schedule_shutdown(datetime(2022, 1, 1, 9, 0, 0))
    assert run_args == [["sudo", "shutdown", "-h", "10:00"]]


def test_signal_done_without_gpio(monkeypatch):
    mod = reload_energy_manager(pin=23)
    monkeypatch.setattr(mod, "GPIO", None)
    called = False

    def fake_sleep(t):
        nonlocal called
        called = True

    monkeypatch.setattr(mod.time, "sleep", fake_sleep)
    mod.signal_done()
    assert not called


def test_signal_done_with_gpio(monkeypatch):
    mod = reload_energy_manager(pin=5)

    class DummyGPIO:
        BCM = "BCM"
        OUT = "OUT"

        def __init__(self):
            self.calls = []

        def setmode(self, mode):
            self.calls.append(("setmode", mode))

        def setup(self, pin, mode):
            self.calls.append(("setup", pin, mode))

        def output(self, pin, val):
            self.calls.append(("output", pin, val))

        def cleanup(self, pin=None):
            self.calls.append(("cleanup", pin))

    dummy = DummyGPIO()
    monkeypatch.setattr(mod, "GPIO", dummy)
    monkeypatch.setattr(mod.time, "sleep", lambda x: None)
    mod.signal_done()
    assert dummy.calls == [
        ("setmode", dummy.BCM),
        ("setup", 5, dummy.OUT),
        ("output", 5, True),
        ("cleanup", 5),
    ]


def test_sun_schedule(tmp_path, monkeypatch):
    sun_file = tmp_path / "sun.json"

    def fake_get_or_update(path, day, lat=None, lon=None):
        if day == datetime(2022, 1, 1).date():
            sunrise = datetime(2022, 1, 1, 7, 0, 0)
            sunset = datetime(2022, 1, 1, 17, 0, 0)
        elif day == datetime(2022, 1, 2).date():
            sunrise = datetime(2022, 1, 2, 7, 0, 0)
            sunset = datetime(2022, 1, 2, 17, 0, 0)
        else:
            sunrise = datetime(2021, 12, 31, 7, 0, 0)
            sunset = datetime(2021, 12, 31, 17, 0, 0)
        return day, sunrise, sunset

    mod = reload_energy_manager(sun_file=sun_file)
    monkeypatch.setattr(mod.sun_times, "get_or_update_sun_times", fake_get_or_update)

    # 20 min before sunset should be active
    assert mod.within_active_period(datetime(2022, 1, 1, 16, 40, 0))
    # 20 min after sunrise still active
    assert mod.within_active_period(datetime(2022, 1, 2, 7, 20, 0))
    # 40 min after sunrise inactive
    assert not mod.within_active_period(datetime(2022, 1, 2, 7, 40, 0))

    stop = mod.next_stop_time(datetime(2022, 1, 1, 12, 0, 0))
    assert stop == datetime(2022, 1, 2, 7, 30, 0)
