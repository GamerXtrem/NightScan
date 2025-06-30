import sys
import importlib
from pathlib import Path
from datetime import datetime
import os

sys.path.append(str(Path(__file__).resolve().parents[1]))

MODULE_PATH = 'NightScanPi.Program.utils.energy_manager'


def reload_energy_manager(pin=None, sun_file=None, offset=None):
    os.environ.pop('NIGHTSCAN_START_HOUR', None)
    os.environ.pop('NIGHTSCAN_STOP_HOUR', None)
    if pin is not None:
        os.environ['NIGHTSCAN_DONE_PIN'] = str(pin)
    else:
        os.environ.pop('NIGHTSCAN_DONE_PIN', None)
    if sun_file is not None:
        os.environ['NIGHTSCAN_SUN_FILE'] = str(sun_file)
    else:
        os.environ['NIGHTSCAN_SUN_FILE'] = str(Path.home() / 'sun_times.json')
    if offset is not None:
        os.environ['NIGHTSCAN_SUN_OFFSET'] = str(offset)
    else:
        os.environ.pop('NIGHTSCAN_SUN_OFFSET', None)
    if MODULE_PATH in sys.modules:
        del sys.modules[MODULE_PATH]
    return importlib.import_module(MODULE_PATH)


def test_within_active_period(monkeypatch, tmp_path):
    sun_file = tmp_path / 'sun.json'

    def fake_get_or_update(path, day, lat=None, lon=None):
        return day, datetime(day.year, day.month, day.day, 7, 0, 0), datetime(day.year, day.month, day.day, 17, 0, 0)

    mod = reload_energy_manager(sun_file=sun_file)
    monkeypatch.setattr(mod.sun_times, 'get_or_update_sun_times', fake_get_or_update)

    assert mod.within_active_period(datetime(2022, 1, 1, 16, 40, 0))
    assert not mod.within_active_period(datetime(2022, 1, 1, 7, 40, 0))


def test_custom_offset(monkeypatch, tmp_path):
    sun_file = tmp_path / 'sun.json'

    def fake_get_or_update(path, day, lat=None, lon=None):
        return day, datetime(day.year, day.month, day.day, 7, 0, 0), datetime(day.year, day.month, day.day, 17, 0, 0)

    mod = reload_energy_manager(sun_file=sun_file, offset=45)
    monkeypatch.setattr(mod.sun_times, 'get_or_update_sun_times', fake_get_or_update)

    assert not mod.within_active_period(datetime(2022, 1, 1, 16, 10, 0))
    assert mod.within_active_period(datetime(2022, 1, 1, 16, 20, 0))


def test_next_stop_time(monkeypatch, tmp_path):
    sun_file = tmp_path / 'sun.json'

    def fake_get_or_update(path, day, lat=None, lon=None):
        return day, datetime(day.year, day.month, day.day, 7, 0, 0), datetime(day.year, day.month, day.day, 17, 0, 0)

    mod = reload_energy_manager(sun_file=sun_file)
    monkeypatch.setattr(mod.sun_times, 'get_or_update_sun_times', fake_get_or_update)
    stop = mod.next_stop_time(datetime(2022, 1, 1, 12, 0, 0))
    assert stop == datetime(2022, 1, 2, 7, 30, 0)


def test_schedule_shutdown(monkeypatch, tmp_path):
    run_args = []
    sun_file = tmp_path / 'sun.json'

    def fake_run(cmd, check=False):
        run_args.append(cmd)

    def fake_get_or_update(path, day, lat=None, lon=None):
        return day, datetime(day.year, day.month, day.day, 7, 0, 0), datetime(day.year, day.month, day.day, 17, 0, 0)

    mod = reload_energy_manager(sun_file=sun_file)
    monkeypatch.setattr(mod.sun_times, 'get_or_update_sun_times', fake_get_or_update)
    monkeypatch.setattr(mod.subprocess, 'run', fake_run)
    mod.schedule_shutdown(datetime(2022, 1, 1, 12, 0, 0))
    assert run_args == [["sudo", "shutdown", "-h", "07:30"]]


def test_signal_done_without_gpio(monkeypatch, tmp_path):
    mod = reload_energy_manager(pin=23, sun_file=tmp_path / 'sun.json')
    monkeypatch.setattr(mod, 'GPIO', None)
    called = False

    def fake_sleep(t):
        nonlocal called
        called = True

    monkeypatch.setattr(mod.time, 'sleep', fake_sleep)
    mod.signal_done()
    assert not called


def test_signal_done_with_gpio(monkeypatch, tmp_path):
    mod = reload_energy_manager(pin=5, sun_file=tmp_path / 'sun.json')

    class DummyGPIO:
        BCM = 'BCM'
        OUT = 'OUT'

        def __init__(self):
            self.calls = []

        def setmode(self, mode):
            self.calls.append(('setmode', mode))

        def setup(self, pin, mode):
            self.calls.append(('setup', pin, mode))

        def output(self, pin, val):
            self.calls.append(('output', pin, val))

        def cleanup(self, pin=None):
            self.calls.append(('cleanup', pin))

    dummy = DummyGPIO()
    monkeypatch.setattr(mod, 'GPIO', dummy)
    monkeypatch.setattr(mod.time, 'sleep', lambda x: None)
    mod.signal_done()
    assert dummy.calls == [
        ('setmode', dummy.BCM),
        ('setup', 5, dummy.OUT),
        ('output', 5, True),
        ('cleanup', 5),
    ]
