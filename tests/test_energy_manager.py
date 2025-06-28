import sys
import importlib
from pathlib import Path
from datetime import datetime
import os

sys.path.append(str(Path(__file__).resolve().parents[1]))

MODULE_PATH = 'NightScanPi.Program.utils.energy_manager'


def reload_energy_manager(start=None, stop=None):
    if start is not None:
        os.environ['NIGHTSCAN_START_HOUR'] = str(start)
    else:
        os.environ.pop('NIGHTSCAN_START_HOUR', None)
    if stop is not None:
        os.environ['NIGHTSCAN_STOP_HOUR'] = str(stop)
    else:
        os.environ.pop('NIGHTSCAN_STOP_HOUR', None)
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

