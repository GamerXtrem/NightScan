import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from NightScanPi.Program import benchmark_wifi_wakeup


def test_benchmark(monkeypatch):
    # Ensure detect_tone runs quickly
    monkeypatch.setattr(benchmark_wifi_wakeup.wifi_wakeup, "detect_tone", lambda: None)
    usage = benchmark_wifi_wakeup.benchmark(10)
    assert 0 <= usage <= 100
