import types
import sys
from pathlib import Path
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from NightScanPi.Program import wifi_wakeup


def test_detect_tone(monkeypatch):
    samples = int(wifi_wakeup.DURATION * wifi_wakeup.RATE)
    t = wifi_wakeup.np.arange(samples) / wifi_wakeup.RATE
    tone = wifi_wakeup.np.sin(2 * wifi_wakeup.np.pi * wifi_wakeup.TARGET_FREQ * t)
    dummy_sd = types.SimpleNamespace(
        rec=lambda *a, **k: tone.astype("float32").reshape(-1, 1),
        wait=lambda: None,
    )
    monkeypatch.setattr(wifi_wakeup, "sd", dummy_sd)
    assert wifi_wakeup.detect_tone()


def test_main_activates_wifi(monkeypatch):
    events = []
    monkeypatch.setattr(wifi_wakeup, "sd", types.SimpleNamespace())
    monkeypatch.setattr(wifi_wakeup, "detect_tone", lambda: True)
    monkeypatch.setattr(wifi_wakeup, "network_available", lambda: False)
    monkeypatch.setattr(wifi_wakeup, "_read_status", lambda: None)
    monkeypatch.setattr(wifi_wakeup, "_write_status", lambda ts: events.append("write"))
    monkeypatch.setattr(wifi_wakeup, "wifi_up", lambda: events.append("up"))
    monkeypatch.setattr(wifi_wakeup, "wifi_down", lambda: events.append("down"))

    def fake_sleep(t):
        raise SystemExit()

    monkeypatch.setattr(wifi_wakeup.time, "sleep", fake_sleep)

    with monkeypatch.context() as m:
        m.setattr(wifi_wakeup, "sd", types.SimpleNamespace())
        with pytest.raises(SystemExit):
            wifi_wakeup.main()

    assert "up" in events
