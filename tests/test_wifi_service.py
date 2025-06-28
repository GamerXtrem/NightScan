import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from NightScanPi.Program import wifi_service, wifi_config


def test_set_wifi(tmp_path, monkeypatch):
    conf = tmp_path / "wpa.conf"
    monkeypatch.setattr(wifi_config, "CONFIG_PATH", conf)
    app = wifi_service.create_app(conf)
    client = app.test_client()
    resp = client.post("/wifi", json={"ssid": "Net", "password": "pass"})
    assert resp.status_code == 200
    text = conf.read_text()
    assert "ssid=\"Net\"" in text
    assert "psk=\"pass\"" in text


def test_set_wifi_missing_fields(tmp_path):
    app = wifi_service.create_app(tmp_path / "conf")
    client = app.test_client()
    resp = client.post("/wifi", json={"ssid": "Net"})
    assert resp.status_code == 400

