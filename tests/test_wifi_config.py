import json
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from NightScanPi.Program import wifi_config


def test_apply_credentials_file(tmp_path, monkeypatch):
    cred = {"ssid": "Net", "password": "pass"}
    cred_file = tmp_path / "cred.json"
    cred_file.write_text(json.dumps(cred))
    out_conf = tmp_path / "wpa.conf"
    monkeypatch.setattr(wifi_config, "CONFIG_PATH", out_conf)
    wifi_config.apply_credentials_file(cred_file)
    text = out_conf.read_text()
    assert "ssid=\"Net\"" in text
    assert "psk=\"pass\"" in text
