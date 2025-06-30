import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from NightScanPi.Program import manual_transfer


def test_network_available(monkeypatch):
    called = False

    def fake_create(*args, **kwargs):
        nonlocal called
        called = True
        class Dummy:
            def __enter__(self):
                return self
            def __exit__(self, *exc):
                pass
        return Dummy()

    monkeypatch.setattr(manual_transfer.socket, "create_connection", fake_create)
    assert manual_transfer.network_available()
    assert called


def test_transfer_exports_success(monkeypatch, tmp_path):
    called = {"sync": False, "notify": False, "network": False}

    monkeypatch.setattr(manual_transfer, "EXPORT_DIR", tmp_path)
    monkeypatch.setattr(manual_transfer, "network_available", lambda: True)

    def fake_sync(path):
        called["sync"] = True

    def fake_notify():
        called["notify"] = True

    monkeypatch.setattr(manual_transfer.sync, "sync_directory", fake_sync)
    monkeypatch.setattr(manual_transfer, "notify_app", fake_notify)

    assert manual_transfer.transfer_exports()
    assert called["sync"]
    assert called["notify"]


def test_create_app_routes(tmp_path):
    app = manual_transfer.create_app()
    client = app.test_client()
    resp = client.get("/")
    assert resp.status_code == 200
    resp = client.post("/transfer")
    assert resp.status_code in {302, 308}
