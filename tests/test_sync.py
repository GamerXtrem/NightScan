import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from NightScanPi.Program import sync


def test_sync_directory_offline(tmp_path, monkeypatch):
    f = tmp_path / "x.txt"
    f.write_text("data")
    called = False

    def fake_upload(path, url=sync.API_URL):
        nonlocal called
        called = True

    monkeypatch.setattr(sync, "upload_file", fake_upload)

    sync.sync_directory(tmp_path, offline=True)
    assert f.exists()
    assert not called


def test_sync_directory_success(tmp_path, monkeypatch):
    f = tmp_path / "x.txt"
    f.write_text("data")
    called = False

    class DummyResp:
        def raise_for_status(self):
            pass

    def fake_post(url, files):
        nonlocal called
        called = True
        return DummyResp()

    monkeypatch.setattr(sync.requests, "post", fake_post)

    sync.sync_directory(tmp_path, url="http://test")
    assert called
    assert not f.exists()


def test_upload_file_fallback_sim(tmp_path, monkeypatch):
    f = tmp_path / "x.txt"
    f.write_text("data")
    called = {
        "serial": False,
    }

    class DummySerial:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            called["serial"] = True
            return self

        def __exit__(self, *exc):
            pass

        def write(self, data):
            pass

        def readline(self):
            return b"OK"

    def fake_post(url, files):
        raise sync.requests.RequestException("fail")

    monkeypatch.setattr(sync.requests, "post", fake_post)
    monkeypatch.setattr(sync, "SIM_DEVICE", "/dev/ttyUSB0")
    monkeypatch.setitem(sys.modules, "serial", type("m", (), {"Serial": DummySerial}))

    sync.upload_file(f, "http://test")
    assert called["serial"]
