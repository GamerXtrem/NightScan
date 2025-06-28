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
