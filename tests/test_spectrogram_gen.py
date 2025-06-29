from pathlib import Path
from datetime import datetime
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

import NightScanPi.Program.spectrogram_gen as sg


def test_scheduled_conversion_skips_before_noon(tmp_path, monkeypatch):
    called = False

    def fake_convert(wav_dir, spec_dir, remove=False, *, sr=22050):
        nonlocal called
        called = True

    monkeypatch.setattr(sg, "convert_directory", fake_convert)
    monkeypatch.setattr(sg, "disk_usage_percent", lambda p: 10)

    wav = tmp_path / "wav"
    wav.mkdir()
    f = wav / "a.wav"
    f.write_bytes(b"data")

    sg.scheduled_conversion(wav, tmp_path / "spec", now=datetime(2022, 1, 1, 11, 0, 0))

    assert not called
    assert f.exists()


def test_scheduled_conversion_deletes_when_threshold(tmp_path, monkeypatch):
    called = False

    def fake_convert(wav_dir, spec_dir, remove=False, *, sr=22050):
        nonlocal called
        called = True

    monkeypatch.setattr(sg, "convert_directory", fake_convert)
    monkeypatch.setattr(sg, "disk_usage_percent", lambda p: 80)

    wav = tmp_path / "wav"
    wav.mkdir()
    f = wav / "a.wav"
    f.write_bytes(b"data")

    sg.scheduled_conversion(wav, tmp_path / "spec", now=datetime(2022, 1, 1, 13, 0, 0))

    assert called
    assert not f.exists()


def test_scheduled_conversion_keeps_when_below_threshold(tmp_path, monkeypatch):
    monkeypatch.setattr(sg, "convert_directory", lambda *a, **k: None)
    monkeypatch.setattr(sg, "disk_usage_percent", lambda p: 50)

    wav = tmp_path / "wav"
    wav.mkdir()
    f = wav / "a.wav"
    f.write_bytes(b"data")

    sg.scheduled_conversion(wav, tmp_path / "spec", now=datetime(2022, 1, 1, 13, 0, 0))

    assert f.exists()


def test_convert_directory_skips_invalid(tmp_path, monkeypatch):
    wav_dir = tmp_path / "wav"
    wav_dir.mkdir()
    good = wav_dir / "good.wav"
    bad = wav_dir / "bad.wav"
    good.write_bytes(b"good")
    bad.write_bytes(b"bad")

    out_dir = tmp_path / "spec"
    calls = []

    def fake_wav_to_spec(src, dst, sr=22050):
        calls.append(src.name)
        if src == bad:
            raise RuntimeError("failed")
        dst.write_text("ok")

    monkeypatch.setattr(sg, "wav_to_spec", fake_wav_to_spec)

    sg.convert_directory(wav_dir, out_dir)

    assert (out_dir / "good.npy").exists()
    assert not (out_dir / "bad.npy").exists()
    assert set(calls) == {"good.wav", "bad.wav"}
