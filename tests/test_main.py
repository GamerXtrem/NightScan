import sys
import types
import pytest
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from NightScanPi.Program import main


def test_run_cycle(tmp_path, monkeypatch):
    audio_called = False
    image_called = False

    def fake_record_segment(duration, out_path):
        nonlocal audio_called
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(b"data")
        audio_called = True

    def fake_capture_image(out_dir):
        nonlocal image_called
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / "image.jpg"
        out.write_bytes(b"data")
        image_called = True
        return out

    monkeypatch.setattr(main, "audio_capture", types.SimpleNamespace(record_segment=fake_record_segment))
    monkeypatch.setattr(main, "camera_trigger", types.SimpleNamespace(capture_image=fake_capture_image))

    main.run_cycle(tmp_path)

    assert audio_called
    assert image_called
    assert any(p.suffix == ".wav" for p in (tmp_path / "audio").iterdir())
    assert any(p.suffix == ".jpg" for p in (tmp_path / "images").iterdir())


def test_main_loop_triggers_on_detection(tmp_path, monkeypatch):
    called = []

    def fake_run_cycle(base_dir):
        called.append(base_dir)

    monkeypatch.setattr(main, "run_cycle", fake_run_cycle)
    monkeypatch.setattr(main.energy_manager, "within_active_period", lambda: True)
    monkeypatch.setattr(
        main,
        "detector",
        types.SimpleNamespace(pir_detected=lambda: True, audio_triggered=lambda: False),
    )

    def fake_sleep(t):
        raise SystemExit()

    monkeypatch.setattr(main.time, "sleep", fake_sleep)

    with pytest.raises(SystemExit):
        main.main(tmp_path)

    assert called == [tmp_path]

