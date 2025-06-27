"""Main orchestrator for NightScanPi."""
from __future__ import annotations

from pathlib import Path
import time
import logging

from . import audio_capture
from . import camera_trigger
from .utils import energy_manager

logging.basicConfig(level=logging.INFO)


def run_cycle(base_dir: Path) -> None:
    """Record audio and capture an image."""
    audio_dir = base_dir / "audio"
    image_dir = base_dir / "images"
    ts = int(time.time())
    audio_capture.record_segment(8, audio_dir / f"{ts}.wav")
    try:
        camera_trigger.capture_image(image_dir)
    except Exception as exc:  # pragma: no cover - camera may be absent
        logging.warning("Camera capture failed: %s", exc)


def main(data_dir: Path = Path("data")) -> None:
    """Continuously capture data during the active period."""
    while True:
        if energy_manager.within_active_period():
            run_cycle(data_dir)
        else:
            time.sleep(60)
        time.sleep(1)


if __name__ == "__main__":
    main()
