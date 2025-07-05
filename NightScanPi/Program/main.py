"""Main orchestrator for NightScanPi."""
from __future__ import annotations

from pathlib import Path
import time
import logging
import os

from . import audio_capture
from . import camera_trigger
from .utils import energy_manager
from .utils import detector
from . import spectrogram_gen

LOG_PATH = Path(os.getenv("NIGHTSCAN_LOG", "nightscan.log"))
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH),
        logging.StreamHandler(),
    ],
)


def run_cycle(base_dir: Path) -> None:
    """Record audio and capture an image with timing metrics."""
    start_time = time.time()
    audio_dir = base_dir / "audio"
    image_dir = base_dir / "images"
    ts = int(time.time())
    
    logging.info(f"Starting capture cycle {ts}")
    
    try:
        # Record audio
        audio_start = time.time()
        audio_capture.record_segment(8, audio_dir / f"{ts}.wav")
        audio_duration = time.time() - audio_start
        logging.info(f"Audio capture completed in {audio_duration:.3f}s")
        
        # Capture image
        camera_start = time.time()
        camera_trigger.capture_image(image_dir)
        camera_duration = time.time() - camera_start
        logging.info(f"Camera capture completed in {camera_duration:.3f}s")
        
    except Exception as exc:  # pragma: no cover - camera may be absent
        logging.warning("Camera capture failed: %s", exc)
    
    total_duration = time.time() - start_time
    logging.info(f"Complete capture cycle took {total_duration:.3f}s")


def main(data_dir: Path = Path("data")) -> None:
    """Continuously capture data during the active period with adaptive sleep."""
    consecutive_no_activity = 0
    base_sleep = 0.1  # Start with 100ms
    max_sleep = 5.0   # Max 5 seconds
    
    while True:
        if energy_manager.within_active_period():
            if detector.pir_detected() or detector.audio_triggered():
                run_cycle(data_dir)
                consecutive_no_activity = 0
                current_sleep = base_sleep  # Reset to fast polling after activity
            else:
                consecutive_no_activity += 1
                # Exponential backoff for sleep time when no activity
                current_sleep = min(
                    base_sleep * (1.5 ** min(consecutive_no_activity // 10, 8)),
                    max_sleep
                )
        else:
            spectrogram_gen.scheduled_conversion(
                data_dir / "audio", data_dir / "spectrograms"
            )
            energy_manager.shutdown()
            break
        
        time.sleep(current_sleep)


if __name__ == "__main__":
    main()
