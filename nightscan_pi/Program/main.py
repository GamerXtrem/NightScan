"""Main orchestrator for NightScanPi with intelligent energy management."""
from __future__ import annotations

from pathlib import Path
import time
import logging
import os

from . import audio_capture
from . import camera_trigger
from .utils import energy_manager
from .utils import detector
from .utils.smart_scheduler import get_energy_scheduler
from . import spectrogram_gen
from filename_utils import FilenameGenerator
from location_manager import location_manager

LOG_PATH = Path(os.getenv("NIGHTSCAN_LOG", "nightscan.log"))
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH),
        logging.StreamHandler(),
    ],
)


def run_cycle(base_dir: Path, camera_enabled: bool = True) -> None:
    """Record audio and optionally capture an image with timing metrics."""
    start_time = time.time()
    audio_dir = base_dir / "audio"
    image_dir = base_dir / "images"
    
    # Initialize filename generator with current location
    filename_gen = FilenameGenerator(location_manager)
    
    # Generate audio filename with GPS metadata
    audio_filename = filename_gen.generate_audio_filename()
    
    mode = "full" if camera_enabled else "audio-only"
    logging.info(f"Starting {mode} capture cycle - Audio: {audio_filename}")
    
    try:
        # Always record audio
        audio_start = time.time()
        audio_capture.record_segment(8, audio_dir / audio_filename)
        audio_duration = time.time() - audio_start
        logging.info(f"Audio capture completed in {audio_duration:.3f}s")
        
        # Capture image only during camera periods
        if camera_enabled:
            camera_start = time.time()
            try:
                camera_trigger.capture_image(image_dir)
                camera_duration = time.time() - camera_start
                logging.info(f"Camera capture completed in {camera_duration:.3f}s")
            except RuntimeError as exc:
                camera_duration = time.time() - camera_start
                logging.warning(f"Camera capture failed after {camera_duration:.3f}s: %s", exc)
        else:
            logging.debug("Camera capture skipped (outside camera period)")
        
    except Exception as exc:  # pragma: no cover - audio may fail
        logging.warning("Audio capture failed: %s", exc)
    
    total_duration = time.time() - start_time
    logging.info(f"Complete {mode} capture cycle took {total_duration:.3f}s")


def main(data_dir: Path = Path("data")) -> None:
    """Continuously capture data with intelligent energy management."""
    consecutive_no_activity = 0
    base_sleep = 0.1  # Start with 100ms
    max_sleep = 5.0   # Max 5 seconds
    scheduler = get_energy_scheduler()
    
    logging.info("Starting NightScanPi with intelligent energy management")
    
    while True:
        operation_mode = scheduler.get_operation_mode()
        
        if operation_mode == "sleep":
            # System should sleep during day hours
            logging.info("Entering day sleep mode")
            spectrogram_gen.scheduled_conversion(
                data_dir / "audio", data_dir / "spectrograms"
            )
            energy_manager.shutdown()
            break
            
        elif operation_mode in ["camera_active", "audio_only"]:
            # Active monitoring periods
            camera_enabled = (operation_mode == "camera_active")
            
            if detector.pir_detected() or detector.audio_triggered():
                run_cycle(data_dir, camera_enabled=camera_enabled)
                consecutive_no_activity = 0
                current_sleep = base_sleep  # Reset to fast polling after activity
            else:
                consecutive_no_activity += 1
                # Exponential backoff for sleep time when no activity
                current_sleep = min(
                    base_sleep * (1.5 ** min(consecutive_no_activity // 10, 8)),
                    max_sleep
                )
                
        elif operation_mode == "minimal":
            # Minimal operation mode - longer sleep periods
            current_sleep = min(max_sleep * 2, 10.0)  # Sleep up to 10 seconds
            consecutive_no_activity += 1
            
        else:
            # Fallback to legacy behavior
            if energy_manager.within_active_period():
                if detector.pir_detected() or detector.audio_triggered():
                    run_cycle(data_dir)
                    consecutive_no_activity = 0
                    current_sleep = base_sleep
                else:
                    consecutive_no_activity += 1
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
