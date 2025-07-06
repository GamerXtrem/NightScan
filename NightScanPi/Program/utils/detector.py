"""Sensor detection helpers for NightScanPi with integrated threshold detection."""
from __future__ import annotations

import os
import time
import logging
from typing import Optional

logger = logging.getLogger(__name__)

try:  # pragma: no cover - hardware not installed in CI
    import RPi.GPIO as GPIO
except Exception:  # pragma: no cover - missing dependency
    GPIO = None

try:  # pragma: no cover - pyaudio might be missing
    import pyaudio
    import numpy as np
except Exception:  # pragma: no cover - missing dependency
    pyaudio = None  # type: ignore
    np = None  # type: ignore

PIR_PIN = int(os.getenv("NIGHTSCAN_PIR_PIN", "17"))
# Keep legacy threshold for backwards compatibility
LEGACY_AUDIO_THRESHOLD = float(os.getenv("NIGHTSCAN_AUDIO_THRESHOLD", "500"))


def pir_detected(pin: Optional[int] = None) -> bool:
    """Return ``True`` if the PIR sensor indicates motion."""
    if GPIO is None:
        return False
    if pin is None:
        pin = PIR_PIN
    try:
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(pin, GPIO.IN)
        state = GPIO.input(pin)
        if state:
            time.sleep(0.1)  # Debounce
        return bool(state)
    except Exception as e:
        logger.debug(f"PIR detection error: {e}")
        return False
    finally:
        try:
            GPIO.cleanup(pin)
        except:
            pass


def audio_triggered(threshold: Optional[float] = None, duration: float = 0.5) -> bool:
    """Return ``True`` if microphone input exceeds configured threshold."""
    # Try to use the new threshold detection system first
    try:
        from ..audio_threshold import audio_triggered as threshold_triggered
        return threshold_triggered()
    except ImportError:
        logger.debug("Audio threshold module not available, using legacy detection")
    except Exception as e:
        logger.warning(f"Audio threshold detection error: {e}, falling back to legacy")
    
    # Fallback to legacy audio detection
    return _legacy_audio_triggered(threshold, duration)


def _legacy_audio_triggered(threshold: Optional[float] = None, duration: float = 0.5) -> bool:
    """Legacy audio trigger detection for backwards compatibility."""
    if pyaudio is None or np is None:
        return False
    if threshold is None:
        threshold = LEGACY_AUDIO_THRESHOLD

    try:
        pa = pyaudio.PyAudio()
        stream = pa.open(format=pyaudio.paInt16, channels=1, rate=22050, input=True, frames_per_buffer=1024)
        samples: list[np.ndarray] = []
        for _ in range(int(22050 / 1024 * duration)):
            data = stream.read(1024, exception_on_overflow=False)
            samples.append(np.frombuffer(data, dtype=np.int16))
        stream.stop_stream()
        stream.close()
        pa.terminate()
        amplitude = np.abs(np.concatenate(samples)).mean()
        return amplitude >= threshold
    except Exception as e:
        logger.error(f"Legacy audio detection error: {e}")
        return False


def get_audio_detection_info() -> dict:
    """Get detailed audio detection information."""
    try:
        from ..audio_threshold import get_threshold_detector
        
        detector = get_threshold_detector()
        
        # Get a quick audio sample for analysis
        from ..audio_capture import get_audio_config
        audio_config = get_audio_config()
        
        if not audio_config:
            return {"error": "Audio configuration not available"}
        
        import pyaudio
        p = pyaudio.PyAudio()
        
        try:
            stream_kwargs = {
                'format': pyaudio.paInt16,
                'channels': audio_config['channels'],
                'rate': audio_config['sample_rate'],
                'input': True,
                'frames_per_buffer': 512
            }
            
            if audio_config['device_id'] is not None:
                stream_kwargs['input_device_index'] = audio_config['device_id']
            
            stream = p.open(**stream_kwargs)
            
            # Get audio sample
            data = stream.read(512, exception_on_overflow=False)
            is_detected, detection_info = detector.is_audio_detected(data)
            
            stream.stop_stream()
            stream.close()
            
            # Add configuration info
            detection_info.update({
                "config": {
                    "threshold_db": detector.config.threshold_db,
                    "environment_preset": detector.config.environment_preset,
                    "adaptive_enabled": detector.config.adaptive_enabled
                }
            })
            
            return detection_info
            
        finally:
            p.terminate()
            
    except ImportError:
        logger.debug("Audio threshold module not available")
        return {"error": "Audio threshold detection not available"}
    except Exception as e:
        logger.error(f"Audio detection info error: {e}")
        return {"error": str(e)}


def get_detection_status() -> dict:
    """Get comprehensive detection status for all sensors."""
    status = {
        "timestamp": time.time(),
        "pir": {
            "available": GPIO is not None,
            "detected": False
        },
        "audio": {
            "available": False,
            "detected": False,
            "info": {}
        }
    }
    
    # Check PIR sensor
    try:
        status["pir"]["detected"] = pir_detected()
    except Exception as e:
        logger.debug(f"PIR detection error: {e}")
    
    # Check audio threshold
    try:
        status["audio"]["detected"] = audio_triggered()
        status["audio"]["available"] = True
        status["audio"]["info"] = get_audio_detection_info()
    except Exception as e:
        logger.debug(f"Audio detection error: {e}")
        status["audio"]["info"] = {"error": str(e)}
    
    return status


def calibrate_audio_detection() -> bool:
    """Perform audio threshold calibration."""
    try:
        from ..audio_threshold import get_threshold_detector
        
        detector = get_threshold_detector()
        logger.info("Starting audio threshold calibration...")
        
        success = detector.calibrate_noise_floor(duration_seconds=5)
        
        if success:
            logger.info("Audio threshold calibration completed successfully")
        else:
            logger.warning("Audio threshold calibration failed")
            
        return success
        
    except ImportError:
        logger.debug("Audio threshold module not available for calibration")
        return False
    except Exception as e:
        logger.error(f"Audio calibration error: {e}")
        return False
