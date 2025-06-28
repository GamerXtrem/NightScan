"""Sensor detection helpers for NightScanPi."""
from __future__ import annotations

import os
import time
from typing import Optional

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
AUDIO_THRESHOLD = float(os.getenv("NIGHTSCAN_AUDIO_THRESHOLD", "500"))


def pir_detected(pin: Optional[int] = None) -> bool:
    """Return ``True`` if the PIR sensor indicates motion."""
    if GPIO is None:
        return False
    if pin is None:
        pin = PIR_PIN
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(pin, GPIO.IN)
    state = GPIO.input(pin)
    GPIO.cleanup(pin)
    return bool(state)


def audio_triggered(threshold: Optional[float] = None, duration: float = 0.5) -> bool:
    """Return ``True`` if microphone input exceeds ``threshold``."""
    if pyaudio is None or np is None:
        return False
    if threshold is None:
        threshold = AUDIO_THRESHOLD

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
