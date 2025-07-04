"""Wi-Fi wakeup via acoustic signal detection."""

from __future__ import annotations

import logging
import os
import socket
import subprocess
import time
from pathlib import Path

import numpy as np

try:  # pragma: no cover - optional dependency
    import sounddevice as sd
except Exception:  # pragma: no cover - missing dependency
    sd = None  # type: ignore


RATE = int(os.getenv("NIGHTSCAN_WAKE_RATE", "22050"))
TARGET_FREQ = int(os.getenv("NIGHTSCAN_WAKE_FREQ", "2100"))
THRESHOLD = float(os.getenv("NIGHTSCAN_WAKE_THRESHOLD", "10"))
DURATION = float(os.getenv("NIGHTSCAN_WAKE_DURATION", "0.5"))
STATUS_FILE = Path(os.getenv("NIGHTSCAN_WAKE_STATUS", "wifi_awake.status"))
DOWN_AFTER = int(os.getenv("NIGHTSCAN_WAKE_TIME", "600"))  # seconds

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)


def _read_status() -> float | None:
    try:
        return float(STATUS_FILE.read_text())
    except Exception:
        return None


def _write_status(ts: float) -> None:
    try:
        STATUS_FILE.write_text(str(ts))
    except Exception:
        pass


def _remove_status() -> None:
    try:
        STATUS_FILE.unlink()
    except FileNotFoundError:
        pass


def wifi_up() -> None:
    """Activate the Wi-Fi interface."""
    subprocess.run(["sudo", "ifconfig", "wlan0", "up"], check=False)
    logger.info("Wi-Fi interface up")


def wifi_down() -> None:
    """Deactivate the Wi-Fi interface and log how long it was up."""
    subprocess.run(["sudo", "ifconfig", "wlan0", "down"], check=False)
    start = _read_status()
    if start is not None:
        duration = time.time() - start
        logger.info("Wi-Fi interface down after %.1f s", duration)


def network_available(host: str = "8.8.8.8", port: int = 53, timeout: int = 3) -> bool:
    try:
        with socket.create_connection((host, port), timeout):
            return True
    except OSError:
        return False


def detect_tone() -> bool:
    """Return True if a TARGET_FREQ tone is detected."""
    if sd is None:
        return False
    rec = sd.rec(int(DURATION * RATE), samplerate=RATE, channels=1, dtype="float32")
    sd.wait()
    data = rec[:, 0]
    window = np.hanning(len(data))
    spec = np.abs(np.fft.rfft(data * window))
    freqs = np.fft.rfftfreq(len(data), d=1 / RATE)
    peak_idx = int(np.argmax(spec))
    peak_freq = freqs[peak_idx]
    amp = spec[peak_idx]
    logger.debug("Detected peak %.1f Hz amplitude %.2f", peak_freq, amp)
    return abs(peak_freq - TARGET_FREQ) < 50 and amp >= THRESHOLD


def main() -> None:
    if sd is None:
        raise RuntimeError("sounddevice not available")
    logger.info("Listening for wake-up tone at %d Hz", TARGET_FREQ)
    wake_time = _read_status()
    while True:
        if wake_time is None:
            if detect_tone():
                logger.info("Wake-up tone detected")
                wifi_up()
                wake_time = time.time()
                _write_status(wake_time)
        else:
            if network_available():
                wake_time = time.time()
                _write_status(wake_time)
            elif time.time() - wake_time > DOWN_AFTER:
                logger.info("Deactivating Wi-Fi after timeout")
                wifi_down()
                wake_time = None
                _remove_status()
        time.sleep(1)


if __name__ == "__main__":
    main()
