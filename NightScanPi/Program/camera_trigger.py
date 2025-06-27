"""Camera capture helpers for NightScanPi."""
from __future__ import annotations

from pathlib import Path
from datetime import datetime

try:
    from picamera import PiCamera
except ImportError:  # pragma: no cover - not available in CI
    PiCamera = None  # type: ignore


def capture_image(out_dir: Path) -> Path:
    """Capture a single JPEG image and return the file path."""
    if PiCamera is None:
        raise RuntimeError("PiCamera library not available")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = datetime.now().strftime("%Y%m%d_%H%M%S.jpg")
    out_path = out_dir / filename

    with PiCamera() as camera:
        camera.resolution = (1920, 1080)
        camera.capture(out_path)

    return out_path


if __name__ == "__main__":
    print(capture_image(Path("images")))
