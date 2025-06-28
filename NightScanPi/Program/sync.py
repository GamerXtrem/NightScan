"""Data synchronisation utilities."""
from __future__ import annotations

from pathlib import Path
import os
import requests

SIM_DEVICE = os.getenv("NIGHTSCAN_SIM_DEVICE")
SIM_BAUDRATE = int(os.getenv("NIGHTSCAN_SIM_BAUDRATE", "115200"))


API_URL = "https://example.com/upload"
OFFLINE = os.getenv("NIGHTSCAN_OFFLINE", "0") in {"1", "true", "True"}


def upload_file(path: Path, url: str = API_URL) -> None:
    """Upload ``path`` to ``url`` via HTTP POST."""
    with path.open("rb") as f:
        try:
            resp = requests.post(url, files={"file": (path.name, f)})
            resp.raise_for_status()
        except requests.RequestException:
            if SIM_DEVICE:
                upload_file_via_sim(path, url, SIM_DEVICE, SIM_BAUDRATE)
            else:
                raise


def upload_file_via_sim(path: Path, url: str = API_URL, device: str | None = None, baudrate: int = 115200) -> None:
    """Upload ``path`` to ``url`` using a SIM modem via AT commands."""
    if device is None:
        raise RuntimeError("SIM device not configured")
    import serial  # lazy import so tests can patch

    with serial.Serial(device, baudrate, timeout=10) as ser, path.open("rb") as f:
        data = f.read()
        ser.write(b"AT+HTTPTERM\r")
        ser.readline()
        ser.write(b"AT+HTTPINIT\r")
        ser.readline()
        ser.write(f'AT+HTTPPARA="URL","{url}"\r'.encode())
        ser.readline()
        ser.write(f"AT+HTTPDATA={len(data)},10000\r".encode())
        ser.readline()
        ser.write(data)
        ser.readline()
        ser.write(b"AT+HTTPACTION=1\r")
        ser.readline()
        ser.write(b"AT+HTTPTERM\r")
        ser.readline()


def sync_directory(dir_path: Path, url: str = API_URL, *, offline: bool | None = None) -> None:
    """Upload all files in ``dir_path`` and remove them on success unless offline."""
    if offline is None:
        offline = OFFLINE
    if offline:
        return
    for p in Path(dir_path).glob("*"):
        try:
            upload_file(p, url)
            p.unlink()
        except requests.RequestException:
            continue


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("dir_path", type=Path)
    parser.add_argument("--url", default=API_URL)
    parser.add_argument("--offline", action="store_true", help="Skip uploads")
    args = parser.parse_args()

    sync_directory(args.dir_path, args.url, offline=args.offline)
