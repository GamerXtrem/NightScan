"""Data synchronisation utilities."""
from __future__ import annotations

from pathlib import Path
import os
import socket
import requests
import logging
import time

SIM_DEVICE = os.getenv("NIGHTSCAN_SIM_DEVICE")
SIM_BAUDRATE = int(os.getenv("NIGHTSCAN_SIM_BAUDRATE", "115200"))


API_URL = os.getenv("NIGHTSCAN_API_URL", "https://api.nightscan.example.com/upload")
API_TOKEN = os.getenv("NIGHTSCAN_API_TOKEN")
UPLOAD_RETRIES = int(os.getenv("NIGHTSCAN_UPLOAD_RETRIES", "3"))
UPLOAD_TIMEOUT = int(os.getenv("NIGHTSCAN_UPLOAD_TIMEOUT", "60"))
OFFLINE = os.getenv("NIGHTSCAN_OFFLINE", "0") in {"1", "true", "True"}

logger = logging.getLogger(__name__)


def network_available(host: str = "8.8.8.8", port: int = 53, timeout: int = 3) -> bool:
    """Return ``True`` if the network appears reachable."""
    try:
        with socket.create_connection((host, port), timeout):
            return True
    except OSError:
        return False


def upload_file(path: Path, url: str = API_URL) -> None:
    """Upload ``path`` to ``url`` via HTTP POST with retry logic and authentication."""
    if not network_available():
        if SIM_DEVICE:
            logger.info(f"Network unavailable, using SIM for {path.name}")
            upload_file_via_sim(path, url, SIM_DEVICE, SIM_BAUDRATE)
            return
        else:
            raise requests.RequestException("Network unavailable and SIM not configured")
    
    # Prepare headers for authentication
    headers = {}
    if API_TOKEN:
        headers["Authorization"] = f"Bearer {API_TOKEN}"
        logger.debug(f"Using API authentication for {path.name}")
    
    # Retry logic for upload
    last_exception = None
    for attempt in range(UPLOAD_RETRIES):
        try:
            with path.open("rb") as f:
                logger.info(f"Uploading {path.name} (attempt {attempt + 1}/{UPLOAD_RETRIES})")
                
                resp = requests.post(
                    url,
                    files={"file": (path.name, f)},
                    headers=headers,
                    timeout=UPLOAD_TIMEOUT
                )
                
                resp.raise_for_status()
                logger.info(f"Successfully uploaded {path.name} ({path.stat().st_size} bytes)")
                return
                
        except requests.exceptions.Timeout as e:
            last_exception = e
            logger.warning(f"Upload timeout for {path.name} (attempt {attempt + 1}): {e}")
            
        except requests.exceptions.HTTPError as e:
            last_exception = e
            if e.response.status_code == 401:
                logger.error(f"Authentication failed for {path.name}: Check API_TOKEN")
                break  # Don't retry auth errors
            elif e.response.status_code == 413:
                logger.error(f"File too large for {path.name}: {e}")
                break  # Don't retry file too large errors
            else:
                logger.warning(f"HTTP error for {path.name} (attempt {attempt + 1}): {e}")
                
        except requests.RequestException as e:
            last_exception = e
            logger.warning(f"Request failed for {path.name} (attempt {attempt + 1}): {e}")
        
        # Exponential backoff between retries
        if attempt < UPLOAD_RETRIES - 1:
            wait_time = 2 ** attempt
            logger.debug(f"Waiting {wait_time}s before retry...")
            time.sleep(wait_time)
    
    # All retries exhausted
    if SIM_DEVICE:
        logger.info(f"HTTP upload failed for {path.name}, trying SIM")
        upload_file_via_sim(path, url, SIM_DEVICE, SIM_BAUDRATE)
    else:
        raise requests.RequestException(f"Upload failed after {UPLOAD_RETRIES} attempts: {last_exception}")


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
