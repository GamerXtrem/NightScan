"""Data synchronisation utilities."""
from __future__ import annotations

from pathlib import Path
import requests


API_URL = "https://example.com/upload"


def upload_file(path: Path, url: str = API_URL) -> None:
    """Upload ``path`` to ``url`` via HTTP POST."""
    with path.open("rb") as f:
        resp = requests.post(url, files={"file": (path.name, f)})
        resp.raise_for_status()


def sync_directory(dir_path: Path, url: str = API_URL) -> None:
    """Upload all files in ``dir_path`` and remove them on success."""
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
    args = parser.parse_args()

    sync_directory(args.dir_path, args.url)
