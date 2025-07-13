"""Wi-Fi configuration helper."""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path


CONFIG_PATH = Path("/etc/wpa_supplicant/wpa_supplicant.conf")


def write_wifi_config(ssid: str, password: str, path: Path = CONFIG_PATH) -> None:
    """Write Wi-Fi credentials to ``path``."""
    conf = (
        "country=US\n"
        "ctrl_interface=DIR=/var/run/wpa_supplicant GROUP=netdev\n"
        "update_config=1\n\n"
        "network={\n"
        f"    ssid=\"{ssid}\"\n"
        f"    psk=\"{password}\"\n"
        "}"
    )
    Path(path).write_text(conf)
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass
    try:
        subprocess.run(["wpa_cli", "-i", "wlan0", "reconfigure"], check=False)
    except Exception:
        pass


def save_credentials(ssid: str, password: str, out_file: Path) -> None:
    """Save credentials in JSON format for later use."""
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps({"ssid": ssid, "password": password}))


def apply_credentials_file(file_path: Path, path: Path | None = None) -> None:
    """Read JSON credentials from ``file_path`` and write the config."""
    if path is None:
        path = CONFIG_PATH
    data = json.loads(Path(file_path).read_text())
    write_wifi_config(data["ssid"], data["password"], path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("ssid", nargs="?")
    parser.add_argument("password", nargs="?")
    parser.add_argument("--out", type=Path, help="Store credentials instead of writing to system")
    parser.add_argument("--apply", type=Path, help="Apply credentials from JSON file")
    args = parser.parse_args()

    if args.apply:
        apply_credentials_file(args.apply, args.out or CONFIG_PATH)
    elif args.ssid and args.password:
        if args.out:
            save_credentials(args.ssid, args.password, args.out)
        else:
            write_wifi_config(args.ssid, args.password)
    else:
        parser.error("SSID and password required unless --apply is used")
