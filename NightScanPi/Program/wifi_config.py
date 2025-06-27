"""Wi-Fi configuration helper."""
from __future__ import annotations

import json
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


def save_credentials(ssid: str, password: str, out_file: Path) -> None:
    """Save credentials in JSON format for later use."""
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps({"ssid": ssid, "password": password}))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("ssid")
    parser.add_argument("password")
    parser.add_argument("--out", type=Path, help="Store credentials instead of writing to system")
    args = parser.parse_args()

    if args.out:
        save_credentials(args.ssid, args.password, args.out)
    else:
        write_wifi_config(args.ssid, args.password)
