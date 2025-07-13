"""Simple web interface for manual data transfer."""
from __future__ import annotations

from pathlib import Path
import os
import socket
import requests

from flask import Flask, render_template_string, redirect, url_for, flash

from . import sync

EXPORT_DIR = Path(os.getenv("NIGHTSCAN_EXPORT_DIR", "/data/exports"))
NOTIFY_URL = os.getenv("NIGHTSCAN_NOTIFY_URL", "")


def network_available(host: str = "8.8.8.8", port: int = 53, timeout: int = 3) -> bool:
    """Return True if the network appears reachable."""
    try:
        with socket.create_connection((host, port), timeout):
            return True
    except OSError:
        return False


def notify_app() -> None:
    """Send completion notification to the mobile app if configured."""
    if NOTIFY_URL:
        try:
            requests.post(NOTIFY_URL, json={"status": "done"}, timeout=5)
        except requests.RequestException:
            pass


def transfer_exports() -> bool:
    """Transfer files from EXPORT_DIR to the VPS."""
    if not network_available():
        return False
    try:
        sync.sync_directory(EXPORT_DIR)
    except Exception:
        return False
    notify_app()
    return True


PAGE = """
<!doctype html>
<title>NightScanPi Transfer</title>
{% with messages = get_flashed_messages() %}
  {% if messages %}
    <ul>{% for m in messages %}<li>{{ m }}</li>{% endfor %}</ul>
  {% endif %}
{% endwith %}
<form method="post" action="/transfer">
  <button type="submit">Transférer les données</button>
</form>
"""


def create_app() -> Flask:
    app = Flask(__name__)
    app.secret_key = os.environ.get("SECRET_KEY", "secret")

    @app.get("/")
    def index():
        return render_template_string(PAGE)

    @app.post("/transfer")
    def transfer():
        if transfer_exports():
            flash("Transfert effectué")
        else:
            flash("Échec du transfert")
        return redirect(url_for("index"))

    return app


if __name__ == "__main__":
    create_app().run(host="0.0.0.0", port=8000)
