from __future__ import annotations

from pathlib import Path

from flask import Flask, request, jsonify

from . import wifi_config


def create_app(config_path: Path | None = None) -> Flask:
    """Return a Flask app that writes Wi-Fi credentials to ``config_path``."""
    if config_path is None:
        config_path = wifi_config.CONFIG_PATH
    app = Flask(__name__)

    @app.post("/wifi")
    def set_wifi():
        data = request.get_json() or {}
        ssid = data.get("ssid")
        password = data.get("password")
        if not ssid or not password:
            return jsonify({"error": "ssid and password required"}), 400
        wifi_config.write_wifi_config(ssid, password, config_path)
        return jsonify({"status": "ok"})

    return app


if __name__ == "__main__":
    create_app().run(host="0.0.0.0", port=5000)
