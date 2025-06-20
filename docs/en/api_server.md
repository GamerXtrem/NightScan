# Prediction API

This module provides a small Flask server exposing the `POST /api/predict` endpoint. It accepts a WAV file and returns predictions in JSON format like those printed by `predict.py --json`.

## Launch the server

Activate the virtual environment. Install `gunicorn` if needed and then run:

```bash
pip install gunicorn
export MODEL_PATH="models/best_model.pth"
export CSV_DIR="data/processed/csv"
gunicorn -w 4 -b 0.0.0.0:8001 \
  Audio_Training.scripts.api_server:application
```

By default the API listens on `0.0.0.0:8001`. The `--host` and `--port` options let you change this address. Make sure not to reuse the Flask app's port to avoid conflicts.

Listening on `0.0.0.0` means the API accepts connections from any interface. When running behind a reverse proxy or a firewall this is generally expected. If you do not want the service to be reachable from outside, bind it to `127.0.0.1` or restrict access with firewall rules (for example using `ufw`).


## Allow the WordPress domain

If the API is called from a third-party site, the browser will reject the
request without CORS headers. Set the `API_CORS_ORIGINS` environment
variable to a comma separated list of allowed origins before starting the
server and make sure `flask_cors` is installed:

```bash
pip install flask_cors
export API_CORS_ORIGINS="https://my-wordpress.example"
```

The server automatically enables CORS for these domains and the
`Access-Control-Allow-Origin` header will contain your WordPress URL so file
uploads from the plugin work correctly.

## File size limit

`api_server.py` rejects files larger than 100 MB. The constant `MAX_FILE_SIZE`
defines this limit and the server checks `request.content_length` before saving
the upload. Clients should keep individual WAV files under this size.

Malformed or truncated WAV files are also rejected. The server attempts to open
the uploaded data before running the model and returns a 400 error when the file
cannot be decoded.

Just like the Flask web app, you should place a reverse proxy (for example
Nginx) in front of Gunicorn and forward requests to port `8001`.
