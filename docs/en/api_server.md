# Prediction API

This module provides a small Flask server exposing the `POST /api/predict` endpoint. It accepts a WAV file and returns predictions in JSON format like those printed by `predict.py --json`.

## Launch the server

Activate the virtual environment then run:

```bash
python Audio_Training/scripts/api_server.py \
  --model_path models/best_model.pth \
  --csv_dir data/processed/csv
```

By default the API listens on `0.0.0.0:8001`. The `--host` and `--port` options let you change this address. Make sure not to reuse the Flask app's port to avoid conflicts.

## Allow the WordPress domain

If the API is called from a third-party site, the browser will reject the request without CORS headers. Install `flask_cors` then add the following in `Audio_Training/scripts/api_server.py`:

```python
from flask_cors import CORS
CORS(app, origins=["https://my-wordpress.example"])
```

Replace the URL with that of your WordPress site. The `Access-Control-Allow-Origin` header will contain this domain so file uploads from the plugin work correctly.
