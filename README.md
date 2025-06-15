# NightScan

NightScan is an experiment in training neural networks to recognize night-time animals.
The project is organized into two main folders:

- **Audio_Training/** – tools for preparing audio clips, generating spectrograms and training models to classify animal sounds.
- **Picture_Training/** – scripts for building image datasets and training image recognition models.

For a quick overview of how to set up and run the audio workflow, see **Manuel.txt** at the repository root.

## VPS setup

Run the setup script to install system packages and create the Python
environment:

```bash
bash setup_vps_infomaniak.sh
```

The script clones the repository if needed, installs `git`, `python3`,
`ffmpeg` and `portaudio`, then creates `env/` and installs the Python
requirements (including `pyaudio`).

If you are deploying on a new Infomaniak VPS, see
[`docs/vps_lite_first_connection.md`](docs/vps_lite_first_connection.md)
for instructions on making the initial SSH connection.

## Web interface

A minimal Flask application in `web/` forwards
uploaded audio clips to a prediction API. The endpoint URL is
read from the `PREDICT_API_URL` environment variable.

Launch it inside the virtual environment:

```bash
source env/bin/activate
python web/app.py
```

The application creates a local SQLite database on first run and stores
predictions for each authenticated user. See
[`docs/flask_app.md`](docs/flask_app.md) for details on the login routes,
database initialization and how to switch to another backend such as MySQL.

Set the `PREDICT_API_URL` environment variable to point to your
prediction service. You must also define `SECRET_KEY` to configure the
Flask session signing; use a random string for production.
The home page exposes a form for manual tests.
When a WAV file is submitted, the server posts it to this API and
displays the JSON response.

### Example Nginx configuration

```
server {
    listen 80;
    server_name example.com;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

This forwards HTTP requests to the Flask server running on port 8000.

## Quick prediction test

After running the setup script (`bash setup_vps_infomaniak.sh`) you can
immediately test the model locally. Activate the virtual environment
created by `setup_vps_infomaniak.sh`:

```bash
source env/bin/activate
```

Ensure you have a trained model (for example
`models/best_model.pth`) and the CSV directory generated during the
preprocessing step (typically `data/processed/csv`). You can then run
the prediction script on one or more WAV files:

```bash
python Audio_Training/scripts/predict.py \
  --model_path models/best_model.pth \
  --csv_dir data/processed/csv \
  path/to/your_audio.wav
```

The script prints the three most probable classes for each audio
segment. Add `--json` to get the result as JSON. No environment
variables are required for this command, but the dependencies installed
in `env/` (PyTorch, torchaudio, pydub, etc.) must be available.

## WordPress plugin

A small plugin located in `wp-plugin/prediction-charts` can display
user prediction statistics inside WordPress. See
[`docs/wordpress_plugin.md`](docs/wordpress_plugin.md) for the expected
database structure, how to export data from the Flask app and an example
of the `[nightscan_chart]` shortcode.
