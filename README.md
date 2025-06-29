# NightScan

NightScan is an experiment in training neural networks to recognize night-time animals.
The project is organized into two main folders:

- **Audio_Training/** – tools for preparing audio clips, generating spectrograms and training models to classify animal sounds.
- **Picture_Training/** – scripts for building image datasets and training image recognition models.

For a quick overview of how to set up and run the audio workflow, see **Manual_en.md** at the repository root.

## VPS setup

Run the setup script with root privileges so that `apt` can install the
required system packages. The script attempts to use `sudo` when it is not
executed as `root` and will exit if neither root nor `sudo` is available:

```bash
sudo bash setup_vps_infomaniak.sh
```

The script clones the repository if needed, installs `git`, `python3`,
`ffmpeg` and `portaudio`, then creates `env/` and installs the Python
requirements (including `pyaudio`).

If you are deploying on a new Infomaniak VPS, see
[`docs/en/vps_lite_first_connection.md`](docs/en/vps_lite_first_connection.md)
for instructions on making the initial SSH connection.

## Web interface

A minimal Flask application in `web/` forwards
uploaded audio clips to a prediction API. The endpoint URL is
read from the `PREDICT_API_URL` environment variable.

Launch it inside the virtual environment. Install `gunicorn` if it is not
already available (it is listed in `requirements.txt`) and start the server
with Gunicorn rather than the built in Flask runner:

```bash
source env/bin/activate
pip install gunicorn
export SECRET_KEY=$(python -c 'import secrets; print(secrets.token_hex(32))')
export WTF_CSRF_SECRET_KEY="$SECRET_KEY"  # optional
export PREDICT_API_URL="https://myserver.example/api/predict"
gunicorn -w 4 -b 0.0.0.0:8000 web.app:application
```

The command above binds the web server to `0.0.0.0`, exposing it on every network interface. This is typical when running behind a reverse proxy or with firewall rules in place. If you prefer to keep the service private, bind to `127.0.0.1` or block the port using a firewall such as `ufw`.

The application connects to a database using the URL in
`SQLALCHEMY_DATABASE_URI`. This variable is **mandatory**—`create_app()` raises
`RuntimeError` when it is missing. Provide secure credentials in the URI and
adjust it if you want to switch to another backend such as SQLite.
See [`docs/en/flask_app.md`](docs/en/flask_app.md) for details on the login
routes and database initialization.

Set the `PREDICT_API_URL` environment variable to point to your
prediction service. If `SECRET_KEY` is not defined the web app
generates a temporary value, but you should configure a stable random
string in production.
`Flask-WTF` provides CSRF protection. It will reuse `SECRET_KEY` unless you
set a dedicated `WTF_CSRF_SECRET_KEY` variable.
To start the prediction API, define the path to the trained model and the
directory containing the training CSV files, then launch the server with
Gunicorn:

```bash
export MODEL_PATH="models/best_model.pth"
export CSV_DIR="data/processed/csv"
gunicorn -w 4 -b 0.0.0.0:8001 \
  Audio_Training.scripts.api_server:application

```

Like the web app, this command listens on `0.0.0.0` so the API is reachable from any interface. Behind a proxy or with firewall rules this is fine. Otherwise consider binding to `127.0.0.1` or restricting the port with a firewall.

`web/app.py` expects this API to listen on `http://localhost:8001/api/predict`
unless you override `PREDICT_API_URL`.
For production deployments set `PREDICT_API_URL` to an `https://` endpoint so
uploads are encrypted in transit.
The home page exposes a form for manual tests.
When a WAV file is submitted, the server posts it to this API and
displays the JSON response.

Each upload may be up to 100 MB. The server also keeps track of the
storage used by every account and refuses new files once a user reaches
10 GB in total.

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

This forwards HTTP requests to the Gunicorn workers listening on port 8000.
Use a similar block for the API server on port 8001 so all traffic goes
through the reverse proxy.
When exposing the service to the internet, terminate HTTPS at the proxy so both
the web app and prediction API are accessed securely.

For convenience the repository provides `setup_nginx.sh`, a helper script that
installs Nginx and writes this configuration. If you prefer an end-to-end setup
including HTTPS certificates, use `setup_nginx_tls.sh` instead. Both helpers are
documented in `docs/en/nginx_setup.md`.

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

## Updating dependencies

All packages in `requirements.txt` are pinned to the versions verified by the
tests. When a dependency needs an upgrade, activate the virtual environment and
install the new versions with `pip --upgrade`. After verifying that `pytest`
passes, regenerate the file using `pip freeze`:

```bash
source env/bin/activate
pip install -U -r requirements.txt
pytest
pip freeze > requirements.txt
```


## Running tests

The automated tests rely on every package listed in `requirements.txt`.
Install them inside the virtual environment and invoke `pytest`:

```bash
source env/bin/activate
pip install -r requirements.txt
pytest
```

If you only need to run the suite on a machine without the heavy
PyTorch stack, use `requirements-ci.txt` instead. The test helper
automatically provides stub versions of the missing libraries so the
tests still execute:

```bash
pip install -r requirements-ci.txt
pytest
```

Commit the updated `requirements.txt` once the tests succeed.

## WordPress plugin

A small plugin located in `wp-plugin/prediction-charts` can display
user prediction statistics inside WordPress. See
[`docs/en/wordpress_plugin.md`](docs/en/wordpress_plugin.md) for the expected
database structure, how to export data from the Flask app and an example
of the `[nightscan_chart]` shortcode. The repository ships with
`export_predictions.py` to copy predictions from the Flask database to the
WordPress table.

## Uploading from WordPress

The repository also includes **NightScan Audio Upload**, a plugin that
lets WordPress send WAV files directly to your prediction API. Copy the
`wp-plugin/audio-upload` folder into `wp-content/plugins/` and activate
it from the admin panel. Set the API endpoint with the `ns_api_endpoint`
option so the plugin knows where to post the files, e.g. using WP‑CLI:

```bash
wp option update ns_api_endpoint https://your-vps.example/api/predict
```

Your WordPress site can run on a different host from the prediction
server. Set the `API_CORS_ORIGINS` environment variable so the API
allows requests from your WordPress domain (see
`docs/en/api_server.md`). HTTPS is also recommended so uploads succeed.

WordPress may enforce stricter file size limits via PHP. In `php.ini`,
set `upload_max_filesize` and `post_max_size` to at least `100M` so the
plugin can accept files up to 100 MB.

## App

For instructions on building a mobile client in React Native, see
[`docs/en/mobile_app.md`](docs/en/mobile_app.md). A minimal starter
project is included under [`ios-app/`](ios-app/).

### Application objectives and user experience

1. **Easy access to observations**
   - Display on an interactive map all detections (sounds or photos) reported by field sensors.
   - Provide a chronological list showing the latest observations with species, time and location.
2. **Quick review and notifications**
   - Receive real-time notifications when a new detection is processed.
   - Filter detections by species or geographic area to obtain relevant results immediately.
3. **Sharing and export**
   - Allow users to share detections by e‑mail or export them (CSV/KMZ) for later analysis.

**Target users**: wildlife photographers, amateur naturalists, researchers or anyone interested in tracking nocturnal fauna.
**Interface**: clean with a bottom navigation bar for the map, detection list, filters and settings.
**Usage**: familiar gestures (pinch to zoom, scrolling, pull to refresh) so the app remains intuitive on iOS and Android.
**Simplicity first**: users quickly see when and where animals were observed without viewing the raw media.

French-speaking contributors can refer to
[`docs/fr/application_objectifs.md`](docs/fr/application_objectifs.md)
for the original summary in French.


## Configuration and logging

The repository provides `config_example.ini` with typical settings such as the
log file path and active hours. Scripts under `NightScanPi/Program` write
informational and error messages to `nightscan.log` by default. Set the
`NIGHTSCAN_LOG` environment variable to override the log file location.
