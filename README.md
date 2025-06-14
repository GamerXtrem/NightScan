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

Set the `PREDICT_API_URL` environment variable to point to your
prediction service. The home page exposes a form for manual tests.
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
