# NightScan User Guide

NightScan classifies animal calls from audio files. This simplified manual explains how to use the project without developer knowledge.

## 1. Prepare your environment
Install Python (version 3.13 or newer)

- **Windows**: download Python from python.org and install it.
- **macOS and Linux**: Python is usually already installed. Check with `python --version` in a terminal.

Install **FFMPEG**, required for converting audio files.

- **Windows**: download "ffmpeg" from ffmpeg.org and follow the instructions.
- **macOS**: you can install it via Homebrew: `brew install ffmpeg`.
- **Linux**: often available in repositories (e.g. `sudo apt install ffmpeg` on Ubuntu).

Create a folder for the project and run:

```bash
git clone https://github.com/GamerXtrem/NightScan.git
# Replace "GamerXtrem" with your own username if using your fork
cd NightScan
```

Install the project dependencies:

```bash
python -m venv env          # create an isolated environment
source env/bin/activate     # on Windows: env\Scripts\activate
pip install -r requirements.txt
```

## 2. Prepare the audio data
Organize your recordings:

Create a `data/raw` directory with one subfolder per animal type,
e.g. `data/raw/cat/`, `data/raw/dog/`, etc. Place your WAV files in the appropriate subfolders.

Launch preprocessing:

```bash
python Audio_Training/scripts/preprocess.py --input_dir data/raw --output_dir data/processed --workers 4
```

This program:

- Copies the WAV files into the output folder.
- Splits sounds into 8‑second segments.
- Creates spectrograms (images representing the sounds).
- Generates three CSV files (`train.csv`, `val.csv`, `test.csv`) describing the data.

You end up with a new `data/processed` folder containing the split WAVs, spectrograms and CSVs.

## 3. Train the model
Run training:

```bash
python Audio_Training/scripts/train.py --csv_dir data/processed/csv --model_dir models --pretrained
```

The program reads the CSVs, loads the spectrograms and trains the model. Training may take several minutes (or more depending on your hardware). The best model is saved as `models/best_model.pth`.

## 4. Identify sounds
Prepare one or more audio files (e.g. `mysound.wav`) in an accessible folder.

Launch prediction:

```bash
python Audio_Training/scripts/predict.py --model_path models/best_model.pth --csv_dir data/processed/csv mysound.wav
```

The script splits the audio, runs the model and prints the three most probable species (e.g., cat, dog, etc.) with their scores. Use `--json` to get JSON output.

## 5. Run the prediction API
With a trained model you can start the small API server to process uploads from the web app or other clients. Inside the project folder run:

```bash
export MODEL_PATH="models/best_model.pth"
export CSV_DIR="data/processed/csv"
gunicorn -w 4 -b 0.0.0.0:8001 \
  Audio_Training.scripts.api_server:application
```

By default the service listens on `0.0.0.0:8001`. Adjust the address with the `--host` and `--port` options if needed. The Flask app expects the API at `http://localhost:8001/api/predict` unless you set the `PREDICT_API_URL` environment variable.

## 6. Tips and troubleshooting
- **No sound detected?** Ensure your recording contains cries or sounds loud enough. Very quiet segments are skipped.
- **FFMPEG error message?** Check that FFMPEG is installed and in your PATH (`ffmpeg -version` in a terminal).
- **Using multiple cores**: the `--workers 4` option speeds up preprocessing on multi‑core machines. Adjust as needed.
- **Model backups**: trained models are stored in the `models` folder. Keep them to avoid retraining each time.

In summary, NightScan converts your recordings into audio images, trains a model to recognize your animals, and lets you identify the contents of new sound files. Only basic command line knowledge is required to follow these steps.
