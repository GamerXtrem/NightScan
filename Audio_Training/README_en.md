# NightScan

## Overview
NightScan is a project for classifying animal sounds using spectrograms generated from audio recordings. The pipeline converts raw audio files into 8‑second spectrograms, trains deep learning models and evaluates performance.

## Key features
- Preprocess audio files into uniform‑length spectrograms.
- Train classification models with PyTorch.
- Evaluate models on validation sets.
- Save the best trained model.

## Project layout
- `README.md` — this file
- `data/` — holds recordings (`raw/`) and spectrograms (`processed/`)
- `models/` — directory for storing trained models
- `scripts/` — `preprocess.py`, `train.py` and `predict.py` for preprocessing, training and prediction
- `utils/` — utility functions
- `setup.sh` — script that automatically creates the `data/`, `models/` and `utils/` folders
- `../requirements.txt` — common Python dependencies (root level)

## Installation
Clone the repository and install dependencies in a virtual environment:

```bash
git clone https://github.com/GamerXtrem/NightScan.git
# Replace "GamerXtrem" with your own username if using your fork
cd NightScan
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
pip install -r requirements.txt
cd Audio_Training
chmod +x setup.sh
./setup.sh
```

The `setup.sh` script, run from `Audio_Training`, creates the required `data/`, `models/` and `utils/` folders.

Before running `preprocess.py`, also install:

```bash
brew install portaudio   # macOS only
pip install pyaudio audioop-lts
```

Preprocessing with `pydub` also requires **ffmpeg** on the system. Check that the `ffmpeg` command is available (`ffmpeg -version` in a terminal). Spectrograms are now computed using **torchaudio**.

## Usage
Preprocess the data and train a model:

```bash
python scripts/preprocess.py --input_dir data/raw --output_dir data/processed --workers 4
python scripts/train.py --csv_dir data/processed/csv --model_dir models/ --pretrained
```

The `data/raw` folder must contain one subdirectory per class (e.g. `data/raw/cat/`, `data/raw/dog/`, ...). Preprocessing keeps this structure and generates spectrograms in `data/processed/spectrograms`. The CSV files in `data/processed/csv` now have two columns: `path` and `label`.

When `preprocess.py` isolates a cry but gets a silent segment (volume below `CHUNK_SILENCE_THRESH`), the resulting file is deleted and ignored.

## How the scripts work

### `preprocess.py`
- Copies WAV files to `output_dir/wav` while preserving folder structure.
- Splits each WAV into 8‑second segments using silence detection (`SPLIT_SILENCE_THRESH` −35 dBFS by default). Short segments are padded with silence and long ones are truncated.
- Segments whose average volume is below `CHUNK_SILENCE_THRESH` (−35 dBFS) are skipped.
- Valid segments are saved under `output_dir/segments` keeping class folders.
- A mel spectrogram is generated for each segment under `output_dir/spectrograms` with **torchaudio**.
  - Paths and labels of these spectrograms are stored in `train.csv`, `val.csv` and `test.csv` inside `output_dir/csv` with a default 70/15/15 split. If you change these ratios in the code, make sure each value is between 0 and 1 and that `train + val` remains strictly below 1, otherwise an error is raised.
- Optional: `--workers` allows parallel conversions and processing.

### `train.py`
- Reads `train.csv` and `val.csv` from the directory given by `--csv_dir`.
- Loads each `.npy` spectrogram, normalizes it to 0‑1, resizes to 224×224 and duplicates it on three channels to use with ResNet18.
- Trains a ResNet18 on GPU or MPS (Apple Silicon) if available. ImageNet weights can be loaded with `--pretrained`.
- After each epoch, displays the loss and validation accuracy and saves to `--model_dir/best_model.pth` the model achieving the best accuracy.
- Optional parameters: `--epochs` (default 10), `--batch_size` (32), `--lr` (1e-3) and `--num_workers` (0).

### `predict.py`
- Loads a trained model and returns the three most probable classes for each audio file. The file is first split into 8‑s segments using the same silence detection as in `preprocess.py`. Almost silent segments are skipped.
- Example usage:

```bash
python scripts/predict.py --model_path models/best_model.pth --csv_dir data/processed/csv sample.wav
```

- To get JSON output:

```bash
python scripts/predict.py --model_path models/best_model.pth --csv_dir data/processed/csv --json sample.wav
```
