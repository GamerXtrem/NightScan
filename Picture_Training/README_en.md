# NightScan - Image Classification

## Overview
NightScan is a project for classifying animals from photos. Images are listed in CSV files and used to train a ResNet18 model with PyTorch.

## Key features
- Structured loading of images organized by folder.
- Unified normalization and resizing of images.
- Train a CNN model (ResNet18) with the `--pretrained` option.
- Evaluate on a validation set and save the best model.

## Project layout
- `README.md` — this file
- `data/` — raw images (`raw/`) and CSVs (`csv/`)
- `models/` — directory for trained models
- `scripts/` — `prepare_csv.py`, `train.py` and `predict.py`
- `utils/` — utility functions
- `Picture_Training/setup.sh` — script to create required folders
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
chmod +x Picture_Training/setup.sh
./Picture_Training/setup.sh
```

## Usage
Prepare the CSV files from images and start training:

```bash
python scripts/prepare_csv.py --input_dir data/raw --output_dir data/csv
python scripts/train.py --csv_dir data/csv --model_dir models/ --pretrained
```

The `data/raw` folder must contain a subdirectory per species (e.g. `data/raw/bubo_bubo/`, `data/raw/capreolus_capreolus/`, ...). Image paths are stored in `train.csv`, `val.csv` and `test.csv` with a default 70/15/15 split. If you change these fractions, each value must be between 0 and 1 and `train + val` must be strictly less than 1 or the program will refuse to generate the CSVs.

### How the scripts work

- **`prepare_csv.py`**
  - Scans all folders in `--input_dir`.
  - Creates `train.csv`, `val.csv` and `test.csv` with columns `path` and `label`.
  - Configurable data split (default: 70/15/15). The values provided must be between 0 and 1 and `train + val` must remain below 1.

- **`train.py`**
  - Reads `train.csv` and `val.csv` from the folder specified by `--csv_dir`.
  - Loads images, resizes them to 224×224 and normalizes them according to ImageNet.
  - Trains a ResNet18 (optionally pre-trained).
  - Automatically saves the best model in `--model_dir/best_model.pth`.
  - Parameters: `--epochs`, `--batch_size`, `--lr`, `--num_workers`.

- **`predict.py`**
  - Loads a trained model and predicts the species of one or more images.
  - Applies the same transformations as during training.
  - Displays the top 3 predicted classes with their confidence score.
  - Requires the `--csv_dir` argument to load class names from `train.csv`.

```bash
python scripts/predict.py --model_path models/best_model.pth --csv_dir data/csv image.jpg
```
