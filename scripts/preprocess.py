#!/usr/bin/env python3
"""Preprocess NightScan audio dataset.

This script converts MP3 files to WAV, isolates animal cries,
normalizes clip length to 8 seconds, generates spectrograms,
creates train/test/val splits and outputs CSV metadata.
"""

import os
import argparse
import random
import csv
from pathlib import Path

from pydub import AudioSegment
from pydub.silence import split_on_silence
import numpy as np
import librosa


def convert_mp3_to_wav(mp3_path: Path, wav_path: Path) -> None:
    audio = AudioSegment.from_file(mp3_path)
    audio = audio.set_frame_rate(16000).set_channels(1)
    audio.export(wav_path, format="wav")


def isolate_cries(audio: AudioSegment) -> AudioSegment:
    """Remove long silences and keep loud segments."""
    chunks = split_on_silence(
        audio,
        min_silence_len=300,
        silence_thresh=audio.dBFS - 20,
        keep_silence=150,
    )
    if not chunks:
        processed = audio
    else:
        processed = AudioSegment.empty()
        for chunk in chunks:
            processed += chunk
    target_ms = 8000
    if len(processed) < target_ms:
        processed += AudioSegment.silent(duration=target_ms - len(processed))
    else:
        processed = processed[:target_ms]
    return processed


def process_audio_files(input_dir: Path, output_dir: Path) -> None:
    for root, _, files in os.walk(input_dir):
        rel = os.path.relpath(root, input_dir)
        out_dir = output_dir / rel
        out_dir.mkdir(parents=True, exist_ok=True)
        for fname in files:
            if fname.lower().endswith(".mp3"):
                mp3_path = Path(root) / fname
                wav_path = out_dir / (Path(fname).stem + ".wav")
                convert_mp3_to_wav(mp3_path, wav_path)
                audio = AudioSegment.from_wav(wav_path)
                processed = isolate_cries(audio)
                processed.export(wav_path, format="wav")


def generate_spectrograms(wav_dir: Path, spec_dir: Path):
    spec_paths = []
    labels = []
    for root, _, files in os.walk(wav_dir):
        rel = os.path.relpath(root, wav_dir)
        label = None if rel == "." else rel.split(os.sep)[0]
        out_dir = spec_dir / rel
        out_dir.mkdir(parents=True, exist_ok=True)
        for fname in files:
            if fname.lower().endswith(".wav"):
                wav_path = Path(root) / fname
                y, sr = librosa.load(wav_path, sr=None, mono=True)
                spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
                spec_db = librosa.power_to_db(spec, ref=np.max)
                spec_path = out_dir / (Path(fname).stem + ".npy")
                np.save(spec_path, spec_db)
                spec_paths.append(str(spec_path))
                labels.append(label)
    return spec_paths, labels


def split_and_save(paths, labels, output_dir: Path) -> None:
    data = list(zip(paths, labels))
    random.shuffle(data)
    n = len(data)
    train_end = int(n * 0.7)
    test_end = train_end + int(n * 0.2)
    splits = {
        "train": data[:train_end],
        "test": data[train_end:test_end],
        "val": data[test_end:],
    }
    for split, items in splits.items():
        csv_path = output_dir / f"{split}.csv"
        with csv_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["path", "label"])
            for p, lbl in items:
                writer.writerow([p, lbl])


def main():
    parser = argparse.ArgumentParser(description="Preprocess audio files")
    parser.add_argument(
        "--input_dir",
        default="Volumes/dataset/NightScan/raw_audio",
        help="Directory containing raw mp3 files",
    )
    parser.add_argument(
        "--output_dir",
        default="Volumes/dataset/NightScan/processed3",
        help="Directory to store processed wavs and spectrograms",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    spec_dir = output_dir / "spectrograms"

    process_audio_files(input_dir, output_dir)
    paths, labels = generate_spectrograms(output_dir, spec_dir)
    split_and_save(paths, labels, output_dir)


if __name__ == "__main__":
    main()
