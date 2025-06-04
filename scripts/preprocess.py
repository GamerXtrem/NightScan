"""Audio preprocessing utilities for NightScan.

This script converts MP3 recordings to WAV, isolates cries using silence
splitting and pads or truncates segments to 8 seconds. It also generates
mel-spectrograms in ``.npy`` format and creates CSV files describing the
train/val/test splits.
"""
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torchaudio
from torchaudio import transforms as T
from pydub import AudioSegment, silence


TARGET_DURATION_MS = 8000  # 8 seconds
SILENCE_THRESH = -40


def _convert_mp3(args: tuple[Path, Path, Path]) -> Path:
    mp3_path, input_dir, wav_dir = args
    audio = AudioSegment.from_mp3(mp3_path)
    relative = mp3_path.relative_to(input_dir).with_suffix(".wav")
    out_path = wav_dir / relative
    out_path.parent.mkdir(parents=True, exist_ok=True)
    audio.export(out_path, format="wav")
    return out_path


def convert_mp3_to_wav(input_dir: Path, wav_dir: Path, workers: int) -> list[Path]:
    """Recursively convert MP3 files in ``input_dir`` to WAV files in ``wav_dir``.

    The directory structure of ``input_dir`` is mirrored in ``wav_dir`` so that
    sub-folder names can be used as class labels later on.
    Returns a list of generated WAV paths.
    """
    input_dir = Path(input_dir)
    wav_dir.mkdir(parents=True, exist_ok=True)

    mp3_files = list(input_dir.rglob("*.mp3"))
    args_list = [(p, input_dir, wav_dir) for p in mp3_files]
    if workers == 1:
        return [_convert_mp3(a) for a in args_list]

    with ProcessPoolExecutor(max_workers=workers) as exe:
        return list(exe.map(_convert_mp3, args_list))


def is_silent(segment: AudioSegment, threshold_db: float = -60.0) -> bool:
    """Return ``True`` if the segment contains almost no sound."""
    return segment.rms == 0 or segment.dBFS < threshold_db


def isolate_cries(audio_path: Path, out_dir: Path) -> list[Path]:
    """Split a WAV file on silence and save 8s segments to ``out_dir``.

    Parameters
    ----------
    audio_path:
        Path to the WAV file to process.
    out_dir:
        Directory where processed segments should be written.
    Returns
    -------
    list[Path]
        List of paths to the generated WAV segments.
    """
    audio = AudioSegment.from_file(audio_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    chunks = silence.split_on_silence(
        audio,
        min_silence_len=500,
        silence_thresh=SILENCE_THRESH,
        keep_silence=250,
    )

    paths: list[Path] = []
    for idx, chunk in enumerate(chunks):
        if len(chunk) > TARGET_DURATION_MS:
            chunk = chunk[:TARGET_DURATION_MS]
        elif len(chunk) < TARGET_DURATION_MS:
            padding = AudioSegment.silent(TARGET_DURATION_MS - len(chunk))
            chunk += padding

        if is_silent(chunk):
            # Skip silent segments as per README recommendation
            continue

        out_path = out_dir / f"{audio_path.stem}_{idx}.wav"
        chunk.export(out_path, format="wav")
        paths.append(out_path)
    return paths


def _isolate_wrapper(args: tuple[Path, Path, Path]) -> list[Path]:
    wav_file, wav_dir, processed_dir = args
    out_dir = processed_dir / wav_file.relative_to(wav_dir).parent
    return isolate_cries(wav_file, out_dir)


def process_wav_files(wav_dir: Path, processed_dir: Path, workers: int) -> list[Path]:
    wav_files = list(wav_dir.rglob("*.wav"))
    processed_dir.mkdir(parents=True, exist_ok=True)
    args_list = [(wf, wav_dir, processed_dir) for wf in wav_files]
    if workers == 1:
        results = [_isolate_wrapper(a) for a in args_list]
    else:
        with ProcessPoolExecutor(max_workers=workers) as exe:
            results = list(exe.map(_isolate_wrapper, args_list))

    processed_paths: list[Path] = []
    for r in results:
        processed_paths.extend(r)
    return processed_paths


def _generate_spec(args: tuple[Path, Path, Path, int]) -> Path:
    wav_path, wav_dir, spec_dir, sr = args
    waveform, _ = torchaudio.load(wav_path)
    mel = T.MelSpectrogram(sample_rate=sr)(waveform)
    mel_db = T.AmplitudeToDB()(mel)
    relative = wav_path.relative_to(wav_dir).with_suffix(".npy")
    out_path = spec_dir / relative
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, mel_db.squeeze(0).numpy())
    return out_path


def generate_spectrograms(wav_dir: Path, spec_dir: Path, sr: int, workers: int) -> list[Path]:
    """Generate mel-spectrograms for all WAV files in ``wav_dir`` in parallel."""
    wav_dir = Path(wav_dir)
    spec_dir.mkdir(parents=True, exist_ok=True)

    wav_files = list(wav_dir.rglob("*.wav"))
    args_list = [(p, wav_dir, spec_dir, sr) for p in wav_files]
    if workers == 1:
        return [_generate_spec(a) for a in args_list]

    with ProcessPoolExecutor(max_workers=workers) as exe:
        return list(exe.map(_generate_spec, args_list))


def split_and_save(files: list[Path], out_dir: Path, train: float = 0.7, val: float = 0.15) -> None:
    """Split ``files`` into train/val/test sets and save CSV metadata including labels."""
    random.shuffle(files)
    n_total = len(files)
    n_train = int(n_total * train)
    n_val = int(n_total * val)

    label_names = sorted({p.parent.name for p in files})
    label_to_idx = {name: idx for idx, name in enumerate(label_names)}

    splits = {
        "train": files[:n_train],
        "val": files[n_train : n_train + n_val],
        "test": files[n_train + n_val :],
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    for split, split_files in splits.items():
        csv_path = out_dir / f"{split}.csv"
        with csv_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["path", "label"])
            for p in split_files:
                writer.writerow([p.as_posix(), label_to_idx[p.parent.name]])


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess audio files")
    parser.add_argument("--input_dir", type=Path, required=True, help="Directory with raw MP3 files")
    parser.add_argument("--output_dir", type=Path, required=True, help="Directory to store processed data")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes for parallel preprocessing",
    )
    args = parser.parse_args()

    wav_dir = args.output_dir / "wav"
    processed_dir = args.output_dir / "segments"
    spec_dir = args.output_dir / "spectrograms"
    csv_dir = args.output_dir / "csv"

    convert_mp3_to_wav(args.input_dir, wav_dir, args.workers)

    processed_paths = process_wav_files(wav_dir, processed_dir, args.workers)

    spec_paths = generate_spectrograms(processed_dir, spec_dir, sr=22050, workers=args.workers)
    split_and_save(spec_paths, csv_dir)


if __name__ == "__main__":
    main()
