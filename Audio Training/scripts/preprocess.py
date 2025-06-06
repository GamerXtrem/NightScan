"""Audio preprocessing utilities for NightScan.

This script expects WAV recordings organised by class folders. It isolates
cries using silence splitting, pads or truncates segments to 8 seconds and
generates mel-spectrograms in ``.npy`` format. Finally, CSV files describing
the train/val/test splits are created.
"""
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import logging
import shutil

import numpy as np
import torchaudio
from torchaudio import transforms as T
from pydub import AudioSegment, silence
from pydub.exceptions import CouldntDecodeError
from pydub.utils import which


TARGET_DURATION_MS = 8000  # 8 seconds

logger = logging.getLogger(__name__)


def ensure_ffmpeg() -> None:
    """Exit with a helpful message if ffmpeg is missing."""
    if which("ffmpeg") is None:
        raise SystemExit(
            "ffmpeg not found. Install it and ensure it is available in your PATH."
        )


def copy_wav_files(input_dir: Path, wav_dir: Path) -> list[Path]:
    """Recursively copy WAV files from ``input_dir`` to ``wav_dir``.

    The directory structure of ``input_dir`` is mirrored in ``wav_dir`` so that
    sub-folder names can be used as class labels later on. Returns a list of
    copied paths.
    """
    input_dir = Path(input_dir)
    wav_dir.mkdir(parents=True, exist_ok=True)

    wav_files = [p for p in input_dir.rglob("*.wav") if not p.name.startswith("._")]

    copied: list[Path] = []
    for p in wav_files:
        relative = p.relative_to(input_dir)
        out_path = wav_dir / relative
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path != p:
            shutil.copy2(p, out_path)
        copied.append(out_path)

    return copied


def is_silent(segment: AudioSegment, threshold_db: float) -> bool:
    """Return ``True`` if the segment contains almost no sound."""
    return segment.dBFS < threshold_db or segment.rms == 0


def isolate_cries(
    audio_path: Path,
    out_dir: Path,
    split_thresh: float,
    chunk_thresh: float,
) -> list[Path]:
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
    try:
        audio = AudioSegment.from_file(audio_path)
        # Normalize audio so that the maximum peak is at 0 dBFS
        if audio.max_dBFS != float("-inf"):
            audio = audio.apply_gain(-audio.max_dBFS)
    except CouldntDecodeError as exc:
        logger.warning("Could not decode %s: %s", audio_path, exc)
        return []
    out_dir.mkdir(parents=True, exist_ok=True)
    chunks = silence.split_on_silence(
        audio,
        min_silence_len=300,
        silence_thresh=split_thresh,
        keep_silence=150,
    )

    paths: list[Path] = []
    for idx, chunk in enumerate(chunks):
        if len(chunk) == 0:
            logger.warning("Chunk vide ignor\xE9 : %s [%d]", audio_path, idx)
            continue
        if len(chunk) > TARGET_DURATION_MS:
            chunk = chunk[:TARGET_DURATION_MS]
        elif len(chunk) < TARGET_DURATION_MS:
            padding = AudioSegment.silent(TARGET_DURATION_MS - len(chunk))
            chunk += padding

        if is_silent(chunk, chunk_thresh):
            # Skip silent segments as per README recommendation
            continue

        out_path = out_dir / f"{audio_path.stem}_{idx}.wav"
        chunk.export(out_path, format="wav")
        paths.append(out_path)
    return paths


def _isolate_wrapper(args: tuple[Path, Path, Path, float, float]) -> list[Path]:
    wav_file, wav_dir, processed_dir, split_thresh, chunk_thresh = args
    out_dir = processed_dir / wav_file.relative_to(wav_dir).parent
    return isolate_cries(wav_file, out_dir, split_thresh, chunk_thresh)


def process_wav_files(
    wav_dir: Path,
    processed_dir: Path,
    workers: int,
    split_thresh: float,
    chunk_thresh: float,
) -> list[Path]:
    wav_files = [p for p in wav_dir.rglob("*.wav") if not p.name.startswith("._")]
    processed_dir.mkdir(parents=True, exist_ok=True)
    args_list = [
        (wf, wav_dir, processed_dir, split_thresh, chunk_thresh) for wf in wav_files
    ]
    if workers == 1:
        results = [_isolate_wrapper(a) for a in args_list]
    else:
        with ProcessPoolExecutor(max_workers=workers) as exe:
            results = list(exe.map(_isolate_wrapper, args_list))

    processed_paths: list[Path] = []
    for r in results:
        processed_paths.extend(r)
    return processed_paths


def _generate_spec(args: tuple[Path, Path, Path, int]) -> Path | None:
    wav_path, wav_dir, spec_dir, sr = args
    try:
        waveform, original_sr = torchaudio.load(wav_path)
    except Exception as exc:  # catch decoding errors
        logger.warning("Could not load %s: %s", wav_path, exc)
        return None
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if original_sr != sr:
        waveform = torchaudio.functional.resample(waveform, original_sr, sr)
    mel = T.MelSpectrogram(sample_rate=sr)(waveform)
    mel_db = T.AmplitudeToDB(top_db=80)(mel)
    relative = wav_path.relative_to(wav_dir).with_suffix(".npy")
    out_path = spec_dir / relative
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, mel_db.squeeze(0).numpy())
    return out_path


def generate_spectrograms(wav_dir: Path, spec_dir: Path, sr: int, workers: int) -> list[Path]:
    """Generate mel-spectrograms for all WAV files in ``wav_dir`` in parallel."""
    wav_dir = Path(wav_dir)
    spec_dir.mkdir(parents=True, exist_ok=True)

    wav_files = [p for p in wav_dir.rglob("*.wav") if not p.name.startswith("._")]
    args_list = [(p, wav_dir, spec_dir, sr) for p in wav_files]
    if workers == 1:
        results = [_generate_spec(a) for a in args_list]
    else:
        with ProcessPoolExecutor(max_workers=workers) as exe:
            results = list(exe.map(_generate_spec, args_list))

    return [r for r in results if r is not None]


def split_and_save(
    files: list[Path],
    out_dir: Path,
    train: float = 0.7,
    val: float = 0.15,
    *,
    seed: int | None = None,
) -> None:
    """Split ``files`` into train/val/test sets and save CSV metadata including labels.

    ``train`` and ``val`` must be in the ``[0, 1]`` range and ``train + val`` must
    be strictly less than 1, leaving some data for the test split.  A
    :class:`ValueError` is raised if the ratios are invalid.

    Parameters
    ----------
    files:
        List of spectrogram paths to split.
    out_dir:
        Directory where ``train.csv``, ``val.csv`` and ``test.csv`` will be written.
    seed:
        Optional random seed ensuring deterministic shuffling.
    """
    if not (0 <= train <= 1) or not (0 <= val <= 1) or train + val >= 1:
        raise ValueError("Invalid split ratios: train and val must be between 0 and 1, with train + val < 1")
    if seed is not None:
        random.seed(seed)
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
    parser.add_argument("--input_dir", type=Path, required=True, help="Directory with WAV files organized by class")
    parser.add_argument("--output_dir", type=Path, required=True, help="Directory to store processed data")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes for parallel preprocessing",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for dataset splitting",
    )
    parser.add_argument(
        "--split_thresh",
        type=float,
        default=-35,
        help="Silence threshold (dBFS) for splitting",
    )
    parser.add_argument(
        "--chunk_thresh",
        type=float,
        default=-35,
        help="Threshold for is_silent()",
    )
    args = parser.parse_args()

    ensure_ffmpeg()

    logging.basicConfig(
        format="%(levelname)s:%(processName)s:%(message)s", level=logging.INFO
    )

    wav_dir = args.output_dir / "wav"
    processed_dir = args.output_dir / "segments"
    spec_dir = args.output_dir / "spectrograms"
    csv_dir = args.output_dir / "csv"

    copy_wav_files(args.input_dir, wav_dir)

    processed_paths = process_wav_files(
        wav_dir,
        processed_dir,
        args.workers,
        args.split_thresh,
        args.chunk_thresh,
    )

    spec_paths = generate_spectrograms(processed_dir, spec_dir, sr=22050, workers=args.workers)
    split_and_save(spec_paths, csv_dir, seed=args.seed)


if __name__ == "__main__":
    main()
