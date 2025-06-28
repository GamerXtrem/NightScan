"""Convert WAV recordings to spectrograms."""
from __future__ import annotations

from pathlib import Path
import os
from datetime import datetime, time as dtime

import numpy as np
import torchaudio
from torchaudio import transforms as T


TARGET_DURATION = 8


def wav_to_spec(wav_path: Path, out_path: Path, sr: int = 22050) -> None:
    """Convert ``wav_path`` to a mel-spectrogram stored as ``out_path``."""
    waveform, original_sr = torchaudio.load(wav_path)
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if original_sr != sr:
        waveform = torchaudio.functional.resample(waveform, original_sr, sr)
    if waveform.shape[1] < sr * TARGET_DURATION:
        pad = sr * TARGET_DURATION - waveform.shape[1]
        waveform = torchaudio.functional.pad(waveform, (0, pad))
    elif waveform.shape[1] > sr * TARGET_DURATION:
        waveform = waveform[:, : sr * TARGET_DURATION]

    mel = T.MelSpectrogram(sample_rate=sr)(waveform)
    mel_db = T.AmplitudeToDB(top_db=80)(mel)
    np.save(out_path, mel_db.squeeze(0).numpy())


def convert_directory(
    wav_dir: Path, out_dir: Path, remove: bool = False, *, sr: int = 22050
) -> None:
    """Convert all WAV files in ``wav_dir`` to ``out_dir``."""
    wav_dir = Path(wav_dir)
    for wav in wav_dir.rglob("*.wav"):
        rel = wav.relative_to(wav_dir)
        spec_path = out_dir / rel.with_suffix(".npy")
        spec_path.parent.mkdir(parents=True, exist_ok=True)
        wav_to_spec(wav, spec_path, sr=sr)
        if remove:
            wav.unlink()


def disk_usage_percent(path: Path) -> float:
    """Return disk usage percentage for ``path``."""
    st = os.statvfs(path)
    total = st.f_blocks * st.f_frsize
    used = (st.f_blocks - st.f_bfree) * st.f_frsize
    return used / total * 100


def scheduled_conversion(
    wav_dir: Path,
    spec_dir: Path,
    *,
    threshold: float = 70.0,
    now: datetime | None = None,
    sr: int = 22050,
) -> None:
    """Convert WAV files after noon and delete them if disk usage is high."""
    if now is None:
        now = datetime.now()
    if now.time() < dtime(12, 0):
        return
    convert_directory(wav_dir, spec_dir, sr=sr)
    if disk_usage_percent(wav_dir) >= threshold:
        for wav in Path(wav_dir).rglob("*.wav"):
            wav.unlink()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("wav_dir", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--remove", action="store_true")
    parser.add_argument(
        "--scheduled",
        action="store_true",
        help="Only run after noon and delete WAV when disk > threshold",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=70.0,
        help="Disk usage percentage triggering WAV deletion",
    )
    args = parser.parse_args()

    if args.scheduled:
        scheduled_conversion(args.wav_dir, args.out_dir, threshold=args.threshold)
    else:
        convert_directory(args.wav_dir, args.out_dir, args.remove)
