"""Convert WAV recordings to spectrograms."""
from __future__ import annotations

from pathlib import Path
import os

import numpy as np
import torchaudio
from torchaudio import transforms as T


TARGET_DURATION = 8


def wav_to_spec(wav_path: Path, out_path: Path) -> None:
    """Convert ``wav_path`` to a mel-spectrogram stored as ``out_path``."""
    waveform, sr = torchaudio.load(wav_path)
    if waveform.shape[1] < sr * TARGET_DURATION:
        pad = sr * TARGET_DURATION - waveform.shape[1]
        waveform = torchaudio.functional.pad(waveform, (0, pad))
    elif waveform.shape[1] > sr * TARGET_DURATION:
        waveform = waveform[:, : sr * TARGET_DURATION]

    spec = T.MelSpectrogram(sample_rate=sr)(waveform)
    np.save(out_path, spec.numpy())


def convert_directory(wav_dir: Path, out_dir: Path, remove: bool = False) -> None:
    """Convert all WAV files in ``wav_dir`` to ``out_dir``."""
    wav_dir = Path(wav_dir)
    for wav in wav_dir.rglob("*.wav"):
        rel = wav.relative_to(wav_dir)
        spec_path = out_dir / rel.with_suffix(".npy")
        spec_path.parent.mkdir(parents=True, exist_ok=True)
        wav_to_spec(wav, spec_path)
        if remove:
            wav.unlink()


def disk_usage_percent(path: Path) -> float:
    """Return disk usage percentage for ``path``."""
    st = os.statvfs(path)
    total = st.f_blocks * st.f_frsize
    used = (st.f_blocks - st.f_bfree) * st.f_frsize
    return used / total * 100


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("wav_dir", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--remove", action="store_true")
    args = parser.parse_args()

    convert_directory(args.wav_dir, args.out_dir, args.remove)
