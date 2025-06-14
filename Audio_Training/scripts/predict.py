import argparse
from pathlib import Path
import csv
from typing import List, Tuple
from io import BytesIO
import json
import sys

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torchaudio import transforms as T
import torchaudio
from pydub import AudioSegment, silence
from pydub.exceptions import CouldntDecodeError
import logging

TARGET_DURATION_MS = 8000  # 8 seconds
SPLIT_SILENCE_THRESH = -35
CHUNK_SILENCE_THRESH = -35

logger = logging.getLogger(__name__)


def iter_wav_files(root: Path):
    """Yield ``.wav`` files under ``root`` skipping AppleDouble artifacts."""
    for p in root.rglob("*.wav"):
        if p.name.startswith("._"):
            continue
        yield p


def is_silent(segment: AudioSegment, threshold_db: float = CHUNK_SILENCE_THRESH) -> bool:
    """Return ``True`` if the segment contains almost no sound."""
    return segment.rms == 0 or segment.dBFS < threshold_db


def extract_segments(path: Path, sr: int) -> List[torch.Tensor]:
    """Split ``path`` on silence and return padded waveforms."""
    try:
        audio = AudioSegment.from_file(path)
    except CouldntDecodeError as e:
        print(f"\u26A0\ufe0f  {path} ignoré : {e}")
        return []
    if audio.max_dBFS != float("-inf"):
        audio = audio.apply_gain(-audio.max_dBFS)
    chunks = silence.split_on_silence(
        audio,
        min_silence_len=300,
        silence_thresh=SPLIT_SILENCE_THRESH,
        keep_silence=150,
    )
    segments: List[torch.Tensor] = []
    for chunk in chunks:
        if len(chunk) > TARGET_DURATION_MS:
            chunk = chunk[:TARGET_DURATION_MS]
        elif len(chunk) < TARGET_DURATION_MS:
            pad_len = TARGET_DURATION_MS - len(chunk)
            chunk = (
                AudioSegment.silent(pad_len // 2)
                + chunk
                + AudioSegment.silent(pad_len - pad_len // 2)
            )

        if is_silent(chunk):
            continue

        buf = BytesIO()
        chunk.export(buf, format="wav")
        buf.seek(0)
        waveform, orig_sr = torchaudio.load(buf)
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if orig_sr != sr:
            waveform = torchaudio.functional.resample(waveform, orig_sr, sr)
        segments.append(waveform)
    return segments


class AudioDataset(Dataset):
    """Dataset loading audio segments and converting them to spectrogram tensors."""

    def __init__(self, files: List[Path], sr: int = 22050) -> None:
        self.samples: List[Tuple[torch.Tensor, str]] = []
        self.sr = sr
        for path in files:
            segments = extract_segments(path, sr)
            for idx, waveform in enumerate(segments):
                name = f"{path.as_posix()}#{idx}" if len(segments) > 1 else path.as_posix()
                self.samples.append((waveform, name))
        self.mel = T.MelSpectrogram(sample_rate=sr)
        self.to_db = T.AmplitudeToDB(top_db=80)
        self.transform = transforms.Compose([
            transforms.Lambda(lambda x: (x - x.min()) / (x.max() - x.min() + 1e-9)),
            transforms.Resize((224, 224)),
        ])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        waveform, name = self.samples[idx]
        num_samples = self.sr * 8
        if waveform.size(1) < num_samples:
            pad = num_samples - waveform.size(1)
            waveform = torch.nn.functional.pad(waveform, (0, pad))
        elif waveform.size(1) > num_samples:
            waveform = waveform[:, :num_samples]
        mel = self.mel(waveform)
        mel = self.to_db(mel)
        mel = mel.squeeze(0)
        mel = mel.unsqueeze(0).repeat(3, 1, 1)
        mel = self.transform(mel)
        return mel, name


def gather_files(inputs: List[Path]) -> List[Path]:
    files: List[Path] = []
    for p in inputs:
        if p.is_dir():
            files.extend(iter_wav_files(p))
        else:
            if p.name.startswith("._"):
                continue
            files.append(p)
    return sorted(files)


def load_labels(csv_dir: Path) -> List[str]:
    csv_path = csv_dir / "train.csv"
    mapping = {}
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            idx = int(row["label"])
            name = Path(row["path"]).parent.name
            mapping[idx] = name
    return [mapping[i] for i in sorted(mapping.keys())]


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict top3 classes for audio files")
    parser.add_argument("--model_path", type=Path, required=True, help="Path to trained model")
    parser.add_argument("--csv_dir", type=Path, required=True, help="Directory containing train.csv")
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="WAV files or directories containing WAV files",
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output predictions as a JSON object instead of plain text",
    )
    args = parser.parse_args()

    logging.basicConfig(
        format="%(levelname)s:%(processName)s:%(message)s", level=logging.INFO
    )

    files = gather_files(args.inputs)
    if not files:
        raise SystemExit("No audio files found")

    labels = load_labels(args.csv_dir)
    num_classes = len(labels)

    dataset = AudioDataset(files)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18()
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    state = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    softmax = torch.nn.Softmax(dim=1)
    results = []
    with torch.no_grad():
        for batch, paths in loader:
            batch = batch.to(device)
            outputs = model(batch)
            probs = softmax(outputs)
            values, indices = torch.topk(probs, k=3, dim=1)
            for path, vals, idxs in zip(paths, values, indices):
                if args.json:
                    seg_idx = 0
                    if "#" in path:
                        try:
                            seg_idx = int(path.rsplit("#", 1)[1])
                        except ValueError:
                            seg_idx = 0
                    result = {
                        "segment": path,
                        "time": seg_idx * (TARGET_DURATION_MS / 1000),
                        "predictions": [
                            {
                                "label": labels[idx.item()] if hasattr(idx, "item") else labels[int(idx)],
                                "probability": float(val),
                            }
                            for val, idx in zip(vals, idxs)
                        ],
                    }
                    results.append(result)
                else:
                    print(path)
                    for rank, (val, idx) in enumerate(zip(vals, idxs), 1):
                        print(f"  {rank}. {labels[idx]} ({val.item():.2f})")

    if args.json:
        json.dump(results, sys.stdout, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
