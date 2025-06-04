import argparse
from pathlib import Path
import csv
from typing import List, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torchaudio import transforms as T
import torchaudio


class AudioDataset(Dataset):
    """Dataset loading audio files and converting them to spectrogram tensors."""

    def __init__(self, files: List[Path], sr: int = 22050) -> None:
        self.files = files
        self.sr = sr
        self.mel = T.MelSpectrogram(sample_rate=sr)
        self.to_db = T.AmplitudeToDB()
        self.transform = transforms.Compose([
            transforms.Lambda(lambda x: (x - x.min()) / (x.max() - x.min() + 1e-9)),
            transforms.Resize((224, 224)),
        ])

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        path = self.files[idx]
        waveform, sample_rate = torchaudio.load(path)
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sample_rate != self.sr:
            waveform = torchaudio.functional.resample(waveform, sample_rate, self.sr)
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
        return mel, path.as_posix()


def gather_files(inputs: List[Path]) -> List[Path]:
    files: List[Path] = []
    for p in inputs:
        if p.is_dir():
            files.extend(list(p.rglob("*.wav")))
            files.extend(list(p.rglob("*.mp3")))
        else:
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
    parser.add_argument("inputs", nargs="+", type=Path, help="Audio files or directories")
    parser.add_argument("--batch_size", type=int, default=1)
    args = parser.parse_args()

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
    with torch.no_grad():
        for batch, paths in loader:
            batch = batch.to(device)
            outputs = model(batch)
            probs = softmax(outputs)
            values, indices = torch.topk(probs, k=3, dim=1)
            for path, vals, idxs in zip(paths, values, indices):
                print(path)
                for rank, (val, idx) in enumerate(zip(vals, idxs), 1):
                    print(f"  {rank}. {labels[idx]} ({val.item():.2f})")


if __name__ == "__main__":
    main()
