"""Predict image classes using a trained ResNet18 model.

Pass image files or directories via ``inputs``. Class names are read from
``--csv_dir`` and the model checkpoint from ``--model_path``. The evaluation
batch size can be set with ``--batch_size``.
"""

import argparse
from pathlib import Path
from typing import List
import csv

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image


class ImageDataset(Dataset):
    """Dataset loading images and applying transforms."""

    def __init__(self, files: List[Path]) -> None:
        self.files = files
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        path = self.files[idx]
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        return img, path.as_posix()


def gather_files(inputs: List[Path]) -> List[Path]:
    files: List[Path] = []
    for p in inputs:
        if p.is_dir():
            for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
                files.extend(p.rglob(ext))
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
    parser = argparse.ArgumentParser(description="Predict classes for images")
    parser.add_argument(
        "--model_path", type=Path, required=True, help="Path to trained model"
    )
    parser.add_argument(
        "--csv_dir", type=Path, required=True, help="Directory containing train.csv"
    )
    parser.add_argument(
        "inputs", nargs="+", type=Path, help="Image files or directories"
    )
    parser.add_argument("--batch_size", type=int, default=1)
    args = parser.parse_args()

    files = gather_files(args.inputs)
    if not files:
        raise SystemExit("No image files found")

    labels = load_labels(args.csv_dir)
    num_classes = len(labels)

    dataset = ImageDataset(files)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    model = models.resnet18()
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    state = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    softmax = torch.nn.Softmax(dim=1)
    top_k = min(3, num_classes)
    with torch.no_grad():
        for batch, paths in loader:
            batch = batch.to(device)
            outputs = model(batch)
            probs = softmax(outputs)
            values, indices = torch.topk(probs, k=top_k, dim=1)
            for path, vals, idxs in zip(paths, values, indices):
                print(path)
                for rank, (val, idx) in enumerate(zip(vals, idxs), 1):
                    print(f"  {rank}. {labels[idx]} ({val.item():.2f})")


if __name__ == "__main__":
    main()
