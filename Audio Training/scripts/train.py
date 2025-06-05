import argparse
from pathlib import Path
import csv
from typing import Tuple, List

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
import torch.nn as nn


class SpectrogramDataset(Dataset):
    """Dataset loading spectrogram paths and labels from a CSV file.

    The CSV must have two columns: ``path`` and ``label``.
    ``path`` points to a ``.npy`` spectrogram file and ``label`` is an
    integer class index.
    """

    def __init__(self, csv_file: Path) -> None:
        self.samples: List[Tuple[Path, int]] = []
        with open(csv_file, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.samples.append((Path(row["path"]), int(row["label"])))

        # Normalize spectrograms to [0, 1]
        self.transform = transforms.Compose([
            transforms.Lambda(lambda x: (x - x.min()) / (x.max() - x.min() + 1e-9)),
            transforms.Resize((224, 224)),
        ])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        mel = np.load(path)
        mel = torch.tensor(mel, dtype=torch.float32)
        # convert to 3 channels for ResNet
        mel = mel.unsqueeze(0).repeat(3, 1, 1)
        mel = self.transform(mel)
        return mel, label


def train_epoch(model: nn.Module, loader: DataLoader, criterion, optimizer, device: torch.device) -> float:
    model.train()
    running_loss = 0.0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    return running_loss / len(loader.dataset)


def evaluate(model: nn.Module, loader: DataLoader, criterion, device: torch.device) -> float:
    model.eval()
    loss = 0.0
    correct = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss += criterion(outputs, targets).item() * inputs.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == targets).sum().item()
    return loss / len(loader.dataset), correct / len(loader.dataset)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ResNet18 on spectrograms")
    parser.add_argument("--csv_dir", type=Path, required=True, help="Directory containing train.csv and val.csv")
    parser.add_argument("--model_dir", type=Path, required=True, help="Where to save the trained model")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of worker processes for data loading",
    )
    parser.add_argument(
        "--pretrained",
        action="store_true",
        help="Use ImageNet pretrained weights for ResNet18",
    )
    args = parser.parse_args()

    train_ds = SpectrogramDataset(args.csv_dir / "train.csv")
    val_ds = SpectrogramDataset(args.csv_dir / "val.csv")
    num_classes = len(set(label for _, label in train_ds.samples))

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    pin_memory = device.type != "cpu"

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    weights = models.ResNet18_Weights.DEFAULT if args.pretrained else None
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    args.model_dir.mkdir(parents=True, exist_ok=True)
    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        print(f"Epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}")
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), args.model_dir / "best_model.pth")


if __name__ == "__main__":
    main()
