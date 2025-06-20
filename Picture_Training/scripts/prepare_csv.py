"""Generate CSV metadata for image classification datasets."""
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
from typing import Sequence

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def gather_images(input_dir: Path) -> list[Path]:
    images: list[Path] = []
    for ext in IMAGE_EXTS:
        images.extend(p for p in input_dir.rglob(f"*{ext}") if not p.name.startswith("._"))
    return sorted(images)


def split_files(files: Sequence[Path], train: float, val: float, *, seed: int | None = None) -> dict[str, list[Path]]:
    """Split ``files`` into train/val/test subsets.

    ``train`` and ``val`` must be between 0 and 1 and ``train + val`` must be
    strictly less than 1. A :class:`ValueError` is raised otherwise.
    """
    if not (0 <= train <= 1) or not (0 <= val <= 1) or train + val >= 1:
        raise ValueError("Invalid split ratios: train and val must be between 0 and 1, with train + val < 1")
    if seed is not None:
        random.seed(seed)
    files = list(files)
    random.shuffle(files)
    n_total = len(files)
    n_train = int(n_total * train)
    n_val = int(n_total * val)
    return {
        "train": files[:n_train],
        "val": files[n_train:n_train + n_val],
        "test": files[n_train + n_val:],
    }


def save_csv(split: dict[str, list[Path]], out_dir: Path) -> None:
    label_names = sorted({p.parent.name for paths in split.values() for p in paths})
    label_to_idx = {name: idx for idx, name in enumerate(label_names)}
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, paths in split.items():
        csv_path = out_dir / f"{name}.csv"
        with csv_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["path", "label"])
            for p in paths:
                writer.writerow([p.as_posix(), label_to_idx[p.parent.name]])


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate train/val/test CSV files from images")
    parser.add_argument("--input_dir", type=Path, required=True, help="Directory with class folders")
    parser.add_argument("--output_dir", type=Path, required=True, help="Directory where CSV files will be created")
    parser.add_argument("--train_split", type=float, default=0.7)
    parser.add_argument("--val_split", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    images = gather_images(args.input_dir)
    split = split_files(images, args.train_split, args.val_split, seed=args.seed)
    save_csv(split, args.output_dir)


if __name__ == "__main__":
    main()
