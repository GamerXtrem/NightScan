import csv
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from audio_training.scripts.preprocess import split_and_save


def test_split_and_save(tmp_path):
    files = []
    spec_dir = tmp_path / "specs"
    for label in ["cat", "dog"]:
        d = spec_dir / label
        d.mkdir(parents=True)
        for i in range(5):
            p = d / f"file{i}.npy"
            p.write_bytes(b"data")
            files.append(p)

    out_dir = tmp_path / "csv"
    split_and_save(files, out_dir, train=0.5, val=0.3, seed=0)

    def read(name):
        with (out_dir / f"{name}.csv").open() as f:
            return list(csv.DictReader(f))

    train_rows = read("train")
    val_rows = read("val")
    test_rows = read("test")

    assert len(train_rows) == 5
    assert len(val_rows) == 3
    assert len(test_rows) == 2

    all_rows = train_rows + val_rows + test_rows
    assert sorted(r["path"] for r in all_rows) == sorted(str(p) for p in files)
    label_map = {"cat": "0", "dog": "1"}
    for row in all_rows:
        expected = label_map[Path(row["path"]).parent.name]
        assert row["label"] == expected
