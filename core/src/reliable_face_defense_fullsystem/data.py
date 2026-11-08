from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import json
import random

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class DatasetManifest:
    seed: int
    train: tuple[str, ...]
    calibration: tuple[str, ...]
    control: tuple[str, ...]
    test: tuple[str, ...]

    def to_dict(self):
        return asdict(self)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path):
        x = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            seed=int(x["seed"]),
            train=tuple(x["train"]),
            calibration=tuple(x["calibration"]),
            control=tuple(x["control"]),
            test=tuple(x["test"]),
        )


def list_images(dataset_dir: str | Path) -> list[Path]:
    root = Path(dataset_dir)
    items = []
    for pat in ("*.jpg", "*.jpeg", "*.png"):
        items.extend(root.glob(pat))
    return sorted(set(items))


def build_manifest(
    *,
    dataset_dir,
    train,
    calibration,
    control,
    test,
    seed,
) -> DatasetManifest:
    items = list_images(dataset_dir)
    total = int(train + calibration + control + test)
    if len(items) < total:
        raise ValueError(
            f"Need {total} images but dataset has only {len(items)}."
        )
    rng = random.Random(int(seed))
    chosen = items[:]
    rng.shuffle(chosen)
    chosen = chosen[:total]

    a = 0
    train_x = chosen[a:a+train]; a += train
    cal_x = chosen[a:a+calibration]; a += calibration
    ctrl_x = chosen[a:a+control]; a += control
    test_x = chosen[a:a+test]

    return DatasetManifest(
        seed=int(seed),
        train=tuple(str(x) for x in train_x),
        calibration=tuple(str(x) for x in cal_x),
        control=tuple(str(x) for x in ctrl_x),
        test=tuple(str(x) for x in test_x),
    )


def load_rgb01(path, size=256) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    if img.size != (size, size):
        img = img.resize((size, size), Image.Resampling.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2,0,1)


class ImagePathDataset(Dataset):
    def __init__(self, paths, size=256):
        self.paths = tuple(paths)
        self.size = int(size)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        return {
            "path": p,
            "image": load_rgb01(p, self.size),
        }
