from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
import csv
import hashlib
import json
import random
import sys

import numpy as np
from PIL import Image
import torch


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def append_jsonl(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(value, ensure_ascii=False) + "\n")


def read_jsonl(path):
    path = Path(path)
    if not path.is_file():
        return []
    out = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def write_csv(path, rows, fieldnames=None):
    rows = list(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        seen = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def stable_seed(*parts):
    raw = ":".join(map(str, parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "little") % (2**31 - 1)


def load_rgb01(path, size=256):
    image = Image.open(path).convert("RGB")
    if image.size != (size, size):
        image = image.resize((size, size), Image.Resampling.BILINEAR)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


def save_rgb01(tensor, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    x = tensor.detach().cpu()
    if x.ndim == 4:
        x = x[0]
    arr = np.rint(x.clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    Image.fromarray(arr, mode="RGB").save(path)


def sha256_file(path, chunk=8 * 1024 * 1024):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


@contextmanager
def isolated_import(root, prefixes):
    root = str(Path(root).resolve())
    old_path = list(sys.path)
    displaced = {}
    prefixes = tuple(prefixes)
    try:
        for name in list(sys.modules):
            if any(name == p or name.startswith(p + ".") for p in prefixes):
                displaced[name] = sys.modules.pop(name)
        sys.path.insert(0, root)
        yield
    finally:
        for name in list(sys.modules):
            if any(name == p or name.startswith(p + ".") for p in prefixes):
                sys.modules.pop(name, None)
        sys.modules.update(displaced)
        sys.path[:] = old_path


@dataclass(frozen=True)
class Pair:
    pair_id: str
    source_identity: int
    source_path: str
    target_identity: int
    target_path: str
    target_index: int


def _read_mapping(path):
    out = {}
    for raw in Path(path).read_text(encoding="utf-8-sig").splitlines():
        p = raw.split()
        if not p:
            continue
        try:
            idx = int(p[0])
        except ValueError:
            continue
        out[idx] = p[-1]
    return out


def _read_identity(path):
    out = {}
    for raw in Path(path).read_text(encoding="utf-8-sig").splitlines():
        p = raw.split()
        if len(p) == 2:
            out[p[0]] = int(p[1])
    return out


def identity_groups(paths, mapping_path, identity_path):
    mapping = _read_mapping(mapping_path)
    identity = _read_identity(identity_path)
    groups = {}
    for raw_path in paths:
        path = Path(raw_path)
        try:
            hq_idx = int(path.stem)
        except ValueError as exc:
            raise ValueError(f"CelebA-HQ filename is not an integer index: {path}") from exc
        original = mapping[hq_idx]
        person = identity[original]
        groups.setdefault(person, []).append(str(path))
    return {k: tuple(sorted(v)) for k, v in groups.items()}


def build_protocol(cfg):
    split = read_json(cfg["official_split_manifest"])
    counts = {k: len(split[k]) for k in ("train", "valid", "test")}
    expected = {"train": 24183, "valid": 2993, "test": 2824}
    if counts != expected:
        raise RuntimeError(f"Official split mismatch: {counts} != {expected}")

    test_groups = identity_groups(
        split["test"], cfg["hq_mapping"], cfg["celeba_identity"]
    )
    valid_groups = identity_groups(
        split["valid"], cfg["hq_mapping"], cfg["celeba_identity"]
    )
    rng = random.Random(int(cfg["seed"]))

    all_test_ids = sorted(test_groups)
    rng.shuffle(all_test_ids)
    source_ids = [x for x in all_test_ids if len(test_groups[x]) >= 2]
    source_ids = source_ids[: int(cfg["source_identities"])]
    if len(source_ids) != int(cfg["source_identities"]):
        raise RuntimeError("Not enough test identities with two images.")

    gallery = {}
    source_paths = {}
    for person in all_test_ids:
        paths = list(test_groups[person])
        gallery[person] = paths[0]
        if person in source_ids:
            source_paths[person] = paths[1]

    pairs = []
    for source_rank, source_id in enumerate(source_ids):
        candidates = [x for x in all_test_ids if x != source_id]
        start = stable_seed(cfg["seed"], "targets", source_id) % len(candidates)
        for target_index in range(int(cfg["targets_per_source"])):
            target_id = candidates[(start + target_index) % len(candidates)]
            pairs.append(Pair(
                pair_id=f"S{source_id}_T{target_index:02d}_{target_id}",
                source_identity=source_id,
                source_path=source_paths[source_id],
                target_identity=target_id,
                target_path=gallery[target_id],
                target_index=target_index,
            ))

    valid_ids = sorted(valid_groups)
    rng.shuffle(valid_ids)
    calibration_paths = [valid_groups[x][0] for x in valid_ids]
    control_paths = []
    for person in valid_ids:
        if len(valid_groups[person]) >= 2:
            control_paths.append(valid_groups[person][1])
        if len(control_paths) >= int(cfg["control_images"]):
            break

    return {
        "seed": int(cfg["seed"]),
        "official_counts": counts,
        "source_identities": source_ids,
        "source_paths": {str(k): v for k, v in source_paths.items()},
        "gallery": {str(k): v for k, v in gallery.items()},
        "calibration_paths": calibration_paths,
        "control_paths": control_paths,
        "pairs": [p.__dict__ for p in pairs],
        "pair_count": len(pairs),
        "pairing_rule": "200 identities with >=2 test images; one source image; five deterministic different-identity targets",
    }
