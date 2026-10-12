"""Five-fold split utilities."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Dict, List


def create_kfold_split(
    data_dir: str,
    split_file: str,
    num_folds: int = 5,
    seed: int = 42,
    overwrite: bool = False,
) -> Dict:
    split_path = Path(split_file)
    if split_path.is_file() and not overwrite:
        with open(split_path, "r", encoding="utf-8") as f:
            return json.load(f)

    files = sorted(f for f in os.listdir(data_dir) if f.endswith(".pkl"))
    if len(files) < num_folds:
        raise ValueError(f"Need at least {num_folds} pkl files, got {len(files)}")

    rng = random.Random(int(seed))
    shuffled = files[:]
    rng.shuffle(shuffled)

    folds: List[Dict[str, List[str]]] = []
    fold_sizes = [len(shuffled) // num_folds] * num_folds
    for i in range(len(shuffled) % num_folds):
        fold_sizes[i] += 1
    start = 0
    for fold_idx, size in enumerate(fold_sizes):
        val = sorted(shuffled[start : start + size])
        train = sorted([f for f in shuffled if f not in set(val)])
        folds.append({"fold": fold_idx, "train": train, "val": val})
        start += size

    payload = {
        "data_dir": data_dir,
        "num_files": len(files),
        "num_folds": num_folds,
        "seed": seed,
        "folds": folds,
    }
    split_path.parent.mkdir(parents=True, exist_ok=True)
    with open(split_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return payload


def load_or_create_split(data_dir: str, split_file: str, num_folds: int = 5, seed: int = 42) -> Dict:
    return create_kfold_split(data_dir, split_file, num_folds=num_folds, seed=seed, overwrite=False)


__all__ = ["create_kfold_split", "load_or_create_split"]
