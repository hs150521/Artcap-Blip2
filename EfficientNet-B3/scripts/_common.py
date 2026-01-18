from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import pandas as pd


IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def list_images(root: str, recursive: bool = True) -> List[str]:
    root_p = Path(root).resolve()
    out: List[str] = []
    if recursive:
        it = root_p.rglob("*")
    else:
        it = root_p.glob("*")
    for p in it:
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            out.append(str(p))
    return out


def split_indices(n: int, seed: int, train_ratio: float = 0.9, val_ratio: float = 0.05) -> Tuple[List[int], List[int], List[int]]:
    if n <= 0:
        return [], [], []
    idx = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(idx)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train = idx[:n_train]
    val = idx[n_train : n_train + n_val]
    test = idx[n_train + n_val :]
    return train, val, test


def ensure_out_dir(out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def write_manifest(rows: List[dict], out_csv: str) -> None:
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
















