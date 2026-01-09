from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from PIL import Image


@dataclass(frozen=True)
class DataCheckResult:
    ok: bool
    num_total: int
    num_by_split: Dict[str, int]
    num_by_source: Dict[str, int]
    checked_images: int
    errors: List[str]


def _read_manifest(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"manifest not found: {path}")
    df = pd.read_csv(p)
    if "image_path" not in df.columns or "label_id" not in df.columns:
        raise ValueError("manifest must contain columns: image_path,label_id (and optionally split/source)")
    return df


def check_manifest(
    manifest_csv: str,
    expected_splits: Tuple[str, ...] = ("train", "val", "test"),
    sample_n: int = 16,
    seed: int = 42,
) -> DataCheckResult:
    errors: List[str] = []
    try:
        df = _read_manifest(manifest_csv)
    except Exception as e:
        return DataCheckResult(
            ok=False,
            num_total=0,
            num_by_split={},
            num_by_source={},
            checked_images=0,
            errors=[str(e)],
        )

    num_total = int(len(df))
    if num_total == 0:
        errors.append("manifest is empty")

    num_by_split: Dict[str, int] = {}
    if "split" in df.columns:
        for s, c in df["split"].astype(str).value_counts().to_dict().items():
            num_by_split[str(s)] = int(c)
        # 如果存在 split 字段，要求至少包含期望 splits（可以多，但不能缺）
        for s in expected_splits:
            if s not in num_by_split:
                errors.append(f"missing split '{s}' in manifest")
    else:
        errors.append("manifest missing 'split' column (required for this pipeline)")

    num_by_source: Dict[str, int] = {}
    if "source" in df.columns:
        for s, c in df["source"].astype(str).value_counts().to_dict().items():
            num_by_source[str(s)] = int(c)

    # 随机抽样检查图像可读
    sample_n = min(int(sample_n), num_total)
    checked = 0
    if sample_n > 0:
        rnd = random.Random(seed)
        idxs = list(range(num_total))
        rnd.shuffle(idxs)
        for i in idxs[:sample_n]:
            p = str(df.iloc[i]["image_path"])
            try:
                with Image.open(p) as im:
                    im.verify()
                checked += 1
            except Exception as e:
                errors.append(f"unreadable image: {p} ({e})")

    return DataCheckResult(
        ok=len(errors) == 0,
        num_total=num_total,
        num_by_split=num_by_split,
        num_by_source=num_by_source,
        checked_images=checked,
        errors=errors,
    )











