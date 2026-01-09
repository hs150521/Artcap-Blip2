from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Iterable, List, Tuple

import pandas as pd

from _common import ensure_out_dir, split_indices
from _path import add_src_to_path

add_src_to_path()

from utils.labels import build_alias_to_canonical, load_classes, load_style_aliases, resolve_style_to_label_id


ART500K_COLS: List[str] = [
    "author_name",
    "painting_name",
    "image_url",
    "Genre",
    "Style",
    "Nationality",
    "Painting School",
    "Art Movement",
    "Field",
    "Date",
    "Influenced by",
    "Media",
    "Influenced on",
    "Family and Relatives",
    "Tag",
    "Pupils",
    "Location",
    "Original Title",
    "Dimensions",
    "Series",
    "Teachers",
    "Friends and Co-workers",
    "Art institution",
    "Period",
    "Theme",
    "Path",
]


def _first_data_line(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.strip():
                return line.rstrip("\n")
    return ""


def _detect_tsv_mode(path: str) -> Tuple[str, bool]:
    line = _first_data_line(path)
    if not line:
        return "unknown", False
    parts = line.split("\t")
    if len(parts) == 2:
        has_header = parts[0].strip().lower() == "image_path" and parts[1].strip().lower() == "style"
        return "two_col", has_header
    if "Path" in parts or "author_name" in parts or "painting_name" in parts:
        return "art500k", True
    if len(parts) >= 20:
        return "art500k", False
    return "unknown", False


def _split_style_tokens(s: str) -> List[str]:
    if not isinstance(s, str):
        return []
    s = s.strip()
    if not s:
        return []
    toks = re.split(r"[;,|/]+", s)
    return [t.strip() for t in toks if t.strip()]


def _pick_style_col(cols: Iterable[str]) -> str:
    cols_set = set(cols)
    if "Style" in cols_set:
        return "Style"
    if "style" in cols_set:
        return "style"
    if "Art Movement" in cols_set:
        return "Art Movement"
    if "art_movement" in cols_set:
        return "art_movement"
    return ""


def _pick_path_col(cols: Iterable[str]) -> str:
    cols_set = set(cols)
    if "image_path" in cols_set:
        return "image_path"
    if "Path" in cols_set:
        return "Path"
    if "path" in cols_set:
        return "path"
    return ""


def _read_tsv_in_chunks(path: str, chunksize: int = 50000) -> Iterable[pd.DataFrame]:
    mode, has_header = _detect_tsv_mode(path)
    if mode == "two_col":
        if has_header:
            return pd.read_csv(path, sep="\t", chunksize=chunksize, dtype=str, keep_default_na=False)
        return pd.read_csv(
            path, sep="\t", header=None, names=["image_path", "style"], chunksize=chunksize, dtype=str, keep_default_na=False
        )
    if mode == "art500k":
        if has_header:
            return pd.read_csv(path, sep="\t", chunksize=chunksize, dtype=str, keep_default_na=False)
        return pd.read_csv(
            path,
            sep="\t",
            header=None,
            names=ART500K_COLS,
            chunksize=chunksize,
            dtype=str,
            keep_default_na=False,
            engine="python",
            quoting=csv.QUOTE_NONE,
            on_bad_lines="skip",
        )
    return pd.read_csv(path, sep="\t", chunksize=chunksize, dtype=str, keep_default_na=False, engine="python", on_bad_lines="skip")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--art500k_root", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--classes_json", default="EfficientNet-B3/labels/classes_28.json")
    ap.add_argument("--style_27_json", default="EfficientNet-B3/labels/style_27.json")
    ap.add_argument("--meta_tsv", required=True, help="TSV metadata (either 2-col: image_path\\tstyle, or ART500K label_list.tsv)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train_ratio", type=float, default=0.9)
    ap.add_argument("--val_ratio", type=float, default=0.05)
    args = ap.parse_args()

    if not Path(args.art500k_root).exists():
        raise SystemExit(f"art500k_root not found: {args.art500k_root}")
    if not Path(args.meta_tsv).exists():
        raise SystemExit(f"meta_tsv not found: {args.meta_tsv}")

    ensure_out_dir(args.out_dir)

    _, name_to_id = load_classes(args.classes_json)
    style_aliases = load_style_aliases(args.style_27_json)
    alias_to_canonical = build_alias_to_canonical(style_aliases)

    image_paths: List[str] = []
    label_ids: List[int] = []

    for chunk in _read_tsv_in_chunks(args.meta_tsv):
        style_col = _pick_style_col(chunk.columns)
        path_col = _pick_path_col(chunk.columns)
        if not style_col or not path_col:
            raise SystemExit(f"meta_tsv must contain path+style columns; got columns={list(chunk.columns)}")

        for img, st in zip(chunk[path_col].astype(str).tolist(), chunk[style_col].astype(str).tolist()):
            if not img or img.lower() == "nan":
                continue
            if not st or st.lower() == "nan":
                continue

            img_path = str((Path(args.art500k_root) / img).resolve()) if not Path(img).is_absolute() else str(Path(img).resolve())

            picked = None
            for tok in _split_style_tokens(st):
                try:
                    picked = resolve_style_to_label_id(tok, alias_to_canonical=alias_to_canonical, name_to_id=name_to_id)
                    break
                except KeyError:
                    continue
            if picked is None:
                continue

            image_paths.append(img_path)
            label_ids.append(picked)

    n = len(image_paths)
    train_idx, val_idx, test_idx = split_indices(n, seed=args.seed, train_ratio=args.train_ratio, val_ratio=args.val_ratio)

    splits = [""] * n
    for i in train_idx:
        splits[i] = "train"
    for i in val_idx:
        splits[i] = "val"
    for i in test_idx:
        splits[i] = "test"

    out_tsv = str((Path(args.out_dir) / "art500k.tsv").resolve())
    with open(out_tsv, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["image_path", "label_id", "source", "split"])
        for p, lid, sp in zip(image_paths, label_ids, splits):
            w.writerow([p, lid, "art500k", sp])

    print(f"wrote: {out_tsv}  (n={n})")


if __name__ == "__main__":
    main()
