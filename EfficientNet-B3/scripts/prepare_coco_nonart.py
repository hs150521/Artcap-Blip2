from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Sequence

from _common import ensure_out_dir, list_images, split_indices, write_manifest
from _path import add_src_to_path

add_src_to_path()

from utils.labels import load_classes


def _parse_subdirs_arg(subdirs: str) -> List[str]:
    return [s.strip() for s in str(subdirs).split(",") if s.strip()]


def _auto_subdirs(coco_root: Path) -> List[str]:
    """
    Try common COCO layouts (2014/2017) and return the subdirs that exist.

    Examples:
    - COCO2017: train2017/, val2017/, test2017/
    - COCO2014: train2014/, val2014/, test2014/
    - Some mirrors: images/train2014/, images/val2014/, ...
    """
    candidates: Sequence[str] = (
        # COCO2017 (canonical)
        "train2017",
        "val2017",
        "test2017",
        # COCO2014 (canonical)
        "train2014",
        "val2014",
        "test2014",
        # Common "images/" layout
        "images/train2017",
        "images/val2017",
        "images/test2017",
        "images/train2014",
        "images/val2014",
        "images/test2014",
    )
    out: List[str] = []
    for sub in candidates:
        if (coco_root / sub).exists():
            out.append(sub)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco_root", required=True, help="COCO root (e.g. contains train2014/val2014 or train2017/val2017)")
    ap.add_argument("--out_dir", required=True, help="manifest output dir")
    ap.add_argument("--classes_json", default="EfficientNet-B3/labels/classes_28.json")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train_ratio", type=float, default=0.9)
    ap.add_argument("--val_ratio", type=float, default=0.05)
    ap.add_argument(
        "--subdirs",
        default="auto",
        help="comma-separated subdirs to scan; use 'auto' to detect COCO2014/2017 layouts (default)",
    )
    args = ap.parse_args()

    coco_root = Path(args.coco_root)
    if not coco_root.exists():
        raise SystemExit(f"coco_root not found: {args.coco_root}")
    ensure_out_dir(args.out_dir)
    id_to_name, name_to_id = load_classes(args.classes_json)
    non_art_id = int(name_to_id.get("non_art", 27))

    images: List[str] = []
    used_subdirs: List[str] = []
    subdirs_raw = str(args.subdirs).strip().lower()
    if subdirs_raw == "auto":
        used_subdirs = _auto_subdirs(coco_root)
    else:
        used_subdirs = _parse_subdirs_arg(str(args.subdirs))

    for sub in used_subdirs:
        sub_path = coco_root / sub
        if sub_path.exists():
            images += list_images(str(sub_path), recursive=True)

    # Fallback: if user passed weird layout or extracted all images directly under coco_root
    if not images and subdirs_raw == "auto":
        images = list_images(str(coco_root), recursive=True)
        used_subdirs = ["<coco_root> (fallback recursive scan)"]

    if not images:
        raise SystemExit(f"No images found under {args.coco_root} with subdirs={args.subdirs}")

    # Ensure stable split and avoid duplicates (e.g. multiple subdirs overlapping via symlinks/mirrors)
    images = sorted(set(images))

    rows = [{"image_path": p, "label_id": non_art_id, "source": "coco"} for p in images]

    train_idx, val_idx, test_idx = split_indices(len(rows), seed=args.seed, train_ratio=args.train_ratio, val_ratio=args.val_ratio)
    for i in train_idx:
        rows[i]["split"] = "train"
    for i in val_idx:
        rows[i]["split"] = "val"
    for i in test_idx:
        rows[i]["split"] = "test"

    out_csv = str((Path(args.out_dir) / "coco_nonart.csv").resolve())
    write_manifest(rows, out_csv)
    print(f"scanned subdirs: {', '.join(used_subdirs) if used_subdirs else '(none)'}")
    print(f"wrote: {out_csv}  (n={len(rows)})")


if __name__ == "__main__":
    main()


