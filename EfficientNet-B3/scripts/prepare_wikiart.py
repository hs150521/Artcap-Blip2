from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd

from _path import add_src_to_path

add_src_to_path()

from utils.labels import build_alias_to_canonical, load_classes, load_style_aliases, normalize_style_name, resolve_style_to_label_id

from _common import ensure_out_dir, list_images, split_indices, write_manifest


def infer_style_from_path(img_path: str, wikiart_root: str) -> str:
    """
    支持常见 WikiArt 组织：root/<style_name>/*.jpg 或 root/style/<style_name>/*.jpg
    取相对路径的第1或第2级目录作为 style 候选。
    """
    p = Path(img_path).resolve()
    root = Path(wikiart_root).resolve()
    rel = p.relative_to(root)
    parts = rel.parts
    if len(parts) >= 2 and normalize_style_name(parts[0]) in {"style", "styles"}:
        return parts[1]
    return parts[0] if parts else ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wikiart_root", required=True, help="WikiArt root directory")
    ap.add_argument("--out_dir", required=True, help="manifest output dir")
    ap.add_argument("--classes_json", default="EfficientNet-B3/labels/classes_28.json")
    ap.add_argument("--style_27_json", default="EfficientNet-B3/labels/style_27.json")
    ap.add_argument("--meta_csv", default="", help="optional metadata csv with columns: image_path,style")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train_ratio", type=float, default=0.9)
    ap.add_argument("--val_ratio", type=float, default=0.05)
    args = ap.parse_args()

    if not Path(args.wikiart_root).exists():
        raise SystemExit(f"wikiart_root not found: {args.wikiart_root}")
    ensure_out_dir(args.out_dir)
    id_to_name, name_to_id = load_classes(args.classes_json)
    style_aliases = load_style_aliases(args.style_27_json)
    alias_to_canonical = build_alias_to_canonical(style_aliases)

    rows: List[dict] = []

    if args.meta_csv:
        df = pd.read_csv(args.meta_csv)
        if "image_path" not in df.columns or "style" not in df.columns:
            raise ValueError("--meta_csv must contain columns: image_path,style")
        image_paths = df["image_path"].astype(str).tolist()
        styles = df["style"].astype(str).tolist()
        for img, st in zip(image_paths, styles):
            img_path = str((Path(args.wikiart_root) / img).resolve()) if not Path(img).is_absolute() else str(Path(img).resolve())
            label_id = resolve_style_to_label_id(st, alias_to_canonical=alias_to_canonical, name_to_id=name_to_id)
            rows.append({"image_path": img_path, "label_id": label_id, "source": "wikiart"})
    else:
        images = list_images(args.wikiart_root, recursive=True)
        for img_path in images:
            st = infer_style_from_path(img_path, wikiart_root=args.wikiart_root)
            label_id = resolve_style_to_label_id(st, alias_to_canonical=alias_to_canonical, name_to_id=name_to_id)
            rows.append({"image_path": img_path, "label_id": label_id, "source": "wikiart"})

    # split
    train_idx, val_idx, test_idx = split_indices(len(rows), seed=args.seed, train_ratio=args.train_ratio, val_ratio=args.val_ratio)
    for i in train_idx:
        rows[i]["split"] = "train"
    for i in val_idx:
        rows[i]["split"] = "val"
    for i in test_idx:
        rows[i]["split"] = "test"

    out_csv = str((Path(args.out_dir) / "wikiart.csv").resolve())
    write_manifest(rows, out_csv)
    print(f"wrote: {out_csv}  (n={len(rows)})")


if __name__ == "__main__":
    main()


