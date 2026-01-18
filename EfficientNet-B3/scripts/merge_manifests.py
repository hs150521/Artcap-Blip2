from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True, help="directory containing wikiart.csv / coco_nonart.csv / art500k.csv etc.")
    ap.add_argument("--out_dir", required=True, help="output directory for train.csv/val.csv/test.csv")
    ap.add_argument("--files", default="", help="comma-separated csv files to merge; default: all *.csv in in_dir")
    args = ap.parse_args()

    in_dir = Path(args.in_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    files: List[Path] = []
    if args.files.strip():
        for fn in [s.strip() for s in args.files.split(",") if s.strip()]:
            files.append((in_dir / fn).resolve())
    else:
        files = sorted([p for p in in_dir.glob("*.csv") if p.is_file()])

    if not files:
        raise SystemExit("No csv files to merge.")

    dfs = []
    for f in files:
        df = pd.read_csv(f)
        if "image_path" not in df.columns or "label_id" not in df.columns:
            raise ValueError(f"{f} missing required columns image_path,label_id")
        if "split" not in df.columns:
            raise ValueError(f"{f} missing required column split")
        dfs.append(df)

    merged = pd.concat(dfs, axis=0, ignore_index=True)

    for split in ["train", "val", "test"]:
        sdf = merged[merged["split"].astype(str) == split].copy()
        sdf.to_csv(str(out_dir / f"{split}.csv"), index=False)

    print(f"merged {len(merged)} rows into {out_dir}")


if __name__ == "__main__":
    main()
















