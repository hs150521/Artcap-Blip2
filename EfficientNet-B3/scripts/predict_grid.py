from __future__ import annotations

import argparse
import csv
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from PIL import Image, ImageDraw, ImageFont

from _path import add_src_to_path

add_src_to_path()

from data.transforms import build_eval_transforms
from model import EfficientNetB3Classifier
from utils.checkpoint import load_checkpoint
from utils.labels import load_classes


@dataclass(frozen=True)
class Sample:
    image_path: str
    label_id: int


def _fmt_pct(p: float) -> str:
    return f"{p * 100.0:.2f}%"


def _load_font(font_size: int) -> ImageFont.ImageFont:
    # 兼容 Linux 常见字体；如果不存在就回退默认字体
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]
    for p in candidates:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, font_size)
            except Exception:
                pass
    return ImageFont.load_default()


def _read_manifest(manifest_csv: str) -> List[Sample]:
    rows: List[Sample] = []
    with open(manifest_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            ip = (r.get("image_path") or "").strip()
            lid_raw = (r.get("label_id") or "").strip()
            if not ip or not lid_raw:
                continue
            try:
                lid = int(lid_raw)
            except Exception:
                continue
            rows.append(Sample(image_path=ip, label_id=lid))
    return rows


def _is_image_readable(path: str) -> bool:
    try:
        p = Path(path)
        if not p.exists():
            return False
        with Image.open(str(p)) as im:
            im.verify()
        return True
    except Exception:
        return False


def _sample_n_distinct_labels(rows: Sequence[Sample], n: int, seed: int) -> List[Sample]:
    """
    采样 n 张图片，要求每张来自不同的 label_id（画风）。
    - 会遍历一次打乱后的 rows，遇到新 label 且图片可读就收下
    - 若最终不足 n 个不同 label，则报错
    """
    if not rows:
        raise RuntimeError("Manifest is empty.")

    idxs = list(range(len(rows)))
    random.Random(seed).shuffle(idxs)

    picked: List[Sample] = []
    used_labels: set[int] = set()

    for idx in idxs:
        s = rows[idx]
        lid = int(s.label_id)
        if lid in used_labels:
            continue
        if not _is_image_readable(s.image_path):
            continue
        used_labels.add(lid)
        picked.append(s)
        if len(picked) >= n:
            break

    if len(picked) < n:
        raise RuntimeError(
            f"Only found {len(picked)}/{n} readable images with distinct label_id in manifest. "
            f"Unique labels found: {len(used_labels)}."
        )
    return picked


@torch.no_grad()
def _predict_topk(
    model: EfficientNetB3Classifier,
    tfm,
    img_path: str,
    device: torch.device,
    k: int = 3,
) -> Tuple[List[int], List[float]]:
    img = Image.open(img_path).convert("RGB")
    x = tfm(img).unsqueeze(0).to(device)
    out = model(x)
    probs = torch.softmax(out.logits[0], dim=-1)
    top_p, top_i = torch.topk(probs, k=int(k))
    return top_i.detach().cpu().tolist(), top_p.detach().cpu().tolist()


def _draw_cell(
    base_img: Image.Image,
    text_lines: List[str],
    cell_w: int,
    cell_h: int,
    pad: int,
    font: ImageFont.ImageFont,
    bg: Tuple[int, int, int] = (255, 255, 255),
    fg: Tuple[int, int, int] = (0, 0, 0),
) -> Image.Image:
    cell = Image.new("RGB", (cell_w, cell_h), bg)
    draw = ImageDraw.Draw(cell)

    # 将图像等比缩放到 cell 的图片区区域
    text_h = int(cell_h * 0.30)
    img_h = cell_h - text_h
    img_area_w = cell_w - 2 * pad
    img_area_h = img_h - 2 * pad

    im = base_img.convert("RGB")
    im.thumbnail((img_area_w, img_area_h), Image.Resampling.LANCZOS)
    x0 = (cell_w - im.width) // 2
    y0 = pad + (img_area_h - im.height) // 2
    cell.paste(im, (x0, y0))

    # 文本区域
    ty = img_h + pad // 2
    tx = pad
    for ln in text_lines:
        draw.text((tx, ty), ln, fill=fg, font=font)
        # PIL 对行高估计不稳定，这里用 bbox 计算更稳
        bbox = draw.textbbox((tx, ty), ln, font=font)
        line_h = max(1, bbox[3] - bbox[1])
        ty += line_h + 2

    # 边框
    draw.rectangle([0, 0, cell_w - 1, cell_h - 1], outline=(220, 220, 220), width=1)
    return cell


def _make_grid(cells: List[Image.Image], rows: int, cols: int, gap: int, bg: Tuple[int, int, int]) -> Image.Image:
    if len(cells) != rows * cols:
        raise ValueError(f"cells size {len(cells)} != rows*cols {rows*cols}")
    cw, ch = cells[0].size
    w = cols * cw + (cols - 1) * gap
    h = rows * ch + (rows - 1) * gap
    canvas = Image.new("RGB", (w, h), bg)
    for r in range(rows):
        for c in range(cols):
            i = r * cols + c
            x = c * (cw + gap)
            y = r * (ch + gap)
            canvas.paste(cells[i], (x, y))
    return canvas


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="/data/artcap-blip2/outputs/effb3_28cls/checkpoints/best.pt")
    ap.add_argument("--manifest", default="/data/artcap-blip2/outputs/effb3_28cls/manifests/all.csv")
    ap.add_argument("--classes_json", default="/data/artcap-blip2/EfficientNet-B3/labels/classes_28.json")
    ap.add_argument("--out", default="/data/artcap-blip2/outputs/effb3_28cls/vis/predict_8.jpg")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--image_size", type=int, default=300)
    ap.add_argument("--embed_dim", type=int, default=256, help="must match training config")
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--gap", type=int, default=12)
    ap.add_argument("--font_size", type=int, default=16)
    ap.add_argument("--pad", type=int, default=10)
    args = ap.parse_args()

    if int(args.n) != 8:
        raise SystemExit("This script is intended for n=8 (2x4). Please keep --n 8.")

    id_to_name, _ = load_classes(args.classes_json)
    num_classes = len(id_to_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = EfficientNetB3Classifier(num_classes=num_classes, pretrained=False, embed_dim=int(args.embed_dim)).to(device)
    ckpt = load_checkpoint(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    tfm = build_eval_transforms(
        image_size=int(args.image_size),
        aug_cfg={"resize_shorter": int(args.image_size * 1.07), "center_crop": int(args.image_size)},
    )

    rows = _read_manifest(args.manifest)
    samples = _sample_n_distinct_labels(rows, n=8, seed=int(args.seed))

    font = _load_font(int(args.font_size))
    cells: List[Image.Image] = []

    cell_w = int(args.image_size * 1.10)
    cell_h = int(args.image_size * 1.45)

    for s in samples:
        top_i, top_p = _predict_topk(model, tfm, s.image_path, device=device, k=int(args.topk))
        gt_name = id_to_name.get(int(s.label_id), f"unknown({s.label_id})")

        pred_lines = []
        for rank, (cid, p) in enumerate(zip(top_i, top_p), start=1):
            cname = id_to_name.get(int(cid), str(cid))
            pred_lines.append(f"Top{rank}: {cname} {_fmt_pct(float(p))}")

        text_lines = [f"GT: {gt_name}"] + pred_lines
        img = Image.open(s.image_path).convert("RGB")
        cell = _draw_cell(
            base_img=img,
            text_lines=text_lines,
            cell_w=cell_w,
            cell_h=cell_h,
            pad=int(args.pad),
            font=font,
            bg=(255, 255, 255),
            fg=(0, 0, 0),
        )
        cells.append(cell)

    grid = _make_grid(cells, rows=2, cols=4, gap=int(args.gap), bg=(245, 245, 245))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(str(out_path), quality=95)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()


