from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import torch
from PIL import Image

from _path import add_src_to_path

add_src_to_path()

from data.transforms import build_eval_transforms
from model import EfficientNetB3Classifier
from utils.checkpoint import load_checkpoint
from utils.labels import load_classes


@torch.no_grad()
def infer_one(
    model: EfficientNetB3Classifier,
    tfm,
    image_path: str,
    device: torch.device,
    id_to_name: Dict[int, str],
) -> Dict[str, Any]:
    img = Image.open(image_path).convert("RGB")
    x = tfm(img).unsqueeze(0).to(device)
    out = model(x)
    probs = torch.softmax(out.logits[0], dim=-1).detach().cpu().tolist()
    pred_id = int(torch.tensor(probs).argmax().item())
    pred_name = id_to_name[pred_id]

    res: Dict[str, Any] = {
        "image_path": str(Path(image_path).resolve()),
        "pred_id": pred_id,
        "pred_name": pred_name,
        "z": probs,  # 概率向量
    }
    if out.embed is not None:
        res["s"] = out.embed[0].detach().cpu().tolist()
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--classes_json", required=True)
    ap.add_argument("--image", default="", help="single image path")
    ap.add_argument("--image_list", default="", help="txt file, one image path per line")
    ap.add_argument("--image_dir", default="", help="directory of images (non-recursive)")
    ap.add_argument("--out", default="", help="write jsonl to file (optional)")
    ap.add_argument("--image_size", type=int, default=300)
    ap.add_argument("--embed_dim", type=int, default=256, help="must match training config")
    args = ap.parse_args()

    id_to_name, _ = load_classes(args.classes_json)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = EfficientNetB3Classifier(num_classes=len(id_to_name), pretrained=False, embed_dim=int(args.embed_dim)).to(device)
    ckpt = load_checkpoint(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    tfm = build_eval_transforms(image_size=int(args.image_size), aug_cfg={"resize_shorter": int(args.image_size * 1.07), "center_crop": int(args.image_size)})

    images: List[str] = []
    if args.image:
        images.append(args.image)
    if args.image_list:
        with open(args.image_list, "r", encoding="utf-8") as f:
            images += [ln.strip() for ln in f if ln.strip()]
    if args.image_dir:
        for p in sorted(Path(args.image_dir).iterdir()):
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
                images.append(str(p))

    if not images:
        raise SystemExit("No images provided. Use --image / --image_list / --image_dir")

    out_f = open(args.out, "w", encoding="utf-8") if args.out else None
    try:
        for img_path in images:
            res = infer_one(model, tfm, img_path, device=device, id_to_name=id_to_name)
            line = json.dumps(res, ensure_ascii=False)
            if out_f:
                out_f.write(line + "\n")
            else:
                print(line)
    finally:
        if out_f:
            out_f.close()


if __name__ == "__main__":
    main()











