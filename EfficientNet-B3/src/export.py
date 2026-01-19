from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import torch

from _path import add_src_to_path

add_src_to_path()

from model import EfficientNetB3Classifier
from utils.checkpoint import load_checkpoint, save_json
from utils.config import is_timestamp_dir, resolve_run_dir
from utils.labels import load_classes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--classes_json", default="EfficientNet-B3/labels/classes_28.json")
    ap.add_argument("--embed_dim", type=int, default=256)
    ap.add_argument("--onnx", action="store_true", help="export onnx (optional)")
    ap.add_argument("--image_size", type=int, default=300)
    args = ap.parse_args()

    base_out = Path(args.out_dir).resolve()
    out_dir = base_out if is_timestamp_dir(base_out) else Path(resolve_run_dir(base_out, strategy="create"))
    out_dir.mkdir(parents=True, exist_ok=True)

    id_to_name, _ = load_classes(args.classes_json)
    ckpt = load_checkpoint(args.checkpoint, map_location="cpu")

    model = EfficientNetB3Classifier(num_classes=len(id_to_name), pretrained=False, embed_dim=int(args.embed_dim))
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    # 1) 保存权重
    torch.save(model.state_dict(), str(out_dir / "model_state_dict.pt"))
    save_json(str(out_dir / "classes_28.json"), {str(k): v for k, v in id_to_name.items()})

    meta: Dict[str, Any] = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "embed_dim": int(args.embed_dim),
        "num_classes": len(id_to_name),
        "image_size": int(args.image_size),
    }
    with open(str(out_dir / "export_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # 2) 可选导出 onnx（仅 logits；embed 可在后续扩展）
    if args.onnx:
        dummy = torch.randn(1, 3, int(args.image_size), int(args.image_size))

        class LogitsOnly(torch.nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m

            def forward(self, x):
                return self.m(x).logits

        wrapper = LogitsOnly(model)
        torch.onnx.export(
            wrapper,
            dummy,
            str(out_dir / "model_logits.onnx"),
            input_names=["image"],
            output_names=["logits"],
            opset_version=17,
            dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
        )

    print(f"exported to: {out_dir}")


if __name__ == "__main__":
    main()

















