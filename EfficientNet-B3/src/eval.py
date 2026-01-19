from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from _path import add_src_to_path

add_src_to_path()

from data.dataset import ImageClassificationDataset
from data.transforms import build_eval_transforms
from model import EfficientNetB3Classifier
from utils.checkpoint import load_checkpoint
from utils.config import ensure_dir, is_timestamp_dir, load_yaml, parse_paths, resolve_run_dir
from utils.labels import load_classes
from utils.metrics import accuracy, confusion_matrix, per_class_accuracy, precision_recall_binary
from utils.data_check import check_manifest


@torch.no_grad()
def evaluate_split(
    cfg: Dict[str, Any],
    checkpoint: str,
    split: str,
    run_ratio: int = 100,
    max_steps: int = 0,
) -> Dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    paths = parse_paths(cfg)
    data_cfg = cfg.get("data", {})
    aug_cfg = cfg.get("augment", {})

    image_size = int(data_cfg.get("image_size", 300))
    batch_size = int(data_cfg.get("batch_size", 64))
    num_workers = int(data_cfg.get("num_workers", 4))
    pin_memory = bool(data_cfg.get("pin_memory", True))

    manifest_name = data_cfg.get(f"{split}_manifest", f"{split}.csv")
    manifest_path = str((Path(paths.manifests_dir) / manifest_name).resolve())

    eval_tf = build_eval_transforms(image_size=image_size, aug_cfg=aug_cfg.get("eval", {}))
    ds = ImageClassificationDataset(manifest_csv=manifest_path, split=split, transform=eval_tf)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    classes_json = cfg.get("labels", {}).get("classes_json", "EfficientNet-B3/labels/classes_28.json")
    id_to_name, name_to_id = load_classes(classes_json)
    non_art_id = int(name_to_id.get(cfg.get("sampling", {}).get("non_art_label_name", "non_art"), 27))

    mcfg = cfg.get("model", {})
    num_classes = int(mcfg.get("num_classes", 28))
    embed_dim = int(mcfg.get("embed_dim", 0))
    model = EfficientNetB3Classifier(num_classes=num_classes, pretrained=False, embed_dim=embed_dim).to(device)
    ckpt = load_checkpoint(checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    y_true = []
    y_pred = []
    if run_ratio < 1 or run_ratio > 100:
        raise ValueError("run_ratio must be in [1,100]")
    if max_steps < 0:
        raise ValueError("max_steps must be >= 0")

    total_batches = len(loader)
    if run_ratio < 100:
        total_batches = max(1, int(total_batches * (run_ratio / 100.0)))
    if max_steps:
        total_batches = min(total_batches, max_steps)

    pbar = tqdm(loader, desc=f"eval:{split}", total=total_batches, dynamic_ncols=True)
    for it, (x, y) in enumerate(pbar):
        if it >= total_batches:
            break
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        out = model(x)
        pred = out.logits.argmax(dim=-1)
        y_true.extend(y.detach().cpu().tolist())
        y_pred.extend(pred.detach().cpu().tolist())

    acc = accuracy(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, num_classes=num_classes)
    per_cls = per_class_accuracy(cm)
    non_prec, non_rec = precision_recall_binary(y_true, y_pred, positive_label=non_art_id)

    return {
        "split": split,
        "acc": acc,
        "per_class_acc": per_cls,
        "non_art_precision": non_prec,
        "non_art_recall": non_rec,
        "confusion": cm,
        "num_samples": len(y_true),
        "classes_json": classes_json,
        "id_to_name": id_to_name,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", default="", help="default: <output_dir>/checkpoints/best.pt")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--run_ratio", type=int, default=100, help="1-100, evaluate on subset for smoke test (seed-fixed)")
    ap.add_argument("--max_steps", type=int, default=0, help="max eval steps (batches); 0 means no limit")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    paths = parse_paths(cfg)
    base_out = Path(paths.output_dir).resolve()
    if is_timestamp_dir(base_out):
        out_dir = str(base_out)
    else:
        # pick latest run dir so default checkpoint works
        try:
            out_dir = resolve_run_dir(base_out, strategy="latest")
        except FileNotFoundError:
            if not args.checkpoint:
                raise SystemExit(
                    f"No timestamped run directory found under {base_out}. "
                    f"Please specify --checkpoint, or run training first."
                )
            out_dir = str(base_out)

    ckpt = args.checkpoint or str((Path(out_dir) / "checkpoints" / "best.pt").resolve())

    # 数据检查（按规则：先检查再使用）
    data_cfg = cfg.get("data", {})
    manifest_name = data_cfg.get(f"{args.split}_manifest", f"{args.split}.csv")
    manifest_path = str((Path(paths.manifests_dir) / manifest_name).resolve())
    dc = check_manifest(manifest_path, sample_n=16, seed=int(cfg.get("seed", 42)))
    logs_dir = ensure_dir(str(Path(out_dir) / "logs"))
    with open(str(Path(logs_dir) / f"data_check_{args.split}.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "ok": dc.ok,
                "num_total": dc.num_total,
                "num_by_split": dc.num_by_split,
                "num_by_source": dc.num_by_source,
                "checked_images": dc.checked_images,
                "errors": dc.errors,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    if not dc.ok:
        raise SystemExit(f"data_check failed, see: {Path(logs_dir) / f'data_check_{args.split}.json'}")

    report = evaluate_split(cfg, checkpoint=ckpt, split=args.split, run_ratio=int(args.run_ratio), max_steps=int(args.max_steps))

    metrics_dir = ensure_dir(str(Path(out_dir) / "metrics"))
    torch.save(torch.tensor(report["confusion"]), str(Path(metrics_dir) / f"confusion_{args.split}.pt"))

    to_json = {k: v for k, v in report.items() if k not in ["confusion", "id_to_name"]}
    with open(str(Path(metrics_dir) / f"report_{args.split}.json"), "w", encoding="utf-8") as f:
        json.dump(to_json, f, ensure_ascii=False, indent=2)

    print(json.dumps({"split": report["split"], "acc": report["acc"], "num_samples": report["num_samples"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()


