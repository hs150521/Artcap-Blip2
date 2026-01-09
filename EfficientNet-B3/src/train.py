from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from _path import add_src_to_path

add_src_to_path()

from data.dataset import ImageClassificationDataset
from data.sampler import ArtNonArtBatchSampler, ArtNonArtSampling
from data.transforms import build_eval_transforms, build_train_transforms
from model import EfficientNetB3Classifier
from utils.checkpoint import CheckpointPayload, load_checkpoint, save_checkpoint, save_json
from utils.config import ensure_dir, load_yaml, parse_paths
from utils.labels import load_classes
from utils.metrics import accuracy, confusion_matrix, per_class_accuracy, precision_recall_binary
from utils.data_check import check_manifest
from utils.seed import seed_everything


def _resolve_manifest_path(manifests_dir: str, filename: str) -> str:
    p = Path(filename)
    if p.is_absolute():
        return str(p)
    return str((Path(manifests_dir) / filename).resolve())


def _build_loaders(cfg: Dict[str, Any], device: torch.device) -> Tuple[DataLoader, DataLoader, Dict[str, Any]]:
    paths = parse_paths(cfg)
    manifests_dir = paths.manifests_dir

    data_cfg = cfg.get("data", {})
    train_manifest = _resolve_manifest_path(manifests_dir, data_cfg.get("train_manifest", "train.csv"))
    val_manifest = _resolve_manifest_path(manifests_dir, data_cfg.get("val_manifest", "val.csv"))
    image_size = int(data_cfg.get("image_size", 300))
    batch_size = int(data_cfg.get("batch_size", 64))
    num_workers = int(data_cfg.get("num_workers", 4))
    pin_memory = bool(data_cfg.get("pin_memory", True))

    aug_cfg = cfg.get("augment", {})
    train_tf = build_train_transforms(image_size=image_size, aug_cfg=aug_cfg.get("train", {}))
    eval_tf = build_eval_transforms(image_size=image_size, aug_cfg=aug_cfg.get("eval", {}))

    train_ds = ImageClassificationDataset(manifest_csv=train_manifest, split="train", transform=train_tf)
    val_ds = ImageClassificationDataset(manifest_csv=val_manifest, split="val", transform=eval_tf)

    sampling_cfg = cfg.get("sampling", {})
    enabled = bool(sampling_cfg.get("enabled", True))
    ratio = int(sampling_cfg.get("art_to_nonart_ratio", 27))

    classes_json = cfg.get("labels", {}).get("classes_json", "EfficientNet-B3/labels/classes_28.json")
    id_to_name, name_to_id = load_classes(classes_json)
    non_art_name = str(sampling_cfg.get("non_art_label_name", "non_art"))
    if non_art_name not in name_to_id:
        raise ValueError(f"non_art_label_name '{non_art_name}' not found in classes_json.")
    non_art_id = int(name_to_id[non_art_name])

    if enabled:
        batch_sampler = ArtNonArtBatchSampler(
            labels=train_ds.labels,
            batch_size=batch_size,
            cfg=ArtNonArtSampling(enabled=True, art_to_nonart_ratio=ratio, non_art_label_id=non_art_id),
            seed=int(cfg.get("seed", 42)),
            drop_last=True,
        )
        train_loader = DataLoader(
            train_ds, batch_sampler=batch_sampler, num_workers=num_workers, pin_memory=pin_memory
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True,
        )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    meta = {"id_to_name": id_to_name, "name_to_id": name_to_id, "non_art_id": non_art_id}
    return train_loader, val_loader, meta


@torch.no_grad()
def _evaluate(model: nn.Module, loader: DataLoader, device: torch.device, num_classes: int, non_art_id: int) -> Dict[str, Any]:
    model.eval()
    y_true = []
    y_pred = []
    for x, y in loader:
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
        "acc": acc,
        "per_class_acc": per_cls,
        "confusion": cm,
        "non_art_precision": non_prec,
        "non_art_recall": non_rec,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="YAML config path, e.g. EfficientNet-B3/configs/train.yaml")
    ap.add_argument("--resume", default="", help="checkpoint path to resume (optional)")
    ap.add_argument("--run_ratio", type=int, default=100, help="1-100, use subset of data for smoke test (seed-fixed)")
    ap.add_argument("--max_steps", type=int, default=0, help="max train steps, 0 means no limit (higher priority than run_ratio)")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    seed = int(cfg.get("seed", 42))
    seed_everything(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    paths = parse_paths(cfg)
    out_dir = ensure_dir(str(Path(paths.output_dir).resolve()))
    ckpt_dir = ensure_dir(str(Path(out_dir) / "checkpoints"))
    metrics_dir = ensure_dir(str(Path(out_dir) / "metrics"))
    logs_dir = ensure_dir(str(Path(out_dir) / "logs"))

    # 记录一次 config 快照
    save_json(str(Path(out_dir) / "config.snapshot.json"), cfg)

    train_loader, val_loader, meta = _build_loaders(cfg, device=device)

    # 规则：训练前必须做数据检查并输出结果（缺失则退出）
    train_manifest_path = _resolve_manifest_path(paths.manifests_dir, cfg.get("data", {}).get("train_manifest", "train.csv"))
    dc = check_manifest(train_manifest_path, sample_n=16, seed=seed)
    with open(str(Path(logs_dir) / "data_check.json"), "w", encoding="utf-8") as f:
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
        raise SystemExit(f"data_check failed, see: {Path(logs_dir) / 'data_check.json'}")

    mcfg = cfg.get("model", {})
    num_classes = int(mcfg.get("num_classes", 28))
    embed_dim = int(mcfg.get("embed_dim", 0))
    pretrained = bool(mcfg.get("pretrained", True))
    model = EfficientNetB3Classifier(num_classes=num_classes, pretrained=pretrained, embed_dim=embed_dim)
    model.to(device)

    tcfg = cfg.get("train", {})
    epochs = int(tcfg.get("epochs", 10))
    lr = float(tcfg.get("lr", 3e-4))
    wd = float(tcfg.get("weight_decay", 0.01))
    amp = bool(tcfg.get("amp", True))
    grad_clip = float(tcfg.get("grad_clip_norm", 1.0))
    log_every = int(tcfg.get("log_every", 50))

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    scaler = GradScaler(enabled=amp)
    criterion = nn.CrossEntropyLoss()

    start_epoch = 0
    global_step = 0
    best_acc = -1.0

    if args.resume:
        ckpt = load_checkpoint(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model"], strict=True)
        if ckpt.get("optimizer"):
            optimizer.load_state_dict(ckpt["optimizer"])
        if ckpt.get("scheduler"):
            scheduler.load_state_dict(ckpt["scheduler"])
        if ckpt.get("scaler") and amp:
            scaler.load_state_dict(ckpt["scaler"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        global_step = int(ckpt.get("step", 0))
        best_acc = float(ckpt.get("best_metric", -1.0))

    non_art_id = int(meta["non_art_id"])

    # run_ratio: 通过限制每个 epoch 的 batch 数实现（保持采样器/随机性稳定）
    run_ratio = int(args.run_ratio)
    if run_ratio < 1 or run_ratio > 100:
        raise ValueError("--run_ratio must be in [1,100]")
    max_steps = int(args.max_steps)
    if max_steps < 0:
        raise ValueError("--max_steps must be >= 0")

    for epoch in range(start_epoch, epochs):
        model.train()
        total_batches = len(train_loader)
        if run_ratio < 100:
            total_batches = max(1, int(total_batches * (run_ratio / 100.0)))
        pbar = tqdm(train_loader, desc=f"epoch {epoch+1}/{epochs}", total=total_batches, dynamic_ncols=True)
        running_loss = 0.0

        for it, (x, y) in enumerate(pbar):
            if it >= total_batches:
                break
            if max_steps and global_step >= max_steps:
                break
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=amp):
                out = model(x)
                loss = criterion(out.logits, y)
            scaler.scale(loss).backward()
            if grad_clip and grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            scaler.step(optimizer)
            scaler.update()

            running_loss += float(loss.item())
            global_step += 1

            if global_step % log_every == 0:
                pbar.set_postfix(loss=running_loss / max(1, it + 1), lr=optimizer.param_groups[0]["lr"])

        if max_steps and global_step >= max_steps:
            # 允许在 max_steps 到达后仍进行一次评测并保存 last/best
            pass
        scheduler.step()

        # eval
        report = _evaluate(model, val_loader, device, num_classes=num_classes, non_art_id=non_art_id)
        val_acc = float(report["acc"])

        # 保存 metrics
        cm = report["confusion"]
        torch.save(torch.tensor(cm), str(Path(metrics_dir) / f"confusion_epoch{epoch+1}.pt"))
        with open(str(Path(metrics_dir) / f"val_report_epoch{epoch+1}.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "acc": val_acc,
                    "per_class_acc": report["per_class_acc"],
                    "non_art_precision": report["non_art_precision"],
                    "non_art_recall": report["non_art_recall"],
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        # save last
        last_path = str(Path(ckpt_dir) / "last.pt")
        save_checkpoint(
            last_path,
            CheckpointPayload(
                model=model.state_dict(),
                optimizer=optimizer.state_dict(),
                scheduler=scheduler.state_dict(),
                scaler=scaler.state_dict() if amp else None,
                epoch=epoch,
                step=global_step,
                best_metric=best_acc,
                config=cfg,
            ),
        )

        if val_acc > best_acc:
            best_acc = val_acc
            best_path = str(Path(ckpt_dir) / "best.pt")
            save_checkpoint(
                best_path,
                CheckpointPayload(
                    model=model.state_dict(),
                    optimizer=optimizer.state_dict(),
                    scheduler=scheduler.state_dict(),
                    scaler=scaler.state_dict() if amp else None,
                    epoch=epoch,
                    step=global_step,
                    best_metric=best_acc,
                    config=cfg,
                ),
            )

        print(
            json.dumps(
                {
                    "epoch": epoch + 1,
                    "val_acc": val_acc,
                    "best_acc": best_acc,
                    "non_art_precision": report["non_art_precision"],
                    "non_art_recall": report["non_art_recall"],
                },
                ensure_ascii=False,
            )
        )

        if max_steps and global_step >= max_steps:
            break


if __name__ == "__main__":
    main()


