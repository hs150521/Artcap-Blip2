#!/usr/bin/env python
"""Train EfficientNet-B3 on the WikiArt dataset for 27 art styles."""
import argparse
import csv
import json
import random
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import unicodedata

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, models, transforms


class SubsetWithTransform(Dataset):
    """A Subset wrapper that applies a transform after fetching the sample."""

    def __init__(self, dataset: Dataset, indices, transform=None):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        image, target = self.dataset[self.indices[idx]]
        if self.transform is not None:
            image = self.transform(image)
        return image, target


def parse_args():
    parser = argparse.ArgumentParser(description="EfficientNet-B3 WikiArt trainer")
    parser.add_argument("--dataset-root", type=Path, default=Path("/data/wikiart"),
                        help="Root folder containing 27 WikiArt style subfolders.")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Where to store checkpoints and logs. If not specified, uses runs/YYYY_MM_DD_HH_MM_SS/")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--val-split", type=float, default=0.1,
                        help="Portion of data held out for validation when no explicit split is provided.")
    parser.add_argument("--subset-ratio", type=float, default=1.0,
                        help="Fraction of the dataset to use (for smoke tests / debugging).")
    parser.add_argument("--mixup-alpha", type=float, default=0.0,
                        help="Set >0.0 to enable MixUp augmentation.")
    parser.add_argument("--cutmix-alpha", type=float, default=0.0,
                        help="Set >0.0 to enable CutMix augmentation.")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--save-every", type=int, default=5,
                        help="Save a checkpoint every N epochs in addition to the best model.")
    parser.add_argument("--image-size", type=int, default=300)
    parser.add_argument("--no-amp", action="store_true",
                        help="Disable mixed precision training.")
    parser.add_argument("--freeze-backbone", action="store_true",
                        help="Freeze EfficientNet feature extractor for linear probing.")
    parser.add_argument("--resume", type=Path, default=None,
                        help="Path to a checkpoint to resume from.")
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_transforms(image_size: int, weights):
    # torchvision weight meta for EfficientNet-B3 does not expose normalization stats,
    # so we fall back to default ImageNet statistics.
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    train_tfms = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.75, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.02)
        ], p=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    eval_tfms = transforms.Compose([
        transforms.Resize(int(image_size * 1.15)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    return train_tfms, eval_tfms


def maybe_mixup(images, targets, alpha):
    if alpha <= 0.0:
        return images, targets, 1.0
    lam = np.random.beta(alpha, alpha)
    batch_size = images.size(0)
    index = torch.randperm(batch_size, device=images.device)
    mixed_images = lam * images + (1 - lam) * images[index]
    targets_a, targets_b = targets, targets[index]
    return mixed_images, (targets_a, targets_b), lam


def cutmix(images, targets, alpha):
    if alpha <= 0.0:
        return images, targets, 1.0
    lam = np.random.beta(alpha, alpha)
    batch_size, _, h, w = images.size()
    index = torch.randperm(batch_size, device=images.device)
    cx = np.random.randint(w)
    cy = np.random.randint(h)
    cut_w = int(w * np.sqrt(1 - lam))
    cut_h = int(h * np.sqrt(1 - lam))
    x1 = np.clip(cx - cut_w // 2, 0, w)
    y1 = np.clip(cy - cut_h // 2, 0, h)
    x2 = np.clip(cx + cut_w // 2, 0, w)
    y2 = np.clip(cy + cut_h // 2, 0, h)
    images[:, :, y1:y2, x1:x2] = images[index, :, y1:y2, x1:x2]
    lam_adjusted = 1 - ((x2 - x1) * (y2 - y1) / (w * h))
    targets_a, targets_b = targets, targets[index]
    return images, (targets_a, targets_b), lam_adjusted


def _normalize_rel_path(path_str: str) -> str:
    normalized = unicodedata.normalize("NFKD", path_str)
    normalized = normalized.encode("ascii", "ignore").decode("ascii")
    return normalized.lower()


def load_indices_from_csv(dataset_root: Path, dataset: datasets.ImageFolder):
    """Build subset index lists from classes.csv if available."""
    csv_path = dataset_root / "classes.csv"
    if not csv_path.exists():
        return None
    rel_to_idx = {}
    norm_rel_to_idx = {}
    for idx, (path, _) in enumerate(dataset.samples):
        try:
            rel_path = Path(path).relative_to(dataset_root).as_posix()
        except ValueError:
            continue
        rel_to_idx[rel_path] = idx
        norm_rel_to_idx[_normalize_rel_path(rel_path)] = idx
    subset_to_indices = defaultdict(list)
    missing_counter = defaultdict(int)
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "subset" not in reader.fieldnames or "filename" not in reader.fieldnames:
            return None
        for row in reader:
            subset = row.get("subset", "").strip().lower()
            filename = row.get("filename", "").strip()
            if not subset or not filename:
                continue
            idx = rel_to_idx.get(filename)
            if idx is None:
                idx = norm_rel_to_idx.get(_normalize_rel_path(filename))
            if idx is None:
                missing_counter[subset] += 1
                continue
            subset_to_indices[subset].append(idx)
    if missing_counter:
        details = ", ".join(f"{key}: {count}" for key, count in sorted(missing_counter.items()))
        print(
            f"Warning: {sum(missing_counter.values())} entries in classes.csv could not be matched to files ({details})."
        )
    return subset_to_indices if subset_to_indices else None


def pick_indices_from_split(subset_indices, seed: int, subset_ratio: float):
    rng = random.Random(seed)
    train_candidates = subset_indices.get("train", [])
    val_candidates = []
    val_key = None
    for key in ("val", "validation", "test"):
        if subset_indices.get(key):
            val_candidates = subset_indices.get(key)
            val_key = key
            break
    ignored = set(subset_indices.keys()) - {"train", "val", "validation", "test"}
    if ignored:
        print(f"Ignoring subsets in classes.csv: {', '.join(sorted(ignored))}")
    if not train_candidates or not val_candidates:
        return None
    train_indices = list(train_candidates)
    val_indices = list(val_candidates)
    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    if subset_ratio < 1.0:
        train_size = max(1, int(len(train_indices) * subset_ratio))
        val_size = max(1, int(len(val_indices) * subset_ratio))
        train_indices = train_indices[:train_size]
        val_indices = val_indices[:val_size]
    return train_indices, val_indices, val_key


def train_epoch(model, loader, criterion, optimizer, scaler, device,
                mixup_alpha=0.0, cutmix_alpha=0.0):
    model.train()
    running_loss = 0.0
    running_correct = 0
    total = 0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        apply_mixup = mixup_alpha > 0.0
        apply_cutmix = cutmix_alpha > 0.0
        lam = 1.0
        target_tuple = targets

        if apply_mixup:
            images, target_tuple, lam = maybe_mixup(images, targets, mixup_alpha)
        if apply_cutmix:
            images, target_tuple, lam = cutmix(images, targets, cutmix_alpha)

        with torch.cuda.amp.autocast(enabled=scaler is not None):
            outputs = model(images)
            if isinstance(target_tuple, tuple):
                targets_a, targets_b = target_tuple
                loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
            else:
                loss = criterion(outputs, targets)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * images.size(0)
        if not isinstance(target_tuple, tuple):
            preds = outputs.argmax(dim=1)
            running_correct += (preds == targets).sum().item()
        total += images.size(0)

    avg_loss = running_loss / total
    acc = running_correct / total if total > 0 else 0.0
    return {"loss": avg_loss, "acc": acc}


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_correct = 0
    total = 0
    class_correct = defaultdict(int)
    class_total = defaultdict(int)
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            outputs = model(images)
            loss = criterion(outputs, targets)
            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            running_correct += (preds == targets).sum().item()
            total += images.size(0)
            for t, p in zip(targets, preds):
                class_total[int(t)] += 1
                if t == p:
                    class_correct[int(t)] += 1
    avg_loss = running_loss / total
    acc = running_correct / total if total > 0 else 0.0
    class_acc = {cls: class_correct[cls] / class_total[cls]
                 if class_total[cls] > 0 else 0.0 for cls in class_total}
    return {"loss": avg_loss, "acc": acc, "per_class_acc": class_acc}


def save_checkpoint(state, path):
    torch.save(state, path)


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory with timestamp if not specified
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        args.output_dir = Path("runs") / timestamp
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {args.output_dir}")

    dataset = datasets.ImageFolder(args.dataset_root, transform=None)
    num_classes = len(dataset.classes)
    if num_classes != 27:
        print(f"Warning: Expected 27 classes but found {num_classes}.")

    train_indices = None
    val_indices = None
    subset_indices = load_indices_from_csv(args.dataset_root, dataset)
    if subset_indices:
        picked = pick_indices_from_split(subset_indices, args.seed, args.subset_ratio)
        if picked is not None:
            train_indices, val_indices, val_key = picked
            val_label = "validation" if val_key in {"val", "validation"} else "test"
            print(
                f"Using classes.csv split with {len(train_indices)} train and {len(val_indices)} {val_label} samples."
            )
        else:
            print("classes.csv found but missing usable train/val/test subsets; falling back to random split.")

    if train_indices is None or val_indices is None:
        total_samples = len(dataset)
        indices = torch.randperm(total_samples).tolist()
        if args.subset_ratio < 1.0:
            subset_size = max(1, int(total_samples * args.subset_ratio))
            indices = indices[:subset_size]
            total_samples = len(indices)
            print(f"Using subset of size {total_samples} samples (ratio={args.subset_ratio}).")
        val_size = max(1, int(total_samples * args.val_split))
        train_size = total_samples - val_size
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        print(f"Train samples: {len(train_indices)}, Val samples: {len(val_indices)}")

    weights = models.EfficientNet_B3_Weights.IMAGENET1K_V1
    train_tfms, eval_tfms = build_transforms(args.image_size, weights)

    base_dataset = datasets.ImageFolder(args.dataset_root, transform=None)
    train_dataset = SubsetWithTransform(base_dataset, train_indices, transform=train_tfms)
    val_dataset = SubsetWithTransform(base_dataset, val_indices, transform=eval_tfms)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    model = models.efficientnet_b3(weights=weights)
    if args.freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                      lr=args.lr, weight_decay=args.weight_decay)
    scaler = None
    if torch.cuda.is_available() and not args.no_amp:
        scaler = torch.cuda.amp.GradScaler()

    start_epoch = 0
    best_acc = 0.0
    history = []

    if args.resume and args.resume.exists():
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        if scaler is not None and "scaler_state" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler_state"])
        start_epoch = checkpoint.get("epoch", 0) + 1
        best_acc = checkpoint.get("best_acc", 0.0)
        print(f"Resumed from {args.resume} at epoch {start_epoch} with best_acc={best_acc:.4f}")

    stats_path = args.output_dir / "training_log.jsonl"

    for epoch in range(start_epoch, args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        train_stats = train_epoch(
            model, train_loader, criterion, optimizer, scaler, device,
            mixup_alpha=args.mixup_alpha, cutmix_alpha=args.cutmix_alpha
        )
        val_stats = evaluate(model, val_loader, criterion, device)
        history.append({
            "epoch": epoch + 1,
            "train_loss": train_stats["loss"],
            "train_acc": train_stats["acc"],
            "val_loss": val_stats["loss"],
            "val_acc": val_stats["acc"],
            "val_per_class_acc": val_stats["per_class_acc"],
            "timestamp": datetime.utcnow().isoformat()
        })
        print(f"train_loss={train_stats['loss']:.4f} train_acc={train_stats['acc']:.4f} "
              f"val_loss={val_stats['loss']:.4f} val_acc={val_stats['acc']:.4f}")

        with stats_path.open("a") as f:
            f.write(json.dumps(history[-1]) + "\n")

        is_best = val_stats["acc"] > best_acc
        if is_best:
            best_acc = val_stats["acc"]
            save_checkpoint({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scaler_state": scaler.state_dict() if scaler is not None else None,
                "best_acc": best_acc,
                "args": vars(args),
            }, args.output_dir / "best.pt")
            print(f"Saved new best model with acc={best_acc:.4f}")

        if (epoch + 1) % args.save_every == 0:
            save_checkpoint({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scaler_state": scaler.state_dict() if scaler is not None else None,
                "best_acc": best_acc,
                "args": vars(args),
            }, args.output_dir / f"checkpoint_epoch_{epoch+1}.pt")

    summary_path = args.output_dir / "summary.json"
    with summary_path.open("w") as f:
        json.dump({
            "best_val_acc": best_acc,
            "epochs": args.epochs,
            "history_file": str(stats_path)
        }, f, indent=2)
    print(f"Training complete. Best val acc: {best_acc:.4f}")


if __name__ == "__main__":
    main()
