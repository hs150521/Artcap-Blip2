#!/usr/bin/env python
"""Train EfficientNet-B3 on WikiArt (27 art styles) + COCO (1 non-art class) = 28 classes."""
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
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset, Subset, ConcatDataset
from torchvision import datasets, models, transforms
from PIL import Image
from tqdm import tqdm


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


class COCODataset(Dataset):
    """Dataset for COCO images, all labeled as non-art class (class index 27)."""

    def __init__(self, coco_root: Path, transform=None):
        self.coco_root = Path(coco_root)
        self.transform = transform
        self.samples = []
        
        # Load images from train2014, val2014, test2014 subfolders
        for split in ["train2014", "val2014", "test2014"]:
            split_dir = self.coco_root / "images" / split
            if split_dir.exists():
                for img_path in split_dir.glob("*.jpg"):
                    self.samples.append(img_path)
        
        print(f"Found {len(self.samples)} COCO images")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = self.samples[idx]
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return a black image as fallback
            image = Image.new("RGB", (224, 224), (0, 0, 0))
        
        if self.transform is not None:
            image = self.transform(image)
        
        # All COCO images are labeled as class 27 (non-art)
        target = 27
        return image, target


class CombinedWikiArtCOCODataset(Dataset):
    """Combined dataset that balances WikiArt and COCO with 27:1 ratio."""

    def __init__(self, wikiart_dataset: Dataset, coco_dataset: Dataset, 
                 wikiart_indices=None, coco_indices=None, transform=None):
        self.wikiart_dataset = wikiart_dataset
        self.coco_dataset = coco_dataset
        self.transform = transform
        
        # Use provided indices or all samples
        if wikiart_indices is None:
            wikiart_indices = list(range(len(wikiart_dataset)))
        if coco_indices is None:
            coco_indices = list(range(len(coco_dataset)))
        
        self.wikiart_indices = wikiart_indices
        self.coco_indices = coco_indices
        
        # Balance: ensure 27:1 ratio (WikiArt:COCO)
        # For each COCO sample, we want 27 WikiArt samples
        # So: WikiArt_count = 27 * COCO_count
        # Or: COCO_count = WikiArt_count / 27
        max_coco_count = len(coco_indices)
        max_wikiart_count = len(wikiart_indices)
        
        # Calculate how many COCO samples we can use based on available WikiArt
        # COCO_count = min(max_coco_count, max_wikiart_count // 27)
        coco_count = min(max_coco_count, max_wikiart_count // 27)
        wikiart_count = coco_count * 27
        
        # Ensure we don't exceed available samples
        coco_count = min(coco_count, max_coco_count)
        wikiart_count = min(wikiart_count, max_wikiart_count)
        
        self.wikiart_indices = wikiart_indices[:wikiart_count]
        self.coco_indices = coco_indices[:coco_count]
        
        self.total_len = len(self.wikiart_indices) + len(self.coco_indices)
        print(f"Combined dataset: {len(self.wikiart_indices)} WikiArt + {len(self.coco_indices)} COCO = {self.total_len} total (ratio {len(self.wikiart_indices)/max(1, len(self.coco_indices)):.1f}:1)")

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        if idx < len(self.wikiart_indices):
            # WikiArt sample
            image, target = self.wikiart_dataset[self.wikiart_indices[idx]]
        else:
            # COCO sample (adjust index)
            coco_idx = idx - len(self.wikiart_indices)
            image, target = self.coco_dataset[self.coco_indices[coco_idx]]
        
        if self.transform is not None:
            image = self.transform(image)
        
        return image, target


def parse_args():
    parser = argparse.ArgumentParser(description="EfficientNet-B3 WikiArt+COCO trainer (28 classes)")
    parser.add_argument("--wikiart-root", type=Path, default=Path("/data/wikiart"),
                        help="Root folder containing 27 WikiArt style subfolders.")
    parser.add_argument("--coco-root", type=Path, default=Path("/data/lavis/coco"),
                        help="Root folder containing COCO images.")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Where to store checkpoints and logs. If not specified, uses runs/YYYY_MM_DD_HH_MM_SS/")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--val-split", type=float, default=0.1,
                        help="Portion of data held out for validation when no explicit split is provided.")
    parser.add_argument("--subset-ratio", type=float, default=1.0,
                        help="Fraction of the dataset to use (for smoke tests / debugging).")
    parser.add_argument("--mixup-alpha", type=float, default=0.0,
                        help="Set >0.0 to enable MixUp augmentation.")
    parser.add_argument("--cutmix-alpha", type=float, default=1.0,
                        help="Set >0.0 to enable CutMix augmentation.")
    parser.add_argument("--dropout", type=float, default=0.3,
                        help="Dropout probability for classifier head.")
    parser.add_argument("--lr-scheduler", type=str, default="cosine",
                        choices=["cosine", "plateau", "none"],
                        help="Learning rate scheduler type.")
    parser.add_argument("--warmup-epochs", type=int, default=0,
                        help="Number of warmup epochs for cosine scheduler.")
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
    args = parser.parse_args()
    
    # Validate label_smoothing range
    if not (0 <= args.label_smoothing < 1):
        raise ValueError(f"label_smoothing must be in [0, 1), got {args.label_smoothing}")
    
    # Validate dropout range
    if not (0 <= args.dropout < 1):
        raise ValueError(f"dropout must be in [0, 1), got {args.dropout}")
    
    return args


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
    pbar = tqdm(loader, desc="Training", unit="batch")
    for images, targets in pbar:
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

        with torch.amp.autocast('cuda', enabled=scaler is not None):
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
        
        # Calculate accuracy: use original targets (not mixed) for accuracy calculation
        # For MixUp/CutMix, use targets_a (primary label) for accuracy
        if isinstance(target_tuple, tuple):
            targets_for_acc = target_tuple[0]  # Use primary label for accuracy
        else:
            targets_for_acc = targets
        
        preds = outputs.argmax(dim=1)
        running_correct += (preds == targets_for_acc).sum().item()
        total += images.size(0)
        
        # Update progress bar
        current_loss = running_loss / total
        current_acc = running_correct / total if total > 0 else 0.0
        pbar.set_postfix({"loss": f"{current_loss:.4f}", "acc": f"{current_acc:.4f}"})

    avg_loss = running_loss / total
    acc = running_correct / total if total > 0 else 0.0
    pbar.close()
    return {"loss": avg_loss, "acc": acc}


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_correct = 0
    total = 0
    class_correct = defaultdict(int)
    class_total = defaultdict(int)
    pbar = tqdm(loader, desc="Validating", unit="batch")
    with torch.no_grad():
        for images, targets in pbar:
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
            
            # Update progress bar
            current_loss = running_loss / total
            current_acc = running_correct / total if total > 0 else 0.0
            pbar.set_postfix({"loss": f"{current_loss:.4f}", "acc": f"{current_acc:.4f}"})
    
    avg_loss = running_loss / total
    acc = running_correct / total if total > 0 else 0.0
    class_acc = {cls: class_correct[cls] / class_total[cls]
                 if class_total[cls] > 0 else 0.0 for cls in class_total}
    pbar.close()
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

    # Load WikiArt dataset
    print(f"Loading WikiArt dataset from {args.wikiart_root}")
    wikiart_dataset = datasets.ImageFolder(args.wikiart_root, transform=None)
    num_wikiart_classes = len(wikiart_dataset.classes)
    print(f"WikiArt: {num_wikiart_classes} classes, {len(wikiart_dataset)} samples")
    
    if num_wikiart_classes != 27:
        print(f"Warning: Expected 27 WikiArt classes but found {num_wikiart_classes}.")

    # Load COCO dataset
    print(f"Loading COCO dataset from {args.coco_root}")
    coco_dataset = COCODataset(args.coco_root, transform=None)
    num_coco_samples = len(coco_dataset)
    
    # Total classes: 27 WikiArt styles + 1 non-art class = 28
    num_classes = 28
    
    # Handle WikiArt train/val split
    wikiart_train_indices = None
    wikiart_val_indices = None
    subset_indices = load_indices_from_csv(args.wikiart_root, wikiart_dataset)
    if subset_indices:
        picked = pick_indices_from_split(subset_indices, args.seed, args.subset_ratio)
        if picked is not None:
            wikiart_train_indices, wikiart_val_indices, val_key = picked
            val_label = "validation" if val_key in {"val", "validation"} else "test"
            print(
                f"Using classes.csv split: {len(wikiart_train_indices)} train and {len(wikiart_val_indices)} {val_label} WikiArt samples."
            )
        else:
            print("classes.csv found but missing usable train/val/test subsets; falling back to random split.")

    if wikiart_train_indices is None or wikiart_val_indices is None:
        total_wikiart = len(wikiart_dataset)
        indices = torch.randperm(total_wikiart).tolist()
        if args.subset_ratio < 1.0:
            subset_size = max(1, int(total_wikiart * args.subset_ratio))
            indices = indices[:subset_size]
            total_wikiart = len(indices)
            print(f"Using subset of WikiArt: {total_wikiart} samples (ratio={args.subset_ratio}).")
        val_size = max(1, int(total_wikiart * args.val_split))
        train_size = total_wikiart - val_size
        wikiart_train_indices = indices[:train_size]
        wikiart_val_indices = indices[train_size:]
        print(f"WikiArt - Train: {len(wikiart_train_indices)}, Val: {len(wikiart_val_indices)}")

    # Split COCO dataset to match WikiArt split ratio
    # Maintain 27:1 ratio (WikiArt:COCO) in both train and val sets
    total_coco = len(coco_dataset)
    coco_indices = list(range(total_coco))
    random.Random(args.seed).shuffle(coco_indices)
    
    wikiart_train_count = len(wikiart_train_indices)
    wikiart_val_count = len(wikiart_val_indices)
    total_wikiart = wikiart_train_count + wikiart_val_count
    
    # Calculate COCO split ratio to match WikiArt
    if total_wikiart > 0:
        train_ratio = wikiart_train_count / total_wikiart
        val_ratio = wikiart_val_count / total_wikiart
    else:
        train_ratio = 1.0 - args.val_split
        val_ratio = args.val_split
    
    # Split COCO with same ratio
    coco_train_size = int(total_coco * train_ratio)
    coco_val_size = int(total_coco * val_ratio)
    
    coco_train_indices = coco_indices[:coco_train_size]
    coco_val_indices = coco_indices[coco_train_size:coco_train_size + coco_val_size]
    
    print(f"COCO - Train: {len(coco_train_indices)}, Val: {len(coco_val_indices)}")
    
    # Balance: ensure 27:1 ratio (WikiArt:COCO) by taking appropriate counts for each split
    # For train: COCO_count = min(available_COCO, WikiArt_count // 27)
    # For val: COCO_count = min(available_COCO, WikiArt_count // 27)
    final_train_coco_count = min(len(coco_train_indices), wikiart_train_count // 27)
    final_val_coco_count = min(len(coco_val_indices), wikiart_val_count // 27)
    
    # Calculate corresponding WikiArt counts to maintain 27:1 ratio
    final_train_wikiart_count = final_train_coco_count * 27
    final_val_wikiart_count = final_val_coco_count * 27
    
    # Ensure we don't exceed available samples
    final_train_wikiart_count = min(final_train_wikiart_count, wikiart_train_count)
    final_val_wikiart_count = min(final_val_wikiart_count, wikiart_val_count)
    
    # Recalculate COCO counts based on actual WikiArt counts to maintain exact ratio
    final_train_coco_count = final_train_wikiart_count // 27
    final_val_coco_count = final_val_wikiart_count // 27
    
    wikiart_train_indices = wikiart_train_indices[:final_train_wikiart_count]
    wikiart_val_indices = wikiart_val_indices[:final_val_wikiart_count]
    coco_train_indices = coco_train_indices[:final_train_coco_count]
    coco_val_indices = coco_val_indices[:final_val_coco_count]
    
    print(f"Final balanced split (27:1 WikiArt:COCO):")
    print(f"  Train: {len(wikiart_train_indices)} WikiArt + {len(coco_train_indices)} COCO = {len(wikiart_train_indices) + len(coco_train_indices)}")
    print(f"  Val: {len(wikiart_val_indices)} WikiArt + {len(coco_val_indices)} COCO = {len(wikiart_val_indices) + len(coco_val_indices)}")

    weights = models.EfficientNet_B3_Weights.IMAGENET1K_V1
    train_tfms, eval_tfms = build_transforms(args.image_size, weights)

    # Create combined datasets
    train_dataset = CombinedWikiArtCOCODataset(
        wikiart_dataset, coco_dataset,
        wikiart_indices=wikiart_train_indices,
        coco_indices=coco_train_indices,
        transform=train_tfms
    )
    val_dataset = CombinedWikiArtCOCODataset(
        wikiart_dataset, coco_dataset,
        wikiart_indices=wikiart_val_indices,
        coco_indices=coco_val_indices,
        transform=eval_tfms
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    # Build model with 28 classes
    model = models.efficientnet_b3(weights=weights)
    if args.freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False
    in_features = model.classifier[1].in_features
    
    # Modify classifier to include/update dropout
    # EfficientNet classifier is typically Sequential(Dropout, Linear)
    if isinstance(model.classifier, nn.Sequential) and len(model.classifier) >= 2:
        # If classifier already has dropout, update its p value
        if isinstance(model.classifier[0], nn.Dropout):
            model.classifier[0].p = args.dropout
            model.classifier[1] = nn.Linear(in_features, num_classes)
        else:
            # Create new Sequential with dropout
            model.classifier = nn.Sequential(
                nn.Dropout(p=args.dropout),
                nn.Linear(in_features, num_classes)
            )
    else:
        # Fallback: create new Sequential with dropout
        model.classifier = nn.Sequential(
            nn.Dropout(p=args.dropout),
            nn.Linear(in_features, num_classes)
        )
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                      lr=args.lr, weight_decay=args.weight_decay)
    scaler = None
    if torch.cuda.is_available() and not args.no_amp:
        scaler = torch.amp.GradScaler('cuda')
    
    # Create learning rate scheduler
    scheduler = None
    if args.lr_scheduler == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif args.lr_scheduler == "plateau":
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

    start_epoch = 0
    best_acc = 0.0
    history = []

    if args.resume and args.resume.exists():
        # PyTorch 2.6+ requires weights_only=False when loading checkpoints with Path objects
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        if scaler is not None and "scaler_state" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler_state"])
        if scheduler is not None and "scheduler_state" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state"])
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

        # Update learning rate scheduler
        if scheduler is not None:
            if args.lr_scheduler == "cosine":
                scheduler.step()
            elif args.lr_scheduler == "plateau":
                scheduler.step(val_stats['loss'])
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Current learning rate: {current_lr:.6f}")

        is_best = val_stats["acc"] > best_acc
        if is_best:
            best_acc = val_stats["acc"]
            save_checkpoint({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scaler_state": scaler.state_dict() if scaler is not None else None,
                "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
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
                "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
                "best_acc": best_acc,
                "args": vars(args),
            }, args.output_dir / f"checkpoint_epoch_{epoch+1}.pt")

    summary_path = args.output_dir / "summary.json"
    with summary_path.open("w") as f:
        json.dump({
            "best_val_acc": best_acc,
            "epochs": args.epochs,
            "num_classes": num_classes,
            "history_file": str(stats_path)
        }, f, indent=2)
    print(f"Training complete. Best val acc: {best_acc:.4f}")


if __name__ == "__main__":
    main()

