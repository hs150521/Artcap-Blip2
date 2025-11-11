"""
ArtQuest dataset loader tailored for gated BLIP-2 training.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


@dataclass
class ArtQuestDatasetConfig:
    image_size: int = 384
    caption_key: str = "question"
    answer_key: str = "answer"
    style_key: str = "style"
    question_id_key: str = "question_id"
    image_key: str = "image"


class ArtQuestSampleDataset(Dataset):
    """Dataset that yields tuples compatible with BLIP-2 gated training."""

    def __init__(
        self,
        csv_path: Path,
        image_root: Path,
        cfg: ArtQuestDatasetConfig,
        transform: Optional[transforms.Compose] = None,
        limit: Optional[int] = None,
    ) -> None:
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file {csv_path} not found")
        if not image_root.exists():
            raise FileNotFoundError(f"Image root {image_root} not found")

        self.data = pd.read_csv(csv_path)
        if limit is not None:
            self.data = self.data.head(limit)

        required_cols = {cfg.image_key, cfg.caption_key}
        missing = required_cols.difference(self.data.columns)
        if missing:
            raise ValueError(f"Missing required columns {missing} in {csv_path}")

        self.cfg = cfg
        self.image_root = image_root
        self.transform = transform

        if cfg.style_key in self.data.columns:
            styles = self.data[cfg.style_key].fillna("unknown").astype(str).tolist()
        else:
            styles = ["unknown"] * len(self.data)
        self.style_vocab = sorted(set(styles))
        self.style_to_id = {style: idx for idx, style in enumerate(self.style_vocab)}

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.data.iloc[idx]

        image_name = str(row[self.cfg.image_key])
        image_path = self._resolve_image_path(image_name)

        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        text_input = str(row[self.cfg.caption_key])
        target = str(row.get(self.cfg.answer_key, ""))
        style_label = str(row.get(self.cfg.style_key, "unknown"))
        style_id = self.style_to_id.get(style_label, -1)

        question_id = row.get(self.cfg.question_id_key, idx)
        if pd.isna(question_id):
            question_id = idx

        return {
            "image": image,
            "text_input": text_input,
            "answer": target,
            "style": style_label,
            "style_id": style_id,
            "question_id": int(question_id),
            "image_path": str(image_path),
        }

    def _resolve_image_path(self, image_name: str) -> Path:
        direct_path = self.image_root / image_name
        if direct_path.exists():
            return direct_path

        for subdir in ["train", "val", "test", "images"]:
            candidate = self.image_root / subdir / image_name
            if candidate.exists():
                return candidate

        raise FileNotFoundError(f"Image {image_name} not found under {self.image_root}")


def _build_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711],
            ),
        ]
    )


def build_artquest_dataloaders(
    train_csv: Path,
    val_csv: Path,
    test_csv: Optional[Path],
    image_root: Path,
    batch_size: int,
    num_workers: int,
    dataset_cfg: Optional[Dict] = None,
    limit: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader], Dict]:
    cfg = ArtQuestDatasetConfig(**(dataset_cfg or {}))
    transform = _build_transform(cfg.image_size)

    train_dataset = ArtQuestSampleDataset(train_csv, image_root, cfg, transform, limit=limit)
    val_dataset = ArtQuestSampleDataset(val_csv, image_root, cfg, transform, limit=limit)
    test_dataset = (
        ArtQuestSampleDataset(test_csv, image_root, cfg, transform, limit=limit)
        if test_csv and test_csv.exists()
        else None
    )

    collate_fn = _default_collate_fn

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    test_loader = (
        DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
        )
        if test_dataset is not None
        else None
    )

    metadata = {
        "style_vocab": train_dataset.style_vocab,
        "style_to_id": train_dataset.style_to_id,
    }
    return train_loader, val_loader, test_loader, metadata


def _default_collate_fn(batch):
    images = torch.stack([item["image"] for item in batch], dim=0)
    text_inputs = [item["text_input"] for item in batch]
    answers = [item["answer"] for item in batch]
    style_ids = torch.tensor([item["style_id"] for item in batch], dtype=torch.long)
    question_ids = torch.tensor([item["question_id"] for item in batch], dtype=torch.long)
    image_paths = [item["image_path"] for item in batch]

    return {
        "image": images,
        "text_input": text_inputs,
        "answer": answers,
        "style_id": style_ids,
        "question_id": question_ids,
        "image_path": image_paths,
    }



