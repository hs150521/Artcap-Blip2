"""
ArtQuest dataset loader for KV training.
"""

import pandas as pd
from pathlib import Path
from typing import Optional

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class ArtQuestDataset(Dataset):
    """Dataset for ArtQuest CSV format compatible with BLIP-2 KV training."""

    def __init__(
        self,
        csv_path: Path,
        image_root: Path,
        image_size: int = 224,
        limit: Optional[int] = None,
    ):
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file {csv_path} not found")
        if not image_root.exists():
            raise FileNotFoundError(f"Image root {image_root} not found")

        self.data = pd.read_csv(csv_path)
        if limit is not None:
            self.data = self.data.head(limit)

        self.image_root = image_root
        self.image_size = image_size
        
        # Build transform
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711],
            ),
        ])

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        row = self.data.iloc[idx]

        image_name = str(row.get("image", row.get("image_path", "")))
        image_path = self._resolve_image_path(image_name)

        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        text_input = str(row.get("question", row.get("text_input", "")))
        answer = str(row.get("answer", ""))

        return {
            "image": image,
            "text_input": text_input,
            "answer": answer,
            "question_id": row.get("question_id", idx),
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


def get_artquest_dataloaders(
    train_data_path: str,
    val_data_path: str,
    test_data_path: Optional[str] = None,
    image_root: str = "",
    cache_path: Optional[str] = None,
    batch_size: int = 4,
    num_workers: int = 4,
    image_size: int = 224,
    limit: Optional[int] = None,
):
    """Build ArtQuest dataloaders."""
    train_dataset = ArtQuestDataset(
        Path(train_data_path),
        Path(image_root),
        image_size=image_size,
        limit=limit,
    )
    val_dataset = ArtQuestDataset(
        Path(val_data_path),
        Path(image_root),
        image_size=image_size,
        limit=limit,
    )
    test_dataset = (
        ArtQuestDataset(
            Path(test_data_path),
            Path(image_root),
            image_size=image_size,
            limit=limit,
        )
        if test_data_path and Path(test_data_path).exists()
        else None
    )

    def collate_fn(batch):
        images = torch.stack([item["image"] for item in batch], dim=0)
        text_inputs = [item["text_input"] for item in batch]
        answers = [item["answer"] for item in batch]
        question_ids = [item.get("question_id", idx) for idx, item in enumerate(batch)]
        image_paths = [item.get("image_path", "") for item in batch]

        return {
            "image": images,
            "text_input": text_inputs,
            "answer": answers,
            "question_id": question_ids,
            "image_path": image_paths,
        }

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

    return train_loader, val_loader, test_loader



