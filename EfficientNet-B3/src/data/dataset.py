from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


@dataclass(frozen=True)
class ManifestRow:
    image_path: str
    label_id: int
    split: str


class ImageClassificationDataset(Dataset):
    def __init__(self, manifest_csv: str, split: str, transform=None):
        df = pd.read_csv(manifest_csv)
        if "split" in df.columns:
            df = df[df["split"].astype(str) == str(split)]
        if "image_path" not in df.columns or "label_id" not in df.columns:
            raise ValueError("manifest must contain columns: image_path,label_id (and optionally split)")

        self.image_paths = df["image_path"].astype(str).tolist()
        self.labels = df["label_id"].astype(int).tolist()
        self.transform = transform

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[object, int]:
        path = self.image_paths[idx]
        y = int(self.labels[idx])
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, y











