"""Data utilities for ArtQuest-style datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

from torch.utils.data import DataLoader

from models.Gated.datasets import build_artquest_dataloaders
from models.Gated.datasets.combined_dataset import create_balanced_dataloader
from models.Gated.datasets.artquest import ArtQuestSampleDataset, _build_transform as build_artquest_transform
from models.Gated.datasets.vqa_dataset import VQASampleDataset


def build_dataloaders(config: Dict) -> Tuple[DataLoader, DataLoader, Optional[DataLoader], Dict]:
    """Create train/val/test dataloaders based on configuration.
    
    Supports both single-dataset (ArtQuest) and multi-dataset (ArtQuest + VQA) modes.
    """

    data_cfg = config.get("data", {})
    use_multi_dataset = data_cfg.get("use_multi_dataset", False)
    
    if use_multi_dataset:
        return _build_multi_dataloaders(config)
    else:
        return _build_single_dataloaders(config)


def _build_single_dataloaders(config: Dict) -> Tuple[DataLoader, DataLoader, Optional[DataLoader], Dict]:
    """Build dataloaders for single dataset (ArtQuest only)."""
    data_cfg = config.get("data", {})
    dataset_root = Path(data_cfg.get("dataset_root", "/data")).expanduser()
    dataset_name = data_cfg.get("dataset_name", "artquest")

    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root {dataset_root} does not exist")

    train_split = data_cfg.get("train_split", "train.csv")
    val_split = data_cfg.get("val_split", "val.csv")
    test_split = data_cfg.get("test_split", "test.csv")

    if not train_split.endswith(".csv"):
        train_split = f"{train_split}.csv"
    if not val_split.endswith(".csv"):
        val_split = f"{val_split}.csv"
    if test_split and not test_split.endswith(".csv"):
        test_split = f"{test_split}.csv"

    train_csv = data_cfg.get("train_csv")
    val_csv = data_cfg.get("val_csv")
    test_csv = data_cfg.get("test_csv")

    dataset_dir = dataset_root / dataset_name
    train_csv_path = Path(train_csv) if train_csv else dataset_dir / train_split
    val_csv_path = Path(val_csv) if val_csv else dataset_dir / val_split
    test_csv_path = Path(test_csv) if test_csv else dataset_dir / test_split

    image_root = Path(data_cfg.get("image_root") or (dataset_dir / "images"))

    batch_size = int(data_cfg.get("batch_size", 16))
    num_workers = int(data_cfg.get("num_workers", 8))
    limit = data_cfg.get("limit")
    limit = int(limit) if limit is not None else None

    dataset_specific_cfg = {
        "image_size": data_cfg.get("image_size", 384),
        "caption_key": data_cfg.get("caption_key", "question"),
        "answer_key": data_cfg.get("answer_key", "answer"),
        "style_key": data_cfg.get("style_key", "style"),
        "question_id_key": data_cfg.get("question_id_key", "question_id"),
        "image_key": data_cfg.get("image_key", "image"),
    }

    train_loader, val_loader, test_loader, metadata = build_artquest_dataloaders(
        train_csv_path,
        val_csv_path,
        test_csv_path if test_csv_path.exists() else None,
        image_root,
        batch_size=batch_size,
        num_workers=num_workers,
        dataset_cfg=dataset_specific_cfg,
        limit=limit,
    )

    return train_loader, val_loader, test_loader, metadata


def _build_multi_dataloaders(config: Dict) -> Tuple[DataLoader, DataLoader, Optional[DataLoader], Dict]:
    """Build dataloaders for multi-dataset (ArtQuest + VQA) with 1:1 ratio."""
    data_cfg = config.get("data", {})
    
    # ArtQuest dataset paths
    artquest_cfg = data_cfg.get("artquest", {})
    dataset_root = Path(artquest_cfg.get("dataset_root", "/data")).expanduser()
    dataset_name = artquest_cfg.get("dataset_name", "artquest")
    
    train_split = artquest_cfg.get("train_split", "train.csv")
    val_split = artquest_cfg.get("val_split", "val.csv")
    if not train_split.endswith(".csv"):
        train_split = f"{train_split}.csv"
    if not val_split.endswith(".csv"):
        val_split = f"{val_split}.csv"
    
    train_csv = artquest_cfg.get("train_csv")
    val_csv = artquest_cfg.get("val_csv")
    
    dataset_dir = dataset_root / dataset_name
    train_csv_path = Path(train_csv) if train_csv else dataset_dir / train_split
    val_csv_path = Path(val_csv) if val_csv else dataset_dir / val_split
    artquest_image_root = Path(artquest_cfg.get("image_root") or (dataset_dir / "images"))
    
    # VQA dataset paths
    vqa_cfg = data_cfg.get("vqa", {})
    vqa_image_root = Path(vqa_cfg.get("image_root", "/data/lavis/coco/images"))
    train_questions_file = Path(vqa_cfg.get("train_questions_file", "/data/lavis/coco/annotations/v2_OpenEnded_mscoco_train2014_questions.json"))
    train_annotations_file = Path(vqa_cfg.get("train_annotations_file", "/data/lavis/coco/annotations/v2_mscoco_train2014_annotations.json"))
    val_questions_file = Path(vqa_cfg.get("val_questions_file", "/data/lavis/coco/annotations/v2_OpenEnded_mscoco_val2014_questions.json"))
    val_annotations_file = Path(vqa_cfg.get("val_annotations_file", "/data/lavis/coco/annotations/v2_mscoco_val2014_annotations.json"))
    
    batch_size = int(data_cfg.get("batch_size", 16))
    num_workers = int(data_cfg.get("num_workers", 8))
    image_size = int(data_cfg.get("image_size", 384))
    limit = data_cfg.get("limit")
    limit = int(limit) if limit is not None else None
    seed = data_cfg.get("seed", 1337)
    
    # Build ArtQuest datasets
    from models.Gated.datasets.artquest import ArtQuestDatasetConfig
    artquest_cfg_obj = ArtQuestDatasetConfig(
        image_size=image_size,
        caption_key=data_cfg.get("caption_key", "question"),
        answer_key=data_cfg.get("answer_key", "answer"),
        style_key=data_cfg.get("style_key", "style"),
        question_id_key=data_cfg.get("question_id_key", "question_id"),
        image_key=data_cfg.get("image_key", "image"),
    )
    artquest_transform = build_artquest_transform(image_size)
    
    artquest_train_dataset = ArtQuestSampleDataset(
        train_csv_path, artquest_image_root, artquest_cfg_obj, artquest_transform, limit=limit
    )
    artquest_val_dataset = ArtQuestSampleDataset(
        val_csv_path, artquest_image_root, artquest_cfg_obj, artquest_transform, limit=limit
    )
    
    # Build VQA datasets
    from models.Gated.datasets.vqa_dataset import VQADatasetConfig
    vqa_cfg_obj = VQADatasetConfig(image_size=image_size)
    vqa_transform = build_artquest_transform(image_size)  # Same transform
    
    vqa_train_dataset = VQASampleDataset(
        train_questions_file, train_annotations_file, vqa_image_root, vqa_cfg_obj, vqa_transform, limit=limit
    )
    vqa_val_dataset = VQASampleDataset(
        val_questions_file, val_annotations_file, vqa_image_root, vqa_cfg_obj, vqa_transform, limit=limit
    )
    
    # Create balanced dataloaders with 1:1 ratio
    train_loader = create_balanced_dataloader(
        artquest_train_dataset,
        vqa_train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        seed=seed,
    )
    
    val_loader = create_balanced_dataloader(
        artquest_val_dataset,
        vqa_val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        seed=seed,
    )
    
    # Combine metadata (prefer ArtQuest style vocab if available)
    metadata = {
        "style_vocab": artquest_train_dataset.style_vocab,
        "style_to_id": artquest_train_dataset.style_to_id,
    }
    
    return train_loader, val_loader, None, metadata

