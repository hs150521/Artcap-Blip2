"""
VQA dataset loader for VQAv2 format (JSON) - KV directory version.
"""

import json
from pathlib import Path
from typing import Dict, Optional

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class VQADataset(Dataset):
    """Dataset for VQAv2 JSON format compatible with BLIP-2 KV training."""

    def __init__(
        self,
        questions_file: Path,
        annotations_file: Optional[Path],
        image_root: Path,
        image_size: int = 224,
        limit: Optional[int] = None,
    ):
        if not questions_file.exists():
            raise FileNotFoundError(f"Questions file {questions_file} not found")
        if not image_root.exists():
            raise FileNotFoundError(f"Image root {image_root} does not exist")

        # Load questions
        with open(questions_file, "r", encoding="utf-8") as f:
            questions_data = json.load(f)
        
        # Load annotations if provided
        annotations_dict = {}
        if annotations_file and annotations_file.exists():
            with open(annotations_file, "r", encoding="utf-8") as f:
                annotations_data = json.load(f)
                # Create mapping from question_id to answer
                for ann in annotations_data.get("annotations", []):
                    qid = ann["question_id"]
                    # Use multiple_choice_answer as primary, fallback to answers[0]
                    answer = ann.get("multiple_choice_answer", "")
                    if not answer and ann.get("answers"):
                        answer = ann["answers"][0].get("answer", "")
                    annotations_dict[qid] = answer

        # Build samples list
        self.samples = []
        for q in questions_data.get("questions", []):
            question_id = q["question_id"]
            image_id = q["image_id"]
            question = q["question"]
            answer = annotations_dict.get(question_id, "")
            
            self.samples.append({
                "question_id": question_id,
                "image_id": image_id,
                "question": question,
                "answer": answer,
            })

        if limit is not None:
            self.samples = self.samples[:limit]

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
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]

        image_id = sample["image_id"]
        image_path = self._resolve_image_path(image_id)

        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        text_input = str(sample["question"])
        answer = str(sample["answer"])

        return {
            "image": image,
            "text_input": text_input,
            "answer": answer,
            "question_id": sample["question_id"],
            "image_path": str(image_path),
        }

    def _resolve_image_path(self, image_id: int) -> Path:
        """Resolve COCO image path from image_id."""
        # COCO images are named as: COCO_train2014_000000000123.jpg or COCO_val2014_000000000123.jpg
        # Try different splits
        for split in ["train2014", "val2014", "test2014"]:
            # Format: COCO_{split}_{image_id:012d}.jpg
            image_name = f"COCO_{split}_{image_id:012d}.jpg"
            candidate = self.image_root / split / image_name
            if candidate.exists():
                return candidate
        
        # Fallback: try direct image root
        for split in ["train2014", "val2014", "test2014"]:
            image_name = f"COCO_{split}_{image_id:012d}.jpg"
            candidate = self.image_root / image_name
            if candidate.exists():
                return candidate

        raise FileNotFoundError(f"Image {image_id} not found under {self.image_root}")


def get_vqa_dataloaders(
    train_questions_file: str,
    train_annotations_file: str,
    val_questions_file: str,
    val_annotations_file: str,
    image_root: str,
    batch_size: int,
    num_workers: int,
    image_size: int = 224,
    limit: Optional[int] = None,
):
    """Build VQA dataloaders compatible with ArtQuest format."""
    from torch.utils.data import DataLoader

    train_dataset = VQADataset(
        Path(train_questions_file),
        Path(train_annotations_file),
        Path(image_root),
        image_size=image_size,
        limit=limit,
    )
    val_dataset = VQADataset(
        Path(val_questions_file),
        Path(val_annotations_file),
        Path(image_root),
        image_size=image_size,
        limit=limit,
    )

    def collate_fn(batch):
        images = torch.stack([item["image"] for item in batch], dim=0)
        text_inputs = [item["text_input"] for item in batch]
        answers = [item["answer"] for item in batch]
        question_ids = [item["question_id"] for item in batch]
        image_paths = [item["image_path"] for item in batch]

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

    return train_loader, val_loader, None



