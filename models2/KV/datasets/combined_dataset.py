"""
Combined dataset for ArtQuest and VQA with 1:1 ratio balancing - KV directory version.
"""

import random
from typing import Dict, Optional

import torch
from torch.utils.data import Dataset


class CombinedArtQuestVQADataset(Dataset):
    """Combined dataset that balances ArtQuest and VQA with 1:1 ratio per batch."""

    def __init__(
        self,
        artquest_dataset: Dataset,
        vqa_dataset: Dataset,
        artquest_indices: Optional[list] = None,
        vqa_indices: Optional[list] = None,
    ):
        self.artquest_dataset = artquest_dataset
        self.vqa_dataset = vqa_dataset

        # Use provided indices or all samples
        if artquest_indices is None:
            artquest_indices = list(range(len(artquest_dataset)))
        if vqa_indices is None:
            vqa_indices = list(range(len(vqa_dataset)))

        self.artquest_indices = artquest_indices
        self.vqa_indices = vqa_indices

        # Balance: ensure 1:1 ratio
        # Use the minimum of both datasets to ensure equal representation
        min_count = min(len(self.artquest_indices), len(self.vqa_indices))
        
        # Take equal numbers from each dataset
        self.artquest_indices = self.artquest_indices[:min_count]
        self.vqa_indices = self.vqa_indices[:min_count]

        self.total_len = len(self.artquest_indices) + len(self.vqa_indices)
        
        # Create interleaved indices for 1:1 sampling
        # This ensures each batch gets equal samples from both datasets
        self.indices = []
        for i in range(min_count):
            self.indices.append(("artquest", i))
            self.indices.append(("vqa", i))
        
        # Shuffle to randomize order while maintaining 1:1 ratio
        random.shuffle(self.indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict:
        dataset_type, local_idx = self.indices[idx]
        
        if dataset_type == "artquest":
            return self.artquest_dataset[self.artquest_indices[local_idx]]
        else:  # vqa
            return self.vqa_dataset[self.vqa_indices[local_idx]]


def create_balanced_dataloader(
    artquest_dataset: Dataset,
    vqa_dataset: Dataset,
    batch_size: int,
    num_workers: int,
    shuffle: bool = True,
    seed: Optional[int] = None,
) -> torch.utils.data.DataLoader:
    """Create a DataLoader with 1:1 balanced sampling from both datasets."""
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)

    combined_dataset = CombinedArtQuestVQADataset(artquest_dataset, vqa_dataset)

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

    return torch.utils.data.DataLoader(
        combined_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )



