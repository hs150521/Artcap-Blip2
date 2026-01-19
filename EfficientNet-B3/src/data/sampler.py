from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, Iterator, List

import numpy as np
from torch.utils.data import Sampler


@dataclass(frozen=True)
class ArtNonArtSampling:
    enabled: bool
    non_art_label_id: int
    # supported: "art_nonart" (legacy) | "class_balanced"
    mode: str = "class_balanced"
    art_to_nonart_ratio: int = 27


class ArtNonArtBatchSampler(Sampler[List[int]]):
    """
    Sampling strategies (configured by ArtNonArtSampling.mode):

    1) mode="class_balanced" (default):
       - epoch 级别近似“类均衡采样”：每个类别在一个 epoch 内出现次数尽量相等（差值<=1），
         通过类内循环索引实现对少数类过采样。

    2) mode="art_nonart" (legacy):
       - 让一个 batch 内 art:non_art ≈ ratio:1。
       - ratio=27 时，一个 batch 默认取 27*k 个艺术样本 + k 个 non_art。
       - 若 batch_size 不能整除 (ratio+1)，会向下取整，剩余位置用艺术样本填充。
    """

    def __init__(
        self,
        labels: List[int],
        batch_size: int,
        cfg: ArtNonArtSampling,
        seed: int = 42,
        drop_last: bool = True,
    ):
        if batch_size < 2:
            raise ValueError("batch_size too small")
        self.labels = labels
        self.batch_size = batch_size
        self.cfg = cfg
        self.seed = seed
        self.drop_last = drop_last

        self.epoch = 0

        # legacy (art/non-art)
        self.art_idx = [i for i, y in enumerate(labels) if int(y) != int(cfg.non_art_label_id)]
        self.non_idx = [i for i, y in enumerate(labels) if int(y) == int(cfg.non_art_label_id)]
        ratio = int(getattr(cfg, "art_to_nonart_ratio", 27))
        self.k = max(1, batch_size // (ratio + 1))  # non_art per batch
        self.non_per_batch = self.k
        self.art_per_batch = batch_size - self.non_per_batch

        # class-balanced
        by_class: Dict[int, List[int]] = {}
        for i, y in enumerate(labels):
            by_class.setdefault(int(y), []).append(i)
        self.by_class = by_class
        self.class_ids = sorted(by_class.keys())
        if cfg.enabled and cfg.mode == "class_balanced" and len(self.class_ids) < 2:
            raise ValueError("class_balanced sampling enabled but only one class present. Check your manifests.")
        if cfg.enabled and cfg.mode == "art_nonart" and (len(self.art_idx) == 0 or len(self.non_idx) == 0):
            raise ValueError("Sampling enabled but art/non_art split is empty. Check your manifests.")

    def set_epoch(self, epoch: int) -> None:
        # mimic DistributedSampler API; train loop can call this to vary shuffling per epoch
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[List[int]]:
        rng = random.Random(self.seed + self.epoch)

        # number of batches per epoch
        num_batches = len(self.labels) // self.batch_size
        if not self.drop_last and (len(self.labels) % self.batch_size) != 0:
            num_batches += 1

        if self.cfg.mode == "class_balanced":
            # shuffle per-class indices
            per_cls = {c: idxs.copy() for c, idxs in self.by_class.items()}
            for c in self.class_ids:
                rng.shuffle(per_cls[c])
            ptr = {c: 0 for c in self.class_ids}

            total = num_batches * self.batch_size
            cls_seq = (self.class_ids * math.ceil(total / max(1, len(self.class_ids))))[:total]
            rng.shuffle(cls_seq)

            for b in range(num_batches):
                batch: List[int] = []
                for c in cls_seq[b * self.batch_size : (b + 1) * self.batch_size]:
                    idxs = per_cls[int(c)]
                    batch.append(idxs[ptr[int(c)] % len(idxs)])
                    ptr[int(c)] += 1
                rng.shuffle(batch)
                yield batch
            return

        # legacy art/non-art
        art = self.art_idx.copy()
        non = self.non_idx.copy()
        rng.shuffle(art)
        rng.shuffle(non)

        a_ptr = 0
        n_ptr = 0
        a_len = len(art)
        n_len = len(non)
        for _ in range(num_batches):
            batch = []
            for _i in range(self.art_per_batch):
                batch.append(art[a_ptr % a_len])
                a_ptr += 1
            for _i in range(self.non_per_batch):
                batch.append(non[n_ptr % n_len])
                n_ptr += 1
            rng.shuffle(batch)
            yield batch

    def __len__(self) -> int:
        return len(self.labels) // self.batch_size if self.drop_last else math.ceil(len(self.labels) / self.batch_size)

















