from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Iterator, List

import numpy as np
from torch.utils.data import Sampler


@dataclass(frozen=True)
class ArtNonArtSampling:
    enabled: bool
    art_to_nonart_ratio: int
    non_art_label_id: int


class ArtNonArtBatchSampler(Sampler[List[int]]):
    """
    让一个 batch 内 art:non_art ≈ ratio:1。
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

        self.art_idx = [i for i, y in enumerate(labels) if int(y) != int(cfg.non_art_label_id)]
        self.non_idx = [i for i, y in enumerate(labels) if int(y) == int(cfg.non_art_label_id)]
        if cfg.enabled and (len(self.art_idx) == 0 or len(self.non_idx) == 0):
            raise ValueError("Sampling enabled but art/non_art split is empty. Check your manifests.")

        ratio = int(cfg.art_to_nonart_ratio)
        # k = non_art per batch
        self.k = max(1, batch_size // (ratio + 1))
        self.non_per_batch = self.k
        self.art_per_batch = batch_size - self.non_per_batch

    def __iter__(self) -> Iterator[List[int]]:
        rng = random.Random(self.seed)

        art = self.art_idx.copy()
        non = self.non_idx.copy()
        rng.shuffle(art)
        rng.shuffle(non)

        # 通过循环索引实现“过采样”
        a_ptr = 0
        n_ptr = 0
        a_len = len(art)
        n_len = len(non)

        # 估计 epoch 中 batch 数：以艺术样本为主（更贴近 27:1 的训练动机）
        num_batches = len(self.labels) // self.batch_size
        if not self.drop_last and (len(self.labels) % self.batch_size) != 0:
            num_batches += 1

        for _ in range(num_batches):
            batch: List[int] = []
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
















