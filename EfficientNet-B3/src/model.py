from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import timm
import torch
import torch.nn as nn


@dataclass(frozen=True)
class ModelOutputs:
    logits: torch.Tensor
    embed: Optional[torch.Tensor] = None


class EfficientNetB3Classifier(nn.Module):
    """
    EfficientNet-B3 分类器：
    - logits: [B, num_classes]
    - embed (可选): [B, embed_dim]，作为风格低维嵌入 s
    """

    def __init__(self, num_classes: int = 28, pretrained: bool = True, embed_dim: int = 0):
        super().__init__()
        # num_classes=0 使 timm 返回特征
        self.backbone = timm.create_model("efficientnet_b3", pretrained=pretrained, num_classes=0, global_pool="avg")

        feat_dim = getattr(self.backbone, "num_features", None)
        if feat_dim is None:
            raise RuntimeError("timm backbone has no num_features")

        self.classifier = nn.Linear(feat_dim, num_classes)
        self.embed_dim = int(embed_dim)
        self.proj = nn.Linear(feat_dim, self.embed_dim) if self.embed_dim > 0 else None

    def forward(self, x: torch.Tensor) -> ModelOutputs:
        feat = self.backbone(x)  # [B, feat_dim]
        logits = self.classifier(feat)
        embed = self.proj(feat) if self.proj is not None else None
        return ModelOutputs(logits=logits, embed=embed)

















