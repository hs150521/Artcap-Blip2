"""
Adapter utilities for loading the EfficientNet-B3 28-class style classifier and
projecting its features into the KV modulation control space.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torchvision.models import efficientnet_b3


_DEFAULT_CHECKPOINT = Path(
    "/data/artcap-blip2-4/models/efficientnet-28/runs/best.pt"
)


def _unwrap_state_dict(state):
    """Recursively unwrap common checkpoint containers."""
    if not isinstance(state, dict):
        return state
    for key in ("state_dict", "model_state_dict", "model", "module", "model_state"):
        if key in state and isinstance(state[key], dict):
            return _unwrap_state_dict(state[key])
    return state


class EfficientNetAdapter(nn.Module):
    """Wrap a pretrained EfficientNet-B3 with frozen weights and feature projection."""

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        output_dim: int = 768,
        convert_from_blip_norm: bool = True,
        enable_feature_grad: bool = False,
    ) -> None:
        super().__init__()
        ckpt_path = Path(checkpoint_path or _DEFAULT_CHECKPOINT).expanduser()
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"EfficientNet checkpoint not found at {ckpt_path}. "
                "Set config.paths.efficientnet_checkpoint to a valid file."
            )

        self.base_model = self._load_backbone(ckpt_path)
        self.base_model.eval()
        if not enable_feature_grad:
            for param in self.base_model.parameters():
                param.requires_grad = False

        feat_dim = self._infer_feature_dim(self.base_model)
        self.proj = nn.Linear(feat_dim, output_dim)
        nn.init.trunc_normal_(self.proj.weight, std=0.02)
        nn.init.zeros_(self.proj.bias)

        self.convert_from_blip_norm = convert_from_blip_norm

        # Register mean/std buffers for BLIP -> EfficientNet normalization conversion
        blip_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073])
        blip_std = torch.tensor([0.26862954, 0.26130258, 0.27577711])
        eff_mean = torch.tensor([0.485, 0.456, 0.406])
        eff_std = torch.tensor([0.229, 0.224, 0.225])
        self.register_buffer("blip_mean", blip_mean.view(1, 3, 1, 1), persistent=False)
        self.register_buffer("blip_std", blip_std.view(1, 3, 1, 1), persistent=False)
        self.register_buffer("eff_mean", eff_mean.view(1, 3, 1, 1), persistent=False)
        self.register_buffer("eff_std", eff_std.view(1, 3, 1, 1), persistent=False)

    @staticmethod
    def _load_backbone(ckpt_path: Path) -> nn.Module:
        # Use weights_only=False for PyTorch 2.6+ compatibility
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state = _unwrap_state_dict(state)

        if not isinstance(state, dict):
            raise ValueError(
                f"Unsupported EfficientNet checkpoint format at {ckpt_path}; expected a state dict."
            )

        # Create EfficientNet with 28 classes (27 art + 1 non-art)
        model = efficientnet_b3(weights=None, num_classes=28)

        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            print(
                f"[EfficientNetAdapter] Missing keys when loading checkpoint ({len(missing)}): {missing[:10]}"
            )
        if unexpected:
            print(
                f"[EfficientNetAdapter] Unexpected keys when loading checkpoint ({len(unexpected)}): {unexpected[:10]}"
            )

        # Replace classifier with identity to expose penultimate features
        if hasattr(model, "classifier"):
            in_features = None
            if isinstance(model.classifier, nn.Sequential):
                for layer in model.classifier:
                    if isinstance(layer, nn.Linear):
                        in_features = layer.in_features
                        break
            elif isinstance(model.classifier, nn.Linear):
                in_features = model.classifier.in_features
            model.classifier = nn.Identity()
            if in_features is not None:
                model.feature_dim = in_features

        return model.float()

    @staticmethod
    def _infer_feature_dim(model: nn.Module) -> int:
        if hasattr(model, "feature_dim"):
            return int(model.feature_dim)
        for module in model.modules():
            if isinstance(module, nn.Linear):
                return module.in_features
        raise AttributeError("Unable to infer EfficientNet feature dimension from checkpoint.")

    def _convert_from_blip(self, images: torch.Tensor) -> torch.Tensor:
        images = images * self.blip_std.to(images.dtype) + self.blip_mean.to(images.dtype)
        images = torch.clamp(images, 0.0, 1.0)
        images = (images - self.eff_mean.to(images.dtype)) / self.eff_std.to(images.dtype)
        return images

    def forward(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.convert_from_blip_norm:
            images = self._convert_from_blip(images)

        backbone_param = next(self.base_model.parameters())
        images = images.to(device=backbone_param.device, dtype=torch.float32)

        with torch.no_grad():
            feats = self.base_model.features(images)
            pooled = self.base_model.avgpool(feats)
            pooled = torch.flatten(pooled, 1)

        embeddings = self.proj(pooled)
        return embeddings, feats


