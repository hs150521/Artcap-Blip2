"""
Prompt mapper that transforms EfficientNet style embeddings into controller
tokens for the KV modulation module, following the formulation described in the
paper (Eq. 2). The mapper emits both per-token control states and a pooled state
that can be used for auxiliary heads (e.g., style logits).
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn


class PromptMapper(nn.Module):
    """Encode EfficientNet/text embeddings into controller states for KV modulation."""

    def __init__(
        self,
        image_dim: int,
        output_dim: int,
        hidden_dim: Optional[int] = None,
        num_tokens: int = 4,
        text_dim: Optional[int] = None,
        dropout: float = 0.1,
        use_layer_norm: bool = True,
    ) -> None:
        super().__init__()
        self.num_tokens = num_tokens
        self.text_dim = text_dim

        hidden_dim = hidden_dim or output_dim

        self.image_proj = nn.Linear(image_dim, output_dim, bias=True)
        nn.init.trunc_normal_(self.image_proj.weight, std=0.02)
        nn.init.zeros_(self.image_proj.bias)

        if text_dim is not None:
            self.text_proj = nn.Linear(text_dim, output_dim, bias=True)
            nn.init.trunc_normal_(self.text_proj.weight, std=0.02)
            nn.init.zeros_(self.text_proj.bias)
        else:
            self.text_proj = None

        layers = []
        for _ in range(2):
            layers.append(nn.Linear(output_dim, hidden_dim, bias=True))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(hidden_dim, output_dim, bias=True))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
        self.mapper = nn.Sequential(*layers)
        self.layer_norm = nn.LayerNorm(output_dim) if use_layer_norm else nn.Identity()

    def forward(
        self,
        image_embeddings: torch.Tensor,
        text_embeddings: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Produce controller tokens and pooled state.

        Args:
            image_embeddings: Tensor with shape (batch, image_dim)
            text_embeddings: Optional tensor with shape (batch, seq_len, text_dim)

        Returns:
            Tuple of (controller_tokens, pooled_state)
        """

        if image_embeddings.dim() != 2:
            raise ValueError(f"image_embeddings should be (batch, dim); got {image_embeddings.shape}")

        batch_size = image_embeddings.size(0)

        controller_tokens = self.image_proj(image_embeddings).unsqueeze(1)
        controller_tokens = controller_tokens.expand(batch_size, self.num_tokens, -1)

        if text_embeddings is not None:
            if self.text_proj is None:
                raise RuntimeError("text_embeddings provided but text_proj is None. Set text_dim in constructor.")
            if text_embeddings.dim() != 3:
                raise ValueError(f"text_embeddings should be (batch, seq_len, dim); got {text_embeddings.shape}")

            text_tokens = self.text_proj(text_embeddings)
            controller_tokens = torch.cat([controller_tokens, text_tokens], dim=1)

        controller_tokens = self.mapper(controller_tokens)
        controller_tokens = self.layer_norm(controller_tokens)
        pooled_state = controller_tokens.mean(dim=1, keepdim=True)
        return controller_tokens, pooled_state


