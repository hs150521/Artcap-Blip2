"""
KV modulation utilities implementing FiLM-style affine control over the key/value
streams of the Q-Former cross-attention layers. The controller consumes the
PromptMapper tokens and emits per-head affine parameters that follow the
formulation: K' = (1 + gamma_K) ⊙ K + beta_K, V' = (1 + gamma_V) ⊙ V + beta_V.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
import math

import torch
import torch.nn as nn


@dataclass
class LoRAConfig:
    enabled: bool = False
    rank: int = 8
    alpha: float = 16.0
    dropout: float = 0.0


class LoRALinear(nn.Module):
    """Minimal LoRA adapter that wraps around an existing Linear layer."""

    def __init__(self, linear: nn.Linear, config: LoRAConfig) -> None:
        super().__init__()
        self.linear = linear
        self.config = config
        self.rank = config.rank
        if not config.enabled or self.rank <= 0:
            self.adapter_A = None
            self.adapter_B = None
            self.scaling = 1.0
            self.dropout = nn.Identity()
            return

        in_features = linear.in_features
        out_features = linear.out_features
        self.adapter_A = nn.Linear(in_features, self.rank, bias=False)
        self.adapter_B = nn.Linear(self.rank, out_features, bias=False)
        nn.init.kaiming_uniform_(self.adapter_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.adapter_B.weight)
        self.scaling = config.alpha / self.rank
        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.linear(x)
        if self.adapter_A is None or self.adapter_B is None:
            return result
        lora_update = self.adapter_B(self.adapter_A(self.dropout(x))) * self.scaling
        return result + lora_update


class KVModulation(nn.Module):
    """Project controller tokens into per-head FiLM parameters."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        controller_dim: int,
        hidden_dim: Optional[int] = None,
        init_scale: float = 0.01,
        use_layer_norm: bool = True,
    ) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(f"hidden_size {hidden_size} must be divisible by num_heads {num_heads}")

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        hidden_dim = hidden_dim or controller_dim

        self.controller_norm = (
            nn.LayerNorm(controller_dim) if use_layer_norm else nn.Identity()
        )
        self.controller_mlp = nn.Sequential(
            nn.Linear(controller_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, controller_dim),
            nn.GELU(),
        )

        proj_out_dim = num_heads * self.head_dim
        self.gamma_k_proj = nn.Linear(controller_dim, proj_out_dim)
        self.beta_k_proj = nn.Linear(controller_dim, proj_out_dim)
        self.gamma_v_proj = nn.Linear(controller_dim, proj_out_dim)
        self.beta_v_proj = nn.Linear(controller_dim, proj_out_dim)

        for layer in (
            self.gamma_k_proj,
            self.beta_k_proj,
            self.gamma_v_proj,
            self.beta_v_proj,
        ):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

        self.init_scale = init_scale

    def forward(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        controller_tokens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            k: (batch, num_heads, seq_len, head_dim)
            v: (batch, num_heads, seq_len, head_dim)
            controller_tokens: (batch, num_tokens, controller_dim)
        """
        if controller_tokens.dim() != 3:
            raise ValueError("controller_tokens must be (batch, num_tokens, dim)")

        pooled = controller_tokens.mean(dim=1)
        pooled = self.controller_norm(pooled)
        pooled = self.controller_mlp(pooled)

        gamma_k = self._reshape_to_heads(self.gamma_k_proj(pooled))
        beta_k = self._reshape_to_heads(self.beta_k_proj(pooled))
        gamma_v = self._reshape_to_heads(self.gamma_v_proj(pooled))
        beta_v = self._reshape_to_heads(self.beta_v_proj(pooled))

        gamma_k = gamma_k * self.init_scale
        beta_k = beta_k * self.init_scale
        gamma_v = gamma_v * self.init_scale
        beta_v = beta_v * self.init_scale

        k = (1.0 + gamma_k) * k + beta_k
        v = (1.0 + gamma_v) * v + beta_v
        return k, v

    def _reshape_to_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        batch = tensor.size(0)
        tensor = tensor.view(batch, self.num_heads, self.head_dim)
        tensor = tensor.unsqueeze(2)  # (batch, heads, 1, head_dim)
        return tensor

