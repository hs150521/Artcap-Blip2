"""Cross-attention wrapper that injects FiLM-style gating into K/V tensors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class GatingConfig:
    """Configuration for the gating module."""

    controller_dim: int = 768
    hidden_dim: Optional[int] = None
    per_head: bool = True
    init_scale: float = 1e-2
    gating_type: str = "film"  # "film" | "gate"
    use_layer_norm: bool = True
    use_bias: bool = True


class GatedCrossAttention(nn.Module):
    """Apply learnable FiLM-style affine or multiplicative gating to K/V."""

    def __init__(
        self,
        kv_dim: int,
        num_heads: int,
        gating_config: Optional[GatingConfig] = None,
    ) -> None:
        super().__init__()
        self.kv_dim = kv_dim
        self.num_heads = num_heads
        if kv_dim % num_heads != 0:
            raise ValueError(f"kv_dim ({kv_dim}) must be divisible by num_heads ({num_heads})")
        self.head_dim = kv_dim // num_heads
        self.cfg = gating_config or GatingConfig()

        controller_dim = self.cfg.controller_dim
        hidden_dim = self.cfg.hidden_dim or controller_dim
        embed_dim = self.num_heads if self.cfg.per_head else kv_dim

        self.controller_norm = (
            nn.LayerNorm(controller_dim) if self.cfg.use_layer_norm else nn.Identity()
        )
        self.controller_mlp = (
            nn.Sequential(
                nn.Linear(controller_dim, hidden_dim, bias=self.cfg.use_bias),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim, bias=self.cfg.use_bias),
            )
            if hidden_dim != controller_dim
            else nn.Identity()
        )

        self.gamma_proj = nn.Linear(hidden_dim, embed_dim, bias=self.cfg.use_bias)
        self.beta_proj = (
            nn.Linear(hidden_dim, embed_dim, bias=self.cfg.use_bias)
            if self.cfg.gating_type == "film"
            else None
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.gamma_proj.weight, std=self.cfg.init_scale)
        if self.gamma_proj.bias is not None:
            nn.init.constant_(self.gamma_proj.bias, 0.0)
        if self.beta_proj is not None:
            nn.init.normal_(self.beta_proj.weight, std=self.cfg.init_scale)
            if self.beta_proj.bias is not None:
                nn.init.constant_(self.beta_proj.bias, 0.0)

        if isinstance(self.controller_mlp, nn.Sequential):
            for module in self.controller_mlp:
                if isinstance(module, nn.Linear):
                    nn.init.normal_(module.weight, std=self.cfg.init_scale)
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0.0)

    def forward(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        controller_state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Gates the K/V tensors before cross-attention.

        Args:
            key: Tensor of shape (batch, seq_len_kv, kv_dim)
            value: Tensor of shape (batch, seq_len_kv, kv_dim)
            controller_state: Conditioning tensor, e.g. PromptMapper output tokens (batch, tokens, dim).

        Returns:
            Tuple of gated key and value tensors.
        """

        gamma, beta = self._compute_gating(controller_state)
        key = self._apply_gate(key, gamma, beta)
        value = self._apply_gate(value, gamma, beta)
        return key, value

    def _compute_gating(
        self, controller_state: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if controller_state.dim() != 3:
            raise ValueError(
                f"controller_state must have shape (batch, tokens, dim); got {controller_state.shape}"
            )
        pooled = controller_state.mean(dim=1)
        pooled = self.controller_norm(pooled)
        hidden = self.controller_mlp(pooled)
        gamma = self.gamma_proj(hidden)
        beta = self.beta_proj(hidden) if self.beta_proj is not None else None
        return gamma, beta

    def _apply_gate(
        self,
        tensor: torch.Tensor,
        gamma: torch.Tensor,
        beta: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if self.cfg.gating_type not in {"film", "gate"}:
            raise ValueError(f"Unsupported gating_type {self.cfg.gating_type}")

        if self.cfg.per_head:
            gamma = gamma.view(-1, 1, self.num_heads, 1)
            if beta is not None:
                beta = beta.view(-1, 1, self.num_heads, 1)

            tensor = tensor.view(tensor.shape[0], tensor.shape[1], self.num_heads, self.head_dim)
            tensor = self._apply_mode(tensor, gamma, beta)
            tensor = tensor.view(tensor.shape[0], tensor.shape[1], self.kv_dim)
        else:
            gamma = gamma.unsqueeze(1)
            beta = beta.unsqueeze(1) if beta is not None else None
            tensor = self._apply_mode(tensor, gamma, beta)
        return tensor

    def _apply_mode(
        self,
        tensor: torch.Tensor,
        gamma: torch.Tensor,
        beta: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if self.cfg.gating_type == "film":
            tensor = tensor * (1 + gamma)
            if beta is not None:
                tensor = tensor + beta
        else:  # gate
            gate = torch.sigmoid(gamma)
            tensor = tensor * gate
        return tensor

