"""Style-related helper functions for auxiliary regularization."""

from __future__ import annotations

from typing import Iterable, Sequence

import torch
import torch.nn.functional as F
from transformers import PreTrainedTokenizerBase


def compute_style_hit_rate(logits: torch.Tensor, style_token_ids: torch.Tensor) -> torch.Tensor:
    """Compute the fraction of generated tokens that match style vocabulary tokens."""

    if style_token_ids.numel() == 0:
        return torch.tensor(0.0, device=logits.device, dtype=logits.dtype)

    pred_tokens = logits.argmax(dim=-1)
    style_token_ids = style_token_ids.to(pred_tokens.device)
    matches = (pred_tokens.unsqueeze(-1) == style_token_ids).any(dim=-1).float()
    return matches.mean()


def compute_consistency_kl(image_style_logits: torch.Tensor, text_style_logits: torch.Tensor) -> torch.Tensor:
    """Symmetric KL divergence between image and text style distributions."""

    p = image_style_logits.log_softmax(dim=-1)
    q = text_style_logits.log_softmax(dim=-1)
    kl_pq = F.kl_div(p, q.exp(), reduction="batchmean")
    kl_qp = F.kl_div(q, p.exp(), reduction="batchmean")
    return 0.5 * (kl_pq + kl_qp)


def build_style_token_ids(
    styles: Sequence[str],
    tokenizer: PreTrainedTokenizerBase,
) -> torch.Tensor:
    """Tokenize style vocabulary and return unique token ids."""

    token_ids = set()
    for style in styles:
        if not style:
            continue
        tokens = tokenizer(style, add_special_tokens=False).input_ids
        token_ids.update(tokens)
    if not token_ids:
        return torch.empty(0, dtype=torch.long)
    return torch.tensor(sorted(token_ids), dtype=torch.long)

