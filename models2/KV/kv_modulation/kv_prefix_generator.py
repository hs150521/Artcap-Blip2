"""
KV-Prefix Generator module.

This module generates Key and Value prefixes from EfficientNet features
for use in cross-attention layers.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


class KVPrefixGenerator(nn.Module):
    """
    Generate K and V prefixes from EfficientNet features.
    
    Args:
        efficientnet_feat_dim: Dimension of EfficientNet features (default: 1536 for B3)
        qformer_hidden_size: Hidden size of Qformer (default: 768)
        num_prefix_tokens: Number of prefix tokens to generate (default: 8)
        num_heads: Number of attention heads (default: 12)
        head_dim: Dimension per attention head (default: 64)
    """
    
    def __init__(
        self,
        efficientnet_feat_dim: int = 1536,
        qformer_hidden_size: int = 768,
        num_prefix_tokens: int = 8,
        num_heads: int = 12,
        head_dim: Optional[int] = None,
    ):
        super().__init__()
        
        self.efficientnet_feat_dim = efficientnet_feat_dim
        self.qformer_hidden_size = qformer_hidden_size
        self.num_prefix_tokens = num_prefix_tokens
        self.num_heads = num_heads
        self.head_dim = head_dim or (qformer_hidden_size // num_heads)
        self.all_head_size = num_heads * self.head_dim
        
        # Project EfficientNet features to Qformer hidden size
        # Use MLP to generate prefix tokens
        self.feature_proj = nn.Sequential(
            nn.Linear(efficientnet_feat_dim, qformer_hidden_size * 2),
            nn.GELU(),
            nn.Linear(qformer_hidden_size * 2, qformer_hidden_size),
        )
        
        # Generate K and V prefixes
        self.k_proj = nn.Linear(qformer_hidden_size, self.all_head_size)
        self.v_proj = nn.Linear(qformer_hidden_size, self.all_head_size)
        
        # Learnable prefix tokens
        self.prefix_tokens = nn.Parameter(
            torch.randn(1, num_prefix_tokens, qformer_hidden_size) * 0.02
        )
    
    def forward(self, efficientnet_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate K and V prefixes from EfficientNet features.
        
        Args:
            efficientnet_features: Tensor of shape (batch_size, efficientnet_feat_dim)
                or (batch_size, seq_len, efficientnet_feat_dim)
        
        Returns:
            k_prefix: Tensor of shape (batch_size, num_heads, num_prefix_tokens, head_dim)
            v_prefix: Tensor of shape (batch_size, num_heads, num_prefix_tokens, head_dim)
        """
        batch_size = efficientnet_features.shape[0]
        
        # Handle different input shapes
        if efficientnet_features.dim() == 2:
            # (batch_size, feat_dim) -> (batch_size, 1, feat_dim)
            efficientnet_features = efficientnet_features.unsqueeze(1)
        
        # Expand prefix tokens to batch size
        prefix_embeds = self.prefix_tokens.expand(batch_size, -1, -1)
        
        # Project EfficientNet features
        if efficientnet_features.shape[1] == 1:
            # Global pooling: average if multiple features
            eff_feat = efficientnet_features.mean(dim=1, keepdim=True)
        else:
            # Use first feature or average
            eff_feat = efficientnet_features[:, 0:1, :] if efficientnet_features.shape[1] > 0 else efficientnet_features.mean(dim=1, keepdim=True)
        
        projected_features = self.feature_proj(eff_feat.squeeze(1))  # (batch_size, hidden_size)
        
        # Add projected features to prefix tokens
        # Expand projected_features to match prefix_embeds shape: (batch_size, num_prefix_tokens, hidden_size)
        projected_features = projected_features.unsqueeze(1).expand(-1, self.num_prefix_tokens, -1)
        prefix_embeds = prefix_embeds + projected_features
        
        # Generate K and V
        k_prefix = self.k_proj(prefix_embeds)  # (batch_size, num_prefix_tokens, all_head_size)
        v_prefix = self.v_proj(prefix_embeds)  # (batch_size, num_prefix_tokens, all_head_size)
        
        # Reshape to (batch_size, num_heads, num_prefix_tokens, head_dim)
        k_prefix = k_prefix.view(batch_size, self.num_prefix_tokens, self.num_heads, self.head_dim)
        k_prefix = k_prefix.transpose(1, 2)  # (batch_size, num_heads, num_prefix_tokens, head_dim)
        
        v_prefix = v_prefix.view(batch_size, self.num_prefix_tokens, self.num_heads, self.head_dim)
        v_prefix = v_prefix.transpose(1, 2)  # (batch_size, num_heads, num_prefix_tokens, head_dim)
        
        return k_prefix, v_prefix
