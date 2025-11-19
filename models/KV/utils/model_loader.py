"""
Utilities for building the KV-modulated BLIP-2 model from a configuration
dictionary. Keeps all path handling and HuggingFace setup in a central place.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from models.KV.modules.blip2_opt_kv import Blip2OPTKV

logger = logging.getLogger(__name__)


def _resolve_path(path_str: Optional[str], repo_root: Path) -> Optional[str]:
    if not path_str:
        return None
    path = Path(path_str)
    if not path.is_absolute():
        path = repo_root / path
    return str(path)


def load_kv_model(
    config: Dict[str, Any],
    device: Optional[torch.device] = None,
) -> Blip2OPTKV:
    """Instantiate the BLIP-2 KV model according to the config dictionary."""

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_cfg = config.get("model", {})
    vit_cfg = model_cfg.get("vision", {})
    prompt_mapper_cfg = model_cfg.get("prompt_mapper", {})
    kv_cfg = model_cfg.get("kv_modulation", {})

    repo_root = Path(__file__).resolve().parents[3]
    paths_cfg = config.get("paths", {})

    efficientnet_ckpt = _resolve_path(
        paths_cfg.get("efficientnet_checkpoint"),
        repo_root,
    )
    opt_path = paths_cfg.get("opt_checkpoint") or model_cfg.get("opt_model", "facebook/opt-2.7b")

    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "300")

    model = Blip2OPTKV(
        vit_model=vit_cfg.get("name", "eva_clip_g"),
        img_size=vit_cfg.get("img_size", 224),
        drop_path_rate=vit_cfg.get("drop_path_rate", 0.0),
        use_grad_checkpoint=vit_cfg.get("use_grad_checkpoint", False),
        vit_precision=vit_cfg.get("precision", "fp16"),
        freeze_vit=vit_cfg.get("freeze", True),
        num_query_token=model_cfg.get("num_query_tokens", 32),
        opt_model=opt_path,
        prompt=model_cfg.get("prompt", ""),
        max_txt_len=model_cfg.get("max_txt_len", 64),
        apply_lemmatizer=model_cfg.get("apply_lemmatizer", False),
        efficientnet_checkpoint=efficientnet_ckpt,
        efficientnet_output_dim=model_cfg.get("efficientnet_output_dim", 768),
        enable_efficientnet_grad=model_cfg.get("enable_efficientnet_grad", False),
        prompt_mapper_cfg=prompt_mapper_cfg,
        kv_modulation_cfg=kv_cfg,
    )

    model = model.to(device)
    return model


