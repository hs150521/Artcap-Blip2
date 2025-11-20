"""
Utilities for building the KV-modulated BLIP-2 model from a configuration
dictionary. Keeps all path handling and HuggingFace setup in a central place.
"""

from __future__ import annotations

import logging
import os
import sys
import types
from pathlib import Path
from typing import Any, Dict, Optional

import torch

# Ensure project root is in sys.path for imports
# Calculate repo root and ensure it's at the front of sys.path
_repo_root = Path(__file__).resolve().parents[3]
_repo_root_str = str(_repo_root)

# Force repo root to be at the front of sys.path
# Remove all occurrences first
while _repo_root_str in sys.path:
    sys.path.remove(_repo_root_str)
# Insert at the very front
sys.path.insert(0, _repo_root_str)

# Verify the path is correct before importing
if not Path(_repo_root_str).exists():
    raise RuntimeError(f"Repository root does not exist: {_repo_root_str}")
_blip2_opt_kv_path = Path(_repo_root_str) / "models" / "KV" / "modules" / "blip2_opt_kv.py"
if not _blip2_opt_kv_path.exists():
    raise RuntimeError(f"Blip2OPTKV module not found at expected path: {_blip2_opt_kv_path}")

# Ensure repo root is definitely in sys.path at the front
if _repo_root_str not in sys.path:
    sys.path.insert(0, _repo_root_str)
elif sys.path[0] != _repo_root_str:
    # Move to front if it's not already there
    sys.path.remove(_repo_root_str)
    sys.path.insert(0, _repo_root_str)

# Set up package structure in sys.modules to help with imports
# This is needed when modules are loaded dynamically
if "models" not in sys.modules:
    _models_module = types.ModuleType("models")
    _models_module.__path__ = [str(Path(_repo_root_str) / "models")]
    sys.modules["models"] = _models_module
if "models.KV" not in sys.modules:
    _kv_module = types.ModuleType("models.KV")
    _kv_module.__path__ = [str(Path(_repo_root_str) / "models" / "KV")]
    sys.modules["models.KV"] = _kv_module
if "models.KV.modules" not in sys.modules:
    _modules_module = types.ModuleType("models.KV.modules")
    _modules_module.__path__ = [str(Path(_repo_root_str) / "models" / "KV" / "modules")]
    sys.modules["models.KV.modules"] = _modules_module

# Now import Blip2OPTKV - should work with proper path and package structure
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


def load_blip2_kv_modulated_model(
    config: Dict[str, Any],
    device: Optional[torch.device] = None,
    checkpoint_path: Optional[str] = None,
) -> Blip2OPTKV:
    """
    Load BLIP2 model with KV modulation support and optional checkpoint.
    
    This function is compatible with the evaluation script's expected interface.
    It loads the model from config and optionally loads a checkpoint.
    
    Args:
        config: Configuration dictionary with model parameters.
            Expected structure:
            - model: Model configuration (opt_model, efficientnet_output_dim, etc.)
            - architecture: Architecture configuration (vit_model, img_size, etc.)
            - kv_modulation: KV modulation configuration
            - paths: Paths configuration (efficientnet_checkpoint, opt_checkpoint)
        device: Device to load model on (default: auto-detect)
        checkpoint_path: Optional path to checkpoint to load
    
    Returns:
        Loaded Blip2OPTKV model
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Convert config format from evaluation script to training script format
    # Evaluation script uses: {model: {...}, architecture: {...}, kv_modulation: {...}}
    # Training script uses: {model: {vision: {...}, prompt_mapper: {...}, kv_modulation: {...}}, paths: {...}}
    
    model_cfg = config.get("model", {})
    arch_cfg = config.get("architecture", {})
    kv_cfg = config.get("kv_modulation", {})
    
    # Build training-style config
    training_config = {
        "model": {
            "opt_model": model_cfg.get("opt_model", "facebook/opt-2.7b"),
            "efficientnet_output_dim": model_cfg.get("efficientnet_output_dim", 768),
            "enable_efficientnet_grad": model_cfg.get("enable_efficientnet_grad", False),
            "num_query_tokens": arch_cfg.get("num_query_token", 32),
            "prompt": model_cfg.get("prompt", ""),
            "max_txt_len": arch_cfg.get("max_txt_len", 32),
            "vision": {
                "name": arch_cfg.get("vit_model", "eva_clip_g"),
                "img_size": arch_cfg.get("img_size", 224),
                "freeze": arch_cfg.get("freeze_vit", True),
                "precision": "fp16",
                "drop_path_rate": 0.0,
                "use_grad_checkpoint": False,
            },
            "prompt_mapper": model_cfg.get("prompt_mapper", {}),
            "kv_modulation": kv_cfg,
        },
        "paths": {
            "efficientnet_checkpoint": model_cfg.get("efficientnet_checkpoint"),
            "opt_checkpoint": model_cfg.get("opt_model"),
        },
    }
    
    # Load model
    model = load_kv_model(training_config, device=device)
    
    # Load checkpoint if provided
    if checkpoint_path:
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.is_absolute():
            repo_root = Path(__file__).resolve().parents[3]
            checkpoint_path = repo_root / checkpoint_path
        
        if not checkpoint_path.exists():
            logger.warning(f"Checkpoint not found at {checkpoint_path}, skipping checkpoint loading")
        else:
            logger.info(f"Loading checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            
            # Handle different checkpoint formats
            if "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"], strict=False)
            elif "model" in checkpoint:
                model.load_state_dict(checkpoint["model"], strict=False)
            else:
                model.load_state_dict(checkpoint, strict=False)
            
            logger.info("Checkpoint loaded successfully")
    
    return model

