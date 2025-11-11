#!/usr/bin/env python3
"""
Model backend factory for ArtQuest evaluation.

This module provides a factory function to load different model backends
based on a model name string.
"""

from typing import Any, Dict, List, Tuple

# Import BLIP2 LAVIS model loader from VQAv2
import sys
from pathlib import Path

# Add VQAv2 models directory to path
vqav2_models_dir = Path(__file__).parent.parent.parent / "VQAv2" / "models"
if str(vqav2_models_dir) not in sys.path:
    sys.path.insert(0, str(vqav2_models_dir))

# Import model backends from VQAv2
from blip2_lavis import load_blip2_lavis_model
from blip2_prompt_aug_lavis import load_blip2_prompt_aug_lavis_model
from blip2_kv import load_blip2_kv_model
from blip2_gated import load_blip2_gated_model

# Import local model backends
from . import blip2_artquest
from . import blip2_prompt_aug_artquest
from . import blip2_kv_artquest
from . import blip2_gated_artquest


def get_model_backend(model_name: str):
    """
    Get model backend functions for the specified model.
    
    Args:
        model_name: Model identifier string. Supported values:
            - "blip2": Standard BLIP2 model via LAVIS
            - "blip2_prompt_aug": BLIP2 with EfficientNet-based prompt augmentation (LAVIS)
            - "blip2_kv": BLIP2 with KV modulation
            - "blip2_gated": BLIP2 with Gated modulation
    
    Returns:
        Tuple of (load_model_func, predict_answers_func)
    """
    if model_name == "blip2":
        return load_blip2_lavis_model, blip2_artquest.predict_answers_blip2_artquest
    elif model_name == "blip2_prompt_aug":
        return load_blip2_prompt_aug_lavis_model, blip2_prompt_aug_artquest.predict_answers_blip2_prompt_aug_artquest
    elif model_name == "blip2_kv":
        return load_blip2_kv_model, blip2_kv_artquest.predict_answers_blip2_kv_artquest
    elif model_name == "blip2_gated":
        return load_blip2_gated_model, blip2_gated_artquest.predict_answers_blip2_gated_artquest
    else:
        raise ValueError(
            f"Unknown model name: {model_name}. "
            f"Supported values: 'blip2', 'blip2_prompt_aug', 'blip2_kv', 'blip2_gated'"
        )
