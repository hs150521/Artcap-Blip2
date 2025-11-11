#!/usr/bin/env python3
"""
Model backend factory for VQAv2 evaluation.

This module provides a factory function to load different model backends
based on a model name string.
"""

from typing import Any, Dict, List, Tuple

# Import model backends
from . import blip2_kv
from . import blip2_lavis
from . import blip2_prompt_aug
from . import blip2_prompt_aug_lavis
from . import blip2_gated


def get_model_backend(model_name: str):
    """
    Get model backend functions for the specified model.
    
    Args:
        model_name: Model identifier string. Supported values:
            - "blip2": Standard BLIP2 model via LAVIS
            - "blip2_prompt_aug": BLIP2 with EfficientNet-based prompt augmentation (transformers)
            - "blip2_prompt_aug_lavis": BLIP2 with EfficientNet-based prompt augmentation (LAVIS)
            - "blip2_kv": BLIP2 with KV modulation
            - "blip2_gated": BLIP2 with Gated modulation
    
    Returns:
        Tuple of (load_model_func, predict_answers_func)
    """
    if model_name == "blip2":
        return blip2_lavis.load_blip2_lavis_model, blip2_lavis.predict_answers_blip2_lavis
    elif model_name == "blip2_prompt_aug":
        return blip2_prompt_aug.load_blip2_prompt_aug_model, blip2_prompt_aug.predict_answers_blip2_prompt_aug
    elif model_name == "blip2_prompt_aug_lavis":
        return blip2_prompt_aug_lavis.load_blip2_prompt_aug_lavis_model, blip2_prompt_aug_lavis.predict_answers_blip2_prompt_aug_lavis
    elif model_name == "blip2_kv":
        return blip2_kv.load_blip2_kv_model, blip2_kv.predict_answers_blip2_kv
    elif model_name == "blip2_gated":
        return blip2_gated.load_blip2_gated_model, blip2_gated.predict_answers_blip2_gated
    else:
        raise ValueError(
            f"Unknown model name: {model_name}. "
            f"Supported values: 'blip2', 'blip2_prompt_aug', 'blip2_prompt_aug_lavis', 'blip2_kv', 'blip2_gated'"
        )

