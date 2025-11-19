"""
Model loading utilities for KV-modulated BLIP2.

This module provides functions to load and initialize the KV-modulated BLIP2 model.
"""

import sys
import time
import logging
import os
from pathlib import Path
from typing import Optional, Dict, Any, TYPE_CHECKING

import torch

# Try to import requests for better SSL error handling
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

logger = logging.getLogger(__name__)

# Add paths
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # Changed from parents[2] to parents[1] to point to KV directory

# Type checking only to avoid circular import
if TYPE_CHECKING:
    from models.blip2_opt_kv_modulated import Blip2OPTKVModulated


def load_blip2_kv_modulated_model(
    config: Dict[str, Any],
    device: Optional[torch.device] = None,
    checkpoint_path: Optional[str] = None,
):
    """
    Load BLIP2 model with KV modulation support.
    
    Args:
        config: Configuration dictionary with model parameters
        device: Device to load model on (default: auto-detect)
        checkpoint_path: Optional path to checkpoint to load
    
    Returns:
        Loaded Blip2OPTKVModulated model
    """
    # Import here to avoid circular import
    from models.blip2_opt_kv_modulated import Blip2OPTKVModulated
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Extract model config
    model_cfg = config.get("model", {})
    arch_cfg = config.get("architecture", {})
    kv_cfg = config.get("kv_modulation", {})
    
    # Get EfficientNet checkpoint path
    repo_root = Path(__file__).resolve().parents[3]
    efficientnet_checkpoint = model_cfg.get("efficientnet_checkpoint")
    if efficientnet_checkpoint and not Path(efficientnet_checkpoint).is_absolute():
        efficientnet_checkpoint = repo_root / efficientnet_checkpoint
    
    # Set environment variables to help with Hugging Face downloads
    # Try to use local cache first, and increase timeout
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "300")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "0")
    
    # Check if using a mirror that might have SSL issues
    hf_endpoint = os.environ.get("HF_ENDPOINT", "")
    if "hf-mirror.com" in hf_endpoint:
        logger.info(f"Detected HF_ENDPOINT={hf_endpoint}. If you encounter SSL errors, try:")
        logger.info("  - Unset HF_ENDPOINT: unset HF_ENDPOINT")
        logger.info("  - Or use official endpoint: export HF_ENDPOINT=https://huggingface.co")
    
    # Initialize model with retry logic for network issues
    max_retries = 5
    retry_delay = 2  # Start with 2 seconds
    
    for attempt in range(max_retries):
        try:
            logger.info(f"Initializing model (attempt {attempt + 1}/{max_retries})...")
            model = Blip2OPTKVModulated(
                vit_model=arch_cfg.get("vit_model", "eva_clip_g"),
                img_size=arch_cfg.get("img_size", 224),
                drop_path_rate=0,
                use_grad_checkpoint=False,
                vit_precision="fp16",
                freeze_vit=arch_cfg.get("freeze_vit", True),
                num_query_token=arch_cfg.get("num_query_token", 32),
                opt_model=model_cfg.get("opt_model", "facebook/opt-2.7b"),
                prompt="",
                max_txt_len=arch_cfg.get("max_txt_len", 32),
                apply_lemmatizer=False,
                efficientnet_checkpoint=str(efficientnet_checkpoint) if efficientnet_checkpoint else None,
                num_prefix_tokens=kv_cfg.get("num_prefix_tokens", 8),
                use_kv_modulation=kv_cfg.get("enabled", True),
            )
            logger.info("Model initialized successfully")
            break
        except Exception as e:
            # Check if it's an SSL or network error
            is_ssl_error = False
            is_network_error = False
            
            # Check for requests SSL errors
            if HAS_REQUESTS and isinstance(e, requests.exceptions.SSLError):
                is_ssl_error = True
            elif HAS_REQUESTS and isinstance(e, (requests.exceptions.ConnectionError, requests.exceptions.Timeout)):
                is_network_error = True
            # Check for built-in network errors
            elif isinstance(e, (ConnectionError, TimeoutError, OSError)):
                is_network_error = True
            # Check error message for SSL indicators
            elif "SSL" in str(e) or "SSLError" in str(type(e).__name__) or "EOF" in str(e) or "SSL:" in str(e):
                is_ssl_error = True
            
            if is_ssl_error or is_network_error:
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (2 ** attempt)  # Exponential backoff
                    logger.warning(
                        f"SSL/Network error during model initialization (attempt {attempt + 1}/{max_retries}): {e}"
                    )
                    logger.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue
                else:
                    logger.error(
                        f"Failed to initialize model after {max_retries} attempts due to SSL/Network errors. "
                        f"Last error: {e}"
                    )
                    error_type = "SSL" if is_ssl_error else "Network"
                    current_endpoint = os.environ.get("HF_ENDPOINT", "not set (using default)")
                    logger.error(
                        f"Possible solutions for {error_type} errors:\n"
                        f"Current HF_ENDPOINT: {current_endpoint}\n"
                        "1. Check your internet connection\n"
                        "2. Try unsetting HF_ENDPOINT: unset HF_ENDPOINT\n"
                        "3. Or use official endpoint: export HF_ENDPOINT=https://huggingface.co\n"
                        "4. Pre-download the models manually: python -c 'from transformers import BertTokenizer; BertTokenizer.from_pretrained(\"bert-base-uncased\")'\n"
                        "5. Use offline mode if models are already cached: export TRANSFORMERS_OFFLINE=1"
                    )
                    raise
            else:
                # For non-network errors, don't retry
                logger.error(f"Error initializing model (non-network error): {e}")
                raise
    
    # Load checkpoint if provided
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        elif "model" in checkpoint:
            model.load_state_dict(checkpoint["model"], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
    
    model = model.to(device)
    # Don't set eval mode here - let the caller decide (train vs eval)
    
    return model
