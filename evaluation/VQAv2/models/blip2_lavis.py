#!/usr/bin/env python3
"""
BLIP2 LAVIS model backend for VQAv2 evaluation.

This module provides functions to load and use BLIP2 models via LAVIS.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List

import torch
from tqdm import tqdm

# Add LAVIS to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "blip2" / "LAVIS"))

from lavis.models import load_model


def load_blip2_lavis_model(config: Dict, device: str) -> Any:
    """
    Load BLIP2 model using LAVIS.
    
    Args:
        config: Configuration dictionary. Expected to have a "model" section with:
            - arch: Model architecture (default: "blip2_opt")
            - model_type: Model type (default: "pretrain_opt2.7b")
            Or "model" can be a string identifier like "blip2" (will use defaults)
        device: Device to load model on
    
    Returns:
        Loaded BLIP2 model
    """
    model_cfg = config.get("model", {})
    
    # Handle case where model is a string identifier (e.g., "blip2")
    if isinstance(model_cfg, str):
        # Use default values when model is specified as a string
        arch = "blip2_opt"
        model_type = "pretrain_opt2.7b"
    elif isinstance(model_cfg, dict):
        # Use values from config dict, with defaults as fallback
        arch = model_cfg.get("arch", "blip2_opt")
        model_type = model_cfg.get("model_type", "pretrain_opt2.7b")
    else:
        # Fallback to defaults if model_cfg is neither string nor dict
        arch = "blip2_opt"
        model_type = "pretrain_opt2.7b"
    
    model = load_model(
        name=arch,
        model_type=model_type,
        is_eval=True,
        device=device
    )
    
    return model


def predict_answers_blip2_lavis(
    model: Any,
    dataset: Any,
    config: Dict,
    batch_size: int,
    device: str
) -> List[Dict]:
    """
    Run BLIP2 LAVIS model inference on VQAv2 dataset.
    
    Args:
        model: Loaded BLIP2 LAVIS model
        dataset: Dataset object that supports __getitem__ and __len__
        config: Configuration dictionary
        batch_size: Batch size for inference
        device: Device to run inference on
    
    Returns:
        List of predictions with question_id and answer
    """
    model.eval()
    model = model.to(device)
    
    run_cfg = config.get("run", {})
    prompt = run_cfg.get("prompt", "")
    num_beams = run_cfg.get("num_beams", 5)
    max_len = run_cfg.get("max_len", 10)
    min_len = run_cfg.get("min_len", 1)
    
    predictions = []
    
    # Process in batches (load images on demand)
    import logging
    logging.info(f"Using blip2_lavis: batch_size={batch_size}")
    
    with torch.no_grad():
        for i in tqdm(range(0, len(dataset), batch_size), desc="Running inference"):
            batch = []
            question_ids = []
            
            # Load batch samples on demand
            for j in range(i, min(i + batch_size, len(dataset))):
                sample = dataset[j]
                batch.append(sample)
                question_ids.append(sample["question_id"])
            
            # Prepare batch
            images = torch.stack([item["image"] for item in batch]).to(device)
            text_inputs = [item["text_input"] for item in batch]
            
            # Create samples dict
            samples = {
                "image": images,
                "text_input": text_inputs,
            }
            
            # Generate answers
            answers = model.predict_answers(
                samples=samples,
                num_beams=num_beams,
                inference_method=run_cfg.get("inference_method", "generate"),
                max_len=max_len,
                min_len=min_len,
                prompt=prompt,
            )
            
            # Collect predictions
            for question_id, answer in zip(question_ids, answers):
                predictions.append({
                    "question_id": int(question_id) if isinstance(question_id, torch.Tensor) else question_id,
                    "answer": answer.strip()
                })
    
    return predictions

