#!/usr/bin/env python3
"""
BLIP2 with EfficientNet prompt augmentation using LAVIS backend.

This module provides an adapter for using EfficientNet-augmented BLIP2
with LAVIS model backend instead of direct transformers.
"""

import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch
from PIL import Image
from tqdm import tqdm

# Add LAVIS to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "blip2" / "LAVIS"))

from lavis.models import load_model


# Add scripts to path for EfficientNet
repo_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(repo_root))

from scripts.prompt_aug.enhanced_blip2_fixed import (
    _classify_image,
    filter_style_labels_for_prompt,
)


def load_blip2_prompt_aug_lavis_model(config: Dict, device: str) -> Any:
    """
    Load BLIP2 model using LAVIS with EfficientNet support.
    
    Args:
        config: Configuration dictionary
        device: Device to load model on
    
    Returns:
        Loaded BLIP2 LAVIS model
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


def _get_image_path(sample: Dict) -> str:
    """
    Extract and resolve image path from dataset sample.
    
    Args:
        sample: Dataset sample dict containing image path information
    
    Returns:
        Resolved absolute path to image file
    
    Raises:
        ValueError: If image path cannot be determined
    """
    if "image_path" in sample:
        return sample["image_path"]
    elif "image_path_in_ann" in sample and "vis_root" in sample:
        vis_root = sample["vis_root"]
        image_path_in_ann = sample["image_path_in_ann"]
        
        # Try direct path first
        direct_path = os.path.join(vis_root, image_path_in_ann)
        if os.path.exists(direct_path):
            return direct_path
        
        # Try searching in common subdirectories
        filename = os.path.basename(image_path_in_ann)
        subdirs = ['val2014', 'train2014', 'test2014', 'val', 'train', 'test']
        for subdir in subdirs:
            candidate_path = os.path.join(vis_root, subdir, filename)
            if os.path.exists(candidate_path):
                return candidate_path
        
        # Search all subdirectories
        if os.path.isdir(vis_root):
            for item in os.listdir(vis_root):
                subdir_path = os.path.join(vis_root, item)
                if os.path.isdir(subdir_path):
                    candidate_path = os.path.join(subdir_path, filename)
                    if os.path.exists(candidate_path):
                        return candidate_path
        
        # Fallback to direct path even if it doesn't exist
        return direct_path
    else:
        raise ValueError(
            "Cannot determine image path from dataset. "
            "The dataset should return 'image_path' in the sample dict."
        )


def _clean_answer(answer: str) -> str:
    """
    Clean answer by removing common prompt artifacts and repetitions.
    
    Args:
        answer: Raw answer from model
    
    Returns:
        Cleaned answer
    """
    if not answer:
        return ""
    
    # Remove leading/trailing whitespace
    cleaned = answer.strip()

    # Remove common prompt artifacts that model might repeat
    artifacts = [
        "give short answer:",
        "give long answer:",
        "short answer:",
        "long answer:",
        "answer:",
        "Question:",
        "question:",
    ]
    
    # Special handling for repeated "Answer:" patterns (common issue)
    # Remove all lines that are just "Answer:" or "answer:"
    lines = cleaned.split('\n')
    filtered_lines = []
    for line in lines:
        line_stripped = line.strip()
        # Skip lines that are just "Answer:" or variations
        if re.match(r'(?i)^answer:?\s*$', line_stripped):
            continue
        filtered_lines.append(line)
    cleaned = '\n'.join(filtered_lines).strip()
    
    # Remove artifacts at the beginning
    for artifact in artifacts:
        if cleaned.lower().startswith(artifact.lower()):
            cleaned = cleaned[len(artifact):].strip()
            # Remove leading colon or comma if present
            cleaned = cleaned.lstrip(":.,; \n\t").strip()
            # If we removed everything, restore the original
            if not cleaned:
                cleaned = answer.strip()
            break  # Only remove the first matching artifact
    
    # Remove artifacts in the middle (common repetition pattern)
    for artifact in artifacts:
        # Replace patterns like ", give short answer:" or " give short answer:"
        pattern = r'[,;]?\s*' + re.escape(artifact) + r'[:]?\s*'
        cleaned = re.sub(pattern, ' ', cleaned, flags=re.IGNORECASE)
    
    # Remove multiple consecutive "Answer:" patterns (handles cases like "Answer:\nAnswer:\nAnswer:")
    # This is a common issue where model repeats the prompt
    cleaned = re.sub(r'(?i)(answer:\s*)+', '', cleaned).strip()
    
    # Remove multiple consecutive spaces
    cleaned = ' '.join(cleaned.split())
    
    # Remove trailing punctuation artifacts
    cleaned = cleaned.rstrip(":.,;")
    
    return cleaned.strip()


def _build_augmented_prompt_fixed(labels: list, template: str, user_prompt: str) -> str:
    """
    Build augmented prompt with context BEFORE the question.
    
    This is critical for LAVIS BLIP2 model to work correctly.
    
    Args:
        labels: List of (label, score) tuples from EfficientNet
        template: Template string with {labels} placeholder
        user_prompt: User's question prompt
    
    Returns:
        Augmented prompt with context before question
    """
    filtered_labels = filter_style_labels_for_prompt(labels)
    if not filtered_labels:
        return user_prompt
    
    # Extract just the label names
    label_names = [label for label, _ in filtered_labels]
    labels_text = ", ".join(label_names)
    
    # Build context part
    context_part = template.format(labels=labels_text)
    
    # Combine: context first, then user prompt
    return f"{context_part}{user_prompt}"


def predict_answers_blip2_prompt_aug_lavis(
    model: Any,
    dataset: Any,
    config: Dict,
    batch_size: int,
    device: str
) -> List[Dict]:
    """
    Run EfficientNet-augmented BLIP2 model inference on VQAv2 dataset using LAVIS backend.
    
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
    base_prompt = run_cfg.get("prompt", "")
    num_beams = run_cfg.get("num_beams", 5)
    max_len = run_cfg.get("max_len", 10)
    min_len = run_cfg.get("min_len", 1)
    
    model_cfg = config.get("model_config", {})
    efficientnet_cfg = model_cfg.get("efficientnet", {})
    topk = efficientnet_cfg.get("topk", 2)
    prompt_template = efficientnet_cfg.get("prompt_template", "Context: {labels}. ")
    
    predictions = []
    
    # Process samples one by one to apply EfficientNet augmentation
    for i in tqdm(range(len(dataset)), desc="Running inference with prompt augmentation"):
        sample = dataset[i]
        question = sample["text_input"]
        question_id = sample["question_id"]
        
        # Load image for EfficientNet classification (need raw PIL image)
        image_path = _get_image_path(sample)
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            logging.warning(f"Failed to load image {image_path}: {e}")
            # Use a fallback: skip EfficientNet and use standard prompt
            labels = []
        else:
            # Classify image and build augmented prompt
            try:
                labels = _classify_image(image, topk)
            except Exception as e:
                logging.warning(f"EfficientNet classification failed for {image_path}: {e}")
                labels = []
        
        # Build user prompt (same as LAVIS behavior)
        if base_prompt:
            try:
                user_prompt = base_prompt.format(question)
            except (KeyError, ValueError):
                user_prompt = f"{base_prompt} {question}"
        else:
            user_prompt = question
        
        # Build augmented prompt with EfficientNet labels
        # Use fixed version that puts context BEFORE the question
        aug_prompt = _build_augmented_prompt_fixed(labels, prompt_template, user_prompt)
        
        # Log the first few prompts for debugging
        if i < 5:
            logging.info(f"Sample {i} - Augmented prompt: {aug_prompt}")
        
        # Use the processed image from dataset for LAVIS inference
        processed_image = sample["image"].unsqueeze(0).to(device)
        
        # Create samples dict for LAVIS
        samples = {
            "image": processed_image,
            "text_input": [aug_prompt],
        }
        
        # Generate answer using LAVIS
        with torch.no_grad():
            answers = model.predict_answers(
                samples=samples,
                num_beams=num_beams,
                inference_method=run_cfg.get("inference_method", "generate"),
                max_len=max_len,
                min_len=min_len,
                prompt="",  # We already built the full prompt with augmentation
            )
        
        # Clean answer
        cleaned_answer = _clean_answer(answers[0] if answers else "")
        
        # Log empty answers with more context for debugging
        if not cleaned_answer:
            logging.warning(f"Empty answer for question_id {question_id}, raw answer: '{answers[0] if answers else ''}', question: '{question}', labels: {labels}")
        
        predictions.append({
            "question_id": question_id,
            "answer": cleaned_answer
        })
        
        # Clear CUDA cache periodically to prevent memory buildup
        if i % 100 == 0 and device == "cuda":
            torch.cuda.empty_cache()
    
    return predictions
