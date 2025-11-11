#!/usr/bin/env python3
"""
BLIP2 with EfficientNet prompt augmentation backend for VQAv2 evaluation.

This module provides an adapter for using EfficientNet-augmented BLIP2
(enhanced_blip2_fixed) in the VQAv2 evaluation pipeline.
"""

import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

from PIL import Image
from tqdm import tqdm

# Add scripts to path
repo_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(repo_root))

from scripts.prompt_aug.enhanced_blip2_fixed import generate_responses_batch, load_models, _is_style_label


def load_blip2_prompt_aug_model(config: Dict, device: str) -> Any:
    """
    Load/Ensure EfficientNet-augmented BLIP2 models are loaded.
    
    This function uses lazy loading, so models are loaded on first use.
    We call load_models() to preload them here.
    
    Args:
        config: Configuration dictionary (may contain model-specific config)
        device: Device to use (for compatibility, but enhanced_blip2_fixed handles this internally)
    
    Returns:
        None (models are loaded globally in enhanced_blip2_fixed)
    """
    # Preload models
    load_models()
    return None  # Return None since models are global


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

    # Aggressively remove context-hint patterns that leak into the answer
    # Match patterns like "Context hints: ..." anywhere in the text, not just at start
    cleaned = re.sub(
        r'(?i)(?:^|\s)(context\s+hints?:?\s*[^\n]*)',
        '',
        cleaned
    ).strip()
    
    # Remove style labels originating from EfficientNet prompts
    if _is_style_label(cleaned):
        cleaned = ""

    # More aggressive: remove any line that starts with "Context hints:" or similar
    lines = cleaned.split('\n')
    filtered_lines = []
    for line in lines:
        line_stripped = line.strip()
        # Skip lines that are just context hints
        if re.match(r'(?i)^(context\s+hints?:?\s*)', line_stripped):
            continue
        # Skip lines that are just style labels (common in WikiArt dataset)
        if _is_style_label(line_stripped):
            continue
        filtered_lines.append(line)
    cleaned = '\n'.join(filtered_lines).strip()
    
    # Final cleanup: if text still contains "context hints" anywhere, try to extract what comes after
    if re.search(r'(?i)context\s+hints', cleaned):
        match = re.search(r'(?i)context\s+hints[^:]*:\s*(.+)', cleaned)
        if match:
            after_hints = match.group(1).strip()
            if after_hints and not _is_style_label(after_hints):
                cleaned = after_hints
            else:
                cleaned = ""
        else:
            cleaned = ""
    
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


def _build_user_prompt(question: str, base_prompt: str) -> str:
    """
    Build user prompt from question and base prompt template.
    
    Matches LAVIS behavior: if prompt exists, use prompt.format(question).
    If format() fails, fall back to simple concatenation.
    
    Args:
        question: The question string
        base_prompt: Base prompt template (may contain "{}" placeholder)
    
    Returns:
        Formatted user prompt
    """
    if not base_prompt:
        return question
    
    # Use format() to match LAVIS behavior exactly
    # If prompt contains "{}", question will replace it
    # If prompt doesn't contain "{}", format() returns prompt unchanged (and question is lost)
    # This matches LAVIS behavior, which expects prompt to contain "{}"
    try:
        return base_prompt.format(question)
    except (KeyError, ValueError) as e:
        # If format() fails (e.g., prompt has other formatting like {name}),
        # fall back to simple concatenation for compatibility
        logging.warning(
            f"Prompt format error: {e}. Using simple concatenation. "
            f"Prompt should contain '{{}}' placeholder for question."
        )
        return f"{base_prompt} {question}"


def predict_answers_blip2_prompt_aug(
    model: Any,
    dataset: Any,
    config: Dict,
    batch_size: int,
    device: str
) -> List[Dict]:
    """
    Run EfficientNet-augmented BLIP2 model inference on VQAv2 dataset.
    
    This implementation follows the simpler approach of blip2_lavis.py
    to avoid complex batch processing issues.
    
    Args:
        model: Not used (models are loaded globally in enhanced_blip2_fixed)
        dataset: Dataset object that supports __getitem__ and __len__
        config: Configuration dictionary
        batch_size: Batch size for inference
        device: Device to run inference on
    
    Returns:
        List of predictions with question_id and answer
    """
    # Preload models
    load_models()
    
    run_cfg = config.get("run", {})
    base_prompt = run_cfg.get("prompt", "")
    num_beams = run_cfg.get("num_beams", 5)
    max_len = run_cfg.get("max_len", 10)
    
    model_cfg = config.get("model_config", {})
    efficientnet_cfg = model_cfg.get("efficientnet", {})
    topk = efficientnet_cfg.get("topk", 3)
    prompt_template = efficientnet_cfg.get("prompt_template", "Image context hints: {labels}. ")
    max_new_tokens = model_cfg.get("max_new_tokens", max_len)
    temperature = model_cfg.get("temperature", 0.0)
    
    predictions = []
    
    # Import required functions
    from scripts.prompt_aug.enhanced_blip2_fixed import _classify_image, _build_augmented_prompt, _run_blip2
    
    # Process samples one by one to avoid batch processing issues
    for i in tqdm(range(len(dataset)), desc="Running inference"):
        sample = dataset[i]
        question = sample["text_input"]
        question_id = sample["question_id"]
        
        # Load image
        image_path = _get_image_path(sample)
        image = Image.open(image_path).convert("RGB")
        
        # Build user prompt
        user_prompt = _build_user_prompt(question, base_prompt)
        
        # Classify image and build augmented prompt
        labels = _classify_image(image, topk)
        aug_prompt = _build_augmented_prompt(labels, prompt_template, user_prompt)
        
        # Generate answer using single sample generation
        answer = _run_blip2(
            image=image,
            prompt=aug_prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            num_beams=num_beams,
        )
        
        # Simple cleaning - just strip whitespace
        cleaned_answer = answer.strip()
        
        # Log empty answers for debugging
        if not cleaned_answer:
            logging.warning(f"Empty answer for question_id {question_id}, raw answer: {repr(answer)}")
        
        predictions.append({
            "question_id": question_id,
            "answer": cleaned_answer
        })
    
    return predictions

