#!/usr/bin/env python3
"""
BLIP2 KV Modulation ArtQuest model backend for ArtQuest evaluation.

This module provides an adapter for using BLIP2 with KV modulation models
on ArtQuest dataset. It reuses the BLIP2 KV modulation model loader
and adapts the prediction function for ArtQuest's specific requirements
(image loading, context handling, etc.).
"""

import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch
from PIL import Image
from tqdm import tqdm

# Add LAVIS to path for processors
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "blip2" / "LAVIS"))

from lavis.processors import load_processor


def predict_answers_blip2_kv_artquest(
    model: Any,
    dataset: Any,
    config: Dict,
    batch_size: int,
    device: str
) -> List[Dict]:
    """
    Run BLIP2 with KV modulation model inference on ArtQuest dataset.
    
    This function adapts BLIP2 KV Modulation for ArtQuest by:
    1. Loading images from image filenames
    2. Processing images using LAVIS processors
    3. Combining question and context into text_input
    4. Running inference for both original and retrieved contexts
    5. Returning predictions with both answers
    
    Args:
        model: Loaded BLIP2 KV Modulation model
        dataset: ArtQuest dataset object that supports __getitem__ and __len__
        config: Configuration dictionary
        batch_size: Batch size for inference
        device: Device to run inference on
    
    Returns:
        List of predictions with question_id, answer, and answer_retrieved
    """
    model.eval()
    model = model.to(device)
    
    run_cfg = config.get("run", {})
    prompt = run_cfg.get("prompt", "Question: {} Short answer:")
    num_beams = run_cfg.get("num_beams", 5)
    max_len = run_cfg.get("max_len", 10)
    min_len = run_cfg.get("min_len", 1)
    inference_method = run_cfg.get("inference_method", "generate")
    
    # Get image root path from config
    data_cfg = config.get("data", {})
    image_root = data_cfg.get("image_root", "artquest/data/SemArt/Images")
    
    # Convert to absolute path if relative
    if not os.path.isabs(image_root):
        image_root = os.path.join(Path(__file__).parent.parent.parent.parent, image_root)
    
    # Load image processor (same as used in VQAv2)
    # We'll use the same processor as BLIP2 for consistency
    try:
        vis_processor = load_processor("blip_image_eval", {"image_size": 224})
    except:
        # Fallback: try to get from model if available
        if hasattr(model, 'vis_processor'):
            vis_processor = model.vis_processor
        else:
            # Simple fallback processor
            from torchvision import transforms
            vis_processor = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                   std=[0.26862954, 0.26130258, 0.27577711])
            ])
    
    predictions = []
    
    import logging
    logging.info(f"Using blip2_kv_artquest: batch_size={batch_size}, image_root={image_root}")
    
    def _normalize_text(value: Any) -> str:
        """Convert NaN/None values to empty strings and strip whitespace."""
        if value is None:
            return ""
        if isinstance(value, float) and math.isnan(value):
            return ""
        text = str(value)
        return text.strip()

    def build_text_input(question: str, context: str, prompt_template: str) -> str:
        """Build text input from question and context using prompt template."""
        question_clean = _normalize_text(question)
        context_clean = _normalize_text(context)
        template = prompt_template or ""

        body = question_clean
        if context_clean:
            body = f"{question_clean}\nContext: {context_clean}".strip()

        text = ""
        try:
            if "{question}" in template or "{context}" in template:
                text = template.format(question=question_clean, context=context_clean)
            else:
                placeholder_count = template.count("{}")
                if placeholder_count >= 2:
                    text = template.format(question_clean, context_clean)
                elif placeholder_count == 1:
                    text = template.format(body)
                else:
                    text = f"{template} {body}".strip()
        except (KeyError, IndexError, ValueError):
            text = ""

        if not text.strip():
            text = body

        return text.strip()
    
    def load_and_process_image(image_name: str, image_root: str, vis_processor):
        """Load image from filename and process it."""
        # Debug: check if image_name is actually a tensor
        if isinstance(image_name, torch.Tensor):
            logging.warning(f"image_name is a tensor, not a string. Shape: {image_name.shape}, dtype: {image_name.dtype}")
            # If it's already a processed image tensor, return it directly
            if image_name.dim() == 3 and image_name.shape[0] == 3:
                return image_name
            else:
                # Try to convert back to string if possible
                try:
                    # Check if it's a scalar tensor that can be converted to string
                    if image_name.numel() == 1:
                        image_name = str(image_name.item())
                        logging.warning(f"Converted scalar tensor to string: {image_name}")
                    else:
                        # If it's a multi-element tensor, we can't use it as filename
                        # Return a dummy image
                        logging.error(f"Cannot convert multi-element tensor to string. Using dummy image.")
                        return torch.zeros((3, 224, 224))
                except:
                    logging.error(f"Failed to convert tensor to string. Using dummy image.")
                    return torch.zeros((3, 224, 224))
        
        # Now image_name should be a string
        if not isinstance(image_name, str):
            logging.error(f"image_name is not a string: {type(image_name)}, value: {image_name}")
            return torch.zeros((3, 224, 224))
        
        image_path = os.path.join(image_root, image_name)
        if not os.path.exists(image_path):
            # Try alternative paths
            alt_paths = [
                os.path.join(image_root, os.path.basename(image_name)),
                image_name,  # Try as absolute path
            ]
            for alt_path in alt_paths:
                if os.path.exists(alt_path):
                    image_path = alt_path
                    break
        
        image = Image.open(image_path).convert("RGB")
        processed_image = vis_processor(image)
        return processed_image
    
    with torch.no_grad():
        for i in tqdm(range(0, len(dataset), batch_size), desc="Running inference"):
            batch = []
            question_ids = []
            processed_images = []
            text_inputs = []
            
            # Load batch samples
            for j in range(i, min(i + batch_size, len(dataset))):
                sample = dataset[j]
                batch.append(sample)
                question_ids.append(sample.get("question_id", j))
                
                # Extract data from sample - handle different dataset formats
                if isinstance(sample, dict):
                    # Standard ArtQuest dataset format
                    image_name = sample.get("image", "")
                    question = sample.get("question", "")
                    context = sample.get("context", "")
                    
                    # Check if image is already a tensor (from adapter dataset)
                    if isinstance(image_name, torch.Tensor):
                        processed_image = image_name
                    else:
                        # Load image from filename
                        try:
                            processed_image = load_and_process_image(image_name, image_root, vis_processor)
                        except Exception as e:
                            logging.warning(f"Failed to load image {image_name}: {e}")
                            processed_image = torch.zeros((3, 224, 224))
                else:
                    # Adapter dataset format (from create_artquest_adapter_dataset)
                    # These samples should already have processed images
                    processed_image = getattr(sample, "image", torch.zeros((3, 224, 224)))
                    raw_text = getattr(sample, "text_input", "")
                    question = raw_text
                    context = ""
                
                # If question is empty, try to extract from text_input
                if not question and hasattr(sample, "text_input"):
                    text_input = sample.text_input
                    # Try to extract question from text_input format: "question Context: context"
                    if "Context:" in text_input:
                        question_part, _, context_part = text_input.partition("Context:")
                        question = question_part.strip()
                        context = context_part.strip()
                    else:
                        question = text_input
                        context = ""
                
                # Convert to tensor if needed and add batch dimension
                if not isinstance(processed_image, torch.Tensor):
                    processed_image = torch.tensor(processed_image)
                if processed_image.dim() == 3:
                    processed_image = processed_image.unsqueeze(0)
                processed_image = processed_image.to(device)
                
                processed_images.append(processed_image)
                
                # Build text input with context
                text_input = build_text_input(question, context, prompt)
                
                text_inputs.append(text_input)
            
            # Stack images for batch processing
            if processed_images:
                images_batch = torch.cat(processed_images, dim=0)
                
                samples = {
                    "image": images_batch,
                    "text_input": text_inputs,
                }
                
                answers = model.predict_answers(
                    samples=samples,
                    num_beams=num_beams,
                    inference_method=inference_method,
                    max_len=max_len,
                    min_len=min_len,
                )
                
                for qid, answer in zip(question_ids, answers):
                    predictions.append({
                        "question_id": qid,
                        "answer": answer.strip() if isinstance(answer, str) else str(answer).strip(),
                    })
    
    return predictions
