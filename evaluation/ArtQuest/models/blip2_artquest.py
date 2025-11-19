#!/usr/bin/env python3
"""
BLIP2 ArtQuest model backend for ArtQuest evaluation.

This module provides an adapter for using BLIP2 models on ArtQuest dataset.
It reuses the BLIP2 LAVIS model loader and adapts the prediction function
for ArtQuest's specific requirements (image loading, context handling, etc.).
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


def predict_answers_blip2_artquest(
    model: Any,
    dataset: Any,
    config: Dict,
    batch_size: int,
    device: str
) -> List[Dict]:
    """
    Run BLIP2 model inference on ArtQuest dataset.
    
    This function adapts BLIP2 for ArtQuest by:
    1. Loading images from image filenames
    2. Processing images using LAVIS processors
    3. Combining question and context into text_input
    4. Running inference for both original and retrieved contexts
    5. Returning predictions with both answers
    
    Args:
        model: Loaded BLIP2 LAVIS model
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

    # Determine how much formatting should be delegated to the underlying LAVIS model.
    # If the template expects both question and context placeholders, we finish formatting
    # inside this adapter and pass an empty prompt to avoid a second formatting pass.
    prompt_for_model = prompt
    if prompt:
        has_named_slots = ("{question}" in prompt) or ("{context}" in prompt)
        if has_named_slots or prompt.count("{}") >= 2:
            prompt_for_model = ""
    
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
    logging.info(f"Using blip2_artquest: batch_size={batch_size}, image_root={image_root}")
    
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

        try:
            if "{question}" in template or "{context}" in template:
                return template.format(question=question_clean, context=context_clean)

            placeholder_count = template.count("{}")
            if placeholder_count >= 2:
                return template.format(question_clean, context_clean)
            if placeholder_count == 1:
                combined = f"Question: {question_clean}\nContext: {context_clean}".strip()
                return template.format(combined)
        except (KeyError, IndexError, ValueError):
            pass

        base = f"Question: {question_clean}\nContext: {context_clean}".strip()
        return f"{base}\nShort answer:".strip() if base else "Short answer:"
    
    def load_and_process_image(image_name: str, image_root: str, vis_processor):
        """Load image from filename and process it."""
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
            
            # Load batch samples
            for j in range(i, min(i + batch_size, len(dataset))):
                sample = dataset[j]
                batch.append(sample)
                question_ids.append(sample.get("question_id", j))
            
            # Process each sample in batch
            for idx, sample in enumerate(batch):
                # Extract data from sample - handle different dataset formats
                if isinstance(sample, dict):
                    # Standard ArtQuest dataset format
                    image_source = sample.get("image", "")
                    question = sample.get("question", "")
                    context = sample.get("context", "")
                    candidate_context = sample.get("candidate_context", context)
                else:
                    # Adapter dataset format (from create_artquest_adapter_dataset)
                    image_source = getattr(sample, "image", "")
                    question = getattr(sample, "text_input", "")
                    context = ""  # Context is already combined in text_input
                    candidate_context = ""
                
                # If question is empty, try to extract from text_input
                if not question and hasattr(sample, "text_input"):
                    text_input = sample.text_input
                    # Try to extract question from text_input format: "question Context: context"
                    if "Context:" in text_input:
                        question = text_input.split("Context:")[0].strip()
                    else:
                        question = text_input
                
                # Load and process image
                processed_image = None
                image_name = None

                if isinstance(image_source, torch.Tensor):
                    processed_image = image_source
                elif isinstance(image_source, (bytes, bytearray)):
                    image_name = image_source.decode("utf-8")
                elif hasattr(image_source, "__fspath__"):
                    image_name = os.fspath(image_source)
                else:
                    image_name = image_source

                if processed_image is None:
                    if not isinstance(image_name, str):
                        image_name = str(image_name) if image_name is not None else ""
                    try:
                        processed_image = load_and_process_image(image_name, image_root, vis_processor)
                    except Exception as e:
                        logging.warning(f"Failed to load image {image_name}: {e}")
                        # Use a dummy image if loading fails
                        processed_image = torch.zeros((3, 224, 224))
                else:
                    # Ensure tensor is on CPU for consistent downstream handling
                    processed_image = processed_image.detach().cpu()
                
                # Convert to tensor if needed and add batch dimension
                if not isinstance(processed_image, torch.Tensor):
                    processed_image = torch.tensor(processed_image)
                if processed_image.dim() == 3:
                    processed_image = processed_image.unsqueeze(0)
                processed_image = processed_image.to(device)
                
                # Build text inputs for both contexts
                text_input_original = build_text_input(question, context, prompt)
                text_input_retrieved = build_text_input(question, candidate_context, prompt)
                
                # Create samples dict for original context
                samples_original = {
                    "image": processed_image,
                    "text_input": [text_input_original],
                }
                
                # Generate answer for original context
                answers_original = model.predict_answers(
                    samples=samples_original,
                    num_beams=num_beams,
                    inference_method=inference_method,
                    max_len=max_len,
                    min_len=min_len,
                    prompt=prompt_for_model,
                )
                answer_original = answers_original[0] if isinstance(answers_original, list) else answers_original
                
                # Create samples dict for retrieved context
                samples_retrieved = {
                    "image": processed_image,
                    "text_input": [text_input_retrieved],
                }
                
                # Generate answer for retrieved context
                answers_retrieved = model.predict_answers(
                    samples=samples_retrieved,
                    num_beams=num_beams,
                    inference_method=inference_method,
                    max_len=max_len,
                    min_len=min_len,
                    prompt=prompt_for_model,
                )
                answer_retrieved = answers_retrieved[0] if isinstance(answers_retrieved, list) else answers_retrieved
                
                predictions.append({
                    "question_id": question_ids[idx],
                    "answer": answer_original.strip() if isinstance(answer_original, str) else str(answer_original).strip(),
                    "answer_retrieved": answer_retrieved.strip() if isinstance(answer_retrieved, str) else str(answer_retrieved).strip()
                })
    
    return predictions

