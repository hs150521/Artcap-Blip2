#!/usr/bin/env python3
"""
Fixed-configuration Enhanced BLIP-2 prompt generation.

This module provides a simplified interface for generating responses using
EfficientNet-augmented BLIP-2. Models are loaded from fixed local paths with
no command-line arguments required.

Usage:
    from scripts.prompt_aug.enhanced_blip2_fixed import generate_response
    from PIL import Image
    
    image = Image.open("path/to/image.jpg")
    response = generate_response(image, "Describe the artistic style.")
"""

import logging
import re
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import torch
from PIL import Image
from torch import nn
from torchvision import transforms

try:
    from transformers import Blip2ForConditionalGeneration, Blip2Processor
except ImportError as exc:
    raise SystemExit(
        "transformers is required. Install it with `pip install transformers`."
    ) from exc


# Fixed model paths
REPO_ROOT = Path(__file__).resolve().parents[2]
BLIP2_MODEL_PATH = REPO_ROOT / "models" / "blip2-opt-2.7b" / "snapshots" / "59a1ef6c1e5117b3f65523d1c6066825bcf315e3"
EFFICIENTNET_CHECKPOINT = REPO_ROOT / "models" / "efficientnet" / "best.pt"

# EfficientNet configuration
EFFICIENTNET_VARIANT = "b3"
EFFICIENTNET_DATASET = "wikiart"

# Canonical WikiArt 27 style labels
WIKIART_STYLES: Tuple[str, ...] = (
    "Abstract_Expressionism",
    "Action_painting",
    "Analytical_Cubism",
    "Art_Nouveau",
    "Baroque",
    "Color_Field_Painting",
    "Contemporary_Realism",
    "Cubism",
    "Early_Renaissance",
    "Expressionism",
    "Fauvism",
    "High_Renaissance",
    "Impressionism",
    "Mannerism_Late_Renaissance",
    "Minimalism",
    "Naive_Art_Primitivism",
    "New_Realism",
    "Northern_Renaissance",
    "Pointillism",
    "Pop_Art",
    "Post_Impressionism",
    "Realism",
    "Rococo",
    "Romanticism",
    "Symbolism",
    "Synthetic_Cubism",
    "Ukiyo_e",
)

STYLE_LABELS_LOWER = {style.lower(): style for style in WIKIART_STYLES}


def _is_style_label(text: str) -> bool:
    """Return True if the text corresponds to a known style label."""
    if not text:
        return False
    normalized = text.strip().lower().replace(' ', '_').rstrip('.!?,')
    return normalized in STYLE_LABELS_LOWER

# Default prompt template
DEFAULT_PROMPT_TEMPLATE = "Image context hints: {labels}. "

# Global variables for lazy-loaded models
_blip2_processor: Optional[Blip2Processor] = None
_blip2_model: Optional[Blip2ForConditionalGeneration] = None
_blip2_dtype: Optional[torch.dtype] = None
_efficientnet_model: Optional[nn.Module] = None
_efficientnet_preprocess: Optional[transforms.Compose] = None
_efficientnet_categories: Optional[List[str]] = None
_device: Optional[torch.device] = None


def _get_device() -> torch.device:
    """Get the appropriate torch device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _infer_num_classes_from_state_dict(state_dict: dict) -> Optional[int]:
    """Infer classifier output dimension from an EfficientNet state dict if possible."""
    candidate_keys = (
        "classifier.1.weight",
        "classifier.weight",
        "classifier.1.bias",
        "classifier.bias",
        "fc.weight",
        "fc.bias",
    )
    for key in candidate_keys:
        tensor = state_dict.get(key)
        if isinstance(tensor, torch.Tensor):
            return tensor.shape[0]
    for key, tensor in state_dict.items():
        if not isinstance(tensor, torch.Tensor):
            continue
        if tensor.ndim == 2 and ("classifier" in key or key.endswith(".weight")):
            return tensor.shape[0]
    return None


def _load_efficientnet_model() -> Tuple[nn.Module, transforms.Compose, List[str]]:
    """Load EfficientNet model from local checkpoint."""
    from torchvision.models import (
        efficientnet_b3,
        EfficientNet_B3_Weights,
    )
    
    device = _get_device()
    variant_key = EFFICIENTNET_VARIANT.lower()
    
    if variant_key != "b3":
        raise ValueError(f"Only EfficientNet-B3 is supported, got {variant_key}")
    
    constructor, weights_enum = efficientnet_b3, EfficientNet_B3_Weights.DEFAULT
    categories = list(WIKIART_STYLES)
    
    # Load checkpoint
    checkpoint_path = Path(EFFICIENTNET_CHECKPOINT).expanduser().resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"EfficientNet checkpoint not found: {checkpoint_path}")
    
    load_kwargs = {"map_location": "cpu", "weights_only": False}
    try:
        state = torch.load(checkpoint_path, **load_kwargs)
    except TypeError:
        load_kwargs.pop("weights_only", None)
        state = torch.load(checkpoint_path, **load_kwargs)
    
    if isinstance(state, dict):
        for key in ("state_dict", "model_state_dict", "model", "model_state"):
            candidate = state.get(key)
            if isinstance(candidate, dict):
                state = candidate
                break
    
    if not isinstance(state, dict):
        raise ValueError("Unsupported EfficientNet checkpoint format; expected a state dict.")
    
    # Determine number of classes
    num_classes_override = len(categories)
    if num_classes_override is None:
        num_classes_override = _infer_num_classes_from_state_dict(state)
    
    # Create model
    model = constructor(weights=None, num_classes=num_classes_override).to(device)
    preprocess = weights_enum.DEFAULT.transforms()
    
    # Load state dict
    load_status = model.load_state_dict(state, strict=False)
    missing, unexpected = load_status.missing_keys, load_status.unexpected_keys
    if missing:
        print(
            f"Warning: EfficientNet checkpoint missing {len(missing)} parameters "
            f"(example: {missing[:5]}).",
            file=sys.stderr,
        )
    if unexpected:
        print(
            f"Warning: EfficientNet checkpoint has {len(unexpected)} unexpected parameters "
            f"(example: {unexpected[:5]}).",
            file=sys.stderr,
        )
    
    model.eval()
    return model, preprocess, categories


def _load_blip2_model() -> Tuple[Blip2Processor, Blip2ForConditionalGeneration, torch.dtype]:
    """Load BLIP2 model from local HuggingFace cache."""
    device = _get_device()
    model_path = str(BLIP2_MODEL_PATH)
    
    if not BLIP2_MODEL_PATH.exists():
        raise FileNotFoundError(f"BLIP2 model path not found: {model_path}")
    
    try:
        processor = Blip2Processor.from_pretrained(model_path, local_files_only=True)
    except Exception as exc:
        if "TokenizerFast" in str(exc) or "ModelWrapper" in str(exc):
            print(
                "Falling back to slow tokenizer for BLIP-2 processor due to fast tokenizer load issue.",
                file=sys.stderr,
            )
            processor = Blip2Processor.from_pretrained(
                model_path, local_files_only=True, use_fast=False
            )
        else:
            raise
    
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    model = Blip2ForConditionalGeneration.from_pretrained(
        model_path,
        local_files_only=True,
        torch_dtype=dtype,
    ).to(device)
    model.eval()
    
    return processor, model, dtype


def _ensure_models_loaded():
    """Ensure all models are loaded (lazy loading)."""
    global _blip2_processor, _blip2_model, _blip2_dtype
    global _efficientnet_model, _efficientnet_preprocess, _efficientnet_categories
    global _device
    
    if _device is None:
        _device = _get_device()
    
    if _blip2_processor is None or _blip2_model is None:
        print("Loading BLIP2 model...", file=sys.stderr)
        _blip2_processor, _blip2_model, _blip2_dtype = _load_blip2_model()
        print("BLIP2 model loaded.", file=sys.stderr)
    
    if _efficientnet_model is None:
        print("Loading EfficientNet model...", file=sys.stderr)
        _efficientnet_model, _efficientnet_preprocess, _efficientnet_categories = _load_efficientnet_model()
        print("EfficientNet model loaded.", file=sys.stderr)


def load_models():
    """
    Explicitly load all models.
    
    This function can be called to preload models before first use,
    or it will be called automatically on first use of generate_response().
    """
    _ensure_models_loaded()


def _looks_like_prompt_echo(text: str) -> bool:
    """
    Heuristic to detect when the model merely echoes the prompt artifacts
    (e.g., repeated 'Answer:' or question text) without producing content.
    """
    if not text:
        return True

    stripped = text.strip().lower()
    if not stripped:
        return True

    if stripped.startswith('answer briefly'):
        return True

    # Remove common artifacts
    cleaned = re.sub(r'(?i)(answer:\s*)+', '', stripped)
    cleaned = re.sub(r'(?i)(question:\s*[^\n]+)', '', cleaned)
    cleaned = re.sub(r'(?i)(optional\s+context\s+hints[^\n]*)', '', cleaned)
    cleaned = re.sub(r'[\s:.,;\-]+', '', cleaned)
    cleaned = cleaned.replace('answerbrieflyinonesentence', '')
    cleaned = cleaned.replace('answerbriefly', '')

    return cleaned == ''


def _extract_primary_answer(text: str) -> str:
    """Return the first segment that does not look like a prompt artifact."""
    if not text:
        return ""

    segments = [seg.strip() for seg in text.split('\n') if seg.strip()]
    for seg in segments:
        if not _looks_like_prompt_echo(seg):
            return seg
    return ""


@torch.no_grad()
def _classify_image(
    image: Image.Image,
    topk: int,
) -> List[Tuple[str, float]]:
    """Classify image using EfficientNet and return top-K labels with scores."""
    _ensure_models_loaded()
    
    tensor = _efficientnet_preprocess(image).unsqueeze(0).to(_device)
    logits = _efficientnet_model(tensor)
    probabilities = logits.softmax(dim=1)
    values, indices = probabilities.topk(topk, dim=1)
    
    results = []
    for score, idx in zip(values.squeeze(0).tolist(), indices.squeeze(0).tolist()):
        label = _efficientnet_categories[idx] if idx < len(_efficientnet_categories) else f"class_{idx}"
        results.append((label, score))
    return results


def _build_augmented_prompt(
    predicted_labels: Sequence[Tuple[str, float]],
    template: str,
    user_prompt: str,
) -> str:
    """
    Build augmented prompt by combining EfficientNet labels with user prompt.
    
    The question text stays first, optional context hints follow, and we ensure
    the prompt ends with an "Answer:" cue so the generator knows where to start
    responding.
    """
    label_str = ", ".join([label for label, _ in predicted_labels])
    template_text = template.format(labels=label_str).strip()
    user_text = user_prompt.strip()

    combined_parts = []
    if user_text:
        combined_parts.append(user_text)
    if template_text:
        combined_parts.append(template_text)

    combined = "\n".join(part for part in combined_parts if part)

    # Remove the automatic "Answer:" addition as it causes repetition
    # The model should learn to answer based on the question structure
    # normalized = combined.strip().lower()
    # if not normalized.endswith("answer:"):
    #     if combined and not combined.endswith("\n"):
    #         combined += "\n"
    #     combined += "Answer:"

    return combined


@torch.no_grad()
def _run_blip2(
    image: Image.Image,
    prompt: str,
    max_new_tokens: int = 40,
    temperature: float = 0.0,
    num_beams: int = 1,
    allow_retry: bool = True,
) -> str:
    """Run BLIP2 generation on image and prompt."""
    _ensure_models_loaded()
    
    inputs = _blip2_processor(images=image, text=prompt, return_tensors="pt")
    moved_inputs = {}
    for key, value in inputs.items():
        if isinstance(value, torch.Tensor):
            if value.dtype.is_floating_point:
                moved_inputs[key] = value.to(_device, dtype=_blip2_dtype)
            else:
                moved_inputs[key] = value.to(_device)
        else:
            moved_inputs[key] = value
    inputs = moved_inputs
    
    # Configure generation with safe defaults
    tokenizer = _blip2_processor.tokenizer
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    pad_token_id = getattr(tokenizer, "pad_token_id", None) or eos_token_id

    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "min_new_tokens": 1,
        "eos_token_id": eos_token_id,
        "pad_token_id": pad_token_id,
    }
    if num_beams > 1:
        generation_kwargs["num_beams"] = num_beams
        generation_kwargs["do_sample"] = False
    elif temperature and temperature > 0:
        generation_kwargs.update({"do_sample": True, "temperature": temperature})
    
    generated_ids = _blip2_model.generate(**inputs, **generation_kwargs)
    
    # Simple decoding like LAVIS models - decode the full sequence
    decoded = tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()
    
    # Remove the input prompt if it appears at the start (simple approach)
    if decoded.startswith(prompt.strip()):
        decoded = decoded[len(prompt.strip()):].strip()
        decoded = decoded.lstrip(":.,; \n\t")
    
    # Store original decoded for final check
    original_decoded = decoded
    
    # Simple cleanup: remove common artifacts
    # More precise pattern to remove context hints with their content
    # This pattern matches "Context hints: [content]" and removes the entire phrase
    decoded = re.sub(
        r'(?i)(?:^|\s)(?:optional\s+)?context\s+hints?:?\s*[^.!?\n]*(?:[.!?]|$)',
        '',
        decoded
    ).strip()
    
    # Additional cleanup for any remaining context hints patterns
    decoded = re.sub(
        r'(?i)context\s+hints?:?\s*[^.!?\n]*(?:[.!?]|$)',
        '',
        decoded
    ).strip()
    
    # Remove lines that contain only style labels (common in WikiArt dataset)
    lines = decoded.split('\n')
    filtered_lines = []
    for line in lines:
        line_stripped = line.strip()
        # Skip lines that are just style labels
        if _is_style_label(line_stripped):
            continue
        # Skip lines that contain context hints
        if re.search(r'(?i)context\s+hints', line_stripped):
            continue
        filtered_lines.append(line)
    decoded = '\n'.join(filtered_lines).strip()
    
    # Final cleanup: if the entire text was just context hints, return empty
    # But if there's actual content before/after context hints, keep it
    if not decoded.strip():
        # Check if the original had content outside context hints
        # Use a more sophisticated approach to extract content
        original_without_hints = re.sub(
            r'(?i)(?:^|\s)(?:optional\s+)?context\s+hints?:?\s*[^.!?\n]*(?:[.!?]|$)',
            '',
            original_decoded
        ).strip()
        # Also remove any remaining context hints
        original_without_hints = re.sub(
            r'(?i)context\s+hints?:?\s*[^.!?\n]*(?:[.!?]|$)',
            '',
            original_without_hints
        ).strip()
        if original_without_hints:
            decoded = original_without_hints
    
    # Remove multiple consecutive "Answer:" patterns
    decoded = re.sub(r'(?i)(answer:\s*)+', '', decoded).strip()
    
    # Remove lines that are just "Answer:" or "Question:"
    lines = decoded.split('\n')
    filtered_lines = []
    for line in lines:
        line_stripped = line.strip()
        # Skip lines that are just "Answer:" or variations
        if re.match(r'(?i)^answer:?\s*$', line_stripped):
            continue
        # Skip lines that are just "Question:" or variations
        if re.match(r'(?i)^question:?\s*$', line_stripped):
            continue
        # Skip lines that contain only prompt instructions (not actual answers)
        if line_stripped.lower() in ['answer briefly in one sentence', 'answer briefly']:
            continue
        filtered_lines.append(line)
    decoded = '\n'.join(filtered_lines).strip()
    
    # If we got nothing meaningful, try a retry with relaxed settings
    if allow_retry and not decoded.strip():
        retry_temperature = temperature if temperature and temperature > 0 else 0.7
        try:
            return _run_blip2(
                image=image,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=retry_temperature,
                num_beams=1,
                allow_retry=False,
            )
        except Exception as retry_exc:  # noqa: F841
            logging.warning(f"Retry generation failed: {retry_exc}")
    
    return decoded.strip()


def generate_response(
    image: Image.Image,
    prompt: str,
    topk: int = 3,
    max_new_tokens: int = 40,
    temperature: float = 0.0,
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
    num_beams: int = 1,
) -> str:
    """
    Generate a response using EfficientNet-augmented BLIP-2.
    
    Args:
        image: PIL Image to process
        prompt: User prompt/question
        topk: Number of top EfficientNet labels to include (default: 3)
        max_new_tokens: Maximum number of tokens to generate (default: 40)
        temperature: Sampling temperature, 0.0 for greedy decoding (default: 0.0)
        prompt_template: Template for building augmented prompt with labels.
                         Must include '{labels}' placeholder (default: "Image context hints: {labels}. ")
    
    Returns:
        Generated text response
    """
    # Ensure models are loaded
    _ensure_models_loaded()
    
    # Classify image with EfficientNet
    labels_with_scores = _classify_image(image, topk=topk)
    
    # Build augmented prompt
    augmented_prompt = _build_augmented_prompt(
        predicted_labels=labels_with_scores,
        template=prompt_template,
        user_prompt=prompt,
    )
    
    # Generate response with BLIP2
    response = _run_blip2(
        image=image,
        prompt=augmented_prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        num_beams=num_beams,
    )
    
    return response


@torch.no_grad()
def _classify_images_batch(
    images: List[Image.Image],
    topk: int,
) -> List[List[Tuple[str, float]]]:
    """Classify multiple images using EfficientNet and return top-K labels with scores for each."""
    _ensure_models_loaded()
    
    # Preprocess all images
    image_tensors = torch.stack([
        _efficientnet_preprocess(img) for img in images
    ]).to(_device)
    
    # Batch inference
    logits = _efficientnet_model(image_tensors)
    probabilities = logits.softmax(dim=1)
    values, indices = probabilities.topk(topk, dim=1)
    
    # Extract results for each image
    results = []
    for img_values, img_indices in zip(values, indices):
        img_results = []
        for score, idx in zip(img_values.tolist(), img_indices.tolist()):
            label = _efficientnet_categories[idx] if idx < len(_efficientnet_categories) else f"class_{idx}"
            img_results.append((label, score))
        results.append(img_results)
    
    return results


@torch.no_grad()
def _run_blip2_batch(
    images: List[Image.Image],
    prompts: List[str],
    max_new_tokens: int = 40,
    temperature: float = 0.0,
    num_beams: int = 1,
) -> List[str]:
    """Run BLIP2 generation on multiple images and prompts (batch processing)."""
    _ensure_models_loaded()
    
    # Process each sample individually to avoid batch processing issues
    responses = []
    for i in range(len(images)):
        try:
            response = _run_blip2(
                image=images[i],
                prompt=prompts[i],
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                num_beams=num_beams,
                allow_retry=False,  # Don't retry within batch to avoid infinite loops
            )
            responses.append(response)
        except Exception as e:
            logging.warning(f"Error processing sample {i}: {e}")
            responses.append("")
    
    return responses


def generate_responses_batch(
    images: List[Image.Image],
    prompts: List[str],
    topk: int = 3,
    max_new_tokens: int = 40,
    temperature: float = 0.0,
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
    num_beams: int = 1,
) -> List[str]:
    """
    Generate responses for multiple images using EfficientNet-augmented BLIP-2 (batch version).
    
    Args:
        images: List of PIL Images to process
        prompts: List of user prompts/questions (must have same length as images)
        topk: Number of top EfficientNet labels to include (default: 3)
        max_new_tokens: Maximum number of tokens to generate (default: 40)
        temperature: Sampling temperature, 0.0 for greedy decoding (default: 0.0)
        prompt_template: Template for building augmented prompt with labels.
                         Must include '{labels}' placeholder (default: "Image context hints: {labels}. ")
    
    Returns:
        List of generated text responses (same length as input images/prompts)
    """
    if len(images) != len(prompts):
        raise ValueError(f"Number of images ({len(images)}) must match number of prompts ({len(prompts)})")
    
    if len(images) == 0:
        return []
    
    # Ensure models are loaded
    _ensure_models_loaded()
    
    # Batch classify images with EfficientNet
    labels_with_scores_list = _classify_images_batch(images, topk=topk)
    
    # Build augmented prompts for each image
    augmented_prompts = []
    for labels_with_scores, user_prompt in zip(labels_with_scores_list, prompts):
        augmented_prompt = _build_augmented_prompt(
            predicted_labels=labels_with_scores,
            template=prompt_template,
            user_prompt=user_prompt,
        )
        augmented_prompts.append(augmented_prompt)
    
    # Batch generate responses with BLIP2
    responses = _run_blip2_batch(
        images=images,
        prompts=augmented_prompts,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        num_beams=num_beams,
    )
    
    return responses

