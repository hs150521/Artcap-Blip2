#!/usr/bin/env python3
"""Gated BLIP2 backend for VQAv2 evaluation."""

from __future__ import annotations

import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from tqdm import tqdm

# Add project root to path to ensure models.Gated can be found
import sys
from pathlib import Path

# Ensure project root is in sys.path for models.Gated import
project_root = Path(__file__).parent.parent.parent.parent  # Go up to project root
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Now try the import
try:
    from models.Gated.modules import Blip2OPTGated  # type: ignore  # noqa: E402
except ImportError:
    # Fallback: try direct import from Gated directory
    gated_dir = project_root / "models" / "Gated"
    if str(gated_dir) not in sys.path:
        sys.path.insert(0, str(gated_dir))
    from modules import Blip2OPTGated  # type: ignore  # noqa: E402


LOGGER = logging.getLogger(__name__)


def _to_abs_path(path_str: str | None) -> str | None:
    if not path_str:
        return None
    path = Path(path_str)
    if path.is_absolute():
        return str(path)
    return str((REPO_ROOT / path).resolve())


def _build_model_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Translate evaluation config into Gated model-friendly structure."""

    backend_cfg: Dict[str, Any] = config.get("model_config", {})
    
    model_cfg = dict(backend_cfg.get("model", {}))
    architecture_cfg = dict(backend_cfg.get("architecture", {}))
    gating_cfg = dict(backend_cfg.get("gating", {}))
    prompt_mapper_cfg = dict(backend_cfg.get("prompt_mapper", {}))
    lora_cfg = dict(backend_cfg.get("lora", {}))
    
    # Handle efficientnet checkpoint path
    efficientnet_ckpt = backend_cfg.get("efficientnet_checkpoint")
    if efficientnet_ckpt:
        efficientnet_ckpt = _to_abs_path(efficientnet_ckpt)
    
    # Build model config dict
    result = {
        "vit_model": architecture_cfg.get("vit_model", "eva_clip_g"),
        "img_size": architecture_cfg.get("img_size", 224),
        "freeze_vit": architecture_cfg.get("freeze_vit", True),
        "num_query_token": architecture_cfg.get("num_query_token", 32),
        "opt_model": model_cfg.get("opt_model", "facebook/opt-2.7b"),
        "prompt": model_cfg.get("prompt", ""),
        "max_txt_len": architecture_cfg.get("max_txt_len", 32),
        "efficientnet_checkpoint": efficientnet_ckpt,
        "efficientnet_output_dim": model_cfg.get("efficientnet_output_dim", 768),
        "convert_from_blip_norm": model_cfg.get("convert_from_blip_norm", True),
        "gating_config": gating_cfg if gating_cfg else None,
        "prompt_mapper_cfg": prompt_mapper_cfg if prompt_mapper_cfg else None,
        "lora_config": lora_cfg if lora_cfg else None,
    }
    
    return result


def _normalize_question_id(question_id: Any) -> Any:
    if isinstance(question_id, torch.Tensor):
        if question_id.numel() == 1:
            return int(question_id.item())
        return question_id.detach().cpu().tolist()
    if isinstance(question_id, (int, np.integer)):
        return int(question_id)
    if isinstance(question_id, float) and question_id.is_integer():
        return int(question_id)
    return question_id


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
    
    # Remove "Long answer:" or "Short answer:" patterns that might appear in output
    # These are common artifacts where the model repeats the prompt
    cleaned = re.sub(
        r'(?i)(?:^|\s)(long\s+answer:?\s*)',
        '',
        cleaned
    ).strip()
    
    cleaned = re.sub(
        r'(?i)(?:^|\s)(short\s+answer:?\s*)',
        '',
        cleaned
    ).strip()
    
    # Remove patterns like "no Long answer: no" or "Yes Long answer: No"
    # Extract the first part before "Long answer:"
    if re.search(r'(?i)long\s+answer:', cleaned):
        match = re.match(r'^(.+?)\s+long\s+answer:.*$', cleaned, re.IGNORECASE)
        if match:
            cleaned = match.group(1).strip()
    
    # Remove patterns like "none Long answer: none"
    if re.search(r'(?i)long\s+answer:', cleaned):
        # Try to extract meaningful content before "Long answer:"
        parts = re.split(r'(?i)\s+long\s+answer:', cleaned, 1)
        if len(parts) > 1:
            before = parts[0].strip()
            after = parts[1].strip()
            # If before and after are similar (like "none" and "none"), use before
            if before.lower() == after.lower():
                cleaned = before
            else:
                # Use the part before "Long answer:" if it's meaningful
                cleaned = before if before else after
    
    # Remove lines that are just "Long answer:" or "Short answer:"
    lines = cleaned.split('\n')
    filtered_lines = []
    for line in lines:
        line_stripped = line.strip()
        # Skip lines that are just prompt artifacts
        if re.match(r'(?i)^(long\s+answer:?|short\s+answer:?)\s*$', line_stripped):
            continue
        filtered_lines.append(line)
    cleaned = '\n'.join(filtered_lines).strip()
    
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
    
    # Remove multiple consecutive spaces
    cleaned = ' '.join(cleaned.split())
    
    # Remove trailing punctuation artifacts
    cleaned = cleaned.rstrip(":.,;")
    
    return cleaned.strip()


def load_blip2_gated_model(config: Dict[str, Any], device: str) -> Any:
    """Load the Gated BLIP2 model for evaluation."""

    torch_device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))

    model_cfg = _build_model_config(config)
    backend_cfg = config.get("model_config", {})

    checkpoint_path = backend_cfg.get("checkpoint") or backend_cfg.get("checkpoint_path")
    checkpoint_path = _to_abs_path(checkpoint_path)

    if not checkpoint_path:
        raise ValueError("Gated backend requires 'model_config.checkpoint' in the config file")

    LOGGER.info("Loading Gated BLIP2 model from %s", checkpoint_path)
    
    # Initialize model
    model = Blip2OPTGated(**model_cfg)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    elif "model" in checkpoint:
        model.load_state_dict(checkpoint["model"], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)

    model.eval()
    model.to(torch_device)
    return model


def predict_answers_blip2_gated(
    model: Any,
    dataset: Any,
    config: Dict[str, Any],
    batch_size: int,
    device: str,
) -> List[Dict[str, Any]]:
    """Run inference with Gated BLIP2 on the VQAv2 dataset."""

    torch_device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model.eval()
    model.to(torch_device)

    run_cfg = config.get("run", {})
    prompt = run_cfg.get("prompt", "")
    num_beams = run_cfg.get("num_beams", 5)
    max_len = run_cfg.get("max_len", 10)
    min_len = run_cfg.get("min_len", 1)
    length_penalty = run_cfg.get("length_penalty", 0.0)

    predictions: List[Dict[str, Any]] = []

    vision_param = next(model.visual_encoder.parameters(), None) if hasattr(model, "visual_encoder") else None
    vision_dtype = vision_param.dtype if vision_param is not None else torch.float32

    # Helper function to apply prompt template
    def _apply_prompt(question: str, prompt_template: str) -> str:
        """Apply prompt template to question."""
        if not prompt_template:
            return question
        try:
            return prompt_template.format(question)
        except (KeyError, ValueError):
            # If format fails, fall back to simple concatenation
            return f"{prompt_template} {question}"

    for start in tqdm(range(0, len(dataset), batch_size), desc="Running inference"):
        end = min(start + batch_size, len(dataset))
        batch_samples = [dataset[idx] for idx in range(start, end)]

        if not batch_samples:
            continue

        images = torch.stack([sample["image"] for sample in batch_samples]).to(torch_device, non_blocking=True)
        if images.dtype != vision_dtype:
            images = images.to(dtype=vision_dtype)

        # Apply prompt template to text inputs
        text_inputs = [_apply_prompt(sample["text_input"], prompt) for sample in batch_samples]
        question_ids = [sample.get("question_id", idx) for idx, sample in enumerate(batch_samples, start=start)]

        with torch.no_grad():
            answers = model.predict_answers(
                samples={
                    "image": images,
                    "text_input": text_inputs,
                },
                num_beams=num_beams,
                max_len=max_len,
                min_len=min_len,
                length_penalty=length_penalty,
            )

        for qid, answer in zip(question_ids, answers):
            # Clean answer: remove common prompt artifacts
            if isinstance(answer, str):
                cleaned_answer = _clean_answer(answer)
                # Log empty answers for debugging
                if not cleaned_answer and answer.strip():
                    LOGGER.warning(
                        f"Answer was cleaned to empty for question_id {qid}, "
                        f"original: {repr(answer)}"
                    )
            else:
                cleaned_answer = answer
            
            predictions.append({
                "question_id": _normalize_question_id(qid),
                "answer": cleaned_answer,
            })

    return predictions

