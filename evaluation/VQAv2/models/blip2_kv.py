#!/usr/bin/env python3
"""KV-modulated BLIP2 backend for VQAv2 evaluation."""

from __future__ import annotations

import importlib.util
import logging
import re
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from tqdm import tqdm

# Ensure the KV package is importable
REPO_ROOT = Path(__file__).resolve().parents[3]
KV_ROOT = REPO_ROOT / "models" / "KV"
if str(KV_ROOT) not in sys.path:
    sys.path.insert(0, str(KV_ROOT))

from utils.model_loader import load_blip2_kv_modulated_model  # type: ignore  # noqa: E402


LOGGER = logging.getLogger(__name__)


def _to_abs_path(path_str: str | None) -> str | None:
    if not path_str:
        return None
    path = Path(path_str)
    if path.is_absolute():
        return str(path)
    return str((REPO_ROOT / path).resolve())


def _build_loader_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Translate evaluation config into loader-friendly structure."""

    backend_cfg: Dict[str, Any] = config.get("model_config", {})

    model_cfg = dict(backend_cfg.get("model", {}))
    architecture_cfg = dict(backend_cfg.get("architecture", {}))
    kv_cfg = dict(backend_cfg.get("kv_modulation", {}))

    efficientnet_ckpt = backend_cfg.get("efficientnet_checkpoint")
    if efficientnet_ckpt and "efficientnet_checkpoint" not in model_cfg:
        model_cfg["efficientnet_checkpoint"] = efficientnet_ckpt

    if "efficientnet_checkpoint" in model_cfg:
        model_cfg["efficientnet_checkpoint"] = _to_abs_path(model_cfg["efficientnet_checkpoint"])

    # Ensure opt_model is passed through
    opt_model_path = backend_cfg.get("opt_model")
    if opt_model_path and "opt_model" not in model_cfg:
        model_cfg["opt_model"] = opt_model_path

    if "opt_model" in model_cfg:
        model_cfg["opt_model"] = _to_abs_path(model_cfg["opt_model"])

    return {
        "model": model_cfg,
        "architecture": architecture_cfg,
        "kv_modulation": kv_cfg,
    }


@contextmanager
def _kv_models_package():
    models_package_path = KV_ROOT / "models"
    previous_module = sys.modules.get("models")
    if previous_module and getattr(previous_module, "__file__", "").startswith(str(models_package_path)):
        # Already set to KV models package
        yield
        return

    spec = importlib.util.spec_from_file_location(
        "models",
        models_package_path / "__init__.py",
        submodule_search_locations=[str(models_package_path)],
    )
    module = importlib.util.module_from_spec(spec)

    try:
        sys.modules["models"] = module
        assert spec.loader is not None
        spec.loader.exec_module(module)
        yield
    finally:
        if previous_module is not None:
            sys.modules["models"] = previous_module
        else:
            sys.modules.pop("models", None)


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


def load_blip2_kv_model(config: Dict[str, Any], device: str) -> Any:
    """Load the KV-modulated BLIP2 model for evaluation."""

    torch_device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))

    loader_cfg = _build_loader_config(config)
    backend_cfg = config.get("model_config", {})

    checkpoint_path = backend_cfg.get("checkpoint") or backend_cfg.get("checkpoint_path")
    checkpoint_path = _to_abs_path(checkpoint_path)

    if not checkpoint_path:
        raise ValueError("KV backend requires 'model_config.checkpoint' in the config file")

    LOGGER.info("Loading KV-modulated BLIP2 model from %s", checkpoint_path)
    with _kv_models_package():
        model = load_blip2_kv_modulated_model(
            config=loader_cfg,
            device=torch_device,
            checkpoint_path=checkpoint_path,
        )

    model.eval()
    return model


def predict_answers_blip2_kv(
    model: Any,
    dataset: Any,
    config: Dict[str, Any],
    batch_size: int,
    device: str,
) -> List[Dict[str, Any]]:
    """Run inference with KV-modulated BLIP2 on the VQAv2 dataset."""

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

    for start in tqdm(range(0, len(dataset), batch_size), desc="Running inference"):
        end = min(start + batch_size, len(dataset))
        batch_samples = [dataset[idx] for idx in range(start, end)]

        if not batch_samples:
            continue

        images = torch.stack([sample["image"] for sample in batch_samples]).to(torch_device, non_blocking=True)
        if images.dtype != vision_dtype:
            images = images.to(dtype=vision_dtype)

        text_inputs = [sample["text_input"] for sample in batch_samples]
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
                prompt=prompt,
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


