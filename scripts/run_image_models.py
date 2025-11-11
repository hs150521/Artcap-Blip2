#!/usr/bin/env python3
"""
Utility script to run EfficientNet classification and BLIP-2 captioning/chat on a remote image.

Both model families can be configured through command-line flags so you can pick different
variants or revisions without touching the underlying LAVIS codebase.
"""

import argparse
import io
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import requests
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Classify an image with EfficientNet and start a BLIP-2 conversation about it."
        )
    )
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--image-url",
        help="HTTP(S) URL of the image to process.",
    )
    source_group.add_argument(
        "--image-path",
        help="Local filesystem path to the image to process.",
    )
    parser.add_argument(
        "--efficientnet-variant",
        default="b0",
        help=(
            "EfficientNet variant to load (e.g. b0, b1, b2, ... , v2_l). "
            "Defaults to b0. See torchvision.models.efficientnet_* docs for options."
        ),
    )
    parser.add_argument(
        "--blip2-model-id",
        default="Salesforce/blip2-opt-2.7b",
        help="Hugging Face model id for BLIP-2 (e.g. Salesforce/blip2-flan-t5-xl).",
    )
    parser.add_argument(
        "--blip2-revision",
        default=None,
        help="Optional git revision/hash/tag for the BLIP-2 checkpoint.",
    )
    parser.add_argument(
        "--prompt",
        default="Describe this image in detail.",
        help="Initial user prompt for BLIP-2 to kick off the conversation.",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Torch device to run inference on (e.g. cuda:0, cuda, cpu). Defaults to cuda:0 with CPU fallback.",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=5,
        help="Number of top EfficientNet classes to display.",
    )
    return parser.parse_args()


def resolve_device(explicit_device: Optional[str]) -> torch.device:
    if explicit_device:
        device = torch.device(explicit_device)
        if device.type == "cuda" and not torch.cuda.is_available():
            print(
                f"Requested CUDA device '{explicit_device}' but CUDA is unavailable. Falling back to CPU.",
                file=sys.stderr,
            )
            return torch.device("cpu")
        return device
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def load_image_from_url(url: str) -> Image.Image:
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise SystemExit(f"Failed to download image from {url}: {exc}") from exc

    try:
        return Image.open(io.BytesIO(response.content)).convert("RGB")
    except OSError as exc:
        raise SystemExit(f"Downloaded content is not a valid image: {exc}") from exc


def load_image_from_path(path_str: str) -> Image.Image:
    path = Path(path_str).expanduser().resolve()
    if not path.exists():
        raise SystemExit(f"Image path does not exist: {path}")
    if not path.is_file():
        raise SystemExit(f"Image path is not a file: {path}")
    try:
        return Image.open(path).convert("RGB")
    except OSError as exc:
        raise SystemExit(f"Failed to open image at {path}: {exc}") from exc


def load_efficientnet(
    variant: str, device: torch.device
) -> Tuple[nn.Module, transforms.Compose, List[str]]:
    """Load EfficientNet variant from torchvision and return model, preprocessor, labels."""
    variant_key = variant.lower()

    # Lazy import to avoid pulling everything in unless needed.
    from torchvision.models import (
        efficientnet_b0,
        efficientnet_b1,
        efficientnet_b2,
        efficientnet_b3,
        efficientnet_b4,
        efficientnet_b5,
        efficientnet_b6,
        efficientnet_b7,
        efficientnet_v2_l,
        efficientnet_v2_m,
        efficientnet_v2_s,
        EfficientNet_B0_Weights,
        EfficientNet_B1_Weights,
        EfficientNet_B2_Weights,
        EfficientNet_B3_Weights,
        EfficientNet_B4_Weights,
        EfficientNet_B5_Weights,
        EfficientNet_B6_Weights,
        EfficientNet_B7_Weights,
        EfficientNet_V2_L_Weights,
        EfficientNet_V2_M_Weights,
        EfficientNet_V2_S_Weights,
    )

    variants = {
        "b0": (efficientnet_b0, EfficientNet_B0_Weights.DEFAULT),
        "b1": (efficientnet_b1, EfficientNet_B1_Weights.DEFAULT),
        "b2": (efficientnet_b2, EfficientNet_B2_Weights.DEFAULT),
        "b3": (efficientnet_b3, EfficientNet_B3_Weights.DEFAULT),
        "b4": (efficientnet_b4, EfficientNet_B4_Weights.DEFAULT),
        "b5": (efficientnet_b5, EfficientNet_B5_Weights.DEFAULT),
        "b6": (efficientnet_b6, EfficientNet_B6_Weights.DEFAULT),
        "b7": (efficientnet_b7, EfficientNet_B7_Weights.DEFAULT),
        "v2_s": (efficientnet_v2_s, EfficientNet_V2_S_Weights.DEFAULT),
        "v2_m": (efficientnet_v2_m, EfficientNet_V2_M_Weights.DEFAULT),
        "v2_l": (efficientnet_v2_l, EfficientNet_V2_L_Weights.DEFAULT),
    }

    if variant_key not in variants:
        available = ", ".join(sorted(variants))
        raise SystemExit(f"Unsupported EfficientNet variant '{variant}'. Choose from: {available}")

    constructor, weights_enum = variants[variant_key]
    weights = weights_enum
    model = constructor(weights=weights).to(device)
    model.eval()
    preprocess = weights.transforms()
    labels = weights.meta.get("categories")
    if not labels:
        raise SystemExit("EfficientNet weights metadata is missing class labels.")
    return model, preprocess, labels


def classify_image(
    model: nn.Module, preprocess: transforms.Compose, labels: List[str], image: Image.Image, device: torch.device, topk: int
) -> List[Tuple[str, float]]:
    tensor = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probabilities = torch.nn.functional.softmax(logits, dim=1)
        values, indices = probabilities.topk(topk)
    results = []
    for score, idx in zip(values[0], indices[0]):
        label = labels[int(idx)]
        results.append((label, float(score)))
    return results


def load_blip2_model(model_id: str, revision: Optional[str], device: torch.device):
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    try:
        processor = Blip2Processor.from_pretrained(model_id, revision=revision)
    except Exception as exc:
        if "PyPreTokenizerTypeWrapper" in str(exc):
            print(
                "Falling back to slow tokenizer for BLIP-2 (requires sentencepiece).",
                file=sys.stderr,
            )
            processor = Blip2Processor.from_pretrained(
                model_id, revision=revision, use_fast=False
            )
        else:
            raise
    model = Blip2ForConditionalGeneration.from_pretrained(
        model_id,
        revision=revision,
        torch_dtype=dtype,
    )
    model.to(device)
    return processor, model


def run_blip2_chat(
    processor: Blip2Processor, model: Blip2ForConditionalGeneration, image: Image.Image, prompt: str, device: torch.device
) -> str:
    inputs = processor(images=image, text=prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    if model.dtype == torch.float16:
        inputs["pixel_values"] = inputs["pixel_values"].to(device=device, dtype=torch.float16)
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=200)
    return processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    if args.image_url:
        print(f"Fetching image from URL: {args.image_url}", file=sys.stderr)
        image = load_image_from_url(args.image_url)
    else:
        print(f"Loading image from path: {args.image_path}", file=sys.stderr)
        image = load_image_from_path(args.image_path)

    print(f"Running EfficientNet-{args.efficientnet_variant} on {device}...", file=sys.stderr)
    eff_model, eff_preprocess, eff_labels = load_efficientnet(args.efficientnet_variant, device)
    eff_results = classify_image(
        eff_model, eff_preprocess, eff_labels, image, device, topk=max(1, args.topk)
    )

    print(f"Loading BLIP-2 model '{args.blip2_model_id}' (revision={args.blip2_revision}) on {device}...", file=sys.stderr)
    blip_processor, blip_model = load_blip2_model(args.blip2_model_id, args.blip2_revision, device)
    blip_response = run_blip2_chat(blip_processor, blip_model, image, args.prompt, device)

    print("\nEfficientNet Classification:")
    for label, score in eff_results:
        print(f"- {label}: {score:.4f}")

    print("\nBLIP-2 Conversation Starter:")
    print(f"[User Prompt] {args.prompt}")
    print(f"[BLIP-2 Reply] {blip_response}")


if __name__ == "__main__":
    main()
