#!/usr/bin/env python3
"""Run BLIP-2 KV-modulated VQA on a single image.

This script loads the KV-modulated BLIP-2 model together with the
EfficientNet checkpoint used to generate KV prefixes, then answers a
provided question about a given image. It mirrors the overall structure of
``scripts/prompt_aug/enhanced_blip2_prompt.py`` while using the specialized
KV modulation utilities that live under ``models/KV``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from PIL import Image
from torchvision import transforms

try:
    import yaml
except ImportError as exc:  # pragma: no cover - dependency guard
    raise SystemExit("PyYAML is required. Install it with `pip install pyyaml`.") from exc


CURRENT_DIR = Path(__file__).resolve().parent
KV_DIR = CURRENT_DIR.parent
REPO_ROOT = CURRENT_DIR.parents[2]

if str(KV_DIR) not in sys.path:
    sys.path.insert(0, str(KV_DIR))

from utils.model_loader import load_blip2_kv_modulated_model


BLIP2_MEAN = (0.48145466, 0.4578275, 0.40821073)
BLIP2_STD = (0.26862954, 0.26130258, 0.27577711)

DEFAULT_CONFIG_PATH = REPO_ROOT / "models" / "KV" / "config" / "config_kv_modulation.yaml"
DEFAULT_CHECKPOINT_PATH = REPO_ROOT / "models" / "KV" / "outputs" / "best_checkpoint.pt"
DEFAULT_EFFICIENTNET_CHECKPOINT = REPO_ROOT / "runs" / "efficientnet-28" / "best.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load the BLIP-2 KV-modulated model and answer a question about a single image."
        )
    )
    parser.add_argument(
        "--image-path",
        required=True,
        help="Path to the image to analyze.",
    )
    parser.add_argument(
        "--question",
        default="Describe the image.",
        help="Question to ask about the image.",
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to a YAML config describing the KV model setup.",
    )
    parser.add_argument(
        "--checkpoint",
        default=str(DEFAULT_CHECKPOINT_PATH),
        help="Path to the trained BLIP-2 KV model checkpoint to load.",
    )
    parser.add_argument(
        "--efficientnet-checkpoint",
        default=str(DEFAULT_EFFICIENTNET_CHECKPOINT),
        help="Path to the EfficientNet checkpoint used for KV prefix generation.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device specifier (e.g. cuda, cuda:0, cpu). Default auto-selects GPU if available.",
    )
    parser.add_argument(
        "--num-beams",
        type=int,
        default=5,
        help="Beam width for answer generation.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=20,
        help="Maximum number of new tokens to generate for the answer.",
    )
    parser.add_argument(
        "--min-new-tokens",
        type=int,
        default=1,
        help="Minimum number of new tokens to generate for the answer.",
    )
    parser.add_argument(
        "--prompt-template",
        default=None,
        help=(
            "Optional prompt template applied to each question before feeding it to the model."
            " Use '{question}' as a placeholder for the raw question."
        ),
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="Image resolution expected by the vision encoder (default: 224).",
    )
    parser.add_argument(
        "--num-query-token",
        type=int,
        default=32,
        help="Number of Q-former query tokens (must match training).",
    )
    parser.add_argument(
        "--max-text-length",
        type=int,
        default=32,
        help="Maximum text length for OPT tokenizer inputs.",
    )
    parser.add_argument(
        "--vit-model",
        default="eva_clip_g",
        help="Vision backbone identifier (should match training config).",
    )
    parser.add_argument(
        "--opt-model",
        default="facebook/opt-2.7b",
        help="OPT language model identifier (should match training config).",
    )
    parser.add_argument(
        "--disable-kv-modulation",
        action="store_true",
        help="Disable EfficientNet-derived KV prefixes (for ablations).",
    )
    return parser.parse_args()


def resolve_device(spec: Optional[str]) -> torch.device:
    if spec:
        device = torch.device(spec)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise SystemExit(f"Requested CUDA device '{spec}' but CUDA is unavailable.")
        return device
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def load_image(path_str: str, image_size: int) -> torch.Tensor:
    path = Path(path_str).expanduser().resolve()
    if not path.exists():
        raise SystemExit(f"Image path does not exist: {path}")
    if not path.is_file():
        raise SystemExit(f"Image path is not a file: {path}")

    image = Image.open(path).convert("RGB")
    preprocess = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=BLIP2_MEAN, std=BLIP2_STD),
        ]
    )
    return preprocess(image)


def load_config(path: str) -> Dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as handle:
            raw_cfg = yaml.safe_load(handle) or {}
    else:
        raw_cfg = {}

    if not isinstance(raw_cfg, dict):
        raise SystemExit(f"Config file at {config_path} does not define a mapping.")

    raw_cfg.setdefault("model", {})
    raw_cfg.setdefault("kv_modulation", {})
    raw_cfg.setdefault("architecture", {})
    raw_cfg.setdefault("generation", {})
    raw_cfg.setdefault("training", {})

    return raw_cfg


@torch.no_grad()
def run_vqa(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)

    image_tensor = load_image(args.image_path, args.image_size).unsqueeze(0).to(device)

    config = load_config(args.config)
    config["model"]["efficientnet_checkpoint"] = str(
        Path(args.efficientnet_checkpoint).expanduser().resolve()
    )
    config["model"].setdefault("opt_model", args.opt_model)
    config["kv_modulation"]["enabled"] = not args.disable_kv_modulation
    config["kv_modulation"].setdefault("num_prefix_tokens", 8)
    config["architecture"].setdefault("vit_model", args.vit_model)
    config["architecture"].setdefault("img_size", args.image_size)
    config["architecture"].setdefault("num_query_token", args.num_query_token)
    config["architecture"].setdefault("max_txt_len", args.max_text_length)
    config["architecture"].setdefault("freeze_vit", True)
    config["architecture"].setdefault("freeze_llm", True)
    config["architecture"].setdefault("freeze_efficientnet", True)

    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint_path.exists():
        raise SystemExit(f"Model checkpoint not found: {checkpoint_path}")

    model = load_blip2_kv_modulated_model(
        config=config,
        device=device,
        checkpoint_path=str(checkpoint_path),
    )
    model.eval()

    question_text = args.question.strip()
    if not question_text:
        raise SystemExit("Question text must be non-empty.")

    samples: Dict[str, Any] = {
        "image": image_tensor,
        "text_input": [question_text],
    }

    prompt_template = args.prompt_template or ""
    min_new_tokens = max(1, args.min_new_tokens)
    max_new_tokens = max(min_new_tokens, args.max_new_tokens)

    answers = model.predict_answers(
        samples,
        num_beams=max(1, args.num_beams),
        max_len=max_new_tokens,
        min_len=min_new_tokens,
        prompt=prompt_template,
    )

    answer = answers[0] if answers else ""
    print("Image:", Path(args.image_path).expanduser().resolve())
    print("Question:", question_text)
    print("Answer:", answer)


def main() -> None:
    args = parse_args()
    run_vqa(args)


if __name__ == "__main__":
    main()

