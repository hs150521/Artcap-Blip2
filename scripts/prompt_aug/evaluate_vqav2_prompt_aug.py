#!/usr/bin/env python3
"""Evaluate the enhanced BLIP-2 prompt pipeline on VQAv2.

This mirrors the intent of
`blip2/LAVIS/run_scripts/blip2/eval/validate_vqa_zeroshot_opt.sh`, but routes the
evaluation through ``scripts/prompt_aug/enhanced_blip2_prompt.py`` so the
EfficientNet-driven prompt augmentation is part of the measurement.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Tuple

from enhanced_blip2_prompt import (
    DEFAULT_WIKIART_CHECKPOINT,
    evaluate_vqa,
    load_blip2,
    load_efficientnet,
    resolve_device,
)


def _detect_lavis_datasets_root() -> Path:
    """Heuristic to locate the dataset root used by the upstream LAVIS scripts."""
    repo_root = Path(__file__).resolve().parents[2]
    env_candidates = [
        os.environ.get("LAVIS_DATASETS_ROOT"),
        os.environ.get("LAVIS_DATA_ROOT"),
        os.environ.get("DATASET_ROOT"),
        os.environ.get("DATA_ROOT"),
    ]

    candidate_paths: list[Path] = [
        Path(candidate).expanduser().resolve()
        for candidate in env_candidates
        if candidate
    ]
    candidate_paths.append(Path("/data/lavis"))
    candidate_paths.extend(
        [
            repo_root / "datasets",
            repo_root / "blip2" / "LAVIS" / "datasets",
            repo_root / "LAVIS" / "datasets",
            repo_root / "data",
        ]
    )

    for candidate in candidate_paths:
        if candidate.exists():
            return candidate
    return repo_root / "blip2" / "LAVIS" / "datasets"


def _guess_vqav2_files(datasets_root: Path) -> Tuple[Path, Path]:
    """Return best-guess paths for VQAv2 questions and annotations."""
    annotations_roots = [
        datasets_root / "vqav2" / "annotations",
        datasets_root / "vqa" / "annotations",
        datasets_root / "coco" / "annotations",
        datasets_root / "vqav2",
        datasets_root / "vqa",
        datasets_root / "coco",
        datasets_root,
    ]

    questions_candidates = []
    annotations_candidates = []

    for root in annotations_roots:
        questions_candidates.extend(
            [
                root / "v2_OpenEnded_mscoco_val2014_questions.json",
                root / "OpenEnded_mscoco_val2014_questions.json",
            ]
        )
        annotations_candidates.extend(
            [
                root / "v2_mscoco_val2014_annotations.json",
                root / "mscoco_val2014_annotations.json",
            ]
        )

    def _first_existing(candidates: list[Path]) -> Path:
        return next((path for path in candidates if path.exists()), candidates[0])

    return _first_existing(questions_candidates), _first_existing(annotations_candidates)


def _guess_coco_image_root(datasets_root: Path) -> Path:
    """Return best-guess root directory for COCO val2014 images."""
    candidates = [
        datasets_root / "coco" / "images" / "val2014",
        datasets_root / "coco" / "val2014",
        datasets_root / "coco" / "images",
        datasets_root / "coco",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _build_parser(default_root: Path) -> argparse.ArgumentParser:
    default_questions, default_annotations = _guess_vqav2_files(default_root)
    default_image_root = _guess_coco_image_root(default_root)

    parser = argparse.ArgumentParser(
        description=(
            "Run VQAv2 validation with the enhanced BLIP-2 prompt pipeline. "
            "Defaults mirror the LAVIS VQA zeroshot OPT evaluation script."
        )
    )
    parser.add_argument(
        "--datasets-root",
        default=str(default_root),
        help=(
            "Root directory for datasets (VQAv2 annotations and COCO images). "
            "Defaults to the location inferred from the LAVIS repo structure."
        ),
    )
    parser.add_argument(
        "--vqa-questions",
        default=str(default_questions),
        help="Path to VQAv2 val question JSON (v2_OpenEnded_mscoco_val2014_questions.json).",
    )
    parser.add_argument(
        "--vqa-annotations",
        default=str(default_annotations),
        help="Path to VQAv2 val annotation JSON (v2_mscoco_val2014_annotations.json).",
    )
    parser.add_argument(
        "--vqa-image-root",
        default=str(default_image_root),
        help="Directory containing COCO val2014 images.",
    )
    parser.add_argument(
        "--vqa-image-template",
        default="COCO_val2014_{image_id:012d}.jpg",
        help="Filename template for locating images under --vqa-image-root.",
    )
    parser.add_argument(
        "--vqa-limit",
        type=int,
        default=None,
        help="Optional limit on evaluation samples (useful for smoke tests).",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/prompt_aug/vqav2",
        help="Directory to store evaluation artifacts (predictions JSON, logs).",
    )
    parser.add_argument(
        "--efficientnet-variant",
        default="b3",
        help="EfficientNet variant to load (e.g. b0, b2, v2_l).",
    )
    parser.add_argument(
        "--efficientnet-dataset",
        choices=("wikiart", "imagenet"),
        default="wikiart",
        help="Classifier label space; use 'wikiart' for the 27-style fine-tuned head.",
    )
    parser.add_argument(
        "--efficientnet-checkpoint",
        default=str(DEFAULT_WIKIART_CHECKPOINT),
        help=(
            "Path to the fine-tuned EfficientNet state dict (.pt/.pth). "
            f"Defaults to {DEFAULT_WIKIART_CHECKPOINT}."
        ),
    )
    parser.add_argument(
        "--efficientnet-labels",
        help="Optional label file (txt/json) if not using the default WikiArt class list.",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=3,
        help="Number of EfficientNet predictions to inject into the prompt.",
    )
    parser.add_argument(
        "--prompt-template",
        default="Image context hints: {labels}. ",
        help="Template describing EfficientNet labels before the VQA prompt.",
    )
    parser.add_argument(
        "--vqa-prompt-template",
        default="Question: {question} Short answer:",
        help="Template applied to each VQAv2 question prior to generation.",
    )
    parser.add_argument(
        "--user-prompt",
        default="Respond with a concise answer based on the image.",
        help="Optional trailing instruction appended after the template.",
    )
    parser.add_argument(
        "--blip2-model-id",
        default="Salesforce/blip2-opt-2.7b",
        help="BLIP-2 checkpoint to evaluate (matches the LAVIS OPT baseline).",
    )
    parser.add_argument(
        "--blip2-revision",
        default=None,
        help="Optional git revision/hash/tag for the BLIP-2 checkpoint.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=10,
        help="Maximum number of tokens to generate for each answer.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature; zero selects greedy decoding.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device spec (e.g. cuda:0, cuda, cpu). Auto-detects GPU when available.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Number of VQA samples to process per inner loop batch.",
    )
    parser.add_argument(
        "--dataset-percent",
        type=float,
        default=100.0,
        help=(
            "Percentage of the validation set to evaluate. "
            "Use 100 to run on the full dataset, 50 for half, etc."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help=(
            "Random seed applied when --dataset-percent is less than 100 to "
            "ensure reproducible sampling."
        ),
    )

    return parser


def _ensure_inputs(questions: Path, annotations: Path, image_root: Path) -> None:
    """Validate dataset paths before launching evaluation."""
    if not questions.exists():
        raise SystemExit(
            f"VQAv2 questions file not found at {questions}. "
            "Override with --vqa-questions or place the file to match the LAVIS layout."
        )
    if not annotations.exists():
        raise SystemExit(
            f"VQAv2 annotations file not found at {annotations}. "
            "Override with --vqa-annotations or download the validation annotations."
        )
    if not image_root.exists():
        raise SystemExit(
            f"COCO image root not found at {image_root}. "
            "Override with --vqa-image-root pointing to val2014 images."
        )


def main() -> None:
    datasets_root = _detect_lavis_datasets_root()
    parser = _build_parser(datasets_root)
    args = parser.parse_args()

    args.image_url = None
    args.image_path = None

    questions_path = Path(args.vqa_questions).expanduser().resolve()
    annotations_path = Path(args.vqa_annotations).expanduser().resolve()
    image_root = Path(args.vqa_image_root).expanduser().resolve()

    _ensure_inputs(questions_path, annotations_path, image_root)

    if args.vqa_limit is not None and args.vqa_limit <= 0:
        raise SystemExit("--vqa-limit must be positive when provided.")

    device = resolve_device(args.device)
    print(f"Using device: {device}")

    model, preprocess, categories = load_efficientnet(
        variant=args.efficientnet_variant,
        device=device,
        dataset=args.efficientnet_dataset,
        checkpoint=args.efficientnet_checkpoint,
        label_path=args.efficientnet_labels,
    )
    processor, blip_model, dtype = load_blip2(
        model_id=args.blip2_model_id,
        revision=args.blip2_revision,
        device=device,
    )

    args.vqa_questions = str(questions_path)
    args.vqa_annotations = str(annotations_path)
    args.vqa_image_root = str(image_root)
    args.output_dir = str(Path(args.output_dir).expanduser().resolve())

    print("Launching VQAv2 evaluation with the enhanced prompt pipeline...")
    evaluate_vqa(
        args=args,
        image_model=model,
        preprocess=preprocess,
        categories=categories,
        processor=processor,
        blip_model=blip_model,
        dtype=dtype,
        device=device,
    )


if __name__ == "__main__":
    main()
