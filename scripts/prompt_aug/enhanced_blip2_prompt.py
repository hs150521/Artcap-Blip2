#!/usr/bin/env python3
"""
EfficientNet-augmented BLIP-2 prompt generation.

Given an input image, we first classify it with EfficientNet. The top-K predicted
class labels are fused into a short textual summary, which is then concatenated
with a user-provided prompt before being passed into BLIP-2 for generation.

Usage example:
    python enhanced_blip2_prompt.py \
        --image-path /path/to/example.jpg \
        --user-prompt "Describe the artistic style." \
        --efficientnet-variant b2 \
        --efficientnet-dataset wikiart \
        --efficientnet-checkpoint /path/to/wikiart_efficientnet_b2.pt \
        --blip2-model-id Salesforce/blip2-opt-2.7b
"""

import argparse
import io
import random
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import requests
import torch
from PIL import Image
from torch import nn
from torchvision import transforms

try:
    from transformers import Blip2ForConditionalGeneration, Blip2Processor
except ImportError as exc:  # pragma: no cover - dependency guard
    raise SystemExit(
        "transformers is required. Install it with `pip install transformers`."
    ) from exc


# Canonical WikiArt 27 style labels. Used by default when --efficientnet-dataset=wikiart.
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


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WIKIART_CHECKPOINT = (
    REPO_ROOT / "efficientnet" / "runs" / "wikiart_b3" / "best.pt"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Classify an image using EfficientNet, feed the predicted labels as context "
            "into BLIP-2, and generate an augmented response."
        )
    )
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--image-url",
        help="HTTP(S) URL of the image to process.",
    )
    mode_group.add_argument(
        "--image-path",
        help="Local filesystem path to the image to process.",
    )
    mode_group.add_argument(
        "--vqa-questions",
        help="Path to the VQA questions JSON file for batch evaluation.",
    )
    parser.add_argument(
        "--efficientnet-variant",
        default="b3",
        help=(
            "EfficientNet variant to load (e.g. b0, b1, b2, ... , v2_l). "
            "Defaults to b3, matching the bundled WikiArt checkpoint. "
            "See torchvision.models.efficientnet_* docs for options."
        ),
    )
    parser.add_argument(
        "--efficientnet-dataset",
        choices=("wikiart", "imagenet"),
        default="wikiart",
        help=(
            "Label space to assume for EfficientNet. "
            "'wikiart' expects 27 WikiArt style classes (default). "
            "Specify 'imagenet' to use the original 1K ImageNet labels."
        ),
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=3,
        help="Number of EfficientNet classes to include in the augmented prompt.",
    )
    parser.add_argument(
        "--efficientnet-checkpoint",
        default=str(DEFAULT_WIKIART_CHECKPOINT),
        help=(
            "Path to a fine-tuned EfficientNet state dict (.pt/.pth). "
            f"Defaults to {DEFAULT_WIKIART_CHECKPOINT}."
        ),
    )
    parser.add_argument(
        "--efficientnet-labels",
        help="Path to class labels for the fine-tuned EfficientNet (txt or json).",
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
        "--user-prompt",
        default="Describe the image.",
        help="Custom user message to append after the EfficientNet summary.",
    )
    parser.add_argument(
        "--prompt-template",
        default="Image context hints: {labels}. ",
        help=(
            "Template used to describe EfficientNet labels. "
            "Must include '{labels}' which will be replaced with a comma-separated list."
        ),
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device to run inference on (e.g. cuda:0, cuda, cpu). Default selects GPU if available.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=40,
        help="Maximum number of new tokens BLIP-2 will generate.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for generation. Defaults to greedy decoding.",
    )
    parser.add_argument(
        "--vqa-annotations",
        help="Path to the VQA annotations JSON file (required when using --vqa-questions).",
    )
    parser.add_argument(
        "--vqa-image-root",
        help="Root directory containing the VQA images (required when using --vqa-questions).",
    )
    parser.add_argument(
        "--vqa-image-template",
        default="COCO_val2014_{image_id:012d}.jpg",
        help="Filename template used to locate images within --vqa-image-root.",
    )
    parser.add_argument(
        "--vqa-limit",
        type=int,
        default=None,
        help="Optional limit on the number of VQA samples to evaluate (for quick sanity checks).",
    )
    parser.add_argument(
        "--vqa-prompt-template",
        default="Question: {question} Short answer:",
        help="Template applied to each VQA question when building the user prompt.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/prompt_aug",
        help="Directory to store evaluation artifacts such as predictions JSON.",
    )
    return parser.parse_args()


def resolve_device(explicit_device: Optional[str]) -> torch.device:
    if explicit_device:
        device = torch.device(explicit_device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise SystemExit(
                f"Requested CUDA device '{explicit_device}' but CUDA is unavailable."
            )
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
    variant: str,
    device: torch.device,
    dataset: str,
    checkpoint: Optional[str] = None,
    label_path: Optional[str] = None,
) -> Tuple[nn.Module, transforms.Compose, Sequence[str]]:
    """Load EfficientNet variant from torchvision and return model, preprocessor, labels."""
    variant_key = variant.lower()

    from torchvision.models import (
        efficientnet_b0,
        efficientnet_b1,
        efficientnet_b2,
        efficientnet_b3,
        efficientnet_b4,
        efficientnet_b5,
        efficientnet_b6,
        efficientnet_b7,
        efficientnet_v2_s,
        efficientnet_v2_m,
        efficientnet_v2_l,
        EfficientNet_B0_Weights,
        EfficientNet_B1_Weights,
        EfficientNet_B2_Weights,
        EfficientNet_B3_Weights,
        EfficientNet_B4_Weights,
        EfficientNet_B5_Weights,
        EfficientNet_B6_Weights,
        EfficientNet_B7_Weights,
        EfficientNet_V2_S_Weights,
        EfficientNet_V2_M_Weights,
        EfficientNet_V2_L_Weights,
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
        raise SystemExit(f"Unsupported EfficientNet variant '{variant}'.")

    constructor, weights_enum = variants[variant_key]

    categories: List[str] = []
    if label_path:
        labels_path = Path(label_path).expanduser().resolve()
        if not labels_path.exists():
            raise SystemExit(f"EfficientNet labels file not found: {labels_path}")
        if labels_path.suffix.lower() in {".json", ".jsonl"}:
            import json

            with open(labels_path, "r") as f:
                categories = json.load(f)
        else:
            with open(labels_path, "r") as f:
                categories = [line.strip() for line in f if line.strip()]

    if not categories and dataset == "wikiart":
        categories = list(WIKIART_STYLES)

    state_dict: Optional[dict] = None
    if checkpoint:
        checkpoint_path = Path(checkpoint).expanduser().resolve()
        if not checkpoint_path.exists():
            raise SystemExit(f"EfficientNet checkpoint not found: {checkpoint_path}")
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
            raise SystemExit("Unsupported EfficientNet checkpoint format; expected a state dict.")
        state_dict = state

    num_classes_override: Optional[int] = len(categories) if categories else None
    if num_classes_override is None and state_dict is not None:
        num_classes_override = infer_num_classes_from_state_dict(state_dict)
    if num_classes_override is None and dataset == "wikiart":
        num_classes_override = len(WIKIART_STYLES)

    use_custom_head = num_classes_override is not None

    if use_custom_head:
        model = constructor(weights=None, num_classes=num_classes_override).to(device)
        preprocess = weights_enum.DEFAULT.transforms()
        if categories:
            categories = list(categories)
            if len(categories) > num_classes_override:
                categories = categories[:num_classes_override]
            elif len(categories) < num_classes_override:
                start_idx = len(categories)
                categories.extend(
                    f"class_{idx}" for idx in range(start_idx, num_classes_override)
                )
        else:
            categories = [f"class_{idx}" for idx in range(num_classes_override)]
    else:
        weights = weights_enum.verify(weights_enum.DEFAULT)
        model = constructor(weights=weights).to(device)
        preprocess = weights.transforms()
        categories = weights.meta.get("categories", []) or categories

    if state_dict is not None:
        load_status = model.load_state_dict(state_dict, strict=False)
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

    if not categories:
        categories = infer_categories_from_model(model)

    model.eval()
    return model, preprocess, categories


def infer_num_classes_from_state_dict(state_dict: dict) -> Optional[int]:
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


def infer_categories_from_model(model: nn.Module) -> List[str]:
    classifier = getattr(model, "classifier", None)
    if isinstance(classifier, nn.Sequential):
        for layer in reversed(classifier):
            if isinstance(layer, nn.Linear):
                return [f"class_{idx}" for idx in range(layer.out_features)]
    if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
        return [f"class_{idx}" for idx in range(model.fc.out_features)]
    return []


@torch.no_grad()
def classify_image(
    image: Image.Image,
    model: nn.Module,
    preprocess: transforms.Compose,
    categories: Sequence[str],
    topk: int,
    device: torch.device,
) -> List[Tuple[str, float]]:
    tensor = preprocess(image).unsqueeze(0).to(device)
    logits = model(tensor)
    probabilities = logits.softmax(dim=1)
    values, indices = probabilities.topk(topk, dim=1)

    results = []
    for score, idx in zip(values.squeeze(0).tolist(), indices.squeeze(0).tolist()):
        label = categories[idx] if idx < len(categories) else f"class_{idx}"
        results.append((label, score))
    return results


def format_labels(labels_with_scores: Iterable[Tuple[str, float]]) -> str:
    labels = []
    for label, score in labels_with_scores:
        labels.append(f"{label} ({score:.2%})")
    return ", ".join(labels)


def build_augmented_prompt(
    predicted_labels: Iterable[Tuple[str, float]],
    template: str,
    user_prompt: str,
) -> str:
    label_str = ", ".join([label for label, _ in predicted_labels])
    template_text = template.format(labels=label_str).strip()
    user_text = user_prompt.strip()
    if template_text and not template_text.endswith((" ", "\n")):
        template_text += " "
    combined = f"{template_text}{user_text}".strip()
    if not combined.lower().endswith("answer:"):
        combined = f"{combined}\nAnswer:"
    return combined


def load_blip2(
    model_id: str,
    revision: Optional[str],
    device: torch.device,
) -> Tuple[Blip2Processor, Blip2ForConditionalGeneration, torch.dtype]:
    try:
        processor = Blip2Processor.from_pretrained(model_id, revision=revision)
    except Exception as exc:
        if "TokenizerFast" in str(exc) or "ModelWrapper" in str(exc):
            print(
                "Falling back to slow tokenizer for BLIP-2 processor due to fast tokenizer load issue.",
                file=sys.stderr,
            )
            processor = Blip2Processor.from_pretrained(
                model_id, revision=revision, use_fast=False
            )
        else:
            raise
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    model = Blip2ForConditionalGeneration.from_pretrained(
        model_id,
        revision=revision,
        torch_dtype=dtype,
    ).to(device)
    model.eval()
    return processor, model, dtype


@torch.no_grad()
def run_blip2(
    image: Image.Image,
    prompt: str,
    model_id: str,
    revision: Optional[str],
    device: torch.device,
    max_new_tokens: int,
    temperature: float,
    processor: Optional[Blip2Processor] = None,
    model: Optional[Blip2ForConditionalGeneration] = None,
    dtype: Optional[torch.dtype] = None,
) -> str:
    owns_model = False
    if processor is None or model is None or dtype is None:
        processor, model, dtype = load_blip2(model_id, revision, device)
        owns_model = True

    inputs = processor(images=image, text=prompt, return_tensors="pt")
    moved_inputs = {}
    for key, value in inputs.items():
        if isinstance(value, torch.Tensor):
            if value.dtype.is_floating_point:
                moved_inputs[key] = value.to(device, dtype=dtype)
            else:
                moved_inputs[key] = value.to(device)
        else:
            moved_inputs[key] = value
    inputs = moved_inputs

    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
    }
    if temperature and temperature > 0:
        generation_kwargs.update({"do_sample": True, "temperature": temperature})

    generated_ids = model.generate(**inputs, **generation_kwargs)
    generated_text = processor.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    output = generated_text.strip()

    if owns_model and device.type == "cuda":
        del model
        torch.cuda.empty_cache()

    return output


def resolve_vqa_image_path(
    image_root: Path,
    image_id: int,
    template: str,
) -> Path:
    candidate = image_root / template.format(image_id=image_id)
    if candidate.exists():
        return candidate
    # Try common alternate naming conventions.
    alternatives = [
        image_root / template.format(image_id=str(image_id)),
        image_root / f"{image_id}.jpg",
        image_root / f"{image_id}.png",
        image_root / f"COCO_train2014_{image_id:012d}.jpg",
    ]
    for path in alternatives:
        if path.exists():
            return path
    raise FileNotFoundError(f"Could not locate image for ID {image_id} using template '{template}'.")


def evaluate_vqa(
    args: argparse.Namespace,
    image_model: nn.Module,
    preprocess: transforms.Compose,
    categories: Sequence[str],
    processor: Blip2Processor,
    blip_model: Blip2ForConditionalGeneration,
    dtype: torch.dtype,
    device: torch.device,
) -> None:
    import json
    from tqdm import tqdm

    from lavis.common.vqa_tools.vqa import VQA
    from lavis.common.vqa_tools.vqa_eval import VQAEval

    questions_path = Path(args.vqa_questions).expanduser().resolve()
    annotations_path = Path(args.vqa_annotations).expanduser().resolve()
    image_root = Path(args.vqa_image_root).expanduser().resolve()

    if not questions_path.exists():
        raise SystemExit(f"VQA questions file not found: {questions_path}")
    if not annotations_path.exists():
        raise SystemExit(f"VQA annotations file not found: {annotations_path}")
    if not image_root.exists():
        raise SystemExit(f"VQA image root not found: {image_root}")

    with open(questions_path, "r") as f:
        questions_data = json.load(f)

    questions = questions_data.get("questions", [])

    dataset_percent = float(getattr(args, "dataset_percent", 100.0))
    if dataset_percent <= 0 or dataset_percent > 100:
        raise SystemExit("--dataset-percent must be in the range (0, 100].")

    batch_size = int(getattr(args, "batch_size", 1))
    if batch_size <= 0:
        raise SystemExit("--batch-size must be a positive integer.")

    seed = getattr(args, "seed", None)
    subset_seed = seed if seed is not None else 42

    if dataset_percent < 100:
        total = len(questions)
        sample_size = max(1, min(total, round(total * dataset_percent / 100.0)))
        if sample_size < total:
            rng = random.Random(subset_seed)
            selected_indices = sorted(rng.sample(range(total), sample_size))
            questions = [questions[idx] for idx in selected_indices]
            print(
                f"Evaluating {sample_size} of {total} questions "
                f"({dataset_percent:.2f}% of the dataset) using seed {subset_seed}."
            )

    if args.vqa_limit is not None:
        questions = questions[: args.vqa_limit]

    def _batched(sequence, size):
        for start in range(0, len(sequence), size):
            yield sequence[start : start + size]

    predictions = []
    progress = tqdm(total=len(questions), desc="Evaluating VQA")
    for batch in _batched(questions, batch_size):
        for entry in batch:
            question = entry["question"]
            image_id = entry["image_id"]
            question_id = entry["question_id"]

            image_path = resolve_vqa_image_path(
                image_root, image_id, args.vqa_image_template
            )
            image = load_image_from_path(str(image_path))

            labels_with_scores = classify_image(
                image=image,
                model=image_model,
                preprocess=preprocess,
                categories=categories,
                topk=max(1, args.topk),
                device=device,
            )

            user_prompt = args.vqa_prompt_template.format(question=question)
            augmented_prompt = build_augmented_prompt(
                predicted_labels=labels_with_scores,
                template=args.prompt_template,
                user_prompt=user_prompt,
            )

            answer = run_blip2(
                image=image,
                prompt=augmented_prompt,
                model_id=args.blip2_model_id,
                revision=args.blip2_revision,
                device=device,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                processor=processor,
                model=blip_model,
                dtype=dtype,
            )

            predictions.append({"question_id": question_id, "answer": answer})

        progress.update(len(batch))

    progress.close()

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "vqa_predictions.json"

    with open(results_path, "w") as f:
        json.dump(predictions, f)

    print(f"Saved predictions to {results_path}")

    vqa = VQA(str(annotations_path), str(questions_path))
    vqa_results = vqa.loadRes(str(results_path), str(questions_path))
    vqa_eval = VQAEval(vqa, vqa_results, n=2)
    vqa_eval.evaluate()

    overall_acc = vqa_eval.accuracy.get("overall", 0.0)
    print(f"Overall VQA accuracy: {overall_acc:.2f}")


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

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

    if args.vqa_questions:
        if not args.vqa_annotations or not args.vqa_image_root:
            raise SystemExit(
                "Both --vqa-annotations and --vqa-image-root are required when using --vqa-questions."
            )
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
        return

    if args.image_url:
        image = load_image_from_url(args.image_url)
        image_source = args.image_url
    else:
        image = load_image_from_path(args.image_path)
        image_source = str(Path(args.image_path).resolve())

    topk = max(1, args.topk)
    labels_with_scores = classify_image(
        image=image,
        model=model,
        preprocess=preprocess,
        categories=categories,
        topk=topk,
        device=device,
    )

    augmented_prompt = build_augmented_prompt(
        predicted_labels=labels_with_scores,
        template=args.prompt_template,
        user_prompt=args.user_prompt,
    )

    response = run_blip2(
        image=image,
        prompt=augmented_prompt,
        model_id=args.blip2_model_id,
        revision=args.blip2_revision,
        device=device,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        processor=processor,
        model=blip_model,
        dtype=dtype,
    )

    print(f"Image source: {image_source}")
    print("Top EfficientNet predictions:")
    for label, score in labels_with_scores:
        print(f"  - {label}: {score:.2%}")
    print("\nAugmented prompt:")
    print(augmented_prompt)
    print("\nBLIP-2 response:")
    print(response)


if __name__ == "__main__":
    main()
