"""
EfficientNet model loader for WikiArt-trained EfficientNet-B3.

This module provides functions to load the pre-trained EfficientNet-B3 model
trained on WikiArt dataset.
"""

import sys
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from torch import nn
from torchvision import transforms
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights


# WikiArt 27 style labels + 1 COCO non-art class = 28 classes
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
    "COCO_Non_Art"  # Additional class for COCO non-art images
)


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


def load_efficientnet_model(
    checkpoint_path: Optional[str] = None,
    repo_root: Optional[str] = None,
) -> Tuple[nn.Module, transforms.Compose, List[str]]:
    """
    Load EfficientNet-B3 model from checkpoint.
    
    Args:
        checkpoint_path: Path to EfficientNet checkpoint. If None, uses default path.
        repo_root: Root directory of the repository. If None, infers from file location.
    
    Returns:
        model: Loaded EfficientNet model (in eval mode)
        preprocess: Image preprocessing transforms
        categories: List of category labels
    """
    if repo_root is None:
        repo_root = Path(__file__).resolve().parents[3]  # Changed from parents[2] to parents[3] to reach repo root
        # If the computed path doesn't exist, try alternative calculation
        if not repo_root.exists():
            # Try going up one more level
            repo_root = Path(__file__).resolve().parents[4]
        # If still not found, use the absolute path
        if not repo_root.exists():
            repo_root = Path("/home/linux/artcap-blip2-4")
    else:
        repo_root = Path(repo_root)
    
    if checkpoint_path is None:
        checkpoint_path = repo_root / "runs" / "efficientnet-28" / "best.pt"
        # If the computed path doesn't exist, use the absolute path
        if not checkpoint_path.exists():
            checkpoint_path = Path("/home/linux/artcap-blip2-4/runs/efficientnet-28/best.pt")
    else:
        checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    
    device = _get_device()
    categories = list(WIKIART_STYLES)
    
    # Load checkpoint
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
    num_classes = len(categories)
    inferred_num_classes = _infer_num_classes_from_state_dict(state)
    if inferred_num_classes is not None and inferred_num_classes != num_classes:
        print(f"Warning: Expected {num_classes} classes but checkpoint has {inferred_num_classes}")
    
    # Create model
    weights_enum = EfficientNet_B3_Weights.DEFAULT
    model = efficientnet_b3(weights=None, num_classes=num_classes).to(device)
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
    
    # Freeze model
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    
    # Ensure model is in fp32 to avoid dtype mismatches with inputs
    # (EfficientNet is frozen, so fp32 is fine and avoids dtype issues)
    model = model.float()
    
    return model, preprocess, categories


def extract_efficientnet_features(
    model: nn.Module,
    images: torch.Tensor,
    return_pooled: bool = True,
) -> torch.Tensor:
    """
    Extract features from EfficientNet model.
    
    Args:
        model: EfficientNet model
        images: Preprocessed images tensor (batch_size, 3, H, W)
        return_pooled: If True, return pooled features (batch_size, feat_dim).
                      If False, return feature map (batch_size, seq_len, feat_dim)
    
    Returns:
        features: Extracted features
    """
    model.eval()
    with torch.no_grad():
        # Extract features before classifier
        features = model.features(images)
        
        if return_pooled:
            # Global average pooling
            features = model.avgpool(features)
            features = features.view(features.size(0), -1)  # (batch_size, feat_dim)
        else:
            # Return feature map
            features = features.view(features.size(0), features.size(1), -1)  # (batch_size, C, H*W)
            features = features.transpose(1, 2)  # (batch_size, H*W, C)
    
    return features
