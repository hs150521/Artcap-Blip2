#!/usr/bin/env python3
"""
ArtQuest Evaluation Script

This script evaluates models on the ArtQuest dataset following the modular
architecture of VQAv2 evaluation, using YAML configuration files and model backend pattern.
"""

import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

# Add artquest directory to path for imports
# __file__ is evaluation/ArtQuest/evaluate_artquest.py
# Need to go up to project root: parent.parent.parent
project_root = Path(__file__).parent.parent.parent
artquest_dir = project_root / "artquest"
if str(artquest_dir) not in sys.path:
    sys.path.insert(0, str(artquest_dir))

# Add project root to path for model imports
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Add current directory to path for models import
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

# Import metrics functions directly to avoid dependency issues
# We only need calc_bleu_sentence and calc_em, which don't require rouge/gleu
try:
    from t5_training.metrics import calc_bleu_sentence, calc_em
except (ImportError, ModuleNotFoundError) as e:
    # Fallback: define the functions directly if import fails
    import warnings
    warnings.warn(f"Could not import from t5_training.metrics: {e}. Using local implementations.")
    
    def calc_em(answer, pred):
        """Calculate Exact Match (EM) score."""
        cnt = 0
        for a, p in zip(answer, pred):
            if a == p:
                cnt += 1
        if len(answer) == 0:
            return 0
        return cnt / len(answer)
    
    def calc_bleu_sentence(answer, pred):
        """Calculate simple string similarity as BLEU approximation."""
        # Simple implementation: use word overlap as approximation
        sum_score = 0
        for a, p in zip(answer, pred):
            a_words = set(a.lower().split())
            p_words = set(p.lower().split())
            if len(a_words) == 0 or len(p_words) == 0:
                score = 0
            else:
                intersection = a_words.intersection(p_words)
                score = len(intersection) / len(a_words.union(p_words))
            sum_score += score
        if len(answer) == 0:
            return 0
        return sum_score / len(answer)

torch.manual_seed(2)

# Add LAVIS to path before importing blip2_lavis
lavis_path = project_root / "blip2" / "LAVIS"
if str(lavis_path) not in sys.path:
    sys.path.insert(0, str(lavis_path))

# Import blip2_lavis for direct use
sys.path.insert(0, str(project_root / "evaluation" / "VQAv2" / "models"))
from blip2_lavis import predict_answers_blip2_lavis


def load_config(config_path: str) -> Dict:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Dictionary containing configuration parameters
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    # Load YAML config
    config = OmegaConf.load(config_path)
    
    # Convert to dict for easier manipulation
    config_dict = OmegaConf.to_container(config, resolve=True)
    
    # Add data_percentage if not present (default to 1.0 for full dataset)
    if "run" not in config_dict:
        config_dict["run"] = {}
    
    if "data_percentage" not in config_dict["run"]:
        config_dict["run"]["data_percentage"] = 1.0
    
    data_percentage = config_dict["run"]["data_percentage"]
    if not (0 < data_percentage <= 1.0):
        raise ValueError(f"data_percentage must be in (0, 1.0], got {data_percentage}")
    
    return config_dict


def create_artquest_adapter_dataset(
    artquest_dataset: Any,
    config_dict: Dict,
    use_retrieved_context: bool = False
) -> Any:
    """
    Create an adapter dataset that converts ArtQuest format to blip2_lavis format.
    
    Args:
        artquest_dataset: Original ArtQuest dataset
        config_dict: Configuration dictionary
        use_retrieved_context: If True, use candidate_context; otherwise use context
    
    Returns:
        Adapter dataset compatible with blip2_lavis
    """
    data_cfg = config_dict.get("data", {})
    image_root = data_cfg.get("image_root", "datasets/artquest/SemArt/Images")
    
    # Convert to absolute path relative to this file if relative
    project_root = Path(__file__).parent.parent.parent
    if not os.path.isabs(image_root):
        image_root = str(project_root / image_root)
    
    # Load image processor
    sys.path.insert(0, str(project_root / "blip2" / "LAVIS"))
    from lavis.processors import load_processor
    from PIL import Image
    
    try:
        vis_processor = load_processor("blip_image_eval", {"image_size": 224})
    except:
        # Fallback processor
        from torchvision import transforms
        vis_processor = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                               std=[0.26862954, 0.26130258, 0.27577711])
        ])
    
    run_cfg = config_dict.get("run", {})
    prompt = run_cfg.get("prompt", "Question: {} Short answer:")
    
    class AdapterDataset:
        def __init__(self, artquest_dataset, image_root, vis_processor, prompt, use_retrieved_context):
            self.artquest_dataset = artquest_dataset
            self.image_root = image_root
            self.vis_processor = vis_processor
            self.prompt = prompt
            self.use_retrieved_context = use_retrieved_context
        
        def __len__(self):
            return len(self.artquest_dataset)
        
        def __getitem__(self, idx):
            sample = self.artquest_dataset[idx]
            image_name = sample["image"]
            
            # Debug: check image_name type at the very beginning
            if not isinstance(image_name, str):
                logging.error(f"CRITICAL: image_name is not string at start of __getitem__! Type: {type(image_name)}, Value: {repr(image_name)}")
                # If it's a tensor, try to convert it back to string
                if hasattr(image_name, 'item'):
                    try:
                        image_name = str(image_name.item())
                        logging.warning(f"Converted tensor image_name to string: {image_name}")
                    except:
                        pass
            
            question = sample["question"]
            context = sample.get("candidate_context", sample["context"]) if self.use_retrieved_context else sample["context"]
            
            # Load and process image
            # Debug: check image_name type before path join
            if not isinstance(image_name, str):
                logging.error(f"CRITICAL: image_name is not string before path join! Type: {type(image_name)}, Value: {repr(image_name)}")
            image_path = Path(self.image_root) / image_name
            if not image_path.exists():
                # Try alternative paths
                alt_paths = [
                    Path(self.image_root) / os.path.basename(image_name),
                    Path(image_name),  # Try as absolute path
                ]
                for alt_path in alt_paths:
                    if alt_path.exists():
                        image_path = alt_path
                        break
            image_path = str(image_path)
            
            try:
                image = Image.open(image_path).convert("RGB")
                # Debug: check image type before processing
                logging.debug(f"Image type before processing: {type(image)}")
                processed_image = self.vis_processor(image)
                # Debug: check processed_image type
                logging.debug(f"Processed image type: {type(processed_image)}")
            except Exception as e:
                logging.warning(f"Failed to load image {image_name}: {e}")
                processed_image = torch.zeros((3, 224, 224))
            
            # Convert to tensor if needed
            if not isinstance(processed_image, torch.Tensor):
                processed_image = torch.tensor(processed_image)
            
            # Ensure image is 3D (C, H, W) - blip2_lavis will stack them into batches
            if processed_image.dim() == 4:
                processed_image = processed_image.squeeze(0)
            elif processed_image.dim() == 2:
                # If somehow 2D, add channel dimension
                processed_image = processed_image.unsqueeze(0)
            
            # Build text input: BLIP2's predict_answers will format it with prompt
            # So we need to combine question and context here, and use simple prompt
            # Or pass only question and let prompt handle context
            # Since we want context in the input, we combine them here
            # and the prompt should be simple "{}" to avoid formatting errors
            text_input = f"{question} Context: {context}"
            
            return {
                "image": processed_image,
                "text_input": text_input,
                "question_id": sample["question_id"]
            }
    
    return AdapterDataset(artquest_dataset, image_root, vis_processor, prompt, use_retrieved_context)


def load_artquest_data(config_dict: Dict, data_percentage: float = 1.0) -> Any:
    """
    Load ArtQuest dataset and apply data sampling if needed.
    
    Args:
        config_dict: Configuration dictionary
        data_percentage: Percentage of data to use (0-1)
        
    Returns:
        Dataset-like object with __getitem__ and __len__ methods
    """
    data_cfg = config_dict.get("data", {})
    
    # Load data files - paths are relative to project root (datasets directory)
    # Default paths assume data is in datasets/artquest/
    artquest_test_path = data_cfg.get("artquest_test", "datasets/artquest/artquest_test.csv")
    semart_cache_path = data_cfg.get("semart_cache", "datasets/artquest/semart_cache.csv")
    retrieved_candidates_path = data_cfg.get("retrieved_candidates", 
        "datasets/artquest/retrieved_candidates/SEMARTCLIP.200BS.IMAGE_TO_TEXT_reference_candidate.pickle")
    
    # Convert to absolute paths relative to this file if relative
    project_root = Path(__file__).parent.parent.parent
    if not os.path.isabs(artquest_test_path):
        artquest_test_path = str(project_root / artquest_test_path)
    if not os.path.isabs(semart_cache_path):
        semart_cache_path = str(project_root / semart_cache_path)
    if not os.path.isabs(retrieved_candidates_path):
        retrieved_candidates_path = str(project_root / retrieved_candidates_path)
    
    logging.info(f"Loading ArtQuest test data from {artquest_test_path}")
    artquest_test = pd.read_csv(artquest_test_path)
    
    logging.info(f"Loading SemArt cache from {semart_cache_path}")
    semart_cache = pd.read_csv(semart_cache_path)
    
    # Join artquest test with semart cache
    artquest_test = artquest_test.join(semart_cache.set_index("image"), on="image")
    
    # Read retrieved context candidates
    logging.info(f"Loading retrieved candidates from {retrieved_candidates_path}")
    ref_candidates_unpickled = pd.read_pickle(retrieved_candidates_path)
    ref_candidates = pd.DataFrame(
        ref_candidates_unpickled["reference_image_names"], 
        columns=["reference_image_names"]
    )
    ref_candidates["image_names_of_candidate_texts"] = ref_candidates_unpickled["image_names_of_candidate_texts"]
    
    retrieval_accuracy = (ref_candidates["image_names_of_candidate_texts"] == 
                         ref_candidates["reference_image_names"]).sum() / len(ref_candidates)
    logging.info(f"Accuracy of retrieved contexts: {retrieval_accuracy:.4f}")
    
    # Map candidate texts to the artquest test data
    candidate_contexts = ref_candidates.join(
        semart_cache.set_index("image"), 
        on="image_names_of_candidate_texts"
    )[["reference_image_names", "context"]]
    candidate_contexts = candidate_contexts.rename(
        columns={"reference_image_names": "image", "context": "candidate_context"}
    )
    artquest_test = artquest_test.join(
        candidate_contexts.set_index("image"), on="image"
    )
    
    # Apply data sampling if needed
    if data_percentage < 1.0:
        seed = config_dict.get("run", {}).get("seed", 2)
        random.seed(seed)
        np.random.seed(seed)
        
        total_samples = len(artquest_test)
        num_samples = int(total_samples * data_percentage)
        
        # Sample indices randomly
        indices = sorted(random.sample(range(total_samples), num_samples))
        artquest_test = artquest_test.iloc[indices].reset_index(drop=True)
        
        logging.info(f"Sampled {num_samples} samples ({data_percentage*100:.1f}%) from {total_samples} total samples")
    
    # Create dataset-like wrapper
    class ArtQuestDataset:
        def __init__(self, df):
            self.df = df.reset_index(drop=True)
        
        def __len__(self):
            return len(self.df)
        
        def __getitem__(self, idx):
            row = self.df.iloc[idx]
            # Debug: check image column type
            image_value = row["image"]
            if not isinstance(image_value, str):
                logging.warning(f"Image column at index {idx} is not string: {type(image_value)}, value: {image_value}")
            return {
                "image": row["image"],
                "question": row["question"],
                "context": row["context"],
                "candidate_context": row.get("candidate_context", row["context"]),
                "img_emb": row["img_emb"],
                "answer": row["answer"],
                "question_id": idx
            }
    
    return ArtQuestDataset(artquest_test)


def evaluate_results(
    predictions: List[Dict],
    dataset: Any,
    output_dir: str
) -> Dict:
    """
    Evaluate predictions against ground truth.
    
    Args:
        predictions: List of predictions with question_id and answer
        dataset: Dataset object to get ground truth answers
        output_dir: Directory to save results
        
    Returns:
        Dictionary with evaluation metrics
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract ground truth and predictions
    answers = []
    preds_original = []
    
    # Create a mapping from question_id to prediction
    pred_dict = {pred["question_id"]: pred for pred in predictions}
    
    for i in range(len(dataset)):
        sample = dataset[i]
        question_id = sample["question_id"]
        
        if question_id in pred_dict:
            answers.append(sample["answer"].lower())
            preds_original.append(pred_dict[question_id].get("answer", "").lower())
    
    # Calculate metrics for original context
    em_original = calc_em(answers, preds_original)
    bleu_original = calc_bleu_sentence(answers, preds_original)
    
    metrics = {
        "original_context": {
            "em": round(em_original, 4),
            "bleu": round(bleu_original, 4)
        }
    }
    
    # Print results
    logging.info("Metrics for the original context:")
    logging.info(f"  EM: {metrics['original_context']['em']:.4f}")
    logging.info(f"  BLEU: {metrics['original_context']['bleu']:.4f}")
    
    # Save metrics to file
    metrics_file = os.path.join(output_dir, "evaluate.txt")
    with open(metrics_file, "w") as f:
        f.write(json.dumps(metrics, indent=2) + "\n")
    
    logging.info(f"Saved metrics to {metrics_file}")
    
    # Save predictions
    result_file = os.path.join(output_dir, "predictions.json")
    with open(result_file, "w") as f:
        json.dump(predictions, f, indent=2)
    
    logging.info(f"Saved predictions to {result_file}")
    
    return metrics


def main():
    """Main function to orchestrate the evaluation pipeline."""
    parser = argparse.ArgumentParser(description="Evaluate models on ArtQuest")
    parser.add_argument(
        "--cfg-path",
        type=str,
        required=True,
        help="Path to configuration YAML file"
    )
    parser.add_argument(
        "--data-percentage",
        type=float,
        default=None,
        help="Percentage of data to use (0-1). Overrides config if provided."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size for inference. Overrides config if provided."
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu). Overrides config if provided."
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Load configuration
    logging.info(f"Loading configuration from {args.cfg_path}")
    config = load_config(args.cfg_path)
    
    # Override config with command line arguments
    if args.data_percentage is not None:
        config["run"]["data_percentage"] = args.data_percentage
    if args.batch_size is not None:
        config["run"]["batch_size_eval"] = args.batch_size
    if args.device is not None:
        config["run"]["device"] = args.device
    
    data_percentage = config["run"]["data_percentage"]
    batch_size = config["run"].get("batch_size_eval", 1)
    device = config["run"].get("device", "cuda" if torch.cuda.is_available() else "cpu")
    output_dir = config["run"].get("output_dir", "output/ArtQuest")
    
    logging.info(f"Configuration loaded:")
    logging.info(f"  Data percentage: {data_percentage}")
    logging.info(f"  Batch size: {batch_size}")
    logging.info(f"  Device: {device}")
    logging.info(f"  Output directory: {output_dir}")
    
    # Determine model type from config
    model_name = config.get("model", "blip2")
    if not isinstance(model_name, str):
        model_name = "blip2"
        config["model"] = model_name
    
    logging.info(f"  Model: {model_name}")
    
    # Get model backend functions
    # Import from local models directory
    try:
        # When run as module, use relative import
        from .models import get_model_backend
    except (ImportError, SystemError):
        # When run as script, use absolute import
        # Add current directory to path first
        current_dir = Path(__file__).parent
        if str(current_dir) not in sys.path:
            sys.path.insert(0, str(current_dir))
        from models import get_model_backend
    
    load_model_func, predict_answers_func = get_model_backend(model_name)
    
    # Load model
    logging.info("Loading model...")
    model = load_model_func(config, device)
    
    # Load data
    logging.info("Loading ArtQuest dataset...")
    artquest_dataset = load_artquest_data(config, data_percentage)
    logging.info(f"Dataset ready with {len(artquest_dataset)} samples")
    
    # Create adapter datasets for original and retrieved contexts
    # Determine whether the backend already handles ArtQuest-specific preprocessing
    predictor_module = getattr(predict_answers_func, "__module__", "")
    native_modules = {
        "evaluation.ArtQuest.models.blip2_artquest",
        "evaluation.ArtQuest.models.blip2_prompt_aug_artquest",
        "evaluation.ArtQuest.models.blip2_kv_artquest",
        "evaluation.ArtQuest.models.blip2_gated_artquest",
        "models.blip2_artquest",
        "models.blip2_prompt_aug_artquest",
        "models.blip2_kv_artquest",
        "models.blip2_gated_artquest",
        "blip2_artquest",
        "blip2_prompt_aug_artquest",
        "blip2_kv_artquest",
        "blip2_gated_artquest",
    }
    uses_artquest_native_backend = predictor_module in native_modules
    
    if uses_artquest_native_backend:
        logging.info("Detected ArtQuest-native backend; running single-pass inference.")
        predictions = predict_answers_func(
            model=model,
            dataset=artquest_dataset,
            config=config,
            batch_size=batch_size,
            device=device
        )
    else:
        logging.info("Creating adapter dataset (no retrieved contexts)...")
        dataset_original = create_artquest_adapter_dataset(
            artquest_dataset, config, use_retrieved_context=False
        )
        
        logging.info("Running model inference...")
        predictions_original = predict_answers_func(
            model=model,
            dataset=dataset_original,
            config=config,
            batch_size=batch_size,
            device=device
        )
        
        pred_dict_original = {pred["question_id"]: pred["answer"] for pred in predictions_original}
        
        predictions = []
        for i in range(len(artquest_dataset)):
            question_id = artquest_dataset[i]["question_id"]
            predictions.append({
                "question_id": question_id,
                "answer": pred_dict_original.get(question_id, ""),
            })
    
    # Evaluate results
    logging.info("Evaluating results...")
    metrics = evaluate_results(
        predictions=predictions,
        dataset=artquest_dataset,
        output_dir=output_dir
    )
    
    logging.info("Evaluation completed successfully!")
    logging.info(f"Original Context - EM: {metrics['original_context']['em']:.4f}, BLEU: {metrics['original_context']['bleu']:.4f}")


if __name__ == "__main__":
    main()

