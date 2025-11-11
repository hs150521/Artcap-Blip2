#!/usr/bin/env python3
"""
VQAv2 Evaluation Script for BLIP2-opt-2.7b

This script converts the shell-based evaluation workflow into a modular Python script
that supports data sampling and model switching.
"""

import argparse
import json
import logging
import os
import random
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from PIL import Image
from omegaconf import OmegaConf
from tqdm import tqdm

# Add project root to path for model imports
REPO_ROOT = Path(__file__).parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Add LAVIS to path
sys.path.insert(0, str(REPO_ROOT / "blip2" / "LAVIS"))

from lavis.common.config import Config as LAVISConfig
from lavis.common.registry import registry
from lavis.common.vqa_tools.vqa import VQA
from lavis.common.vqa_tools.vqa_eval import VQAEval
from lavis.datasets.builders import *
from lavis.processors import *
import lavis.tasks as tasks

# Note: lavis.models.* is imported only when needed in model backends


def find_image_path(vis_root: str, image_path_in_ann: str) -> str:
    """
    Intelligently find image file path, handling cases where images are in subdirectories.
    
    Args:
        vis_root: Root directory of images (e.g., /data/lavis/coco/images)
        image_path_in_ann: Image path from annotation (e.g., "COCO_val2014_000000000391.jpg" or "val2014/COCO_val2014_000000000391.jpg")
        
    Returns:
        Full path to the image file
    """
    # Try direct path first
    direct_path = os.path.join(vis_root, image_path_in_ann)
    if os.path.exists(direct_path):
        return direct_path
    
    # If not found, try searching in subdirectories
    # Common COCO subdirectories: train2014, val2014, test2014
    filename = os.path.basename(image_path_in_ann)
    
    # Try common subdirectory patterns
    subdirs = ['val2014', 'train2014', 'test2014', 'val', 'train', 'test']
    for subdir in subdirs:
        candidate_path = os.path.join(vis_root, subdir, filename)
        if os.path.exists(candidate_path):
            return candidate_path
    
    # If still not found, search all subdirectories
    if os.path.isdir(vis_root):
        for item in os.listdir(vis_root):
            subdir_path = os.path.join(vis_root, item)
            if os.path.isdir(subdir_path):
                candidate_path = os.path.join(subdir_path, filename)
                if os.path.exists(candidate_path):
                    return candidate_path
    
    # If still not found, return the direct path (will raise error later)
    return direct_path


def load_config(config_path: str) -> Dict:
    """
    Load configuration from YAML file and add data_percentage support.
    
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


def load_vqav2_data(config_dict: Dict, data_percentage: float = 1.0) -> Tuple[List[Dict], Dict]:
    """
    Load VQAv2 dataset and apply data sampling if needed.
    
    Args:
        config_dict: Configuration dictionary
        data_percentage: Percentage of data to use (0-1)
        
    Returns:
        Tuple of (data_samples, dataset_info)
        - data_samples: List of samples with image, question, question_id
        - dataset_info: Dictionary with question_file and annotation_file paths
    """
    # Create a copy of config for LAVIS (LAVIS requires model to be a dict)
    # If model is a string, convert it to dict format for LAVIS dataset loading
    lavis_config_dict = config_dict.copy()
    model_config = config_dict.get("model")
    
    if isinstance(model_config, str):
        # For dataset loading, we need a model dict for LAVIS
        # Use default BLIP2 config since dataset loading mainly needs processors
        lavis_config_dict["model"] = {
            "arch": "blip2_opt",
            "model_type": "pretrain_opt2.7b",
            "use_grad_checkpoint": False
        }
    elif isinstance(model_config, dict):
        # Already in dict format, use as is
        pass
    else:
        # Default fallback
        lavis_config_dict["model"] = {
            "arch": "blip2_opt",
            "model_type": "pretrain_opt2.7b",
            "use_grad_checkpoint": False
        }
    
    # Convert dict to OmegaConf
    omega_config = OmegaConf.create(lavis_config_dict)
    
    # Use a workaround: save to temp file then load (LAVIS Config requires a file path)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        OmegaConf.save(omega_config, f.name)
        temp_config_path = f.name
    
    try:
        class Args:
            def __init__(self, cfg_path):
                self.cfg_path = cfg_path
                self.options = None
        
        args = Args(temp_config_path)
        lavis_config = LAVISConfig(args)
        
        # Setup task and build datasets (similar to evaluate.py)
        task = tasks.setup_task(lavis_config)
        datasets = task.build_datasets(lavis_config)
    finally:
        # Clean up temp file
        if os.path.exists(temp_config_path):
            os.remove(temp_config_path)
    
    # Get the eval dataset (typically 'val' split)
    eval_datasets = {}
    dataset_info = {}
    
    test_splits = config_dict.get("run", {}).get("test_splits", ["val"])
    
    for dataset_name, dataset in datasets.items():
        for split_name, split_dataset in dataset.items():
            if split_name in test_splits:
                eval_datasets[split_name] = split_dataset
                
                # Get question and annotation file paths from task
                if hasattr(task, "ques_files") and split_name in task.ques_files:
                    dataset_info["question_file"] = task.ques_files[split_name]
                    dataset_info["annotation_file"] = task.anno_files[split_name]
                elif hasattr(split_dataset, "coco_fmt_qust_file") and split_dataset.coco_fmt_qust_file:
                    dataset_info["question_file"] = split_dataset.coco_fmt_qust_file
                    dataset_info["annotation_file"] = split_dataset.coco_fmt_anno_file
                else:
                    # Fallback: try to find from cache
                    cache_root = registry.get_path("cache_root")
                    dataset_info["question_file"] = os.path.join(
                        cache_root, f"{dataset_name}_gt", f"{dataset_name}_{split_name}_questions.json"
                    )
                    dataset_info["annotation_file"] = os.path.join(
                        cache_root, f"{dataset_name}_gt", f"{dataset_name}_{split_name}_annotations.json"
                    )
    
    if not eval_datasets:
        raise ValueError("No evaluation datasets found")
    
    # Use the first eval split (typically 'val')
    split_name = list(eval_datasets.keys())[0]
    eval_dataset = eval_datasets[split_name]
    
    # Create a wrapper dataset that handles subdirectory image paths
    # We'll modify the dataset's __getitem__ method to use our find_image_path function
    class WrappedDataset:
        def __init__(self, dataset, vis_root, find_image_path_func):
            self.dataset = dataset
            self.vis_root = vis_root
            self.find_image_path = find_image_path_func
            self.vis_processor = dataset.vis_processor
            self.text_processor = dataset.text_processor
            
        def __len__(self):
            return len(self.dataset)
        
        def __getitem__(self, idx):
            # Get annotation to access image path
            ann = self.dataset.annotation[idx]
            image_path_in_ann = ann.get("image", "")
            
            # Find the actual image path (handling subdirectories)
            actual_image_path = self.find_image_path(self.vis_root, image_path_in_ann)
            
            # Load and process image
            image = Image.open(actual_image_path).convert("RGB")
            processed_image = self.vis_processor(image)
            
            # Get processed question
            question = self.text_processor(ann.get("question", ""))
            
            return {
                "image": processed_image,
                "text_input": question,
                "question_id": ann.get("question_id", idx),
                "instance_id": ann.get("instance_id", idx),
                # Store image path info for models that need raw PIL Images
                "image_path": actual_image_path,
                "image_path_in_ann": image_path_in_ann,
                "vis_root": self.vis_root,
            }
    
    # Wrap the dataset
    wrapped_dataset = WrappedDataset(eval_dataset, eval_dataset.vis_root, find_image_path)
    
    # Apply data sampling if needed (sample indices instead of loading all data)
    indices = list(range(len(wrapped_dataset)))
    if data_percentage < 1.0:
        seed = config_dict.get("run", {}).get("seed", 42)
        random.seed(seed)
        np.random.seed(seed)
        
        total_samples = len(indices)
        num_samples = int(total_samples * data_percentage)
        
        # Sample indices randomly
        indices = sorted(random.sample(indices, num_samples))
        
        logging.info(f"Sampled {num_samples} samples ({data_percentage*100:.1f}%) from {total_samples} total samples")
    
    # Create a subset dataset using sampled indices
    class SubsetDataset:
        def __init__(self, dataset, indices):
            self.dataset = dataset
            self.indices = indices
        
        def __len__(self):
            return len(self.indices)
        
        def __getitem__(self, idx):
            return self.dataset[self.indices[idx]]
    
    final_dataset = SubsetDataset(wrapped_dataset, indices) if data_percentage < 1.0 else wrapped_dataset
    
    return final_dataset, dataset_info


def evaluate_results(
    predictions: List[Dict],
    dataset_info: Dict,
    output_dir: str
) -> Dict:
    """
    Evaluate predictions against ground truth using VQA evaluation tools.
    
    Args:
        predictions: List of predictions with question_id and answer
        dataset_info: Dictionary with question_file and annotation_file paths
        output_dir: Directory to save results
        
    Returns:
        Dictionary with evaluation metrics
    """
    # Save predictions to temporary file
    os.makedirs(output_dir, exist_ok=True)
    result_file = os.path.join(output_dir, "vqa_predictions.json")
    
    with open(result_file, "w") as f:
        json.dump(predictions, f)
    
    logging.info(f"Saved predictions to {result_file}")
    
    # Load VQA evaluation tools
    question_file = dataset_info["question_file"]
    annotation_file = dataset_info["annotation_file"]
    
    if not os.path.exists(question_file):
        raise FileNotFoundError(f"Question file not found: {question_file}")
    if not os.path.exists(annotation_file):
        raise FileNotFoundError(f"Annotation file not found: {annotation_file}")
    
    # Load original annotations and questions
    with open(annotation_file, 'r') as f:
        original_annotations = json.load(f)
    with open(question_file, 'r') as f:
        original_questions = json.load(f)
    
    # Extract question_ids from predictions
    predicted_question_ids = set(pred["question_id"] for pred in predictions)
    
    # Filter annotations and questions to only include predicted question_ids
    filtered_annotations = {
        "info": original_annotations.get("info", {}),
        "task_type": original_annotations.get("task_type", ""),
        "data_type": original_annotations.get("data_type", ""),
        "license": original_annotations.get("license", ""),
        "data_subtype": original_annotations.get("data_subtype", ""),
        "annotations": [ann for ann in original_annotations["annotations"] 
                       if ann["question_id"] in predicted_question_ids]
    }
    
    filtered_questions = {
        "info": original_questions.get("info", {}),
        "task_type": original_questions.get("task_type", ""),
        "data_type": original_questions.get("data_type", ""),
        "license": original_questions.get("license", ""),
        "data_subtype": original_questions.get("data_subtype", ""),
        "questions": [q for q in original_questions["questions"] 
                     if q["question_id"] in predicted_question_ids]
    }
    
    # Create temporary filtered files
    temp_annotation_file = os.path.join(output_dir, "temp_annotations.json")
    temp_question_file = os.path.join(output_dir, "temp_questions.json")
    
    with open(temp_annotation_file, "w") as f:
        json.dump(filtered_annotations, f)
    with open(temp_question_file, "w") as f:
        json.dump(filtered_questions, f)
    
    logging.info(f"Filtered to {len(filtered_annotations['annotations'])} annotations for evaluation")
    
    # Use filtered files for evaluation
    vqa = VQA(temp_annotation_file, temp_question_file)
    vqa_result = vqa.loadRes(result_file, temp_question_file)
    vqa_eval = VQAEval(vqa, vqa_result, n=2)
    
    logging.info("Starting VQA evaluation...")
    vqa_eval.evaluate()
    
    # Clean up temporary files
    try:
        os.remove(temp_annotation_file)
        os.remove(temp_question_file)
    except:
        pass
    
    # Extract metrics
    # VQA eval returns percentages (0-100), normalize to 0-1 range
    # Also ensure accuracy values don't exceed 1.0 (100%)
    overall_percent = vqa_eval.accuracy["overall"]
    # If value is already in [0, 1] range, use as-is; otherwise divide by 100
    if overall_percent <= 1.0:
        overall_acc = overall_percent
    else:
        overall_acc = overall_percent / 100.0
    overall_acc = min(1.0, max(0.0, overall_acc))  # Clamp to [0, 1]
    
    per_answer_type = {}
    for ans_type, acc_percent in vqa_eval.accuracy["perAnswerType"].items():
        # Convert from percentage to [0, 1] range and clamp to valid range
        # Handle both percentage format (0-100) and normalized format (0-1)
        if acc_percent <= 1.0:
            acc_normalized = acc_percent
        else:
            acc_normalized = acc_percent / 100.0
        
        # Log warning if original value was > 100% (indicating a bug)
        if acc_percent > 100.0:
            logging.warning(
                f"Answer type '{ans_type}' had accuracy > 100% ({acc_percent:.2f}%), "
                f"clamped to 100%. This indicates a bug in the VQA evaluation logic."
            )
        
        acc_normalized = min(1.0, max(0.0, acc_normalized))  # Clamp to [0, 1]
        per_answer_type[ans_type] = round(acc_normalized, 4)
    
    metrics = {
        "agg_metrics": round(overall_acc, 4),
        "per_answer_type": per_answer_type
    }
    
    # Print results
    logging.info(f"Overall Accuracy: {metrics['agg_metrics']:.2f}")
    logging.info("Per Answer Type Accuracy:")
    for ans_type, acc in metrics["per_answer_type"].items():
        logging.info(f"  {ans_type}: {acc:.2f}")
    
    # Save metrics to file
    metrics_file = os.path.join(output_dir, "evaluate.txt")
    with open(metrics_file, "w") as f:
        f.write(json.dumps(metrics) + "\n")
    
    logging.info(f"Saved metrics to {metrics_file}")
    
    return metrics


def main():
    """Main function to orchestrate the evaluation pipeline."""
    parser = argparse.ArgumentParser(description="Evaluate BLIP2 on VQAv2")
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
    batch_size = config["run"].get("batch_size_eval", 64)
    device = config["run"].get("device", "cuda")
    output_dir = config["run"].get("output_dir", "output/BLIP2/VQA")
    
    logging.info(f"Configuration loaded:")
    logging.info(f"  Data percentage: {data_percentage}")
    logging.info(f"  Batch size: {batch_size}")
    logging.info(f"  Device: {device}")
    logging.info(f"  Output directory: {output_dir}")
    
    # Determine model type from config
    # Support both old format (model: {arch: ..., model_type: ...}) and new format (model: "blip2")
    model_name = config.get("model")
    if isinstance(model_name, str):
        # New format: model is a string identifier
        pass
    elif isinstance(model_name, dict):
        # Old format: model is a dict with arch/model_type
        # Preserve old config dict for backward compatibility (blip2_lavis will use it)
        # But use "blip2" as the model name for factory lookup
        model_name = "blip2"
        # Keep the old dict in config["model"] for blip2_lavis to access
    else:
        # Default to blip2 if not specified
        model_name = "blip2"
        config["model"] = model_name
    
    logging.info(f"  Model: {model_name}")
    
    # Get model backend functions
    # Handle both script execution and module import
    # Ensure REPO_ROOT is first in sys.path so models.Gated imports work correctly
    repo_root_str = str(REPO_ROOT)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)
    elif sys.path.index(repo_root_str) != 0:
        # Move repo_root to the front if it's not already there
        sys.path.remove(repo_root_str)
        sys.path.insert(0, repo_root_str)
    
    try:
        from .models import get_model_backend
    except ImportError:
        # When run as script, import from the evaluation/VQAv2/models directory
        # Add the evaluation/VQAv2 directory to path AFTER repo_root
        vqav2_dir = Path(__file__).parent
        vqav2_dir_str = str(vqav2_dir)
        if vqav2_dir_str not in sys.path:
            # Insert after repo_root (at index 1)
            sys.path.insert(1, vqav2_dir_str)
        # Now import - this will find evaluation/VQAv2/models first, but
        # blip2_gated.py ensures repo_root is first before importing models.Gated
        from models import get_model_backend
    load_model_func, predict_answers_func = get_model_backend(model_name)
    
    # Load model
    logging.info("Loading model...")
    model = load_model_func(config, device)
    
    # Load data (only metadata, images loaded on demand)
    logging.info("Loading VQAv2 dataset...")
    dataset, dataset_info = load_vqav2_data(config, data_percentage)
    logging.info(f"Dataset ready with {len(dataset)} samples (images will be loaded on demand)")
    
    # Run inference
    logging.info("Running model inference...")
    predictions = predict_answers_func(
        model=model,
        dataset=dataset,
        config=config,
        batch_size=batch_size,
        device=device
    )
    
    # Evaluate results
    logging.info("Evaluating results...")
    metrics = evaluate_results(
        predictions=predictions,
        dataset_info=dataset_info,
        output_dir=output_dir
    )
    
    logging.info("Evaluation completed successfully!")
    logging.info(f"Final Overall Accuracy: {metrics['agg_metrics']:.2f}")


if __name__ == "__main__":
    main()
