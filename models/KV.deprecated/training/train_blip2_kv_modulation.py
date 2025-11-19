#!/usr/bin/env python3
"""
Training script for BLIP2 with KV modulation.

This script trains the KV-modulated BLIP2 model on ArtQuest dataset.
"""

import argparse
import logging
import os
import sys
import shutil
import time
from pathlib import Path
from typing import Dict, Any, Optional
import re
import json
import csv
from datetime import datetime

# Suppress TensorFlow/MediaPipe verbose logs before importing torch
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0=all, 1=info, 2=warnings, 3=errors only
os.environ['GLOG_minloglevel'] = '2'  # Suppress glog (used by MediaPipe)

# Filter stderr to suppress MediaPipe verbose logs
class StderrFilter:
    """Filter stderr to suppress MediaPipe verbose logs."""
    def __init__(self, original_stderr):
        self.original_stderr = original_stderr
        # Patterns to filter out (match MediaPipe glog format)
        self.filter_patterns = [
            r'inference_feedback_manager\.cc',
            r'gl_context\.cc',
            r'gl_context_egl\.cc',
            r'Successfully initialized EGL',
            r'GL version:',
            r'Feedback manager requires',
            r'renderer: NVIDIA',
        ]
        self.compiled_patterns = [re.compile(pattern) for pattern in self.filter_patterns]
        self.buffer = ''  # Buffer for incomplete lines
    
    def write(self, text):
        if not text:
            return
        
        # Add to buffer
        self.buffer += text
        
        # Process complete lines
        while '\n' in self.buffer:
            line, self.buffer = self.buffer.split('\n', 1)
            line += '\n'
            
            # Check if line should be filtered
            should_filter = False
            for pattern in self.compiled_patterns:
                if pattern.search(line):
                    should_filter = True
                    break
            
            # Only write if not filtered
            if not should_filter:
                self.original_stderr.write(line)
    
    def flush(self):
        # Write any remaining buffer content if it doesn't match filter patterns
        if self.buffer:
            should_filter = False
            for pattern in self.compiled_patterns:
                if pattern.search(self.buffer):
                    should_filter = True
                    break
            if not should_filter:
                self.original_stderr.write(self.buffer)
            self.buffer = ''
        self.original_stderr.flush()
    
    def __getattr__(self, name):
        # Delegate all other attributes to original stderr
        return getattr(self.original_stderr, name)

# Install stderr filter
sys.stderr = StderrFilter(sys.stderr)

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
from torch.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

# Add paths
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # Changed from parents[2] to parents[1] to point to KV directory
from models.blip2_opt_kv_modulated import Blip2OPTKVModulated
from datasets.artquest_dataset import get_artquest_dataloaders
from utils.model_loader import load_blip2_kv_modulated_model


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create file handler for detailed training logs
log_dir = Path("KV/training_logs")
log_dir.mkdir(parents=True, exist_ok=True)
file_handler = logging.FileHandler(log_dir / "training_detailed.log")
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

# Create CSV logger for detailed metrics
metrics_file = log_dir / "training_metrics.csv"
csv_header_written = False
if not metrics_file.exists():
    with open(metrics_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'timestamp', 'epoch', 'batch', 'train_loss', 'val_loss', 
            'learning_rate', 'grad_norm', 'skipped_batches'
        ])


def setup_training(config: Dict[str, Any], device: torch.device) -> tuple:
    """
    Setup model, dataloaders, and optimizer for training.
    
    Returns:
        model, train_loader, val_loader, optimizer, scheduler, scaler, config
    """
    # Load model
    logger.info("Loading model...")
    model = load_blip2_kv_modulated_model(config, device=device)
    
    # Set model to train mode
    model.train()
    
    # Freeze specified components
    freeze_vit = config.get("architecture", {}).get("freeze_vit", True)
    freeze_llm = config.get("architecture", {}).get("freeze_llm", True)
    freeze_efficientnet = config.get("architecture", {}).get("freeze_efficientnet", True)
    
    # Only train KV-Prefix generator and Qformer (if not frozen)
    trainable_params = []
    for name, param in model.named_parameters():
        if freeze_vit and "visual_encoder" in name:
            param.requires_grad = False
        elif freeze_llm and "opt_model" in name:
            param.requires_grad = False
        elif freeze_efficientnet and "efficientnet_model" in name:
            param.requires_grad = False
        elif param.requires_grad:
            trainable_params.append(param)
            logger.info(f"Training parameter: {name}")
    
    logger.info(f"Total trainable parameters: {sum(p.numel() for p in trainable_params):,}")
    
    # Check if there are any trainable parameters
    if len(trainable_params) == 0:
        raise ValueError(
            "No trainable parameters found! All parameters are frozen. "
            "Check freeze_vit, freeze_llm, and freeze_efficientnet settings."
        )
    
    # Setup dataloaders
    logger.info("Loading datasets...")
    dataset_cfg = config.get("dataset", {})
    multi_dataset_cfg = config.get("multi_dataset", {})
    use_multi_dataset = multi_dataset_cfg.get("enabled", False)
    
    batch_size = config.get("training", {}).get("batch_size", 4)
    num_workers = config.get("training", {}).get("num_workers", 4)
    image_size = config.get("architecture", {}).get("img_size", 224)
    seed = config.get("project", {}).get("seed", 1337)
    
    if use_multi_dataset:
        # Multi-dataset mode: ArtQuest + VQA with 1:1 ratio
        logger.info("Using multi-dataset mode (ArtQuest + VQA)")
        
        # ArtQuest dataset paths
        artquest_cfg = config.get("artquest_dataset", {})
        artquest_train_path = artquest_cfg.get("train_data")
        artquest_val_path = artquest_cfg.get("val_data")
        artquest_image_root = artquest_cfg.get("image_root")
        
        # VQA dataset paths
        vqa_cfg = config.get("vqa_dataset", {})
        vqa_train_questions = vqa_cfg.get("train_question_file")
        vqa_train_annotations = vqa_cfg.get("train_annotation_file")
        vqa_val_questions = vqa_cfg.get("val_question_file")
        vqa_val_annotations = vqa_cfg.get("val_annotation_file")
        vqa_image_root = vqa_cfg.get("image_root")
        
        # Validate paths
        if not artquest_train_path or not Path(artquest_train_path).exists():
            raise ValueError(f"ArtQuest training data path not found: {artquest_train_path}")
        if not artquest_val_path or not Path(artquest_val_path).exists():
            raise ValueError(f"ArtQuest validation data path not found: {artquest_val_path}")
        if not artquest_image_root or not Path(artquest_image_root).exists():
            raise ValueError(f"ArtQuest image root not found: {artquest_image_root}")
        
        if not vqa_train_questions or not Path(vqa_train_questions).exists():
            raise ValueError(f"VQA training questions file not found: {vqa_train_questions}")
        if not vqa_train_annotations or not Path(vqa_train_annotations).exists():
            raise ValueError(f"VQA training annotations file not found: {vqa_train_annotations}")
        if not vqa_val_questions or not Path(vqa_val_questions).exists():
            raise ValueError(f"VQA validation questions file not found: {vqa_val_questions}")
        if not vqa_val_annotations or not Path(vqa_val_annotations).exists():
            raise ValueError(f"VQA validation annotations file not found: {vqa_val_annotations}")
        if not vqa_image_root or not Path(vqa_image_root).exists():
            raise ValueError(f"VQA image root not found: {vqa_image_root}")
        
        # Load ArtQuest datasets
        from datasets.artquest_dataset import ArtQuestDataset
        artquest_train_dataset = ArtQuestDataset(
            Path(artquest_train_path),
            Path(artquest_image_root),
            image_size=image_size,
        )
        artquest_val_dataset = ArtQuestDataset(
            Path(artquest_val_path),
            Path(artquest_image_root),
            image_size=image_size,
        )
        
        # Load VQA datasets
        from datasets.coco_captions_dataset import COCOCaptionsDataset
        vqa_train_dataset = COCOCaptionsDataset(
            Path(vqa_train_questions),
            Path(vqa_image_root),
            image_size=image_size,
            limit=vqa_cfg.get("max_samples"),
        )
        vqa_val_dataset = COCOCaptionsDataset(
            Path(vqa_val_questions),
            Path(vqa_image_root),
            image_size=image_size,
            limit=vqa_cfg.get("max_samples"),
        )
        
        # Create balanced dataloaders with 1:1 ratio
        from datasets.combined_dataset import create_balanced_dataloader
        train_loader = create_balanced_dataloader(
            artquest_train_dataset,
            vqa_train_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True,
            seed=seed,
        )
        val_loader = create_balanced_dataloader(
            artquest_val_dataset,
            vqa_val_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
            seed=seed,
        )
        test_loader = None
        
        logger.info(f"ArtQuest train samples: {len(artquest_train_dataset)}, VQA train samples: {len(vqa_train_dataset)}")
        logger.info(f"Combined train samples: {len(train_loader.dataset)} (1:1 ratio)")
    else:
        # Single dataset mode: ArtQuest only
        logger.info("Using single-dataset mode (ArtQuest only)")
        dataset_cfg = config.get("dataset", {})
        
        # Validate dataset paths
        train_data_path = dataset_cfg.get("train_data")
        val_data_path = dataset_cfg.get("val_data")
        image_root = dataset_cfg.get("image_root")
        
        if not train_data_path or not Path(train_data_path).exists():
            raise ValueError(f"Training data path not found: {train_data_path}")
        if not val_data_path or not Path(val_data_path).exists():
            raise ValueError(f"Validation data path not found: {val_data_path}")
        if not image_root or not Path(image_root).exists():
            raise ValueError(f"Image root directory not found: {image_root}")
        
        train_loader, val_loader, test_loader = get_artquest_dataloaders(
            train_data_path=train_data_path,
            val_data_path=val_data_path,
            test_data_path=dataset_cfg.get("test_data"),
            image_root=image_root,
            cache_path=dataset_cfg.get("cache_path"),
            batch_size=batch_size,
            num_workers=num_workers,
            image_size=image_size,
        )
    
    # Setup optimizer
    training_cfg = config.get("training", {})
    # Ensure learning_rate and weight_decay are floats (YAML might read them as strings)
    learning_rate = float(training_cfg.get("learning_rate", 1e-4))
    weight_decay = float(training_cfg.get("weight_decay", 0.01))
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    
    # Setup learning rate scheduler (warmup + linear decay)
    num_epochs = training_cfg.get("num_epochs", 10)
    num_training_steps = len(train_loader) * num_epochs
    warmup_steps = training_cfg.get("warmup_steps", 500)
    
    # Use linear schedule with warmup
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - warmup_steps)))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Setup mixed precision training
    use_amp = training_cfg.get("use_amp", True) and device.type == "cuda"
    scaler = GradScaler() if use_amp else None
    if use_amp:
        logger.info("Using mixed precision training (AMP)")
    
    return model, train_loader, val_loader, optimizer, scheduler, scaler, config


def train_epoch(
    model: Blip2OPTKVModulated,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    scaler: Optional[GradScaler],
    device: torch.device,
    epoch: int,
    config: Dict[str, Any],
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    skipped_batches = 0
    
    training_cfg = config.get("training", {})
    max_grad_norm = training_cfg.get("max_grad_norm", 1.0)
    gradient_accumulation_steps = training_cfg.get("gradient_accumulation_steps", 1)
    use_amp = scaler is not None
    autocast_dtype = torch.float16 if device.type == "cuda" else torch.bfloat16
    
    optimizer.zero_grad()
    accumulated_steps = 0  # Track accumulated gradient steps
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False, position=1)
    for batch_idx, batch in enumerate(pbar):
        try:
            # Move batch to device
            images = batch["image"].to(device)
            text_inputs = batch["text_input"]
            
            # Prepare samples dict
            samples = {
                "image": images,
                "text_input": text_inputs,
                "answers": batch["answer"],
            }
            
            # Note: KV prefix will be set automatically in model.forward()
            # No need to clear it manually as forward() sets it fresh each time
            
            # Forward pass with mixed precision
            with autocast(device_type=device.type, dtype=autocast_dtype, enabled=use_amp):
                outputs = model(samples)
                loss = outputs["loss"]
                
                # Scale loss for gradient accumulation
                loss = loss / gradient_accumulation_steps
            
            # Check for NaN loss
            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning(f"NaN/Inf loss detected at batch {batch_idx + 1}, skipping...")
                skipped_batches += 1
                # Clear gradients on NaN/Inf to prevent accumulation issues
                # Also reset accumulation counter since we're skipping this batch
                optimizer.zero_grad()
                accumulated_steps = 0
                continue
            
            # Backward pass
            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            accumulated_steps += 1
            
            # Update weights every gradient_accumulation_steps
            if accumulated_steps >= gradient_accumulation_steps:
                # Gradient clipping
                grad_norm = 0.0
                if max_grad_norm > 0:
                    trainable_params_list = [p for p in model.parameters() if p.requires_grad]
                    if trainable_params_list:
                        if use_amp:
                            scaler.unscale_(optimizer)
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            trainable_params_list,
                            max_grad_norm
                        ).item()
                
                # Optimizer step
                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                
                scheduler.step()
                optimizer.zero_grad()
                accumulated_steps = 0  # Reset after optimizer step
            
            # Update metrics (use unscaled loss for logging)
            scaled_loss = loss.item() * gradient_accumulation_steps
            total_loss += scaled_loss
            num_batches += 1
            
            # Update progress bar
            current_lr = scheduler.get_last_lr()[0] if scheduler else training_cfg.get("learning_rate", 1e-4)
            pbar.set_postfix({
                "loss": scaled_loss,
                "avg_loss": total_loss / num_batches if num_batches > 0 else 0.0,
                "lr": f"{current_lr:.2e}",
                "skipped": skipped_batches
            })
            
            # Log detailed metrics to CSV
            with open(metrics_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    datetime.now().isoformat(),
                    epoch,
                    batch_idx + 1,
                    scaled_loss,
                    None,  # val_loss will be filled during validation
                    current_lr,
                    grad_norm,
                    skipped_batches
                ])
            
            # Logging
            if (batch_idx + 1) % training_cfg.get("logging_steps", 100) == 0:
                logger.info(
                    f"Epoch {epoch}, Batch {batch_idx + 1}/{len(train_loader)}, "
                    f"Loss: {scaled_loss:.4f}, Avg Loss: {total_loss / num_batches:.4f}, "
                    f"LR: {current_lr:.2e}, Grad Norm: {grad_norm:.4f}, Skipped: {skipped_batches}"
                )
        except Exception as e:
            logger.error(f"Error processing batch {batch_idx + 1}: {e}", exc_info=True)
            skipped_batches += 1
            # Clear gradients on error to prevent accumulation issues
            # Also reset accumulation counter since we're skipping this batch
            optimizer.zero_grad()
            accumulated_steps = 0
            continue
    
    # Handle remaining gradients if not evenly divisible by gradient_accumulation_steps
    # Only process if we have accumulated some gradients
    if accumulated_steps > 0:
        if max_grad_norm > 0:
            trainable_params_list = [p for p in model.parameters() if p.requires_grad]
            if trainable_params_list:
                if use_amp:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    trainable_params_list,
                    max_grad_norm
                )
        if use_amp:
            # scaler.step() will skip optimizer step if overflow occurred
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        # Update scheduler only after optimizer step
        scheduler.step()
        optimizer.zero_grad()
    
    if num_batches == 0:
        error_msg = f"No valid batches processed in epoch {epoch}! All batches were skipped."
        logger.error(error_msg)
        raise RuntimeError(
            error_msg + " This indicates a serious problem with the data or model. "
            "Please check for NaN/Inf values, data loading issues, or model configuration."
        )
    
    if skipped_batches > 0:
        logger.warning(f"Skipped {skipped_batches} batches in epoch {epoch} due to errors or NaN/Inf loss")
    
    avg_loss = total_loss / num_batches
    return {"loss": avg_loss}


@torch.no_grad()
def validate(
    model: Blip2OPTKVModulated,
    val_loader: DataLoader,
    device: torch.device,
    config: Dict[str, Any],
    scaler: Optional[GradScaler] = None,
) -> Dict[str, float]:
    """Validate model."""
    model.eval()
    # Ensure KV prefix generator is also in eval mode if it exists
    if hasattr(model, 'kv_prefix_generator') and model.kv_prefix_generator is not None:
        model.kv_prefix_generator.eval()
    
    total_loss = 0.0
    num_batches = 0
    skipped_batches = 0
    use_amp = scaler is not None if scaler else False
    autocast_dtype = torch.float16 if device.type == "cuda" else torch.bfloat16
    
    pbar = tqdm(val_loader, desc="Validation", leave=False, position=1)
    for batch in pbar:
        try:
            # Move batch to device
            images = batch["image"].to(device)
            text_inputs = batch["text_input"]
            
            # Prepare samples dict
            samples = {
                "image": images,
                "text_input": text_inputs,
                "answers": batch["answer"],
            }
            
            # Note: KV prefix will be set automatically in model.forward()
            # No need to clear it manually as forward() sets it fresh each time
            
            # Forward pass with mixed precision
            with autocast(device_type=device.type, dtype=autocast_dtype, enabled=use_amp):
                outputs = model(samples)
                loss = outputs["loss"]
            
            # Check for NaN loss
            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning("NaN/Inf loss detected in validation, skipping...")
                skipped_batches += 1
                continue
            
            # Update metrics
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                "loss": loss.item(),
                "avg_loss": total_loss / num_batches if num_batches > 0 else 0.0,
                "skipped": skipped_batches
            })
        except Exception as e:
            logger.error(f"Error processing validation batch: {e}", exc_info=True)
            skipped_batches += 1
            continue
    
    if num_batches == 0:
        logger.error("No valid batches processed in validation! All batches were skipped.")
        return {"loss": float('inf')}
    
    if skipped_batches > 0:
        logger.warning(f"Skipped {skipped_batches} batches in validation due to errors or NaN/Inf loss")
    
    avg_loss = total_loss / num_batches
    
    # Update CSV with validation loss for this epoch
    if num_batches > 0:
        # Read the last few rows and update validation loss
        rows = []
        with open(metrics_file, 'r', newline='') as f:
            reader = csv.reader(f)
            rows = list(reader)
        
        # Update the last rows that have None for val_loss
        updated = False
        for i in range(len(rows)-1, max(0, len(rows)-100), -1):  # Check last 100 rows
            if len(rows[i]) > 4 and rows[i][4] == '':
                rows[i][4] = str(avg_loss)
                updated = True
        
        if updated:
            with open(metrics_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(rows)
    
    return {"loss": avg_loss}


def check_disk_space(path: Path, required_gb: float = 5.0) -> bool:
    """Check if there's enough disk space at the given path."""
    try:
        stat = shutil.disk_usage(path)
        free_gb = stat.free / (1024 ** 3)  # Convert to GB
        if free_gb < required_gb:
            logger.warning(f"Low disk space: {free_gb:.2f} GB free (need at least {required_gb} GB)")
            return False
        return True
    except Exception as e:
        logger.warning(f"Could not check disk space: {e}")
        return True  # Assume OK if we can't check


def save_checkpoint_atomic(
    checkpoint: Dict[str, Any],
    checkpoint_path: Path,
    max_retries: int = 3,
) -> bool:
    """
    Save checkpoint using atomic write (write to temp file, then rename).
    
    Returns:
        True if successful, False otherwise
    """
    checkpoint_dir = checkpoint_path.parent
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Create temporary file in the same directory (for atomic rename)
    temp_path = checkpoint_path.with_suffix('.tmp')
    
    for attempt in range(max_retries):
        try:
            # Remove temp file if it exists from previous failed attempt
            if temp_path.exists():
                temp_path.unlink()
            
            # Write to temporary file
            torch.save(checkpoint, temp_path)
            
            # Verify the file was written correctly by checking its size
            if not temp_path.exists() or temp_path.stat().st_size == 0:
                raise RuntimeError("Checkpoint file is empty or doesn't exist after save")
            
            # Atomic rename (this is typically atomic on most filesystems)
            temp_path.replace(checkpoint_path)
            
            # Verify final file exists and is not empty
            if not checkpoint_path.exists() or checkpoint_path.stat().st_size == 0:
                raise RuntimeError("Final checkpoint file is empty or doesn't exist")
            
            return True
            
        except Exception as e:
            logger.warning(f"Attempt {attempt + 1}/{max_retries} failed to save checkpoint: {e}")
            # Clean up temp file if it exists
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except Exception:
                    pass
            
            if attempt == max_retries - 1:
                logger.error(f"Failed to save checkpoint after {max_retries} attempts")
                return False
            
            # Wait a bit before retrying (exponential backoff)
            time.sleep(2 ** attempt)
    
    return False


def save_checkpoint(
    model: Blip2OPTKVModulated,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    scaler: Optional[GradScaler],
    epoch: int,
    loss: float,
    checkpoint_dir: Path,
    is_best: bool = False,
):
    """Save model checkpoint with robust error handling."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Check disk space before attempting to save (estimate 5GB needed for large models)
    if not check_disk_space(checkpoint_dir, required_gb=5.0):
        logger.error("Insufficient disk space to save checkpoint. Aborting save.")
        raise RuntimeError("Insufficient disk space to save checkpoint")
    
    # Build checkpoint dictionary
    try:
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "loss": loss,
        }
        
        # Save scaler state if using mixed precision
        if scaler is not None:
            checkpoint["scaler_state_dict"] = scaler.state_dict()
    except Exception as e:
        logger.error(f"Failed to build checkpoint dictionary: {e}")
        raise
    
    # Save regular checkpoint
    checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
    if not save_checkpoint_atomic(checkpoint, checkpoint_path):
        raise RuntimeError(f"Failed to save checkpoint to {checkpoint_path}")
    logger.info(f"Saved checkpoint to {checkpoint_path}")
    
    # Save best checkpoint (only if regular checkpoint succeeded)
    if is_best:
        best_path = checkpoint_dir / "best_checkpoint.pt"
        if not save_checkpoint_atomic(checkpoint, best_path):
            logger.error(f"Failed to save best checkpoint to {best_path}, but regular checkpoint was saved")
            # Don't raise here - at least we saved the regular checkpoint
        else:
            logger.info(f"Saved best checkpoint to {best_path}")


def main():
    parser = argparse.ArgumentParser(description="Train BLIP2 with KV modulation")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config YAML file",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    args = parser.parse_args()
    
    # Load config
    if not Path(args.config).exists():
        logger.error(f"Config file not found: {args.config}")
        sys.exit(1)
    
    try:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Failed to load config file: {e}")
        sys.exit(1)
    
    # Validate config
    required_keys = ["model", "dataset", "training"]
    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        logger.error(f"Missing required config keys: {missing_keys}")
        sys.exit(1)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Setup training
    model, train_loader, val_loader, optimizer, scheduler, scaler, config = setup_training(config, device)
    
    # Resume from checkpoint if provided
    start_epoch = 0
    best_val_loss = float("inf")
    if args.resume:
        try:
            checkpoint = torch.load(args.resume, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"], strict=False)
            
            # Check if optimizer state matches current model parameters
            try:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            except Exception as e:
                logger.warning(f"Failed to load optimizer state (parameters may have changed): {e}")
                logger.info("Continuing with new optimizer state...")
            
            if "scheduler_state_dict" in checkpoint:
                try:
                    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
                except Exception as e:
                    logger.warning(f"Failed to load scheduler state: {e}")
                    logger.info("Continuing with new scheduler state...")
            
            if "scaler_state_dict" in checkpoint and scaler is not None:
                try:
                    scaler.load_state_dict(checkpoint["scaler_state_dict"])
                except Exception as e:
                    logger.warning(f"Failed to load scaler state: {e}")
                    logger.info("Continuing with new scaler state...")
            
            start_epoch = checkpoint.get("epoch", 0) + 1
            best_val_loss = checkpoint.get("loss", float("inf"))
            logger.info(f"Resumed from epoch {start_epoch}, best val loss: {best_val_loss:.4f}")
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}", exc_info=True)
            logger.info("Starting training from scratch...")
    
    # Training loop
    training_cfg = config.get("training", {})
    num_epochs = training_cfg.get("num_epochs", 10)
    output_dir = Path(training_cfg.get("output_dir", "KV/outputs"))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Starting training for {num_epochs} epochs...")
    
    # Create overall training progress bar
    epoch_pbar = tqdm(
        range(start_epoch, num_epochs),
        desc="Training Progress",
        unit="epoch",
        position=0,
        leave=True
    )
    
    for epoch in epoch_pbar:
        # Ensure model is in train mode (including KV prefix generator)
        model.train()
        if hasattr(model, 'kv_prefix_generator') and model.kv_prefix_generator is not None:
            model.kv_prefix_generator.train()
        
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, scheduler, scaler, device, epoch + 1, config)
        
        # Validate
        val_metrics = validate(model, val_loader, device, config, scaler)
        
        # Save checkpoint
        is_best = val_metrics["loss"] < best_val_loss
        if is_best:
            best_val_loss = val_metrics["loss"]
            logger.info(f"New best validation loss: {best_val_loss:.4f}")
        
        # Update overall progress bar
        epoch_pbar.set_postfix({
            "Epoch": f"{epoch + 1}/{num_epochs}",
            "Train Loss": f"{train_metrics['loss']:.4f}",
            "Val Loss": f"{val_metrics['loss']:.4f}",
            "Best Val": f"{best_val_loss:.4f}" if best_val_loss != float('inf') else "N/A"
        })
        
        logger.info(f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {train_metrics['loss']:.4f}, Val Loss: {val_metrics['loss']:.4f}")
        
        # Ensure model is back in train mode for next epoch
        model.train()
        if hasattr(model, 'kv_prefix_generator') and model.kv_prefix_generator is not None:
            model.kv_prefix_generator.train()
        
        # Save checkpoint with error handling
        try:
            save_checkpoint(
                model,
                optimizer,
                scheduler,
                scaler,
                epoch + 1,
                val_metrics["loss"],
                output_dir,
                is_best=is_best,
            )
        except Exception as e:
            logger.error(f"Failed to save checkpoint for epoch {epoch + 1}: {e}", exc_info=True)
            logger.warning("Training will continue, but checkpoint was not saved. Please check disk space and permissions.")
            # Continue training even if checkpoint save fails
    
    epoch_pbar.close()
    logger.info("Training completed!")


if __name__ == "__main__":
    main()
