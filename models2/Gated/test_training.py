#!/usr/bin/env python
"""Test script for Gated BLIP-2 training with small batches."""

import torch
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from models.Gated.trainers import GatedTrainer
from models.Gated.configs import load_config

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def create_mock_data():
    """Create mock data files for testing."""
    import pandas as pd
    import os
    from PIL import Image
    import numpy as np
    
    # Create mock data directory with dataset name subdirectory
    mock_dir = Path("/tmp/gated_test_data")
    dataset_dir = mock_dir / "artquest"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    
    # Create mock CSV files
    train_data = {
        "question_id": [1, 2, 3, 4],
        "image": ["img1.jpg", "img2.jpg", "img3.jpg", "img4.jpg"],
        "question": ["What style is this?", "Describe this painting", "What style is this?", "Describe this artwork"],
        "answer": ["Impressionism", "Baroque", "Renaissance", "Cubism"],
        "style": ["Impressionism", "Baroque", "Renaissance", "Cubism"]
    }
    
    val_data = {
        "question_id": [5, 6],
        "image": ["img5.jpg", "img6.jpg"],
        "question": ["What style is this?", "Describe this painting"],
        "answer": ["Surrealism", "Realism"],
        "style": ["Surrealism", "Realism"]
    }
    
    train_df = pd.DataFrame(train_data)
    val_df = pd.DataFrame(val_data)
    
    train_csv = dataset_dir / "train.csv"
    val_csv = dataset_dir / "val.csv"
    
    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)
    
    # Create mock image directory
    image_dir = dataset_dir / "images"
    image_dir.mkdir(exist_ok=True)
    
    # Create dummy image files (small RGB images)
    for img_name in ["img1.jpg", "img2.jpg", "img3.jpg", "img4.jpg", "img5.jpg", "img6.jpg"]:
        # Create a small random RGB image
        img_array = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        img.save(image_dir / img_name)
    
    return mock_dir


def test_training_pipeline():
    """Test the full training pipeline with minimal settings."""
    logger.info("Testing training pipeline...")
    
    try:
        # Create mock data
        mock_dir = create_mock_data()
        
        # Load default config
        config = load_config("default")
        
        # Modify config for testing
        config["data"]["dataset_root"] = str(mock_dir)
        config["data"]["image_root"] = str(mock_dir / "artquest" / "images")
        config["data"]["image_size"] = 224  # Match model input size
        config["data"]["batch_size"] = 2  # Small batch size
        config["data"]["num_workers"] = 0  # No multiprocessing for testing
        config["training"]["max_epochs"] = 1  # Just one epoch
        config["training"]["logging_steps"] = 1  # Log every step
        config["training"]["use_amp"] = False  # Disable AMP for stability
        
        # Create trainer
        trainer = GatedTrainer(config)
        
        logger.info("Trainer created successfully!")
        logger.info(f"Train loader length: {len(trainer.train_loader)}")
        logger.info(f"Val loader length: {len(trainer.val_loader)}")
        
        # Test one training step
        logger.info("Testing one training step...")
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            initial_memory = torch.cuda.memory_allocated() / 1024**3  # GB
            logger.info(f"Initial GPU memory: {initial_memory:.2f} GB")
        
        # Run one epoch
        train_metrics = trainer._run_epoch(0)
        
        if torch.cuda.is_available():
            peak_memory = torch.cuda.max_memory_allocated() / 1024**3  # GB
            logger.info(f"Peak GPU memory during training: {peak_memory:.2f} GB")
        
        logger.info(f"Training step successful! Loss: {train_metrics['loss']:.4f}")
        
        # Test evaluation
        logger.info("Testing evaluation...")
        val_metrics = trainer.evaluate(trainer.val_loader)
        logger.info(f"Evaluation successful! Loss: {val_metrics['loss']:.4f}")
        
        # Don't clean up for testing purposes
        # import shutil
        # shutil.rmtree(mock_dir)
        
        return True
        
    except Exception as e:
        logger.error(f"Training pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run training pipeline test."""
    logger.info("Starting Gated BLIP-2 training pipeline test...")
    
    success = test_training_pipeline()
    
    if success:
        logger.info("Training pipeline test passed successfully!")
    else:
        logger.error("Training pipeline test failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
