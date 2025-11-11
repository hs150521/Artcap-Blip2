#!/usr/bin/env python
"""Test script for Gated BLIP-2 with small batches to verify GPU memory usage."""

import torch
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from models.Gated.modules import Blip2OPTGated
from models.Gated.configs import load_config

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def test_model_loading():
    """Test if the model can be loaded successfully."""
    logger.info("Testing model loading...")
    
    try:
        # Load default config
        config = load_config("default")
        
        # Create model with minimal settings for testing
        model = Blip2OPTGated(
            vit_model="eva_clip_g",
            img_size=224,
            freeze_vit=True,
            num_query_token=32,
            opt_model="facebook/opt-2.7b",
            prompt="",
            max_txt_len=32,
            efficientnet_output_dim=768,
            convert_from_blip_norm=True,
            gating_config={
                "type": "film",
                "per_head": True,
                "hidden_dim": 512,
                "init_scale": 0.01,
                "use_layer_norm": True,
            },
            prompt_mapper_cfg={
                "num_tokens": 4,
                "hidden_dim": 1024,
                "dropout": 0.1,
                "use_layer_norm": True,
            },
            lora_config={
                "enabled": False,
            }
        )
        
        logger.info("Model loaded successfully!")
        
        # Count trainable parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        
        logger.info(f"Trainable parameters: {trainable_params:,}")
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable ratio: {trainable_params/total_params:.4f}")
        
        return model
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


def test_forward_pass(model, batch_size=2):
    """Test forward pass with small batch to check memory usage."""
    logger.info(f"Testing forward pass with batch_size={batch_size}...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Create dummy batch
    images = torch.randn(batch_size, 3, 224, 224, device=device)
    text_inputs = ["What style is this painting?" for _ in range(batch_size)]
    
    samples = {
        "image": images,
        "text_input": text_inputs,
    }
    
    try:
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            initial_memory = torch.cuda.memory_allocated() / 1024**3  # GB
            logger.info(f"Initial GPU memory: {initial_memory:.2f} GB")
        
        # Forward pass
        with torch.no_grad():
            outputs = model(samples)
        
        if torch.cuda.is_available():
            peak_memory = torch.cuda.max_memory_allocated() / 1024**3  # GB
            logger.info(f"Peak GPU memory during forward pass: {peak_memory:.2f} GB")
        
        logger.info(f"Forward pass successful! Loss: {outputs['loss'].item():.4f}")
        logger.info(f"Output keys: {list(outputs.keys())}")
        
        return outputs
        
    except Exception as e:
        logger.error(f"Forward pass failed: {e}")
        raise


def test_generation(model, batch_size=1):
    """Test generation with small batch."""
    logger.info(f"Testing generation with batch_size={batch_size}...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Create dummy batch
    images = torch.randn(batch_size, 3, 224, 224, device=device)
    text_inputs = ["What style is this painting?" for _ in range(batch_size)]
    
    samples = {
        "image": images,
        "text_input": text_inputs,
    }
    
    try:
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            initial_memory = torch.cuda.memory_allocated() / 1024**3  # GB
            logger.info(f"Initial GPU memory: {initial_memory:.2f} GB")
        
        # Generate
        with torch.no_grad():
            generated = model.generate(
                samples,
                use_nucleus_sampling=False,
                num_beams=1,  # Use single beam to save memory
                max_length=20,
                min_length=1,
            )
        
        if torch.cuda.is_available():
            peak_memory = torch.cuda.max_memory_allocated() / 1024**3  # GB
            logger.info(f"Peak GPU memory during generation: {peak_memory:.2f} GB")
        
        logger.info(f"Generation successful! Generated: {generated}")
        
        return generated
        
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise


def main():
    """Run all tests."""
    logger.info("Starting Gated BLIP-2 tests...")
    
    try:
        # Test 1: Model loading
        model = test_model_loading()
        
        # Test 2: Forward pass with small batch
        outputs = test_forward_pass(model, batch_size=2)
        
        # Test 3: Generation with small batch
        generated = test_generation(model, batch_size=1)
        
        logger.info("All tests passed successfully!")
        
    except Exception as e:
        logger.error(f"Tests failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
