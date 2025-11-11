#!/usr/bin/env python3
"""Test script to verify style vocabulary construction."""

import sys
from pathlib import Path

# Add the project root to Python path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Now import after adding to path
from models.Gated.configs import load_config
from models.Gated.trainers import GatedTrainer


def main():
    # Test with original config (should have empty style vocab)
    print("Testing with original config (style_key: style)...")
    config_orig = load_config("artquest_local")
    trainer_orig = GatedTrainer(config_orig)
    
    print(f"Original style vocab: {trainer_orig.metadata.get('style_vocab', [])}")
    print(f"Original style token ids shape: {trainer_orig.style_token_ids.shape}")
    print(f"Original style token to label map: {trainer_orig.style_token_to_label_map}")
    
    # Test with fixed config (should have proper style vocab)
    print("\nTesting with fixed config (style_key: question_type)...")
    config_fixed = load_config("artquest_local_fixed")
    trainer_fixed = GatedTrainer(config_fixed)
    
    print(f"Fixed style vocab: {trainer_fixed.metadata.get('style_vocab', [])}")
    print(f"Fixed style token ids shape: {trainer_fixed.style_token_ids.shape}")
    print(f"Fixed style token to label map size: {len(trainer_fixed.style_token_to_label_map)}")
    
    # Show some examples from the fixed config
    if trainer_fixed.style_token_to_label_map:
        print("\nSample style token mappings:")
        for i, (token_id, label_id) in enumerate(list(trainer_fixed.style_token_to_label_map.items())[:5]):
            token = trainer_fixed.model.opt_tokenizer.decode([token_id])
            style_label = trainer_fixed.metadata.get('style_vocab', [])[label_id] if label_id < len(trainer_fixed.metadata.get('style_vocab', [])) else "unknown"
            print(f"  Token {token_id} ('{token}') -> Label {label_id} ('{style_label}')")


if __name__ == "__main__":
    main()
