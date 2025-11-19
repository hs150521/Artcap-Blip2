# BLIP2-EfficientNet KV Modulation

This directory contains the implementation of BLIP2-OPT-2.7b with KV modulation using EfficientNet-B3 features trained on WikiArt.

## Overview

This implementation integrates EfficientNet-B3 (trained on WikiArt) with BLIP2-OPT-2.7b through Cross-Attention KV modulation. The key idea is:

1. **Freeze** visual encoder (EVA CLIP) and LLM (OPT-2.7b)
2. **Extract features** from EfficientNet-B3 for each image
3. **Generate KV prefixes** from EfficientNet features using a learnable generator
4. **Inject KV prefixes** into Qformer cross-attention layers: `K = concat([K_prefix, K_img])`, `V = concat([V_prefix, V_img])`
5. **Train only** the KV-Prefix generator (and optionally LoRA adapters)

## Directory Structure

```
KV/
├── kv_modulation/          # KV modulation components
│   ├── __init__.py
│   └── kv_prefix_generator.py  # Generates K and V prefixes from EfficientNet features
├── models/                  # Modified model implementations
│   ├── __init__.py
│   ├── qformer_kv_modulated.py  # Modified Qformer with KV modulation support
│   └── blip2_opt_kv_modulated.py  # Modified BLIP2-OPT with KV modulation
├── datasets/                # Dataset loaders
│   ├── __init__.py
│   └── artquest_dataset.py  # ArtQuest dataset loader
├── utils/                   # Utility modules
│   ├── __init__.py
│   ├── efficientnet_loader.py  # EfficientNet model loader
│   └── model_loader.py      # Model loading utilities
├── training/                # Training scripts
│   ├── __init__.py
│   └── train_blip2_kv_modulation.py  # Main training script
└── config/                  # Configuration files
    └── config_kv_modulation.yaml  # Training configuration
```

## Installation

Make sure you have the required dependencies:

```bash
pip install torch torchvision transformers
pip install lavis  # For BLIP2 models
```

## Usage

### Training

1. **Update configuration**: Edit `config/config_kv_modulation.yaml` with your dataset paths and training parameters.

2. **Run training**:
```bash
python KV/training/train_blip2_kv_modulation.py --config KV/config/config_kv_modulation.yaml
```

3. **Resume training**:
```bash
python KV/training/train_blip2_kv_modulation.py \
    --config KV/config/config_kv_modulation.yaml \
    --resume KV/outputs/checkpoint_epoch_N.pt
```

### Configuration

Key configuration parameters in `config/config_kv_modulation.yaml`:

- **Model paths**: Paths to BLIP2, EfficientNet checkpoints
- **KV modulation**: `num_prefix_tokens` (default: 8), `enabled` (true/false)
- **Dataset paths**: Paths to ArtQuest train/val/test CSV files and image root
- **Training parameters**: batch_size, learning_rate, num_epochs, etc.

### Dataset Format

The ArtQuest dataset should have CSV files with the following columns:
- `image`: Image filename
- `question`: Question text
- `answer`: Answer text
- `question_id`: (Optional) Question identifier

Images should be stored in the `image_root` directory (or subdirectories).

## Model Architecture

### KV-Prefix Generator

The `KVPrefixGenerator` takes EfficientNet-B3 features and generates K and V prefixes:

```
EfficientNet Features (1536) 
  → MLP Projection 
  → K_prefix, V_prefix (8 tokens × 768 hidden)
```

### Modified Qformer

The cross-attention layers in Qformer are modified to:

1. Accept KV prefixes from the generator
2. Concatenate with image K and V: `K = concat([K_prefix, K_img])`
3. Extend attention mask accordingly

### Training Strategy

- **Frozen**: Visual encoder, LLM, EfficientNet
- **Trainable**: KV-Prefix generator, Qformer cross-attention (if not frozen)

## Files Description

- **kv_prefix_generator.py**: Implements the learnable module that generates K and V prefixes from EfficientNet features
- **qformer_kv_modulated.py**: Modified Qformer classes supporting KV prefix injection
- **blip2_opt_kv_modulated.py**: Main BLIP2 model with KV modulation integrated
- **efficientnet_loader.py**: Loads WikiArt-trained EfficientNet-B3 model
- **artquest_dataset.py**: Dataset loader for ArtQuest format
- **train_blip2_kv_modulation.py**: Main training loop

## Notes

- All files are self-contained in the `KV/` directory
- No modifications to files outside `KV/` are required
- The implementation uses imports from LAVIS and transformers libraries but doesn't modify them
- EfficientNet preprocessing may need adjustment based on your image preprocessing pipeline

## Troubleshooting

1. **Import errors**: Make sure LAVIS is installed and accessible
2. **Path issues**: Update paths in config file to match your setup
3. **Memory issues**: Reduce batch_size or use gradient accumulation
4. **EfficientNet features**: Ensure EfficientNet checkpoint path is correct
