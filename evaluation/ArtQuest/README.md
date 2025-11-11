# ArtQuest Evaluation

This directory contains the refactored ArtQuest evaluation script following the modular architecture of VQAv2 evaluation.

## Structure

```
ArtQuest/
├── evaluate_artquest.py      # Main evaluation script
├── config_blip2.yaml         # Configuration for BLIP2 model
├── config_prompt_aug.yaml    # Configuration for Prompt Augmentation model
├── config_kv.yaml            # Configuration for KV Modulation model
├── config_gated.yaml         # Configuration for Gated Modulation model
├── models/                   # Model backends
│   ├── __init__.py           # Model backend factory
│   ├── blip2_artquest.py     # BLIP2 ArtQuest model backend
│   ├── blip2_prompt_aug_artquest.py  # Prompt Augmentation ArtQuest backend
│   ├── blip2_kv_artquest.py  # KV Modulation ArtQuest backend
│   └── blip2_gated_artquest.py # Gated Modulation ArtQuest backend
└── README.md                  # This file
```

## Usage

### Basic Usage

Run the evaluation script with a configuration file:

```bash
cd ArtQuest
python evaluate_artquest.py --cfg-path config_blip2.yaml
```

### Command Line Arguments

- `--cfg-path`: (Required) Path to configuration YAML file
- `--data-percentage`: (Optional) Percentage of data to use (0-1). Overrides config if provided.
- `--batch-size`: (Optional) Batch size for inference. Overrides config if provided.
- `--device`: (Optional) Device to use (cuda/cpu). Overrides config if provided.

### Examples

```bash
# Use BLIP2 model
python evaluate_artquest.py --cfg-path config_blip2.yaml

# Use Prompt Augmentation model
python evaluate_artquest.py --cfg-path config_prompt_aug.yaml

# Use KV Modulation model
python evaluate_artquest.py --cfg-path config_kv.yaml

# Use Gated Modulation model
python evaluate_artquest.py --cfg-path config_gated.yaml

# Use 50% of the data
python evaluate_artquest.py --cfg-path config_blip2.yaml --data-percentage 0.5

# Override device
python evaluate_artquest.py --cfg-path config_blip2.yaml --device cpu

# Override batch size
python evaluate_artquest.py --cfg-path config_blip2.yaml --batch-size 2
```

## Configuration

Edit the configuration YAML files to configure:

- **Model**: Model identifier and model-specific settings (checkpoint path, base model)
- **Data paths**: Paths to ArtQuest test data, SemArt cache, and retrieved candidates
- **Run settings**: Batch size, device, data percentage, output directory, etc.

All paths in the config file are relative to the project root directory.

### Supported Models

- **BLIP2**: Standard BLIP2 model via LAVIS (`config_blip2.yaml`)
- **Prompt Augmentation**: BLIP2 with EfficientNet-based prompt augmentation (`config_prompt_aug.yaml`)
- **KV Modulation**: BLIP2 with KV modulation (`config_kv.yaml`)
- **Gated Modulation**: BLIP2 with Gated modulation (`config_gated.yaml`)

## Output

The evaluation script will:

1. Load the model and data according to the configuration
2. Run inference on the dataset
3. Calculate EM and BLEU metrics for both original and retrieved contexts
4. Save results to the output directory:
   - `evaluate.txt`: Evaluation metrics in JSON format
   - `predictions.json`: All predictions with question IDs

## Requirements

- Python packages: `pandas`, `torch`, `transformers`, `omegaconf`, `tqdm`
- ArtQuest package installed: `pip install -e ../artquest`
- Required data files and model checkpoints as specified in the config

## Model Backend Architecture

The ArtQuest evaluation system follows the same modular architecture as VQAv2:

- **Model Factory**: `models/__init__.py` provides a unified interface to load different model backends
- **Model Backends**: Each model variant has its own backend module that handles model loading and inference
- **Configuration**: YAML files specify model type and parameters
- **Dataset Adapter**: Converts ArtQuest format to model-compatible format

This architecture makes it easy to add new model variants while maintaining consistency with the VQAv2 evaluation system.

