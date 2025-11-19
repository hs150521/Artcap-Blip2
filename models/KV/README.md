# KV-Modulated BLIP-2

This project implements the classifier-guided KV modulation pipeline described in the paper. EfficientNet-B3 style embeddings are transformed into controller tokens that FiLM the Q-Former cross-attention streams, while BLIP-2 OPT remains frozen for efficient adaptation.

## Layout

- `config/`: YAML configs (dataset paths, hyper-parameters, checkpoints)
- `datasets/`: ArtQuest and VQA dataloaders with BLIP-2 preprocessing
- `modules/`: Trainable components (EfficientNet adapter, Prompt Mapper, KV modulation, BLIP-2 model)
- `trainers/kv_trainer.py`: High-level training loop, logging each epoch to `runs/<timestamp>/training_log.json`
- `scripts/train_kv.py`: CLI entry point: `python models/KV/scripts/train_kv.py --config models/KV/config/artquest.yaml`
- `runs/`: Auto-generated training directories (checkpoints + JSON logs)

## Quickstart

```bash
conda activate lavis2
cd /data/artcap-blip2-4
python models/KV/scripts/train_kv.py --config models/KV/config/artquest.yaml
```

Each run creates `models/KV/runs/YYYY_MM_DD_HH_MM_SS/training_log.json` with:

```json
{
  "epoch": 1,
  "train_loss": 2.34,
  "train_acc": 0.41,
  "val_loss": 2.01,
  "val_acc": 0.48,
  "timestamp": "2025-11-18T09:30:00"
}
```

Update `config/*.yaml` with real dataset paths under `/data/artcap-blip2-4/datasets/` before launching training.


