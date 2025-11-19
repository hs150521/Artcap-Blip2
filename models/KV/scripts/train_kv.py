#!/usr/bin/env python3
"""Entry point for training the KV-modulated BLIP-2 model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.KV.trainers.kv_trainer import KVTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=str,
        default="models/KV/config/artquest.yaml",
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Optional path to checkpoint (.pt) for resuming training.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    resume_path = args.resume or config.get("training", {}).get("resume_from")
    trainer = KVTrainer(config, resume_path=resume_path)
    trainer.train()


if __name__ == "__main__":
    main()

