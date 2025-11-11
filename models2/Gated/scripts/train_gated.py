#!/usr/bin/env python3
"""Training entry point for gated BLIP-2 project."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict

import yaml

# Add the project root to Python path for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Now we can import from the models.Gated package
try:
    from models.Gated.configs import load_config
    from models.Gated.trainers import GatedTrainer
    print("✓ Imports successful")
except ImportError as e:
    print(f"✗ Import failed: {e}")
    print(f"Project root: {project_root}")
    print(f"Python path: {sys.path}")
    sys.exit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="artquest",
        help="配置名称（与 configs 下文件同名，不含后缀）或 YAML 文件绝对路径。",
    )
    return parser.parse_args()


def load_config_from_arg(arg: str) -> Dict:
    path = Path(arg)
    if path.suffix in {".yml", ".yaml"} and path.exists():
        with path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return load_config(arg)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    config = load_config_from_arg(args.config)
    trainer = GatedTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()

