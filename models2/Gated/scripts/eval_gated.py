"""Evaluation entry point for gated BLIP-2 project."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict

import yaml
import torch

# Add the project root to Python path for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from models.Gated.configs import load_config
from models.Gated.trainers import GatedTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="artquest",
        help="配置名称（configs 下文件名）或 YAML 文件路径。",
    )
    parser.add_argument(
        "--checkpoint",
        default="/home/linux/artcap-blip2-4/models/Gated/checkpoints/best.pt",
        help="模型权重路径（默认指向 best.pt）。",
    )
    parser.add_argument(
        "--split",
        default="val",
        choices=["val", "test"],
        help="评估数据集划分。",
    )
    parser.add_argument(
        "--dump_predictions",
        type=str,
        default=None,
        help="可选：将预测结果写入 JSON 文件。",
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=5,
        help="预测时的 beam size。",
    )
    return parser.parse_args()


def load_config_from_arg(arg: str) -> Dict:
    path = Path(arg)
    if path.suffix in {".yml", ".yaml"} and path.exists():
        with path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return load_config(arg)


def dump_predictions(trainer: GatedTrainer, loader, output_path: Path, num_beams: int) -> None:
    trainer.model.eval()
    preds = []
    with torch.no_grad():  # type: ignore[name-defined]
        for batch in loader:
            samples = trainer._move_batch_to_device(batch)  # pylint: disable=protected-access
            answers = trainer.model.predict_answers(samples, num_beams=num_beams)
            for qid, ans in zip(batch["question_id"], answers):
                preds.append({"question_id": int(qid), "answer": ans})

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(preds, f, ensure_ascii=False, indent=2)
    logging.info("Saved %d predictions to %s", len(preds), output_path)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    config = load_config_from_arg(args.config)
    trainer = GatedTrainer(config)
    trainer.load_checkpoint(args.checkpoint, strict=False)

    loader = trainer.val_loader if args.split == "val" else trainer.test_loader
    if loader is None:
        raise ValueError(f"Requested split '{args.split}' has no dataloader configured.")

    metrics = trainer.evaluate(loader)
    logging.info("Evaluation metrics on %s: %s", args.split, metrics)

    if args.dump_predictions:
        dump_predictions(
            trainer,
            loader,
            Path(args.dump_predictions),
            num_beams=args.num_beams,
        )


if __name__ == "__main__":
    main()

