from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch


@dataclass
class CheckpointPayload:
    model: Dict[str, Any]
    optimizer: Optional[Dict[str, Any]] = None
    scheduler: Optional[Dict[str, Any]] = None
    scaler: Optional[Dict[str, Any]] = None
    epoch: int = 0
    step: int = 0
    best_metric: float = -1.0
    config: Optional[Dict[str, Any]] = None


def save_checkpoint(path: str, payload: CheckpointPayload) -> None:
    to_save = {
        "model": payload.model,
        "optimizer": payload.optimizer,
        "scheduler": payload.scheduler,
        "scaler": payload.scaler,
        "epoch": payload.epoch,
        "step": payload.step,
        "best_metric": payload.best_metric,
        "config": payload.config,
    }
    torch.save(to_save, path)


def load_checkpoint(path: str, map_location: str = "cpu") -> Dict[str, Any]:
    return torch.load(path, map_location=map_location)


def save_json(path: str, data: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
















