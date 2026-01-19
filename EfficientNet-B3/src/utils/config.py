from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def as_abs(path: str, base_dir: str | None = None) -> str:
    p = Path(path)
    if p.is_absolute():
        return str(p)
    if base_dir is None:
        return str(p.resolve())
    return str((Path(base_dir) / p).resolve())


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


@dataclass(frozen=True)
class CfgPaths:
    datasets_root: str
    manifests_dir: str
    output_dir: str


def parse_paths(cfg: Dict[str, Any]) -> CfgPaths:
    paths = cfg.get("paths", {})
    datasets_root = paths.get("datasets_root", "")
    manifests_dir = paths.get("manifests_dir", "")
    output_dir = paths.get("output_dir", "")
    if not datasets_root or not manifests_dir or not output_dir:
        raise ValueError("config.paths must include datasets_root, manifests_dir, output_dir")
    return CfgPaths(
        datasets_root=str(datasets_root),
        manifests_dir=str(manifests_dir),
        output_dir=str(output_dir),
    )

















