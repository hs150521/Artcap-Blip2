from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import datetime
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


_TS_DIR_RE = re.compile(r"^\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2}$")


def make_timestamp_dirname() -> str:
    # yyyy_mm_dd_hh_mm_ss
    return datetime.now().strftime("%Y_%m_%d_%H_%M_%S")


def is_timestamp_dir(path: str | Path) -> bool:
    p = Path(path)
    return bool(_TS_DIR_RE.match(p.name))


def resolve_run_dir(base_output_dir: str | Path, strategy: str = "create") -> str:
    """
    Resolve the concrete run directory under an output base directory.

    - If base_output_dir already ends with a timestamp folder, return it as-is.
    - strategy='create': create a new timestamp folder under base_output_dir.
    - strategy='latest': pick the latest timestamp folder under base_output_dir (if any).
    """
    base = Path(base_output_dir).resolve()
    if is_timestamp_dir(base):
        return str(base)

    if strategy not in {"create", "latest"}:
        raise ValueError("strategy must be one of: create, latest")

    if strategy == "latest":
        if base.exists():
            ts_dirs = [p for p in base.iterdir() if p.is_dir() and is_timestamp_dir(p)]
            if ts_dirs:
                return str(sorted(ts_dirs, key=lambda p: p.name)[-1])
        raise FileNotFoundError(f"No timestamped run directory found under: {base}")

    # create
    return str(base / make_timestamp_dirname())


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

















