from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple


def load_classes(classes_json: str) -> Tuple[Dict[int, str], Dict[str, int]]:
    with open(classes_json, "r", encoding="utf-8") as f:
        raw = json.load(f)
    id_to_name: Dict[int, str] = {int(k): str(v) for k, v in raw.items()}
    name_to_id: Dict[str, int] = {v: k for k, v in id_to_name.items()}
    return id_to_name, name_to_id


def normalize_style_name(s: str) -> str:
    s = s.strip().lower()
    s = s.replace("&", "and")
    s = re.sub(r"[\s\-]+", "_", s)
    s = re.sub(r"[^a-z0-9_]+", "", s)
    return s


def load_style_aliases(style_27_json: str) -> Dict[str, List[str]]:
    with open(style_27_json, "r", encoding="utf-8") as f:
        raw = json.load(f)
    out: Dict[str, List[str]] = {}
    for canonical, aliases in raw.items():
        out[normalize_style_name(canonical)] = [normalize_style_name(a) for a in aliases]
    return out


def build_alias_to_canonical(style_aliases: Dict[str, List[str]]) -> Dict[str, str]:
    alias_to_canonical: Dict[str, str] = {}
    for canonical, aliases in style_aliases.items():
        alias_to_canonical[canonical] = canonical
        for a in aliases:
            alias_to_canonical[a] = canonical
    return alias_to_canonical


def resolve_style_to_label_id(
    style_name: str,
    alias_to_canonical: Dict[str, str],
    name_to_id: Dict[str, int],
) -> int:
    n = normalize_style_name(style_name)
    canonical = alias_to_canonical.get(n, n)
    if canonical not in name_to_id:
        raise KeyError(f"Unknown style '{style_name}' -> '{canonical}'. Please update labels/style_27.json.")
    return int(name_to_id[canonical])


def find_project_root_from_file(path: str) -> str:
    p = Path(path).resolve()
    for parent in [p] + list(p.parents):
        if (parent / "EfficientNet-B3").exists():
            return str(parent)
    return str(Path.cwd().resolve())











