"""Configuration presets for gated BLIP-2 training."""

from importlib import resources
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Set


def load_config(name: str) -> Dict[str, Any]:
    """Load a YAML config from the local configs package with optional _base_."""

    return _resolve_config(name, seen=set())


def _resolve_config(name: str, seen: Set[str]) -> Dict[str, Any]:
    if name in seen:
        raise ValueError(f"Circular config reference detected: {' -> '.join(seen)} -> {name}")
    seen.add(name)

    resource = resources.files(__package__) / f"{name}.yaml"
    if not resource.is_file():
        raise FileNotFoundError(f"Config {name}.yaml not found in {__package__}")

    with resources.as_file(resource) as path:
        config = _load_yaml_file(path)

    base_name = config.pop("_base_", None)
    if base_name:
        base_config = _resolve_config(base_name, seen)
        config = _merge_dicts(base_config, config)

    seen.remove(name)
    return config


def _load_yaml_file(path: Path) -> Dict[str, Any]:
    import yaml

    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    if not isinstance(data, MutableMapping):
        raise ValueError(f"Config file {path} must define a mapping")
    return dict(data)


def _merge_dicts(base: Mapping[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], MutableMapping)
            and isinstance(value, Mapping)
        ):
            result[key] = _merge_dicts(result[key], value)
        else:
            result[key] = value
    return result


__all__ = ["load_config"]

