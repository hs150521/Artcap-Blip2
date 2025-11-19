"""Metrics helpers for KV model training."""

from __future__ import annotations

from typing import Iterable, Sequence


def exact_match(predictions: Sequence[str], references: Sequence[str]) -> float:
    if not predictions:
        return 0.0
    matches = sum(int(pred.strip().lower() == ref.strip().lower()) for pred, ref in zip(predictions, references))
    return matches / len(predictions)


def average(values: Iterable[float]) -> float:
    total = 0.0
    count = 0
    for value in values:
        total += float(value)
        count += 1
    return total / max(count, 1)


