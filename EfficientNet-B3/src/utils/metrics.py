from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class EvalReport:
    acc: float
    per_class_acc: Dict[int, float]
    confusion: np.ndarray
    non_art_precision: float | None = None
    non_art_recall: float | None = None


def confusion_matrix(y_true: List[int], y_pred: List[int], num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def accuracy(y_true: List[int], y_pred: List[int]) -> float:
    if not y_true:
        return 0.0
    correct = sum(int(t == p) for t, p in zip(y_true, y_pred))
    return float(correct) / float(len(y_true))


def per_class_accuracy(cm: np.ndarray) -> Dict[int, float]:
    out: Dict[int, float] = {}
    for i in range(cm.shape[0]):
        denom = cm[i, :].sum()
        out[i] = float(cm[i, i]) / float(denom) if denom > 0 else 0.0
    return out


def precision_recall_binary(y_true: List[int], y_pred: List[int], positive_label: int) -> Tuple[float, float]:
    tp = fp = fn = 0
    for t, p in zip(y_true, y_pred):
        if p == positive_label and t == positive_label:
            tp += 1
        elif p == positive_label and t != positive_label:
            fp += 1
        elif p != positive_label and t == positive_label:
            fn += 1
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return float(prec), float(rec)

















