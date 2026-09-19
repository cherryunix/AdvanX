"""Small, dependency-free helpers for human-calibrated pose classification."""

from __future__ import annotations

import numpy as np


def confusion(labels: np.ndarray, predictions: np.ndarray) -> dict[str, int]:
    truth = np.asarray(labels, dtype=bool)
    guess = np.asarray(predictions, dtype=bool)
    if truth.shape != guess.shape:
        raise ValueError("labels and predictions must have the same shape")
    return {
        "true_negative": int(np.count_nonzero(~truth & ~guess)),
        "false_positive": int(np.count_nonzero(~truth & guess)),
        "false_negative": int(np.count_nonzero(truth & ~guess)),
        "true_positive": int(np.count_nonzero(truth & guess)),
    }


def balanced_accuracy(counts: dict[str, int]) -> float:
    negative = counts["true_negative"] + counts["false_positive"]
    positive = counts["true_positive"] + counts["false_negative"]
    if not negative or not positive:
        return 0.0
    specificity = counts["true_negative"] / negative
    sensitivity = counts["true_positive"] / positive
    return (specificity + sensitivity) * 0.5


def best_upper_threshold(values: np.ndarray, labels: np.ndarray) -> tuple[float, dict]:
    """Fit `value <= threshold` for a positive class using balanced accuracy."""
    x = np.asarray(values, dtype=np.float64)
    y = np.asarray(labels, dtype=bool)
    valid = np.isfinite(x)
    x, y = x[valid], y[valid]
    if len(x) < 2 or not y.any() or y.all():
        raise ValueError("need finite examples from both classes")
    unique = np.unique(x)
    candidates = (unique[:-1] + unique[1:]) * 0.5
    best: tuple[float, float, dict] | None = None
    for threshold in candidates:
        counts = confusion(y, x <= threshold)
        score = balanced_accuracy(counts)
        candidate = (score, float(threshold), counts)
        if best is None or candidate[:2] > best[:2]:
            best = candidate
    assert best is not None
    return best[1], {"balanced_accuracy": best[0], **best[2]}
