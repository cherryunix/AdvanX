"""Temporal mask operations used by pose-driven source segmentation."""

from __future__ import annotations

import math

import numpy as np


def true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return half-open intervals for contiguous true values."""
    values = np.asarray(mask, dtype=bool)
    padded = np.pad(values.astype(np.int8), (1, 1))
    edges = np.diff(padded)
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    return [(int(start), int(end)) for start, end in zip(starts, ends)]


def fill_short_gaps(
    mask: np.ndarray, max_gap: int, blockers: np.ndarray | None = None
) -> np.ndarray:
    """Close bounded false runs without crossing explicitly blocked frames."""
    result = np.asarray(mask, dtype=bool).copy()
    blocked = (
        np.zeros(len(result), dtype=bool)
        if blockers is None
        else np.asarray(blockers, dtype=bool)
    )
    if blocked.shape != result.shape:
        raise ValueError("blockers must match mask shape")
    if max_gap < 0:
        raise ValueError("max_gap must be non-negative")
    for start, end in true_runs(~result):
        if (
            end - start <= max_gap
            and start > 0
            and end < len(result)
            and result[start - 1]
            and result[end]
            and not blocked[start:end].any()
        ):
            result[start:end] = True
    result[blocked] = False
    return result


def split_run(
    start: int,
    end: int,
    target_frames: int,
    max_frames: int,
    min_frames: int,
) -> list[tuple[int, int]]:
    """Split a run into near-equal, non-overlapping intervals."""
    length = end - start
    if length < min_frames:
        return []
    if min(target_frames, max_frames, min_frames) <= 0:
        raise ValueError("frame limits must be positive")
    if not min_frames <= target_frames <= max_frames:
        raise ValueError("expected min_frames <= target_frames <= max_frames")

    count = max(1, round(length / target_frames))
    count = max(count, math.ceil(length / max_frames))
    while count > 1 and length / count < min_frames:
        count -= 1
    boundaries = [start + round(index * length / count) for index in range(count + 1)]
    output = [
        (left, right)
        for left, right in zip(boundaries, boundaries[1:])
        if right - left >= min_frames
    ]
    return output
