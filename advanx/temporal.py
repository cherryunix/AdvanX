"""Exact source-frame selection for a regular output timeline.

This module deliberately has no GPU or video-runtime dependencies so the
sampling contract can be tested on any machine.
"""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np


def target_source_indices(
    source_frames: int, source_fps: float, target_fps: float
) -> np.ndarray:
    """Map a regular target timeline to nearest, unique source frames."""
    if source_frames <= 0 or source_fps <= 0 or target_fps <= 0:
        raise ValueError("Frame counts and rates must be positive")
    if target_fps > source_fps + 1e-6:
        raise ValueError("Target FPS cannot exceed source FPS without duplication")
    output_frames = math.floor((source_frames - 1) * target_fps / source_fps) + 1
    indices = np.floor(
        np.arange(output_frames, dtype=np.float64) * source_fps / target_fps + 0.5
    ).astype(np.int32)
    if indices[-1] >= source_frames:
        raise RuntimeError("Temporal sampling exceeded source frames")
    if len(indices) > 1 and np.any(np.diff(indices) <= 0):
        raise RuntimeError("Temporal sampling duplicated source frames")
    return indices


def periodic_select_pattern(
    source_fps: float, target_fps: float
) -> tuple[int, tuple[int, ...]]:
    """Return the repeating exact-nearest frame pattern for FFmpeg select."""
    ratio = Fraction(source_fps / target_fps).limit_denominator(1001)
    source_period, target_count = ratio.numerator, ratio.denominator
    offsets = tuple(
        math.floor(index * source_period / target_count + 0.5)
        for index in range(target_count)
    )
    if len(set(offsets)) != len(offsets):
        raise ValueError("Software decode sampling pattern contains duplicates")
    return source_period, offsets


def ffmpeg_select_filter(
    source_fps: float, target_fps: float, source_frames: int | None = None
) -> str:
    """Build an FFmpeg select expression identical to target_source_indices."""
    ratio = Fraction(source_fps / target_fps).limit_denominator(1_000_000)
    p, q = ratio.numerator, ratio.denominator
    candidate_k = f"floor((2*n*{q}+{p})/{2*p})"
    mapped_n = f"floor((2*{candidate_k}*{p}+{q})/{2*q})"
    expression = f"eq(n\\,{mapped_n})"
    if source_frames is not None:
        last_target = math.floor((source_frames - 1) * target_fps / source_fps)
        expression += f"*lte({candidate_k}\\,{last_target})"
    return f"select={expression}"
