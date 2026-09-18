#!/usr/bin/env python3
"""Profile decode, GPU assembly, and fused Holistic inference on a real source."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import PyNvVideoCodec as nvc
import torch

from gaze_iris_audit_trt import FusedHolisticTrt
from nv12_to_rgb_triton import nv12_surfaces_to_rgb_nchw


def rotate_tensor(frame: torch.Tensor, clockwise: int) -> torch.Tensor:
    clockwise %= 360
    if clockwise == 90:
        return torch.rot90(frame, -1, (1, 2))
    if clockwise == 180:
        return torch.rot90(frame, 2, (1, 2))
    if clockwise == 270:
        return torch.rot90(frame, 1, (1, 2))
    if clockwise:
        raise ValueError(f"Unsupported rotation: {clockwise}")
    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument(
        "--face-geometry-metadata",
        type=Path,
        default=Path("models/mediapipe/face_geometry_procrustes.npz"),
    )
    parser.add_argument("--rotation", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--decode-buffer", type=int, default=96)
    parser.add_argument("--max-batches", type=int, default=200)
    parser.add_argument("--native-nv12", action="store_true")
    args = parser.parse_args()

    pipeline = FusedHolisticTrt(args.engine, args.face_geometry_metadata)
    decoder = nvc.ThreadedDecoder(
        str(args.source),
        buffer_size=args.decode_buffer,
        gpu_id=0,
        use_device_memory=True,
        output_color_type=(
            nvc.OutputColorType.NATIVE
            if args.native_nv12
            else nvc.OutputColorType.RGBP
        ),
    )
    timing = {"decode_wait": 0.0, "assembly": 0.0, "inference": 0.0}
    frames = 0
    started = time.perf_counter()
    try:
        for _ in range(args.max_batches):
            stage = time.perf_counter()
            decoded = decoder.get_batch_frames(args.batch_size)
            timing["decode_wait"] += time.perf_counter() - stage
            if not decoded:
                break
            stage = time.perf_counter()
            if args.native_nv12:
                rgb_nchw = nv12_surfaces_to_rgb_nchw(decoded, args.rotation)
                tensors = []
            else:
                tensors = [
                    rotate_tensor(torch.from_dlpack(frame), args.rotation)
                    for frame in decoded
                ]
                rgb_nchw = torch.stack(tensors)
            timing["assembly"] += time.perf_counter() - stage
            stage = time.perf_counter()
            rows = pipeline.infer_rgb_nchw(rgb_nchw)
            timing["inference"] += time.perf_counter() - stage
            frames += len(rows)
    finally:
        decoder.end()
    elapsed = time.perf_counter() - started
    print(
        json.dumps(
            {
                "source": str(args.source),
                "affinity": sorted(os.sched_getaffinity(0)),
                "frames": frames,
                "wall_seconds": elapsed,
                "wall_fps": frames / elapsed,
                "inference_fps": frames / timing["inference"],
                "timing_seconds": timing,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
