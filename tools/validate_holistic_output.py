#!/usr/bin/env python3
"""Validate an AdvanX per-frame cache against its source catalog."""

from __future__ import annotations

import argparse
import json
import math
from fractions import Fraction
from pathlib import Path

import numpy as np

from advanx.temporal import target_source_indices


BASE_ARRAYS = {
    "frame",
    "time_s",
    "source_time_s",
    "face_detected",
    "pose_detected",
    "face_candidates",
    "left_hand_detected",
    "right_hand_detected",
    "face_landmarks",
    "pose_landmarks",
    "pose_world_landmarks",
    "left_hand_landmarks",
    "left_hand_world_landmarks",
    "right_hand_landmarks",
    "right_hand_world_landmarks",
    "head_rotation_matrix",
    "right_eye",
    "left_eye",
}
SCALARS = {
    "face_score",
    "pose_score",
    "head_pitch_deg",
    "head_yaw_deg",
    "head_roll_deg",
    "head_pose_rigidity_error",
    "eye_horizontal",
    "eye_vertical",
    "delta_horizontal",
    "delta_vertical",
    "delta_head_pitch_deg",
    "delta_head_yaw_deg",
    "delta_head_roll_deg",
    "left_hand_score",
    "right_hand_score",
}
TRACE_FIELDS = {
    "delta_horizontal",
    "delta_vertical",
    "delta_head_pitch_deg",
    "delta_head_yaw_deg",
    "delta_head_roll_deg",
}
REQUIRED_ARRAYS = BASE_ARRAYS | SCALARS | {
    f"{field}_{suffix}"
    for field in TRACE_FIELDS
    for suffix in ("smooth", "p10", "p90")
}


def validate_source(item: dict, source_dir: Path, requested_fps: float | None) -> dict:
    summary_path = source_dir / "summary.json"
    if not summary_path.is_file():
        raise ValueError("missing summary.json")
    summary = json.loads(summary_path.read_text())
    source_fps = float(Fraction(str(item["fps"])))
    target_fps = requested_fps or source_fps
    source_frames = int(summary["source_frames"])
    expected = target_source_indices(source_frames, source_fps, target_fps)

    if summary.get("signature") != item.get("signature"):
        raise ValueError("catalog signature does not match")
    if not math.isclose(float(summary["target_fps"]), target_fps, abs_tol=1e-6):
        raise ValueError("target FPS does not match")
    if int(summary.get("decoded_source_frames", -1)) != source_frames:
        raise ValueError("decoder did not consume the complete source")
    if not summary.get("all_source_frames_processed"):
        raise ValueError("source is not marked complete")
    if int(summary.get("processed_frames", -1)) != len(expected):
        raise ValueError("processed frame count does not match the target timeline")

    present = {path.stem for path in source_dir.glob("*.npy")}
    missing = sorted(REQUIRED_ARRAYS - present)
    if missing:
        raise ValueError(f"missing arrays: {', '.join(missing)}")

    for name in sorted(REQUIRED_ARRAYS):
        array = np.load(source_dir / f"{name}.npy", mmap_mode="r")
        if array.shape[0] != len(expected):
            raise ValueError(
                f"{name}.npy has {array.shape[0]} rows; expected {len(expected)}"
            )

    frame = np.load(source_dir / "frame.npy", mmap_mode="r")
    if not np.array_equal(frame, expected):
        raise ValueError("frame.npy does not match exact nearest-frame sampling")
    time_s = np.load(source_dir / "time_s.npy", mmap_mode="r")
    expected_time = np.arange(len(expected), dtype=np.float64) / target_fps
    if not np.allclose(time_s, expected_time, rtol=0, atol=2e-5):
        raise ValueError("time_s.npy does not match the target timeline")

    return {
        "id": item["id"],
        "frames": len(expected),
        "source_frames": source_frames,
        "target_fps": target_fps,
        "arrays": len(REQUIRED_ARRAYS),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("catalog", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--target-fps",
        type=float,
        help="Expected output timeline. Defaults to each source's own frame rate.",
    )
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    catalog = json.loads(args.catalog.read_text())
    results, failures = [], []
    for index, item in enumerate(catalog, 1):
        try:
            result = validate_source(
                item, args.output_dir / "sources" / item["id"], args.target_fps
            )
            results.append(result)
            print(f"[{index}/{len(catalog)}] {item['id']} ok", flush=True)
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
            failures.append({"id": item.get("id"), "error": str(error)})
            print(f"[{index}/{len(catalog)}] {item.get('id')} FAIL: {error}", flush=True)

    report = {
        "ok": not failures and len(results) == len(catalog),
        "catalog_sources": len(catalog),
        "validated_sources": len(results),
        "validated_frames": sum(row["frames"] for row in results),
        "arrays_per_source": len(REQUIRED_ARRAYS),
        "failures": failures,
    }
    payload = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(payload)
    print(payload, end="")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
