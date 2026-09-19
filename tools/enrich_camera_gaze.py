#!/usr/bin/env python3
"""Add a camera-relative head/iris gaze proxy to segment manifests."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from advanx.classification import best_upper_threshold


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class TraceCache:
    def __init__(self, root: Path):
        self.root = root
        self.loaded: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    def get(self, source_id: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if source_id not in self.loaded:
            directory = self.root / "sources" / source_id
            self.loaded[source_id] = tuple(
                np.load(directory / name, mmap_mode="r")
                for name in (
                    "delta_horizontal_smooth.npy",
                    "delta_vertical_smooth.npy",
                    "delta_head_yaw_deg_smooth.npy",
                )
            )
        return self.loaded[source_id]

    def segment(self, row: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        begin, end = row["start_index"], row["end_index_exclusive"]
        return tuple(np.asarray(values[begin:end], dtype=np.float64) for values in self.get(row["source_id"]))


def metric(row: dict, traces: TraceCache, slope: float) -> dict:
    horizontal, vertical, yaw = traces.segment(row)
    valid = np.isfinite(horizontal) & np.isfinite(yaw)
    vertical = vertical[np.isfinite(vertical)]
    if not np.count_nonzero(valid) or not len(vertical):
        return {
            "p95_abs_gaze_horizontal": None,
            "p95_abs_gaze_vertical": None,
            "camera_gaze_horizontal_residual_p95": None,
        }
    return {
        "p95_abs_gaze_horizontal": float(np.quantile(np.abs(horizontal[valid]), 0.95)),
        "p95_abs_gaze_vertical": float(np.quantile(np.abs(vertical), 0.95)),
        "camera_gaze_horizontal_residual_p95": float(
            np.quantile(np.abs(horizontal[valid] - slope * yaw[valid]), 0.95)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("segments", type=Path)
    parser.add_argument("cache_root", type=Path)
    parser.add_argument("--calibration-segments", type=Path, required=True)
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    target = json.loads(args.segments.read_text())
    calibration = json.loads(args.calibration_segments.read_text())
    by_id = {row["segment_id"]: row for row in calibration}
    payload = json.loads(args.decisions.read_text())
    decisions = payload.get("decisions", payload)
    labelled = [by_id[ident] for ident in decisions]
    labels = np.asarray([decisions[row["segment_id"]] == "keep" for row in labelled])
    traces = TraceCache(args.cache_root)
    kept_sequences = [traces.segment(row) for row, keep in zip(labelled, labels) if keep]

    slopes = np.linspace(-0.008, 0.002, 1001)
    loss = []
    for slope in slopes:
        per_segment = []
        for horizontal, _, yaw in kept_sequences:
            valid = np.isfinite(horizontal) & np.isfinite(yaw)
            per_segment.append(
                np.quantile(np.abs(horizontal[valid] - slope * yaw[valid]), 0.95)
            )
        loss.append(np.median(per_segment))
    slope = float(slopes[int(np.argmin(loss))])

    labelled_metrics = [metric(row, traces, slope) for row in labelled]
    residuals = np.asarray(
        [row["camera_gaze_horizontal_residual_p95"] for row in labelled_metrics]
    )
    horizontal_threshold, horizontal_fit = best_upper_threshold(residuals, labels)
    kept_vertical = np.asarray(
        [row["p95_abs_gaze_vertical"] for row, keep in zip(labelled_metrics, labels) if keep]
    )
    vertical_reference = float(np.quantile(kept_vertical, 0.95))

    output = []
    for source in target:
        row = dict(source)
        values = metric(row, traces, slope)
        row.update(values)
        if values["camera_gaze_horizontal_residual_p95"] is None:
            risk = None
        else:
            risk = max(
                values["camera_gaze_horizontal_residual_p95"] / horizontal_threshold,
                values["p95_abs_gaze_vertical"] / vertical_reference,
            )
        row.update(
            {
                "camera_gaze_horizontal_compensation_per_yaw_degree": slope,
                "camera_gaze_horizontal_threshold": horizontal_threshold,
                "camera_gaze_vertical_reference": vertical_reference,
                "camera_gaze_risk": risk,
            }
        )
        output.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    write_csv(args.output.with_suffix(".csv"), output)
    model = {
        "calibration": "global camera-looking iris baseline plus human-kept head/eye compensation",
        "horizontal_compensation_per_yaw_degree": slope,
        "horizontal_residual_threshold": horizontal_threshold,
        "horizontal_fit": horizontal_fit,
        "vertical_reference_kept_p95": vertical_reference,
        "segments": len(output),
        "warning": "This is a camera-gaze proxy, not a calibrated 3D optical-axis angle.",
    }
    args.output.with_name(args.output.stem + "_model.json").write_text(
        json.dumps(model, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(model, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
