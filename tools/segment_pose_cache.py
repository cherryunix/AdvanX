#!/usr/bin/env python3
"""Build ranked video intervals from an AdvanX per-frame pose cache."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np

from advanx.segments import fill_short_gaps, split_run, true_runs


@dataclass(frozen=True)
class Profile:
    name: str
    min_seconds: float
    target_seconds: float
    max_seconds: float
    fill_gap_seconds: float
    cut_buffer_seconds: float
    face_score: float
    rigidity_error: float
    max_abs_pitch: float
    max_abs_yaw: float
    max_abs_roll: float
    max_abs_gaze_horizontal: float
    min_gaze_vertical: float
    max_gaze_vertical: float
    min_face_scale: float
    center_x_min: float
    center_x_max: float
    center_y_min: float
    center_y_max: float
    eye_open_fraction: float
    minimum_raw_pass_ratio: float


PROFILES = {
    "loose": Profile(
        "loose", 6, 16, 24, 0.75, 0.125, 0.50, 0.025,
        30, 50, 25, 0.20, -0.10, 0.15, 0.06,
        -0.05, 1.05, -0.05, 0.90, 0.25, 0.72,
    ),
    "standard": Profile(
        "standard", 8, 14, 20, 0.50, 0.25, 0.90, 0.015,
        24, 40, 18, 0.14, -0.06, 0.12, 0.10,
        0.05, 0.95, 0.05, 0.80, 0.35, 0.84,
    ),
    "strict": Profile(
        "strict", 8, 12, 16, 0.25, 0.25, 0.99, 0.011,
        18, 28, 13, 0.10, -0.03, 0.08, 0.13,
        0.15, 0.85, 0.12, 0.72, 0.45, 0.92,
    ),
}


def load(source_dir: Path, name: str) -> np.ndarray:
    return np.load(source_dir / f"{name}.npy", mmap_mode="r")


def pose_fingerprint(source_dir: Path) -> str:
    """Fingerprint decoded timeline and pose signals to catch re-encoded copies."""
    digest = hashlib.sha256()
    for name in (
        "time_s",
        "face_detected",
        "pose_detected",
        "face_score",
        "head_yaw_deg",
        "delta_vertical",
    ):
        values = load(source_dir, name)
        digest.update(name.encode())
        digest.update(str(values.dtype).encode())
        digest.update(np.asarray(values.shape, dtype=np.int64).tobytes())
        digest.update(np.asarray(values).tobytes())
    return digest.hexdigest()


def deduplicate_catalog(catalog: list[dict], cache_root: Path) -> tuple[list[dict], list[dict]]:
    """Keep one source per exact pose trajectory and record excluded copies."""
    canonical: dict[str, dict] = {}
    unique: list[dict] = []
    duplicates: list[dict] = []
    for item in catalog:
        fingerprint = pose_fingerprint(cache_root / "sources" / item["id"])
        previous = canonical.get(fingerprint)
        if previous is None:
            row = dict(item)
            row["pose_fingerprint"] = fingerprint
            row["duplicate_aliases"] = []
            canonical[fingerprint] = row
            unique.append(row)
            continue
        alias = {
            "source_id": item["id"],
            "relative_path": item.get("relative_path", item["path"]),
            "source_path": item["path"],
        }
        previous["duplicate_aliases"].append(alias)
        duplicates.append(
            {
                **alias,
                "canonical_source_id": previous["id"],
                "canonical_relative_path": previous.get(
                    "relative_path", previous["path"]
                ),
                "pose_fingerprint": fingerprint,
            }
        )
    return unique, duplicates


def face_geometry(source_dir: Path, detected: np.ndarray) -> tuple[np.ndarray, ...]:
    """Compute robust face center and scale in bounded chunks."""
    landmarks = load(source_dir, "face_landmarks")
    count = len(detected)
    center_x = np.full(count, np.nan, dtype=np.float32)
    center_y = np.full(count, np.nan, dtype=np.float32)
    scale = np.full(count, np.nan, dtype=np.float32)
    for begin in range(0, count, 1024):
        end = min(count, begin + 1024)
        valid = np.flatnonzero(np.asarray(detected[begin:end], dtype=bool))
        if not len(valid):
            continue
        points = np.asarray(landmarks[begin:end][valid, :, :2], dtype=np.float32)
        low = np.nanpercentile(points, 2, axis=1)
        high = np.nanpercentile(points, 98, axis=1)
        center = (low + high) * 0.5
        size = np.maximum(high - low, 0)
        positions = begin + valid
        center_x[positions] = center[:, 0]
        center_y[positions] = center[:, 1]
        scale[positions] = np.sqrt(size[:, 0] * size[:, 1])
    return center_x, center_y, scale


def finite(values: np.ndarray) -> np.ndarray:
    return np.isfinite(values)


def padded_breaks(breaks: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return breaks
    kernel = np.ones(2 * radius + 1, dtype=np.int16)
    return np.convolve(breaks.astype(np.int16), kernel, mode="same") > 0


def robust_median(values: np.ndarray) -> float | None:
    values = values[np.isfinite(values)]
    return float(np.median(values)) if len(values) else None


def quantile(values: np.ndarray, q: float) -> float | None:
    values = values[np.isfinite(values)]
    return float(np.quantile(values, q)) if len(values) else None


def ratio(values: np.ndarray) -> float:
    return float(np.mean(values)) if len(values) else 0.0


def analyze_source(
    item: dict,
    source_dir: Path,
    profile: Profile,
    fps: float,
    geometry: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> tuple[list[dict], dict]:
    face_detected = np.asarray(load(source_dir, "face_detected"), dtype=bool)
    pose_detected = np.asarray(load(source_dir, "pose_detected"), dtype=bool)
    face_score = np.asarray(load(source_dir, "face_score"), dtype=np.float32)
    pose_score = np.asarray(load(source_dir, "pose_score"), dtype=np.float32)
    rigidity = np.asarray(load(source_dir, "head_pose_rigidity_error"), dtype=np.float32)
    pitch = np.asarray(load(source_dir, "delta_head_pitch_deg_smooth"), dtype=np.float32)
    yaw = np.asarray(load(source_dir, "delta_head_yaw_deg_smooth"), dtype=np.float32)
    roll = np.asarray(load(source_dir, "delta_head_roll_deg_smooth"), dtype=np.float32)
    gaze_h = np.asarray(load(source_dir, "delta_horizontal_smooth"), dtype=np.float32)
    gaze_v = np.asarray(load(source_dir, "delta_vertical_smooth"), dtype=np.float32)
    left_eye = np.asarray(load(source_dir, "left_eye"), dtype=np.float32)
    right_eye = np.asarray(load(source_dir, "right_eye"), dtype=np.float32)
    source_time = np.asarray(load(source_dir, "source_time_s"), dtype=np.float32)
    source_frame = np.asarray(load(source_dir, "frame"), dtype=np.int32)
    center_x, center_y, face_scale = (
        geometry if geometry is not None else face_geometry(source_dir, face_detected)
    )

    openness = np.minimum(left_eye[:, 2], right_eye[:, 2])
    finite_open = openness[np.isfinite(openness)]
    open_reference = float(np.quantile(finite_open, 0.80)) if len(finite_open) else np.nan
    eye_open = finite(openness)
    if np.isfinite(open_reference) and open_reference > 0:
        eye_open &= openness >= open_reference * profile.eye_open_fraction

    criteria = {
        "face": face_detected,
        "face_score": finite(face_score) & (face_score >= profile.face_score),
        "rigidity": finite(rigidity) & (rigidity <= profile.rigidity_error),
        "pitch": finite(pitch) & (np.abs(pitch) <= profile.max_abs_pitch),
        "yaw": finite(yaw) & (np.abs(yaw) <= profile.max_abs_yaw),
        "roll": finite(roll) & (np.abs(roll) <= profile.max_abs_roll),
        "gaze_h": finite(gaze_h) & (np.abs(gaze_h) <= profile.max_abs_gaze_horizontal),
        "gaze_v": finite(gaze_v)
        & (gaze_v >= profile.min_gaze_vertical)
        & (gaze_v <= profile.max_gaze_vertical),
        "eye_open": eye_open,
        "face_scale": finite(face_scale) & (face_scale >= profile.min_face_scale),
        "framing": finite(center_x)
        & finite(center_y)
        & (center_x >= profile.center_x_min)
        & (center_x <= profile.center_x_max)
        & (center_y >= profile.center_y_min)
        & (center_y <= profile.center_y_max),
    }
    raw_pass = np.logical_and.reduce(tuple(criteria.values()))

    previous_valid = face_detected[1:] & face_detected[:-1]
    center_jump = np.zeros(len(raw_pass), dtype=bool)
    scale_jump = np.zeros(len(raw_pass), dtype=bool)
    rotation_jump = np.zeros(len(raw_pass), dtype=bool)
    center_step = np.hypot(np.diff(center_x), np.diff(center_y))
    scale_step = np.abs(np.diff(np.log(np.maximum(face_scale, 1e-6))))
    center_jump[1:] = previous_valid & finite(center_step) & (center_step > 0.08)
    scale_jump[1:] = previous_valid & finite(scale_step) & (scale_step > 0.25)
    rotation_jump[1:] = previous_valid & (
        (np.abs(np.diff(pitch)) > 15)
        | (np.abs(np.diff(yaw)) > 20)
        | (np.abs(np.diff(roll)) > 20)
    )
    breaks = center_jump | scale_jump | rotation_jump
    blockers = padded_breaks(breaks, round(profile.cut_buffer_seconds * fps))
    eligible = fill_short_gaps(
        raw_pass & ~blockers,
        round(profile.fill_gap_seconds * fps),
        blockers,
    )

    min_frames = round(profile.min_seconds * fps)
    target_frames = round(profile.target_seconds * fps)
    max_frames = round(profile.max_seconds * fps)
    intervals = []
    for run_start, run_end in true_runs(eligible):
        intervals.extend(
            split_run(run_start, run_end, target_frames, max_frames, min_frames)
        )

    results = []
    for start, end in intervals:
        selection = slice(start, end)
        raw_ratio = ratio(raw_pass[selection])
        if raw_ratio < profile.minimum_raw_pass_ratio:
            continue
        median_scale = robust_median(face_scale[selection])
        shot = (
            "unknown" if median_scale is None
            else "close" if median_scale >= 0.20
            else "medium" if median_scale >= 0.15
            else "wide"
        )
        head_score = np.nanmean(
            np.stack(
                (
                    np.clip(1 - np.abs(pitch[selection]) / profile.max_abs_pitch, 0, 1),
                    np.clip(1 - np.abs(yaw[selection]) / profile.max_abs_yaw, 0, 1),
                    np.clip(1 - np.abs(roll[selection]) / profile.max_abs_roll, 0, 1),
                )
            )
        )
        gaze_score = np.nanmean(
            np.stack(
                (
                    np.clip(1 - np.abs(gaze_h[selection]) / profile.max_abs_gaze_horizontal, 0, 1),
                    np.clip(
                        1 - np.abs(gaze_v[selection])
                        / max(abs(profile.min_gaze_vertical), abs(profile.max_gaze_vertical)),
                        0,
                        1,
                    ),
                )
            )
        )
        score = 100 * (
            0.30 * raw_ratio
            + 0.20 * ratio(face_detected[selection])
            + 0.10 * ratio(pose_detected[selection])
            + 0.20 * float(head_score)
            + 0.15 * float(gaze_score)
            + 0.05 * ratio(~blockers[selection])
        )
        results.append(
            {
                "profile": profile.name,
                "source_id": item["id"],
                "source_path": item["path"],
                "relative_path": item.get("relative_path", item["path"]),
                "rotation_clockwise": int(item.get("rotation_correction_clockwise", 0)),
                "duplicate_aliases": item.get("duplicate_aliases", []),
                "start_index": start,
                "end_index_exclusive": end,
                "target_start_s": start / fps,
                "target_end_s": end / fps,
                "duration_s": (end - start) / fps,
                "source_start_s": float(source_time[start]),
                "source_end_s": float(
                    source_time[end - 1]
                    + 1 / float(Fraction(str(item.get("fps", fps))))
                ),
                "source_start_frame": int(source_frame[start]),
                "source_end_frame_inclusive": int(source_frame[end - 1]),
                "score": float(score),
                "raw_pass_ratio": raw_ratio,
                "face_coverage": ratio(face_detected[selection]),
                "pose_coverage": ratio(pose_detected[selection]),
                "eye_open_coverage": ratio(eye_open[selection]),
                "median_face_scale": median_scale,
                "shot": shot,
                "median_pitch_deg": robust_median(pitch[selection]),
                "median_yaw_deg": robust_median(yaw[selection]),
                "median_roll_deg": robust_median(roll[selection]),
                "median_gaze_horizontal": robust_median(gaze_h[selection]),
                "median_gaze_vertical": robust_median(gaze_v[selection]),
                "p95_abs_pitch_deg": quantile(np.abs(pitch[selection]), 0.95),
                "p95_abs_yaw_deg": quantile(np.abs(yaw[selection]), 0.95),
                "p95_abs_roll_deg": quantile(np.abs(roll[selection]), 0.95),
                "p05_gaze_vertical": quantile(gaze_v[selection], 0.05),
                "p95_gaze_vertical": quantile(gaze_v[selection], 0.95),
            }
        )

    rejected = {
        name: int(np.count_nonzero(~values)) for name, values in criteria.items()
    }
    source_summary = {
        "source_id": item["id"],
        "relative_path": item.get("relative_path", item["path"]),
        "frames": len(raw_pass),
        "seconds": len(raw_pass) / fps,
        "raw_pass_frames": int(np.count_nonzero(raw_pass)),
        "eligible_frames": int(np.count_nonzero(eligible)),
        "selected_frames": sum(row["end_index_exclusive"] - row["start_index"] for row in results),
        "segments": len(results),
        "hard_breaks": int(np.count_nonzero(breaks)),
        "eye_open_reference": open_reference if np.isfinite(open_reference) else None,
        "rejected_frame_counts": rejected,
    }
    return results, source_summary


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze_source_profiles(
    item: dict, cache_root: Path, profile_names: list[str]
) -> tuple[str, dict[str, tuple[list[dict], dict]]]:
    """Load heavyweight face geometry once, then evaluate every profile."""
    source_dir = cache_root / "sources" / item["id"]
    summary = json.loads((source_dir / "summary.json").read_text())
    fps = float(summary["target_fps"])
    detected = np.asarray(load(source_dir, "face_detected"), dtype=bool)
    geometry = face_geometry(source_dir, detected)
    output = {
        name: analyze_source(item, source_dir, PROFILES[name], fps, geometry)
        for name in profile_names
    }
    return item["id"], output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("catalog", type=Path)
    parser.add_argument("cache_root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--profile", choices=tuple(PROFILES), action="append",
        help="Repeat to generate several thresholds; defaults to all profiles.",
    )
    parser.add_argument(
        "--workers", type=int, default=min(8, max(1, (os.cpu_count() or 2) // 2))
    )
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be >= 1")
    catalog = json.loads(args.catalog.read_text())
    catalog_sources = len(catalog)
    catalog, duplicates = deduplicate_catalog(catalog, args.cache_root)
    profiles = args.profile or list(PROFILES)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "duplicates.json").write_text(
        json.dumps(duplicates, ensure_ascii=False, indent=2) + "\n"
    )
    print(
        f"pose-trajectory deduplication: {catalog_sources} paths -> "
        f"{len(catalog)} unique sources ({len(duplicates)} excluded copies)",
        flush=True,
    )

    collected = {
        name: {"segments": [], "sources": []}
        for name in profiles
    }
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(analyze_source_profiles, item, args.cache_root, profiles): item
            for item in catalog
        }
        for index, future in enumerate(as_completed(futures), 1):
            ident, output = future.result()
            counts = []
            for profile_name, (source_segments, source_summary) in output.items():
                collected[profile_name]["segments"].extend(source_segments)
                collected[profile_name]["sources"].append(source_summary)
                counts.append(f"{profile_name}={len(source_segments)}")
            print(
                f"[{index}/{len(catalog)}] {ident} " + " ".join(counts),
                flush=True,
            )

    all_summary = {}
    for profile_name in profiles:
        profile = PROFILES[profile_name]
        segments = collected[profile_name]["segments"]
        sources = collected[profile_name]["sources"]
        sources.sort(key=lambda row: row["source_id"])
        segments.sort(key=lambda row: (-row["score"], row["source_id"], row["target_start_s"]))
        for rank, row in enumerate(segments, 1):
            row["rank"] = rank
            row["segment_id"] = (
                f"{row['source_id']}_{round(row['target_start_s'] * 1000):010d}_"
                f"{round(row['target_end_s'] * 1000):010d}"
            )
        total_frames = sum(row["frames"] for row in sources)
        selected_frames = sum(row["selected_frames"] for row in sources)
        summary = {
            "profile": asdict(profile),
            "catalog_sources": catalog_sources,
            "sources": len(catalog),
            "duplicate_sources_excluded": len(duplicates),
            "sources_with_segments": sum(row["segments"] > 0 for row in sources),
            "segments": len(segments),
            "input_frames": total_frames,
            "selected_frames": selected_frames,
            "selected_seconds": selected_frames / 24,
            "selected_ratio": selected_frames / total_frames if total_frames else 0,
            "duration_seconds": {
                "min": min((row["duration_s"] for row in segments), default=0),
                "median": float(np.median([row["duration_s"] for row in segments])) if segments else 0,
                "max": max((row["duration_s"] for row in segments), default=0),
            },
            "shot_counts": {
                shot: sum(row["shot"] == shot for row in segments)
                for shot in ("close", "medium", "wide", "unknown")
            },
        }
        directory = args.output_dir / profile_name
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "segments.json").write_text(
            json.dumps(segments, ensure_ascii=False, indent=2) + "\n"
        )
        write_csv(directory / "segments.csv", segments)
        (directory / "source_summary.json").write_text(
            json.dumps(sources, ensure_ascii=False, indent=2) + "\n"
        )
        (directory / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
        )
        all_summary[profile_name] = summary
        print(json.dumps(summary, ensure_ascii=False), flush=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(all_summary, ensure_ascii=False, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
