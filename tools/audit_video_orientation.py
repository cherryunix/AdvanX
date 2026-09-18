#!/usr/bin/env python3
"""Estimate per-video rotation from the first decoded frame.

Each unique file is evaluated at 0/90/180/270 degrees.  The selected rotation
maximizes upright face and body evidence from the same Holistic graph used by
the full-source audit.  Ambiguous first frames are retained for manual review.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

from run_holistic_sources_trt import GeometryHolisticTrt


ROTATIONS = (0, 90, 180, 270)


def rotate(frame: np.ndarray, clockwise: int) -> np.ndarray:
    if clockwise == 0:
        return frame
    if clockwise == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if clockwise == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    if clockwise == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    raise ValueError(clockwise)


def first_frame(path: Path) -> np.ndarray:
    capture = cv2.VideoCapture(str(path))
    orientation_auto = getattr(cv2, "CAP_PROP_ORIENTATION_AUTO", None)
    if orientation_auto is not None:
        capture.set(orientation_auto, 0)
    ok, frame = capture.read()
    capture.release()
    if not ok or frame is None:
        raise RuntimeError(f"Could not decode first frame: {path}")
    return frame


def wrapped_degrees(value: float) -> float:
    return (value + 180.0) % 360.0 - 180.0


def orientation_score(row: dict) -> tuple[float, dict]:
    face_score = float(row.get("face_score", 0.0))
    pose_score = float(row.get("pose_score", 0.0))
    face_valid = bool(row.get("detected"))
    pose_valid = bool(row.get("pose_detected"))

    roll_bonus = 0.0
    if face_valid and row.get("head_roll_deg") is not None:
        roll = abs(wrapped_degrees(float(row["head_roll_deg"])))
        roll_bonus = max(0.0, 1.0 - roll / 45.0)

    torso_down = 0.0
    points = row.get("pose_landmarks")
    if pose_valid and points is not None:
        points = np.asarray(points, dtype=np.float32)
        shoulders = points[[11, 12], :2].mean(axis=0)
        hips = points[[23, 24], :2].mean(axis=0)
        torso = hips - shoulders
        norm = float(np.linalg.norm(torso))
        if norm > 1e-6:
            torso_down = float(torso[1] / norm)

    score = (
        4.0 * face_score
        + 1.5 * pose_score
        + roll_bonus
        + max(0.0, torso_down)
    )
    if not face_valid:
        score -= 2.0
    if not pose_valid:
        score -= 0.5
    return score, {
        "score": score,
        "face_detected": face_valid,
        "face_score": face_score if face_valid else None,
        "pose_detected": pose_valid,
        "pose_score": pose_score if pose_valid else None,
        "head_roll_deg": row.get("head_roll_deg"),
        "torso_down": torso_down if pose_valid else None,
    }


def infer_groups(
    pipeline: GeometryHolisticTrt,
    requests: list[tuple[str, int, np.ndarray]],
    batch_size: int,
) -> dict[tuple[str, int], dict]:
    groups: dict[tuple[int, int], list[tuple[str, int, np.ndarray]]] = defaultdict(list)
    for request in requests:
        groups[request[2].shape[:2]].append(request)

    output: dict[tuple[str, int], dict] = {}
    for geometry, group in groups.items():
        for begin in range(0, len(group), batch_size):
            chunk = group[begin : begin + batch_size]
            bgr = np.stack([request[2] for request in chunk])
            rgb = np.ascontiguousarray(bgr[..., ::-1])
            tensor = (
                torch.from_numpy(rgb)
                .to("cuda", non_blocking=True)
                .permute(0, 3, 1, 2)
            )
            rows = pipeline.infer_rgb_nchw(tensor)
            for (ident, rotation, _), row in zip(chunk, rows):
                output[(ident, rotation)] = row
            print(
                f"orientation geometry={geometry[0]}x{geometry[1]} "
                f"{min(begin + len(chunk), len(group))}/{len(group)}",
                flush=True,
            )
            del tensor, rgb, bgr, rows
    return output


def thumbnail(frame: np.ndarray, label: str, max_width: int = 640) -> np.ndarray:
    if frame.shape[1] > max_width:
        scale = max_width / frame.shape[1]
        frame = cv2.resize(
            frame,
            (max_width, round(frame.shape[0] * scale)),
            interpolation=cv2.INTER_AREA,
        )
    result = frame.copy()
    cv2.rectangle(result, (0, 0), (result.shape[1], 42), (0, 0, 0), -1)
    cv2.putText(
        result,
        label,
        (10, 29),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inventory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--thumbnail-dir", type=Path)
    parser.add_argument(
        "--verified-json",
        type=Path,
        help="Optional key-to-clockwise-rotation decisions from visual review.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--fused-engine", type=Path, action="append", required=True
    )
    parser.add_argument(
        "--face-geometry-metadata",
        type=Path,
        default=Path("models/mediapipe/face_geometry_procrustes.npz"),
    )
    args = parser.parse_args()
    verified = {}
    if args.verified_json and args.verified_json.is_file():
        verified = {
            str(key): int(value)
            for key, value in json.loads(args.verified_json.read_text()).items()
        }

    document = json.loads(args.inventory.read_text())
    items = document["items"] if isinstance(document, dict) else document
    eligible = [
        item
        for item in items
        if item.get("probe_ok", True) and item.get("stable_3s", True)
    ]

    unique: dict[str, dict] = {}
    aliases: dict[str, list[dict]] = defaultdict(list)
    for item in eligible:
        key = item.get("edge_blake2b16") or str(Path(item["path"]).resolve())
        aliases[key].append(item)
        unique.setdefault(key, item)

    decoded: dict[str, np.ndarray] = {}
    failures: dict[str, str] = {}
    requests: list[tuple[str, int, np.ndarray]] = []
    for index, (key, item) in enumerate(unique.items(), 1):
        try:
            frame = first_frame(Path(item["path"]))
            decoded[key] = frame
            for rotation in ROTATIONS:
                requests.append((key, rotation, rotate(frame, rotation)))
            print(f"decode [{index}/{len(unique)}] {item.get('relative_path', item['path'])}", flush=True)
        except Exception as error:  # retain per-file failure in the audit
            failures[key] = str(error)

    pipeline = GeometryHolisticTrt(
        fallback=None,
        fused_engines=args.fused_engine,
        face_geometry_metadata=args.face_geometry_metadata,
    )
    inferred = infer_groups(pipeline, requests, args.batch_size)

    if args.thumbnail_dir:
        args.thumbnail_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for key, item in unique.items():
        alias_paths = [alias.get("relative_path", alias["path"]) for alias in aliases[key]]
        if key in failures:
            results.append(
                {
                    "key": key,
                    "canonical_path": item["path"],
                    "aliases": alias_paths,
                    "state": "decode_failed",
                    "error": failures[key],
                }
            )
            continue
        candidates = []
        for rotation in ROTATIONS:
            score, detail = orientation_score(inferred[(key, rotation)])
            candidates.append({"rotation_clockwise": rotation, **detail})
        candidates.sort(key=lambda value: value["score"], reverse=True)
        best, second = candidates[:2]
        gap = float(best["score"] - second["score"])
        evidence = bool(best["face_detected"] or best["pose_detected"])
        auto_rotation = int(best["rotation_clockwise"])
        selected_rotation = int(verified.get(key, auto_rotation))
        confidence = (
            "visual_verified"
            if key in verified
            else "high" if evidence and gap >= 0.75 else "review"
        )
        ident = hashlib.sha1(item["path"].encode()).hexdigest()[:12]
        thumb_path = None
        if args.thumbnail_dir:
            thumb_path = args.thumbnail_dir / f"{ident}.jpg"
            label = (
                f"rot={selected_rotation} gap={gap:.2f} "
                f"{item.get('relative_path', Path(item['path']).name)}"
            )
            cv2.imwrite(
                str(thumb_path),
                thumbnail(rotate(decoded[key], selected_rotation), label),
                [cv2.IMWRITE_JPEG_QUALITY, 92],
            )
        results.append(
            {
                "key": key,
                "canonical_path": item["path"],
                "aliases": alias_paths,
                "state": "estimated",
                "rotation_correction_clockwise": selected_rotation,
                "auto_rotation_correction_clockwise": auto_rotation,
                "visual_override": (
                    selected_rotation != auto_rotation if key in verified else None
                ),
                "confidence": confidence,
                "score_gap": gap,
                "thumbnail": str(thumb_path) if thumb_path else None,
                "candidates": candidates,
            }
        )

    summary = {
        "files": len(eligible),
        "unique_files": len(unique),
        "estimated": sum(row["state"] == "estimated" for row in results),
        "high_confidence": sum(row.get("confidence") == "high" for row in results),
        "visual_verified": sum(
            row.get("confidence") == "visual_verified" for row in results
        ),
        "manual_review": sum(row.get("confidence") == "review" for row in results),
        "decode_failed": sum(row["state"] == "decode_failed" for row in results),
        "rotation_counts": {
            str(rotation): sum(
                row.get("rotation_correction_clockwise") == rotation for row in results
            )
            for rotation in ROTATIONS
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"summary": summary, "items": results}, ensure_ascii=False, indent=2)
        + "\n"
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
