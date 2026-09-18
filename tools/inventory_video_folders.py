#!/usr/bin/env python3
"""Build an incremental inventory for explicitly selected video folders."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from fractions import Fraction
from pathlib import Path


VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".avi", ".m4v"}
EDGE_BYTES = 1 << 20


def stat_key(path: Path) -> tuple[int, int]:
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns


def edge_digest(path: Path) -> str:
    size = path.stat().st_size
    digest = hashlib.blake2b(digest_size=16)
    digest.update(size.to_bytes(8, "little"))
    with path.open("rb") as source:
        digest.update(source.read(EDGE_BYTES))
        if size > EDGE_BYTES:
            source.seek(max(0, size - EDGE_BYTES))
            digest.update(source.read(EDGE_BYTES))
    return digest.hexdigest()


def probe(path: Path) -> dict:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_streams",
            "-show_format",
            "-of",
            "json",
            str(path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        return {"probe_ok": False, "probe_error": result.stderr.strip()}
    payload = json.loads(result.stdout)
    video = next(
        (stream for stream in payload.get("streams", []) if stream.get("codec_type") == "video"),
        None,
    )
    audio = [
        stream for stream in payload.get("streams", []) if stream.get("codec_type") == "audio"
    ]
    if video is None:
        return {"probe_ok": False, "probe_error": "No video stream"}
    rate = video.get("avg_frame_rate") or video.get("r_frame_rate") or "0/1"
    try:
        fps = float(Fraction(rate))
    except (ValueError, ZeroDivisionError):
        fps = 0.0
    return {
        "probe_ok": True,
        "video": video,
        "audio": audio,
        "format": payload.get("format", {}),
        "fps": fps,
        "is_1080p100": (
            int(video.get("width", 0)) == 1920
            and int(video.get("height", 0)) == 1080
            and abs(fps - 100.0) < 0.01
        ),
    }


def inspect(path: Path, relative: str, stable: bool, cached: dict | None) -> dict:
    size, mtime_ns = stat_key(path)
    base = {
        "path": str(path),
        "relative_path": relative,
        "bytes": size,
        "mtime_ns": mtime_ns,
        "stable": stable,
        "stable_3s": stable,
    }
    if not stable:
        return {**base, "probe_ok": False, "probe_error": "File is still changing"}
    if (
        cached
        and int(cached.get("bytes", -1)) == size
        and int(cached.get("mtime_ns", -1)) == mtime_ns
        and cached.get("probe_ok")
    ):
        return {
            **base,
            **{
                key: cached[key]
                for key in (
                    "edge_blake2b16",
                    "probe_ok",
                    "video",
                    "audio",
                    "format",
                    "fps",
                    "is_1080p100",
                )
                if key in cached
            },
        }
    return {**base, "edge_blake2b16": edge_digest(path), **probe(path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--folder", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stability-seconds", type=float, default=10.0)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    root = args.root.resolve()
    selected = [root / folder for folder in args.folder]
    missing = [str(path) for path in selected if not path.is_dir()]
    if missing:
        parser.error(f"Missing folders: {missing}")
    paths = sorted({
        path
        for directory in selected
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES
    })
    before = {path: stat_key(path) for path in paths}
    time.sleep(args.stability_seconds)
    current_paths = sorted({
        path
        for directory in selected
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES
    })
    after = {path: stat_key(path) for path in current_paths}
    stable = {path: before.get(path) == after[path] for path in current_paths}

    cached_by_path = {}
    if args.output.is_file():
        try:
            old = json.loads(args.output.read_text())
            cached_by_path = {item["path"]: item for item in old.get("items", [])}
        except (OSError, json.JSONDecodeError, KeyError):
            pass

    items = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                inspect,
                path,
                str(path.relative_to(root)),
                stable[path],
                cached_by_path.get(str(path)),
            ): path
            for path in current_paths
        }
        for index, future in enumerate(as_completed(futures), 1):
            item = future.result()
            items.append(item)
            print(
                f"[{index}/{len(futures)}] {item['relative_path']} "
                f"stable={item['stable']} probe={item['probe_ok']}",
                flush=True,
            )
    items.sort(key=lambda item: item["relative_path"])

    duplicate_groups: dict[tuple[int, str], list[str]] = {}
    for item in items:
        digest = item.get("edge_blake2b16")
        if item.get("probe_ok") and digest:
            duplicate_groups.setdefault((item["bytes"], digest), []).append(
                item["relative_path"]
            )
    duplicates = [group for group in duplicate_groups.values() if len(group) > 1]
    duplicate_excess = sum(len(group) - 1 for group in duplicates)
    stable_items = [item for item in items if item["stable"] and item.get("probe_ok")]
    duplicate_bytes = sum(
        (len(group) - 1)
        * next(item["bytes"] for item in items if item["relative_path"] == group[0])
        for group in duplicates
    )
    total_bytes = sum(item["bytes"] for item in items)
    summary = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "root": str(root),
        "folders": args.folder,
        "files": len(items),
        "stable": sum(item["stable"] for item in items),
        "probe_ok": sum(item.get("probe_ok", False) for item in items),
        "probe_failed": sum(not item.get("probe_ok", False) for item in items),
        "is_1080p100": sum(item.get("is_1080p100", False) for item in items),
        "unique_stable_probable": len(stable_items) - duplicate_excess,
        "duplicate_groups": len(duplicates),
        "duplicate_files": duplicate_excess,
        "total_gib": round(total_bytes / (1 << 30), 3),
        "unique_stable_gib": round(
            (sum(item["bytes"] for item in stable_items) - duplicate_bytes) / (1 << 30),
            3,
        ),
        "duplicate_candidates": duplicates,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"summary": summary, "items": items}, ensure_ascii=False, indent=2)
        + "\n"
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
