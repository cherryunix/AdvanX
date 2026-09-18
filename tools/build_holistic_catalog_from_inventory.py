#!/usr/bin/env python3
"""Build a deduplicated full-source Holistic catalog with fixed orientation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inventory", type=Path)
    parser.add_argument("orientation_audit", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--training-target-fps", type=float, default=24.0)
    args = parser.parse_args()

    inventory = json.loads(args.inventory.read_text())
    audit = json.loads(args.orientation_audit.read_text())
    orientation = {
        row["key"]: row
        for row in audit["items"]
        if row.get("state") == "estimated"
    }

    groups: dict[tuple[int, str], list[dict]] = {}
    for item in inventory["items"]:
        if not item.get("stable") or not item.get("probe_ok"):
            continue
        key = item.get("edge_blake2b16")
        if not key:
            raise RuntimeError(f"Missing content key for {item['path']}")
        groups.setdefault((int(item["bytes"]), key), []).append(item)

    catalog = []
    for (_, key), aliases in sorted(
        groups.items(), key=lambda entry: min(row["relative_path"] for row in entry[1])
    ):
        aliases.sort(key=lambda item: item["relative_path"])
        item = aliases[0]
        decision = orientation.get(key)
        if decision is None:
            raise RuntimeError(f"Missing orientation decision for {item['relative_path']}")
        rotation = int(decision["rotation_correction_clockwise"])
        video = item["video"]
        width, height = int(video["width"]), int(video["height"])
        inference_width, inference_height = (
            (height, width) if rotation in (90, 270) else (width, height)
        )
        duration = float(video.get("duration") or item["format"].get("duration"))
        fps = video.get("avg_frame_rate") or video.get("r_frame_rate")
        signature_payload = "\0".join(
            (
                item["path"],
                str(item["bytes"]),
                str(item["mtime_ns"]),
                str(rotation),
            )
        )
        signature = hashlib.sha256(signature_payload.encode()).hexdigest()
        catalog.append(
            {
                "id": f"NM_{key[:12]}",
                "path": item["path"],
                "relative_path": item["relative_path"],
                "aliases": [alias["relative_path"] for alias in aliases],
                "origin": "selected_new_material_folders",
                "duration": duration,
                "width": width,
                "height": height,
                "inference_width": inference_width,
                "inference_height": inference_height,
                "fps": fps,
                "training_target_fps": args.training_target_fps,
                "rotation_correction_clockwise": rotation,
                "orientation_confidence": decision.get("confidence"),
                "bytes": int(item["bytes"]),
                "mtime_ns": int(item["mtime_ns"]),
                "content_key": key,
                "signature": signature,
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(catalog, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            {
                "sources": len(catalog),
                "aliases": sum(len(item["aliases"]) for item in catalog),
                "target_fps": args.training_target_fps,
                "rotation_counts": {
                    str(rotation): sum(
                        item["rotation_correction_clockwise"] == rotation
                        for item in catalog
                    )
                    for rotation in (0, 90, 180, 270)
                },
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
