#!/usr/bin/env python3
"""Reclassify pose segments from human keep/drop review decisions."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from advanx.classification import balanced_accuracy, best_upper_threshold, confusion


FEATURE = "p95_abs_yaw_deg"


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def stratified_folds(labels: np.ndarray, count: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    buckets: list[list[int]] = [[] for _ in range(count)]
    for value in (False, True):
        indices = np.flatnonzero(labels == value)
        rng.shuffle(indices)
        for position, index in enumerate(indices):
            buckets[position % count].append(int(index))
    return [np.asarray(sorted(bucket), dtype=np.int64) for bucket in buckets]


def cross_validate(values: np.ndarray, labels: np.ndarray, folds: int) -> dict:
    predictions = np.zeros(len(labels), dtype=bool)
    thresholds = []
    all_indices = np.arange(len(labels))
    for test in stratified_folds(labels, folds, seed=42):
        train = np.setdiff1d(all_indices, test, assume_unique=True)
        threshold, _ = best_upper_threshold(values[train], labels[train])
        thresholds.append(threshold)
        predictions[test] = values[test] <= threshold
    counts = confusion(labels, predictions)
    return {
        "folds": folds,
        "balanced_accuracy": balanced_accuracy(counts),
        "accuracy": float(np.mean(predictions == labels)),
        "threshold_min": min(thresholds),
        "threshold_median": float(np.median(thresholds)),
        "threshold_max": max(thresholds),
        **counts,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("segments", type=Path)
    parser.add_argument("decisions", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--review-margin-deg", type=float, default=2.0,
        help="Send unlabelled rows within this distance of the learned threshold to review.",
    )
    args = parser.parse_args()
    if args.review_margin_deg < 0:
        parser.error("--review-margin-deg must be non-negative")

    rows = json.loads(args.segments.read_text())
    payload = json.loads(args.decisions.read_text())
    decisions = payload.get("decisions", payload)
    invalid = {key: value for key, value in decisions.items() if value not in {"keep", "drop"}}
    if invalid:
        parser.error(f"invalid decisions: {invalid}")
    by_id = {row["segment_id"]: row for row in rows}
    missing = sorted(set(decisions) - set(by_id))
    if missing:
        parser.error(f"decision IDs absent from manifest: {missing}")
    if not decisions:
        parser.error("no decisions found")

    labelled = [by_id[ident] for ident in decisions]
    values = np.asarray([row[FEATURE] for row in labelled], dtype=np.float64)
    labels = np.asarray([decisions[row["segment_id"]] == "keep" for row in labelled])
    threshold, fitted = best_upper_threshold(values, labels)
    cv = cross_validate(values, labels, min(10, int(min(labels.sum(), (~labels).sum()))))
    review_low = threshold - args.review_margin_deg
    review_high = threshold + args.review_margin_deg

    classified = []
    for source in rows:
        row = dict(source)
        value = float(row[FEATURE])
        suggested = "keep" if value <= threshold else "drop"
        human = decisions.get(row["segment_id"])
        if human:
            classification = human
            decision_source = "human"
        elif review_low <= value <= review_high:
            classification = "review"
            decision_source = "rule_review_band"
        else:
            classification = suggested
            decision_source = "yaw_rule"
        row.update(
            {
                "classification": classification,
                "suggested_classification": suggested,
                "decision_source": decision_source,
                "classification_feature": FEATURE,
                "classification_value": value,
                "classification_threshold": threshold,
                "distance_from_threshold": value - threshold,
            }
        )
        classified.append(row)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name in ("keep", "review", "drop"):
        subset = [row for row in classified if row["classification"] == name]
        (args.output_dir / f"{name}.json").write_text(
            json.dumps(subset, ensure_ascii=False, indent=2) + "\n"
        )
        write_csv(args.output_dir / f"{name}.csv", subset)
    (args.output_dir / "all_segments.json").write_text(
        json.dumps(classified, ensure_ascii=False, indent=2) + "\n"
    )
    write_csv(args.output_dir / "all_segments.csv", classified)
    (args.output_dir / "decisions.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    )

    source_summary: dict[str, dict] = {}
    for row in classified:
        summary = source_summary.setdefault(
            row["source_id"],
            {
                "source_id": row["source_id"],
                "relative_path": row["relative_path"],
                "keep_segments": 0,
                "keep_seconds": 0.0,
                "review_segments": 0,
                "review_seconds": 0.0,
                "drop_segments": 0,
                "drop_seconds": 0.0,
            },
        )
        name = row["classification"]
        summary[f"{name}_segments"] += 1
        summary[f"{name}_seconds"] += row["duration_s"]
    source_rows = sorted(source_summary.values(), key=lambda row: row["relative_path"])
    (args.output_dir / "source_summary.json").write_text(
        json.dumps(source_rows, ensure_ascii=False, indent=2) + "\n"
    )
    write_csv(args.output_dir / "source_summary.csv", source_rows)

    groups: dict[str, Counter] = defaultdict(Counter)
    for row in classified:
        group = row["relative_path"].split("/", 1)[0]
        groups[group][f"{row['classification']}_segments"] += 1
        groups[group][f"{row['classification']}_seconds"] += row["duration_s"]
    counts = Counter(row["classification"] for row in classified)
    seconds = Counter()
    sources: dict[str, set[str]] = defaultdict(set)
    for row in classified:
        name = row["classification"]
        seconds[name] += row["duration_s"]
        sources[name].add(row["source_id"])
    summary = {
        "input_segments": len(rows),
        "human_labels": dict(Counter(decisions.values())),
        "feature": FEATURE,
        "learned_threshold": threshold,
        "review_margin_deg": args.review_margin_deg,
        "review_band": [review_low, review_high],
        "fit": fitted,
        "cross_validation": cv,
        "classification_counts": dict(counts),
        "classification_seconds": dict(seconds),
        "classification_sources": {name: len(value) for name, value in sources.items()},
        "groups": {name: dict(values) for name, values in sorted(groups.items())},
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
