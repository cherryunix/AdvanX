#!/usr/bin/env python3
"""Dispatch whole-source Holistic jobs across a local and a remote GPU node.

The unit of scheduling is one complete source video.  Remote inputs are staged
to the desktop's ext4 filesystem before decode, and only finished numerical
outputs are copied back.  No source is split at GOP or frame boundaries.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shlex
import shutil
import subprocess
import sys
import time
from fractions import Fraction
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
WORKER = REPO / "tools/run_holistic_sources_trt.py"
SUPPORT_FILES = (
    REPO / "tools/run_holistic_sources_trt.py",
    REPO / "tools/gaze_iris_audit_trt.py",
    REPO / "tools/nv12_to_rgb_triton.py",
)
SUPPORT_PACKAGE_FILES = (
    REPO / "advanx/__init__.py",
    REPO / "advanx/temporal.py",
)


def estimated_frames(item: dict, target_fps: float | None = None) -> int:
    source_fps = float(Fraction(str(item["fps"])))
    effective_fps = target_fps or source_fps
    return max(1, round(float(item["duration"]) * effective_fps))


def is_complete(output_dir: Path, item: dict, target_fps: float | None) -> bool:
    summary = output_dir / "sources" / item["id"] / "summary.json"
    if not summary.is_file():
        return False
    try:
        saved = json.loads(summary.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    effective_fps = target_fps or float(Fraction(str(item["fps"])))
    return bool(
        saved.get("signature") == item.get("signature")
        and math.isclose(
            float(saved.get("target_fps", saved.get("fps", -1))),
            effective_fps,
            rel_tol=0,
            abs_tol=1e-6,
        )
        and saved.get("all_source_frames_processed")
        and saved.get("decoded_source_frames", saved.get("source_frames"))
        == saved.get("source_frames")
        and saved.get("processed_frames")
        == saved.get("selected_source_frames", saved.get("source_frames"))
    )


def partition(
    items: list[dict],
    local_fps: float,
    remote_fps: float,
    transfer_mib_s: float,
    target_fps: float | None,
) -> tuple[dict[str, list[dict]], dict[str, float]]:
    transfer_bytes_s = transfer_mib_s * 1024 * 1024

    def costs(item: dict) -> tuple[float, float]:
        frames = estimated_frames(item, target_fps)
        local = frames / local_fps
        remote = float(item.get("bytes", 0)) / transfer_bytes_s + frames / remote_fps
        return local, remote

    assignments: dict[str, list[dict]] = {"local": [], "remote": []}
    finish = {"local": 0.0, "remote": 0.0}
    for item in sorted(items, key=lambda row: max(costs(row)), reverse=True):
        local_cost, remote_cost = costs(item)
        local_finish = finish["local"] + local_cost
        remote_finish = finish["remote"] + remote_cost
        node = "local" if local_finish <= remote_finish else "remote"
        assignments[node].append(item)
        finish[node] += local_cost if node == "local" else remote_cost
    return assignments, finish


def split_lanes(
    items: list[dict], lanes: int, target_fps: float | None
) -> list[list[dict]]:
    """Balance whole videos across concurrent decoder/inference lanes."""
    output = [[] for _ in range(lanes)]
    loads = [0 for _ in range(lanes)]
    for item in sorted(
        items, key=lambda row: estimated_frames(row, target_fps), reverse=True
    ):
        lane = min(range(lanes), key=loads.__getitem__)
        output[lane].append(item)
        loads[lane] += estimated_frames(item, target_fps)
    return output


def split_weighted_lanes(
    items: list[dict], rates: list[float], target_fps: float | None
) -> list[list[dict]]:
    """Balance whole videos across lanes with different measured rates."""
    if not rates or any(rate <= 0 for rate in rates):
        raise ValueError("Lane rates must be positive")
    output = [[] for _ in rates]
    finish = [0.0 for _ in rates]
    for item in sorted(
        items, key=lambda row: estimated_frames(row, target_fps), reverse=True
    ):
        lane = min(range(len(rates)), key=finish.__getitem__)
        output[lane].append(item)
        finish[lane] += estimated_frames(item, target_fps) / rates[lane]
    return output


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def run_checked(command: list[str], log=None) -> None:
    subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)


def copy_remote_results(
    remote_host: str, remote_sources: str, local_sources: Path
) -> None:
    """Stream a result tree through tar; avoids scp's rejected trailing '/.'."""
    local_sources.mkdir(parents=True, exist_ok=True)
    sender = subprocess.Popen(
        ["ssh", remote_host, "tar", "-C", remote_sources, "-cf", "-", "."],
        stdout=subprocess.PIPE,
    )
    if sender.stdout is None:
        raise RuntimeError("Could not open remote tar stream")
    try:
        receiver = subprocess.run(
            ["tar", "-C", str(local_sources), "-xf", "-"],
            stdin=sender.stdout,
            check=False,
        )
    finally:
        sender.stdout.close()
    sender_code = sender.wait()
    if sender_code or receiver.returncode:
        raise RuntimeError(
            f"Remote result transfer failed: sender={sender_code}, "
            f"receiver={receiver.returncode}"
        )


def worker_command(
    python: str,
    worker: str,
    catalog: str,
    output: str,
    baseline: str,
    engines: list[str],
    metadata: str,
    batch_size: int,
    decode_buffer: int,
    target_fps: float | None,
    software_decode: bool = False,
    igpu_decode: bool = False,
    windows_ffmpeg: str | None = None,
    software_decode_threads: int = 16,
    cpu_affinity: str | None = None,
) -> list[str]:
    command = [
        python,
        worker,
        catalog,
        "--output-dir",
        output,
        "--baseline-json",
        baseline,
        "--fused-only",
        "--face-geometry-metadata",
        metadata,
        "--batch-size",
        str(batch_size),
        "--decode-buffer",
        str(decode_buffer),
        "--native-nv12",
    ]
    for engine in engines:
        command.extend(["--fused-engine", engine])
    if target_fps is not None:
        command.extend(["--target-fps", str(target_fps)])
    if software_decode:
        command.extend(
            ["--software-decode", "--software-decode-threads", str(software_decode_threads)]
        )
    if igpu_decode:
        command.append("--igpu-decode")
        if windows_ffmpeg:
            command.extend(["--windows-ffmpeg", windows_ffmpeg])
    if cpu_affinity:
        command.extend(["--cpu-affinity", cpu_affinity])
    return command


def deploy_remote_assets(args: argparse.Namespace) -> None:
    run_checked(["ssh", args.remote_host, "mkdir", "-p", f"{args.remote_root}/tools", f"{args.remote_root}/models", f"{args.remote_root}/advanx"])
    run_checked(
        ["scp", *(str(path) for path in SUPPORT_FILES), f"{args.remote_host}:{args.remote_root}/tools/"]
    )
    run_checked(
        ["scp", *(str(path) for path in SUPPORT_PACKAGE_FILES), f"{args.remote_host}:{args.remote_root}/advanx/"]
    )
    run_checked(
        [
            "scp",
            *(str(engine) for engine in args.engine),
            str(args.face_geometry_metadata),
            f"{args.remote_host}:{args.remote_root}/models/",
        ]
    )


def stage_remote(
    items: list[dict],
    args: argparse.Namespace,
    run_dir: Path,
    remote_job: str,
) -> list[dict]:
    run_checked(["ssh", args.remote_host, "mkdir", "-p", f"{remote_job}/inputs", f"{remote_job}/output"])
    remapped = []
    for index, item in enumerate(items, 1):
        source = Path(item["path"])
        if not source.is_file():
            raise FileNotFoundError(source)
        remote_name = f"{item['id']}{source.suffix.lower()}"
        remote_path = f"{remote_job}/inputs/{remote_name}"
        print(f"stage [{index}/{len(items)}] {item['id']} {source.name}", flush=True)
        run_checked(["scp", "-p", str(source), f"{args.remote_host}:{remote_path}"])
        if int(item.get("bytes", source.stat().st_size)) != source.stat().st_size:
            raise RuntimeError(f"Catalog size changed for {item['id']}: {source}")
        remapped.append({**item, "path": remote_path})
    manifest = run_dir / "remote.staged.json"
    write_json(manifest, remapped)
    baseline_remote = f"{remote_job}/baseline.json"
    run_checked(
        [
            "scp",
            str(manifest),
            str(args.baseline_json),
            f"{args.remote_host}:{remote_job}/",
        ]
    )
    # The second scp operand keeps its basename; normalize the expected name.
    if args.baseline_json.name != "baseline.json":
        run_checked(
            [
                "ssh",
                args.remote_host,
                "mv",
                f"{remote_job}/{args.baseline_json.name}",
                baseline_remote,
            ]
        )
    return remapped


def merge_results(
    catalog: list[dict], output_dir: Path, run_started: float, plan: dict,
    target_fps: float | None,
) -> dict:
    summaries = []
    for item in catalog:
        path = output_dir / "sources" / item["id"] / "summary.json"
        if not path.is_file():
            continue
        summary = json.loads(path.read_text())
        effective_fps = target_fps or float(Fraction(str(item["fps"])))
        if (
            summary.get("signature") == item.get("signature")
            and math.isclose(
                float(summary.get("target_fps", summary.get("fps", -1))),
                effective_fps,
                rel_tol=0,
                abs_tol=1e-6,
            )
        ):
            summaries.append(summary)
    write_json(output_dir / "source_summaries.json", summaries)
    if summaries:
        fields = list(summaries[0])
        with (output_dir / "source_summaries.csv").open(
            "w", encoding="utf-8-sig", newline=""
        ) as target:
            writer = csv.DictWriter(target, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(summaries)
    elapsed = time.perf_counter() - run_started
    total_frames = sum(int(row["processed_frames"]) for row in summaries)
    source_wall_sum = sum(float(row.get("source_wall_seconds", 0)) for row in summaries)
    dispatched_ids = {
        ident
        for identifiers in plan["assignments"].values()
        for ident in identifiers
    }
    dispatched_frames = sum(
        int(row["processed_frames"])
        for row in summaries
        if row.get("id") in dispatched_ids
    )
    aggregate = {
        "backend": "multi-node TensorRT FP16",
        "nodes": ["local", "remote"],
        "target_fps": target_fps,
        "sampling": (
            "every decoded source frame"
            if target_fps is None
            else f"nearest source frame on a {target_fps:g} fps timeline"
        ),
        "sources": len(summaries),
        "frames": total_frames,
        "wall_seconds": source_wall_sum,
        "wall_fps": total_frames / source_wall_sum if source_wall_sum else None,
        "dispatch_wall_seconds": elapsed,
        "dispatched_frames": dispatched_frames,
        "dispatch_fps": dispatched_frames / elapsed if dispatched_frames else None,
        "all_source_frames_processed": len(summaries) == len(catalog)
        and all(row.get("all_source_frames_processed") for row in summaries),
        "plan": plan,
    }
    write_json(output_dir / "aggregate.json", aggregate)
    write_json(output_dir / "distributed_aggregate.json", aggregate)
    return aggregate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("catalog", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--baseline-json", type=Path)
    parser.add_argument("--run-dir", type=Path, default=REPO / "runs/holistic-distributed")
    parser.add_argument(
        "--engine",
        type=Path,
        action="append",
        help="Fused engine for one input geometry; repeat for mixed orientations.",
    )
    parser.add_argument("--face-geometry-metadata", type=Path, default=REPO / "models/mediapipe/face_geometry_procrustes.npz")
    parser.add_argument("--local-python", default=sys.executable)
    parser.add_argument("--remote-host", default="gpu-worker")
    parser.add_argument("--remote-root", default="/tmp/advanx-runtime")
    parser.add_argument("--remote-python", default="/tmp/advanx-runtime/.venv/bin/python")
    parser.add_argument(
        "--remote-extra-ld-library-path",
        default="",
        help="Extra remote library path, for example a TensorRT wheel's tensorrt_libs directory.",
    )
    parser.add_argument("--local-fps", type=float, default=454.6)
    parser.add_argument("--remote-fps", type=float, default=614.5)
    parser.add_argument("--transfer-mib-s", type=float, default=38.9)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--decode-buffer", type=int, default=384)
    parser.add_argument("--target-fps", type=float)
    parser.add_argument("--workers-per-node", type=int, default=2)
    parser.add_argument("--software-workers-per-node", type=int, default=0)
    parser.add_argument(
        "--local-igpu-workers", type=int, default=0,
        help="Local Intel D3D11VA decode lanes. This path is available under WSL.",
    )
    parser.add_argument(
        "--local-igpu-fps", type=float, default=120.0,
        help="Measured output rate for one local iGPU decode lane.",
    )
    parser.add_argument(
        "--windows-ffmpeg", type=Path,
        help="WSL path to a Windows ffmpeg.exe with D3D11VA support.",
    )
    parser.add_argument(
        "--local-software-fps", type=float, default=118.0,
        help="Measured 24-fps output rate of one local CPU decode lane.",
    )
    parser.add_argument(
        "--remote-software-fps", type=float, default=78.0,
        help="Measured 24-fps output rate of one remote CPU decode lane.",
    )
    parser.add_argument("--local-software-threads", type=int, default=16)
    parser.add_argument("--remote-software-threads", type=int, default=12)
    parser.add_argument(
        "--local-software-affinity", action="append", default=[],
        help="CPU list for a local software lane; repeat per lane.",
    )
    parser.add_argument(
        "--remote-software-affinity", action="append", default=[],
        help="CPU list for a remote software lane; repeat per lane.",
    )
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--plan-all", action="store_true", help="Include completed sources in a non-executing capacity plan.")
    parser.add_argument("--keep-remote", action="store_true")
    args = parser.parse_args()
    if args.target_fps is not None and args.target_fps <= 0:
        parser.error("--target-fps must be > 0")
    if args.workers_per_node < 1:
        parser.error("--workers-per-node must be >= 1")
    if args.software_workers_per_node < 0:
        parser.error("--software-workers-per-node must be >= 0")
    if args.local_igpu_workers < 0:
        parser.error("--local-igpu-workers must be >= 0")
    if args.local_software_fps <= 0 or args.remote_software_fps <= 0:
        parser.error("Software lane rates must be > 0")
    if args.local_igpu_fps <= 0:
        parser.error("iGPU lane rate must be > 0")
    if args.plan_all and not args.plan_only:
        parser.error("--plan-all is only valid with --plan-only")
    args.catalog = args.catalog.resolve()
    args.output_dir = args.output_dir.resolve()
    if not args.engine:
        args.engine = [
            REPO / "models/mediapipe/fused/holistic_native_full_b32_1080.engine"
        ]
    args.engine = [engine.resolve() for engine in args.engine]
    args.face_geometry_metadata = args.face_geometry_metadata.resolve()
    args.baseline_json = (
        args.baseline_json.resolve()
        if args.baseline_json
        else args.output_dir / "calibration.json"
    )
    required_paths = [args.catalog]
    if not args.plan_only:
        required_paths.extend((*args.engine, args.face_geometry_metadata, args.baseline_json))
    for path in required_paths:
        if not path.is_file():
            parser.error(f"Missing required file: {path}")

    catalog = json.loads(args.catalog.read_text())
    pending = catalog if args.plan_all else [
        item for item in catalog if not is_complete(args.output_dir, item, args.target_fps)
    ]
    local_total_fps = (
        args.local_fps
        + args.software_workers_per_node * args.local_software_fps
        + args.local_igpu_workers * args.local_igpu_fps
    )
    remote_total_fps = (
        args.remote_fps
        + args.software_workers_per_node * args.remote_software_fps
    )
    assignments, predicted = partition(
        pending, local_total_fps, remote_total_fps,
        args.transfer_mib_s, args.target_fps
    )

    def affinity(values: list[str], index: int) -> str | None:
        return values[index % len(values)] if values else None

    lane_specs = {
        "local": [
            {"backend": "nvdec", "rate": args.local_fps / args.workers_per_node}
            for _ in range(args.workers_per_node)
        ] + [
            {
                "backend": "software",
                "rate": args.local_software_fps,
                "threads": args.local_software_threads,
                "affinity": affinity(args.local_software_affinity, index),
            }
            for index in range(args.software_workers_per_node)
        ] + [
            {
                "backend": "igpu",
                "rate": args.local_igpu_fps,
            }
            for _ in range(args.local_igpu_workers)
        ],
        "remote": [
            {"backend": "nvdec", "rate": args.remote_fps / args.workers_per_node}
            for _ in range(args.workers_per_node)
        ] + [
            {
                "backend": "software",
                "rate": args.remote_software_fps,
                "threads": args.remote_software_threads,
                "affinity": affinity(args.remote_software_affinity, index),
            }
            for index in range(args.software_workers_per_node)
        ],
    }
    lanes = {
        node: split_weighted_lanes(
            rows, [spec["rate"] for spec in lane_specs[node]], args.target_fps
        )
        for node, rows in assignments.items()
    }
    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = args.run_dir / stamp
    run_dir.mkdir(parents=True, exist_ok=True)
    for node, node_lanes in lanes.items():
        for lane_index, rows in enumerate(node_lanes):
            write_json(run_dir / f"{node}.{lane_index}.json", rows)
    plan = {
        "catalog": str(args.catalog),
        "already_complete": len(catalog) - len(pending) if not args.plan_all else 0,
        "pending": len(pending),
        "target_fps": args.target_fps,
        "workers_per_node": args.workers_per_node,
        "assignments": {name: [item["id"] for item in rows] for name, rows in assignments.items()},
        "lanes": {
            node: [[item["id"] for item in rows] for rows in node_lanes]
            for node, node_lanes in lanes.items()
        },
        "lane_specs": lane_specs,
        "predicted_seconds": predicted,
        "measured_rates": {
            "local_nvdec_fps": args.local_fps,
            "remote_nvdec_fps": args.remote_fps,
            "local_software_lane_fps": args.local_software_fps,
            "remote_software_lane_fps": args.remote_software_fps,
            "local_igpu_lane_fps": args.local_igpu_fps,
            "local_total_fps": local_total_fps,
            "remote_total_fps": remote_total_fps,
            "lan_mib_s": args.transfer_mib_s,
        },
    }
    write_json(run_dir / "dispatch_plan.json", plan)
    print(json.dumps(plan, ensure_ascii=False), flush=True)
    if args.plan_only:
        return
    if not pending:
        aggregate = merge_results(
            catalog, args.output_dir, time.perf_counter(), plan, args.target_fps
        )
        print(json.dumps(aggregate, ensure_ascii=False), flush=True)
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "sources").mkdir(exist_ok=True)
    remote_job = f"{args.remote_root}/jobs/{stamp}"
    started = time.perf_counter()
    if assignments["remote"]:
        deploy_remote_assets(args)

    local_processes = []
    local_logs = []
    for lane_index, rows in enumerate(lanes["local"]):
        if not rows:
            continue
        local_command = worker_command(
            args.local_python,
            str(WORKER),
            str(run_dir / f"local.{lane_index}.json"),
            str(args.output_dir),
            str(args.baseline_json),
            [str(engine) for engine in args.engine],
            str(args.face_geometry_metadata),
            args.batch_size,
            args.decode_buffer,
            args.target_fps,
            software_decode=lane_specs["local"][lane_index]["backend"] == "software",
            igpu_decode=lane_specs["local"][lane_index]["backend"] == "igpu",
            windows_ffmpeg=str(args.windows_ffmpeg) if args.windows_ffmpeg else None,
            software_decode_threads=int(
                lane_specs["local"][lane_index].get("threads", 16)
            ),
            cpu_affinity=lane_specs["local"][lane_index].get("affinity"),
        )
        log = (run_dir / f"local.{lane_index}.log").open("w")
        local_logs.append(log)
        local_processes.append(
            (
                lane_index,
                subprocess.Popen(
                    local_command, stdout=log, stderr=subprocess.STDOUT
                ),
            )
        )

    remote_processes = []
    remote_logs = []
    completed_successfully = False
    try:
        if assignments["remote"]:
            remapped = stage_remote(
                assignments["remote"], args, run_dir, remote_job
            )
            remapped_by_id = {item["id"]: item for item in remapped}
            remote_worker = f"{args.remote_root}/tools/run_holistic_sources_trt.py"
            remote_engines = [
                f"{args.remote_root}/models/{engine.name}" for engine in args.engine
            ]
            remote_metadata = f"{args.remote_root}/models/{args.face_geometry_metadata.name}"
            for lane_index, rows in enumerate(lanes["remote"]):
                if not rows:
                    continue
                lane_manifest = run_dir / f"remote.{lane_index}.staged.json"
                write_json(
                    lane_manifest, [remapped_by_id[item["id"]] for item in rows]
                )
                run_checked(
                    [
                        "scp",
                        str(lane_manifest),
                        f"{args.remote_host}:{remote_job}/",
                    ]
                )
                remote_command = worker_command(
                    args.remote_python,
                    remote_worker,
                    f"{remote_job}/{lane_manifest.name}",
                    f"{remote_job}/output",
                    f"{remote_job}/baseline.json",
                    remote_engines,
                    remote_metadata,
                    args.batch_size,
                    args.decode_buffer,
                    args.target_fps,
                    software_decode=lane_specs["remote"][lane_index]["backend"] == "software",
                    software_decode_threads=int(
                        lane_specs["remote"][lane_index].get("threads", 12)
                    ),
                    cpu_affinity=lane_specs["remote"][lane_index].get("affinity"),
                )
                library_paths = ["/usr/lib/wsl/lib"]
                if args.remote_extra_ld_library_path:
                    library_paths.append(args.remote_extra_ld_library_path)
                remote_shell = (
                    f"cd {shlex.quote(args.remote_root)} && "
                    f"export PYTHONPATH={shlex.quote(args.remote_root)}:$PYTHONPATH; "
                    f"export LD_LIBRARY_PATH={shlex.quote(':'.join(library_paths))}:$LD_LIBRARY_PATH; "
                    + shlex.join(remote_command)
                )
                log = (run_dir / f"remote.{lane_index}.log").open("w")
                remote_logs.append(log)
                remote_processes.append(
                    (
                        lane_index,
                        subprocess.Popen(
                            ["ssh", args.remote_host, remote_shell],
                            stdout=log,
                            stderr=subprocess.STDOUT,
                        ),
                    )
                )

        failures = []
        for lane_index, process in local_processes:
            if process.wait() != 0:
                failures.append(
                    f"local worker {lane_index} failed; see "
                    f"{run_dir / f'local.{lane_index}.log'}"
                )
        for lane_index, process in remote_processes:
            if process.wait() != 0:
                failures.append(
                    f"remote worker {lane_index} failed; see "
                    f"{run_dir / f'remote.{lane_index}.log'}"
                )
        if failures:
            raise RuntimeError("; ".join(failures))

        if assignments["remote"]:
            copy_remote_results(
                args.remote_host,
                f"{remote_job}/output/sources",
                args.output_dir / "sources",
            )
        aggregate = merge_results(
            catalog, args.output_dir, started, plan, args.target_fps
        )
        print(json.dumps(aggregate, ensure_ascii=False), flush=True)
        completed_successfully = True
    finally:
        for log in (*local_logs, *remote_logs):
            log.close()
        if (
            completed_successfully
            and not args.keep_remote
            and assignments["remote"]
        ):
            subprocess.run(
                ["ssh", args.remote_host, "rm", "-rf", remote_job], check=False
            )
        elif assignments["remote"] and not completed_successfully:
            print(
                f"Remote job preserved after failure: "
                f"{args.remote_host}:{remote_job}",
                file=sys.stderr,
                flush=True,
            )


if __name__ == "__main__":
    main()
