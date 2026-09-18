#!/usr/bin/env python3
"""Profile multiple NVDEC producers feeding one fused Holistic engine."""

from __future__ import annotations

import argparse
import json
import os
import queue
import threading
import time
from pathlib import Path

import PyNvVideoCodec as nvc
import torch
from cuda.bindings import driver as cuda_driver

from gaze_iris_audit_trt import FusedHolisticTrt
from nv12_to_rgb_triton import nv12_surfaces_to_rgb_nchw


def parse_cpus(value: str) -> list[int]:
    cpus: list[int] = []
    for part in value.split(","):
        bounds = part.split("-", 1)
        if len(bounds) == 1:
            cpus.append(int(bounds[0]))
        else:
            cpus.extend(range(int(bounds[0]), int(bounds[1]) + 1))
    return sorted(set(cpus))


def split_cpus(cpus: list[int], groups: int) -> list[list[int]]:
    return [
        cpus[(len(cpus) * index) // groups : (len(cpus) * (index + 1)) // groups]
        for index in range(groups)
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path, nargs="+")
    parser.add_argument("--rotation", type=int, action="append", required=True)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument(
        "--face-geometry-metadata",
        type=Path,
        default=Path("models/mediapipe/face_geometry_procrustes.npz"),
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--decode-buffer", type=int, default=96)
    parser.add_argument("--queue-batches", type=int, default=6)
    parser.add_argument("--max-batches", type=int, default=200)
    parser.add_argument("--consumer-cpus", default="0-15")
    parser.add_argument("--producer-cpus", default="16-31")
    args = parser.parse_args()
    if len(args.rotation) != len(args.source):
        parser.error("Supply one --rotation for each source")

    consumer_cpus = parse_cpus(args.consumer_cpus)
    producer_cpu_groups = split_cpus(
        parse_cpus(args.producer_cpus), len(args.source)
    )
    os.sched_setaffinity(0, consumer_cpus)
    pipeline = FusedHolisticTrt(args.engine, args.face_geometry_metadata)
    # TensorRT, Triton and PyTorch use the device primary context.  Passing that
    # context into every decoder keeps all NVDEC surfaces addressable from the
    # fused Triton conversion even when several decoder threads are active.
    torch.empty(0, device="cuda")
    status, cuda_context = cuda_driver.cuCtxGetCurrent()
    if status != cuda_driver.CUresult.CUDA_SUCCESS or int(cuda_context) == 0:
        raise RuntimeError(f"Could not obtain the CUDA primary context: {status}")
    cuda_context_handle = int(cuda_context)

    ready: queue.Queue = queue.Queue(maxsize=args.queue_batches)
    stop = threading.Event()
    producer_stats = [
        {"frames": 0, "decoder_wait_seconds": 0.0} for _ in args.source
    ]

    started = time.perf_counter()
    decoders: list[nvc.ThreadedDecoder] = []
    decoder_streams: list[torch.cuda.Stream] = []
    try:
        for index, source in enumerate(args.source):
            # The decoder's native worker inherits the constructing thread's
            # affinity.  Put those workers on the E-core set and reserve the
            # P-core set for TRT output handling and numerical postprocessing.
            os.sched_setaffinity(0, producer_cpu_groups[index])
            stream = torch.cuda.Stream()
            decoder_streams.append(stream)
            decoders.append(
                nvc.ThreadedDecoder(
                    str(source),
                    buffer_size=args.decode_buffer,
                    gpu_id=0,
                    cuda_context=cuda_context_handle,
                    cuda_stream=stream.cuda_stream,
                    use_device_memory=True,
                    output_color_type=nvc.OutputColorType.NATIVE,
                )
            )
    finally:
        os.sched_setaffinity(0, consumer_cpus)

    def produce(
        index: int,
        decoder: nvc.ThreadedDecoder,
        rotation: int,
    ) -> None:
        os.sched_setaffinity(0, producer_cpu_groups[index])
        try:
            for _ in range(args.max_batches):
                stage = time.perf_counter()
                decoded = decoder.get_batch_frames(args.batch_size)
                producer_stats[index]["decoder_wait_seconds"] += (
                    time.perf_counter() - stage
                )
                if not decoded:
                    break
                producer_stats[index]["frames"] += len(decoded)
                while not stop.is_set():
                    try:
                        ready.put((index, rotation, decoded), timeout=0.2)
                        break
                    except queue.Full:
                        pass
        finally:
            decoder.end()
            while not stop.is_set():
                try:
                    ready.put((index, rotation, None), timeout=0.2)
                    break
                except queue.Full:
                    pass

    threads = [
        threading.Thread(
            target=produce,
            args=(index, decoders[index], rotation),
            name=f"nvdec-{index}",
        )
        for index, (source, rotation) in enumerate(
            zip(args.source, args.rotation, strict=True)
        )
    ]
    for thread in threads:
        thread.start()
    timing = {"queue_wait": 0.0, "assembly": 0.0, "inference": 0.0}
    completed = 0
    consumed = [0] * len(args.source)
    try:
        while completed < len(threads):
            stage = time.perf_counter()
            index, rotation, decoded = ready.get()
            timing["queue_wait"] += time.perf_counter() - stage
            if decoded is None:
                completed += 1
                continue
            stage = time.perf_counter()
            rgb_nchw = nv12_surfaces_to_rgb_nchw(decoded, rotation)
            timing["assembly"] += time.perf_counter() - stage
            stage = time.perf_counter()
            rows = pipeline.infer_rgb_nchw(rgb_nchw)
            timing["inference"] += time.perf_counter() - stage
            consumed[index] += len(rows)
    finally:
        stop.set()
        for thread in threads:
            thread.join(timeout=5)
    elapsed = time.perf_counter() - started
    frames = sum(consumed)
    print(
        json.dumps(
            {
                "sources": [str(path) for path in args.source],
                "affinity": sorted(os.sched_getaffinity(0)),
                "producer_affinity": producer_cpu_groups,
                "frames_by_source": consumed,
                "frames": frames,
                "wall_seconds": elapsed,
                "wall_fps": frames / elapsed,
                "inference_fps": frames / timing["inference"],
                "timing_seconds": timing,
                "producer_stats": producer_stats,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
