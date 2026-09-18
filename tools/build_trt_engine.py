#!/usr/bin/env python3
"""Build a TensorRT engine from a local ONNX graph without downloading data."""

from __future__ import annotations

import argparse
from pathlib import Path

import tensorrt as trt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--workspace-gib", type=float, default=6.0)
    parser.add_argument("--builder-optimization-level", type=int, default=3)
    parser.add_argument("--max-aux-streams", type=int, default=0)
    parser.add_argument("--timing-cache", type=Path)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logger = trt.Logger(trt.Logger.VERBOSE if args.verbose else trt.Logger.INFO)
    builder = trt.Builder(logger)
    flag = trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED
    network = builder.create_network(1 << int(flag))
    onnx_parser = trt.OnnxParser(network, logger)
    if not onnx_parser.parse(args.source.read_bytes()):
        messages = "\n".join(
            str(onnx_parser.get_error(i)) for i in range(onnx_parser.num_errors)
        )
        raise RuntimeError(f"TensorRT could not parse {args.source}:\n{messages}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE, int(args.workspace_gib * (1 << 30))
    )
    config.builder_optimization_level = args.builder_optimization_level
    config.max_aux_streams = args.max_aux_streams
    fp16 = getattr(trt.BuilderFlag, "FP16", None)
    if fp16 is not None:
        config.set_flag(fp16)
    cache_bytes = b""
    if args.timing_cache and args.timing_cache.is_file():
        cache_bytes = args.timing_cache.read_bytes()
    timing_cache = config.create_timing_cache(cache_bytes)
    if not config.set_timing_cache(timing_cache, ignore_mismatch=False):
        raise RuntimeError("TensorRT rejected the timing cache")

    # The fused graph is deliberately fixed-shape.  Retain support for a
    # symbolic batch input so the utility remains useful for smaller models.
    dynamic_inputs = []
    for index in range(network.num_inputs):
        tensor = network.get_input(index)
        if any(dimension == -1 for dimension in tensor.shape):
            dynamic_inputs.append(tensor)
    if dynamic_inputs:
        profile = builder.create_optimization_profile()
        for tensor in dynamic_inputs:
            dims = tuple(tensor.shape)
            if dims[0] != -1 or any(value == -1 for value in dims[1:]):
                raise RuntimeError(
                    f"Only a dynamic batch axis is supported: {tensor.name} {dims}"
                )
            profile.set_shape(
                tensor.name, (1, *dims[1:]), (32, *dims[1:]), (64, *dims[1:])
            )
        config.add_optimization_profile(profile)

    print(
        f"building {args.source} nodes={network.num_layers} "
        f"inputs={network.num_inputs} outputs={network.num_outputs}",
        flush=True,
    )
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT engine build failed")
    args.engine.parent.mkdir(parents=True, exist_ok=True)
    args.engine.write_bytes(serialized)
    if args.timing_cache:
        args.timing_cache.parent.mkdir(parents=True, exist_ok=True)
        args.timing_cache.write_bytes(bytes(config.get_timing_cache().serialize()))
    print(
        f"wrote {args.engine} ({args.engine.stat().st_size / (1 << 20):.1f} MiB)",
        flush=True,
    )


if __name__ == "__main__":
    main()
