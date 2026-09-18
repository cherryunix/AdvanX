#!/usr/bin/env python3
"""Patch MediaPipe ONNX exports for batching and build FP16 TensorRT engines.

Run this with the TensorRT Python environment.  The script does not download
weights; its inputs are ONNX conversions of the models embedded in the local
Holistic and Face Landmarker ``.task`` archives.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import onnx
from onnxconverter_common import float16
from onnx import numpy_helper
import tensorrt as trt


def _set_dynamic_batch(value_info) -> None:
    shape = value_info.type.tensor_type.shape
    if not shape.dim:
        return
    shape.dim[0].ClearField("dim_value")
    shape.dim[0].dim_param = "batch"


def make_batch_dynamic(source: Path, destination: Path, fp16: bool) -> None:
    """Replace fixed batch dimensions and optionally convert the graph to FP16."""
    model = onnx.load(str(source))
    initializers = {item.name: item for item in model.graph.initializer}
    for value_info in (*model.graph.input, *model.graph.output):
        _set_dynamic_batch(value_info)

    for node in model.graph.node:
        if node.op_type != "Reshape" or len(node.input) < 2:
            continue
        initializer = initializers.get(node.input[1])
        if initializer is None:
            continue
        values = numpy_helper.to_array(initializer).copy()
        if values.ndim == 1 and len(values) and values[0] == 1:
            # ONNX Reshape uses zero to copy the corresponding input axis.
            # Using -1 here would collide with an existing inferred axis in
            # detector heads shaped [1, -1, channels].
            values[0] = 0
            initializer.CopyFrom(numpy_helper.from_array(values, initializer.name))

    onnx.checker.check_model(model)
    if fp16:
        model = float16.convert_float_to_float16(
            model,
            keep_io_types=False,
            disable_shape_infer=True,
        )
        onnx.checker.check_model(model)
    destination.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(destination))


def build_engine(
    source: Path,
    destination: Path,
    min_batch: int,
    opt_batch: int,
    max_batch: int,
    workspace_gib: float,
) -> None:
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    strongly_typed = getattr(
        trt.NetworkDefinitionCreationFlag, "STRONGLY_TYPED", None
    )
    explicit_batch = getattr(trt.NetworkDefinitionCreationFlag, "EXPLICIT_BATCH", None)
    flag = strongly_typed if strongly_typed is not None else explicit_batch
    network_flags = 0 if flag is None else 1 << int(flag)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(source.read_bytes()):
        messages = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise RuntimeError(f"TensorRT could not parse {source}:\n{messages}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE, int(workspace_gib * (1 << 30))
    )
    fp16_flag = getattr(trt.BuilderFlag, "FP16", None)
    if fp16_flag is not None:
        config.set_flag(fp16_flag)

    profile = builder.create_optimization_profile()
    for index in range(network.num_inputs):
        tensor = network.get_input(index)
        dims = tuple(tensor.shape)
        if not dims or dims[0] != -1:
            raise RuntimeError(f"Expected dynamic batch input, got {tensor.name}: {dims}")
        tail = dims[1:]
        profile.set_shape(
            tensor.name,
            (min_batch, *tail),
            (opt_batch, *tail),
            (max_batch, *tail),
        )
    config.add_optimization_profile(profile)

    print(
        f"building {source.name}: batch {min_batch}/{opt_batch}/{max_batch}, "
        f"workspace={workspace_gib:g} GiB",
        flush=True,
    )
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError(f"TensorRT engine build failed for {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(serialized)
    print(f"wrote {destination} ({destination.stat().st_size / (1 << 20):.1f} MiB)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--dynamic-onnx", type=Path, required=True)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--min-batch", type=int, default=1)
    parser.add_argument("--opt-batch", type=int, default=64)
    parser.add_argument("--max-batch", type=int, default=128)
    parser.add_argument("--workspace-gib", type=float, default=1.5)
    parser.add_argument("--fp32", action="store_true", help="Keep the ONNX graph in FP32.")
    args = parser.parse_args()
    if not (1 <= args.min_batch <= args.opt_batch <= args.max_batch):
        parser.error("batch sizes must satisfy 1 <= min <= opt <= max")
    make_batch_dynamic(args.source, args.dynamic_onnx, fp16=not args.fp32)
    build_engine(
        args.dynamic_onnx,
        args.engine,
        args.min_batch,
        args.opt_batch,
        args.max_batch,
        args.workspace_gib,
    )


if __name__ == "__main__":
    main()
