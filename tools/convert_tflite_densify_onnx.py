#!/usr/bin/env python3
"""Convert sparse TFLite models to ONNX by materializing DENSIFY constants.

MediaPipe's pose detector stores convolution weights sparsely.  The stock
``tflite2onnx`` converter does not implement TFLite opcode 124 (DENSIFY).  This
adapter asks the local TFLite interpreter to materialize those constant tensors
and emits them as ONNX Constant nodes.  No model weights are downloaded.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper
import tensorflow as tf
import tflite
import tflite2onnx
from tflite2onnx.op.common import OpFactory, Operator


DENSE_VALUES: dict[int, np.ndarray] = {}


class Densify(Operator):
    TypeMapping = {tflite.BuiltinOperator.DENSIFY: "Constant"}

    def __init__(self, tensor_factory, index):
        super().__init__(tensor_factory, index)
        self.setInited()

    @property
    def type(self):
        return "Constant"

    def parse(self):
        if self.tflite.InputsLength() != 1 or self.tflite.OutputsLength() != 1:
            raise RuntimeError("Unexpected DENSIFY signature")
        output_index = int(self.tflite.Outputs(0))
        self.value = DENSE_VALUES[output_index].copy()
        self.parseOutput(0)
        self.setParsed()

    def propagatableTensors(self):
        return []

    def transform(self):
        output = self.outputs[0]
        if output.layout is None:
            return
        self.value = self.value.transpose(output.layout.perm)
        output.shape = output.layout.transform(output.shape)

    def convert(self):
        output = self.outputs[0]
        output.convert()
        value = numpy_helper.from_array(self.value, f"{output.name}_dense_value")
        self.onnx = helper.make_node(
            "Constant", [], [output.name], name=self.name, value=value
        )
        self.setConverted()


class DepthToSpace(Operator):
    TypeMapping = {tflite.BuiltinOperator.DEPTH_TO_SPACE: "DepthToSpace"}

    def __init__(self, tensor_factory, index):
        super().__init__(tensor_factory, index)
        self.setInited()

    @property
    def type(self):
        return "DepthToSpace"

    def parse(self):
        options = tflite.DepthToSpaceOptions()
        builtin = self.tflite.BuiltinOptions()
        options.Init(builtin.Bytes, builtin.Pos)
        self.attrs["blocksize"] = int(options.BlockSize())
        self.attrs["mode"] = "DCR"
        self.parseInput(0)
        self.parseOutput(0)
        self.setParsed()

    def propagatableTensors(self):
        return self.inputs + self.outputs

    def transform(self):
        pass


def materialize_densify(path: Path) -> dict[int, np.ndarray]:
    interpreter = tf.lite.Interpreter(
        model_path=str(path), experimental_preserve_all_tensors=True
    )
    interpreter.allocate_tensors()
    for detail in interpreter.get_input_details():
        interpreter.set_tensor(
            detail["index"], np.zeros(detail["shape"], dtype=detail["dtype"])
        )
    interpreter.invoke()
    values = {}
    for operation in interpreter._get_ops_details():
        if operation["op_name"] != "DENSIFY":
            continue
        index = int(operation["outputs"][0])
        values[index] = interpreter.get_tensor(index).copy()
    if not values:
        raise RuntimeError(f"No DENSIFY tensors found in {path}")
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    DENSE_VALUES.update(materialize_densify(args.source))
    OpFactory.register(Densify)
    OpFactory.register(DepthToSpace)
    args.destination.parent.mkdir(parents=True, exist_ok=True)
    tflite2onnx.convert(str(args.source), str(args.destination))
    model = onnx.load(str(args.destination))
    onnx.checker.check_model(model)
    print(
        f"wrote {args.destination}; materialized {len(DENSE_VALUES)} sparse tensors"
    )


if __name__ == "__main__":
    main()
