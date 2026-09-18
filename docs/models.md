# Model assets

AdvanX does not download or redistribute model weights. Expected assets are conversions of a face detector, face landmarker with iris points, pose landmarker with world landmarks, and hand landmarker, plus face-geometry Procrustes metadata.

A typical private model tree is:

```text
models/mediapipe/
├── onnx_dynamic/
├── fused/
│   ├── portrait.engine
│   └── landscape.engine
└── face_geometry_procrustes.npz
```

`tools/export_fused_holistic_onnx.py` assembles fixed-geometry fused graphs from locally supplied components. `tools/build_trt_engine.py` builds a TensorRT engine from an ONNX graph. The engine batch and input geometry must match the worker arguments and oriented source geometry.

`face_geometry_procrustes.npz` is expected to contain the canonical face geometry and basis used by the rigid fit. It is treated as a model-derived asset and is intentionally excluded from this repository.

TensorRT engines are not portable across arbitrary TensorRT versions or GPU architectures. Build and test them on each execution class.
