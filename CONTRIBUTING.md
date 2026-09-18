# Contributing

Keep timing semantics exact and keep models out of the repository.

Before opening a change:

```bash
python -m unittest discover -s tests -v
python -m compileall -q advanx tools tests
```

Changes to sampling must add a case to `tests/test_temporal_sampling.py`. Runtime changes should report source fps, target fps, source frame count, selected indices, decode backend, and measured end-to-end fps.

Do not commit videos, model weights, ONNX graphs, TensorRT engines, task bundles, run outputs, or manifests containing private filesystem paths.
