# Runtime

The validated development host used:

- WSL2 on Windows
- NVIDIA GeForce RTX 4090 Laptop GPU, 16 GB
- NVIDIA driver 591.86
- Intel Core i9-14900HX, 32 logical CPUs
- Python 3.10+
- NumPy 2.2.6
- TensorRT 11.0.0.114
- PyTorch 2.13.0+cu130, torchvision 0.28.0+cu130
- Triton 3.7.1
- PyNvVideoCodec 2.2.0
- OpenCV headless 5.0.0.93
- ONNX 1.22.0, onnx2torch 1.5.15

These versions describe the tested setup. They are not a claim that every combination is supported. Match CUDA, TensorRT, PyTorch, PyNvVideoCodec, and the NVIDIA driver as one stack.

The Intel path launches a Windows `ffmpeg.exe` from WSL and requests `d3d11va=igpu` with Intel's vendor ID. Frames are selected on the exact target timeline before they cross into the Linux process.

For hybrid CPUs, use `--cpu-affinity` on individual workers or the local/remote software-affinity options in the dispatcher. Keep decode workers on the intended P-core or E-core sets and measure each lane before entering its rate into the scheduler.
