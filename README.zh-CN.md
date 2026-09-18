# AdvanX

**Accelerated Decoding and Vision Across Nodes**，读作 “advance”。X 只作为跨设备执行结构的视觉标记，不单独发音。

AdvanX 把长视频变成严格对齐时间轴的逐帧数值缓存：人脸 478 点与瞳孔、全身 33 点与世界坐标、双手各 21 点、3D 头部位姿、视线和时序包络。系统可同时调度 NVDEC、Intel D3D11VA、CPU 软解、TensorRT 推理和多台机器。

![AdvanX 性能](assets/advanx-performance.png)

核心能力：

- 每个原片只在第一帧比较 0° / 90° / 180° / 270°，整段复用方向结论。
- 100 fps 等任意高帧率原片直接映射到精确 24 fps 时间轴，不先重编码素材。
- 以整段原片为调度单位，根据实测 lane 吞吐和远端传输成本分配任务。
- 在 WSL 下混用 NVIDIA NVDEC、Windows Intel D3D11VA 与 FFmpeg CPU 解码。
- 用 TensorRT FP16 固定分辨率图执行完整 Holistic 推理。
- 输出可续跑的 memory-mapped NumPy 列，并生成平滑值与 P10/P90 包络。
- 校验源文件签名、完整解码、目标帧索引、时间戳和全部数组长度。

![AdvanX 架构](assets/advanx-architecture.png)

仓库包含 `uv.toml`、`.python-version` 和 `uv.lock`，执行 `uv sync --locked` 后即可运行基础测试。安装与完整命令请看英文 [README](README.md)。模型、ONNX 和 TensorRT engine 均不随仓库发布，也不会自动下载；请使用自己有权使用的模型资产。运行环境、模型目录和性能口径分别见 [runtime](docs/runtime.md)、[models](docs/models.md) 与 [benchmark](docs/benchmark.md)。

两页 1920×1080 的 HTML 幻灯片位于 [slides/advanx-keynote.html](slides/advanx-keynote.html)，方向键切换，URL 加 `?slide=1&export=1` 或 `?slide=2&export=1` 可稳定导出。朋友圈竖向长图提供 [HTML 源文件](slides/advanx-social.html) 和 [1920×2160 PNG](assets/advanx-social-vertical.png)。
