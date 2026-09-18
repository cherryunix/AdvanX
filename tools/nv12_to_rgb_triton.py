#!/usr/bin/env python3
"""Fused CUDA conversion of NVDEC NV12 surfaces to rotated RGB FP16 NCHW."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import triton
import triton.language as tl


@triton.jit
def _nv12_to_rgb_fp16_kernel(
    source,
    destination,
    destination_batch_offset,
    source_height: tl.constexpr,
    source_width: tl.constexpr,
    destination_height: tl.constexpr,
    destination_width: tl.constexpr,
    rotation: tl.constexpr,
    chroma_planar: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    pixels = destination_height * destination_width
    mask = offsets < pixels
    output_y = offsets // destination_width
    output_x = offsets - output_y * destination_width

    if rotation == 0:
        source_y = output_y
        source_x = output_x
    elif rotation == 90:
        source_y = source_height - 1 - output_x
        source_x = output_y
    elif rotation == 180:
        source_y = source_height - 1 - output_y
        source_x = source_width - 1 - output_x
    else:
        source_y = output_x
        source_x = source_width - 1 - output_y

    y_offset = source_y * source_width + source_x
    y = tl.load(source + y_offset, mask=mask, other=16).to(tl.float32)
    if chroma_planar:
        chroma_offset = (source_y // 2) * (source_width // 2) + source_x // 2
        u_offset = source_height * source_width + chroma_offset
        v_offset = source_height * source_width * 5 // 4 + chroma_offset
        u = tl.load(source + u_offset, mask=mask, other=128).to(tl.float32)
        v = tl.load(source + v_offset, mask=mask, other=128).to(tl.float32)
    else:
        uv_offset = (
            source_height * source_width
            + (source_y // 2) * source_width
            + (source_x // 2) * 2
        )
        u = tl.load(source + uv_offset, mask=mask, other=128).to(tl.float32)
        v = tl.load(source + uv_offset + 1, mask=mask, other=128).to(tl.float32)

    c = tl.maximum(y - 16.0, 0.0)
    d = u - 128.0
    e = v - 128.0
    red = tl.minimum(tl.maximum(tl.floor(1.164383 * c + 1.792741 * e), 0.0), 255.0)
    green = tl.minimum(
        tl.maximum(
            tl.floor(1.164383 * c - 0.213249 * d - 0.532909 * e), 0.0
        ),
        255.0,
    )
    blue = tl.minimum(tl.maximum(tl.floor(1.164383 * c + 2.112402 * d), 0.0), 255.0)

    plane = destination_height * destination_width
    base = destination_batch_offset + offsets
    tl.store(destination + base, red, mask=mask)
    tl.store(destination + base + plane, green, mask=mask)
    tl.store(destination + base + 2 * plane, blue, mask=mask)


def nv12_surfaces_to_rgb_nchw(
    surfaces: Sequence[object], rotation: int = 0, chroma_layout: str = "nv12"
) -> torch.Tensor:
    """Convert batched NV12/I420 surfaces without an intermediate RGBP batch."""
    if isinstance(surfaces, torch.Tensor):
        if surfaces.ndim == 2:
            surfaces = surfaces.unsqueeze(0)
        if surfaces.ndim != 3 or surfaces.shape[0] == 0:
            raise ValueError(
                "A batched NV12 tensor must have shape [batch, height * 3/2, width]"
            )
        # Software decode supplies one pinned host batch.  Upload it with one
        # H2D transaction rather than one small transfer per frame.
        batch = (
            surfaces
            if surfaces.is_cuda
            else surfaces.to("cuda", non_blocking=surfaces.is_pinned())
        )
        tensors = list(batch.unbind(0))
    else:
        if not surfaces:
            raise ValueError("At least one NV12 surface is required")
        tensors = [
            surface
            if isinstance(surface, torch.Tensor)
            else torch.from_dlpack(surface)
            for surface in surfaces
        ]
    rotation %= 360
    if rotation not in (0, 90, 180, 270):
        raise ValueError(f"Unsupported rotation: {rotation}")
    if chroma_layout not in ("nv12", "i420"):
        raise ValueError(f"Unsupported chroma layout: {chroma_layout}")
    # Some decoder/driver combinations can return a host DLPack tensor when
    # several decoders are active.  Keep that rare fallback correct.
    tensors = [
        tensor if tensor.is_cuda else tensor.to("cuda", non_blocking=False)
        for tensor in tensors
    ]
    shape = tuple(tensors[0].shape)
    if len(shape) != 2 or shape[0] % 3:
        raise ValueError(f"Expected an NV12 [height * 3/2, width] tensor: {shape}")
    if any(
        tuple(tensor.shape) != shape
        or tensor.dtype != torch.uint8
        or not tensor.is_cuda
        for tensor in tensors
    ):
        raise ValueError("All NV12 surfaces in a batch must share uint8 geometry")
    source_height = shape[0] * 2 // 3
    source_width = shape[1]
    if rotation in (90, 270):
        destination_height, destination_width = source_width, source_height
    else:
        destination_height, destination_width = source_height, source_width
    output = torch.empty(
        (len(tensors), 3, destination_height, destination_width),
        dtype=torch.float16,
        device=tensors[0].device,
    )
    pixels = destination_height * destination_width
    grid = (triton.cdiv(pixels, 256),)
    for index, tensor in enumerate(tensors):
        _nv12_to_rgb_fp16_kernel[grid](
            tensor,
            output,
            index * 3 * pixels,
            source_height=source_height,
            source_width=source_width,
            destination_height=destination_height,
            destination_width=destination_width,
            rotation=rotation,
            chroma_planar=chroma_layout == "i420",
            BLOCK=256,
        )
    return output


__all__ = ["nv12_surfaces_to_rgb_nchw"]
