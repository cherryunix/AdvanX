#!/usr/bin/env python3
"""Solve each original source once and cache full-frame Holistic columns."""

from __future__ import annotations

import argparse
import bisect
import csv
import gc
import json
import math
import os
import shutil
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np
import PyNvVideoCodec as nvc
import torch

from gaze_iris_audit_trt import FusedHolisticTrt, HolisticTrt, calibration_baseline
from nv12_to_rgb_triton import nv12_surfaces_to_rgb_nchw
from advanx.temporal import (
    ffmpeg_select_filter,
    periodic_select_pattern,
    target_source_indices,
)


SCALARS = (
    "face_score", "pose_score", "head_pitch_deg", "head_yaw_deg", "head_roll_deg",
    "head_pose_rigidity_error", "eye_horizontal", "eye_vertical", "delta_horizontal",
    "delta_vertical", "delta_head_pitch_deg", "delta_head_yaw_deg", "delta_head_roll_deg",
    "left_hand_score", "right_hand_score",
)
TRACE_FIELDS = (
    "delta_horizontal", "delta_vertical", "delta_head_pitch_deg",
    "delta_head_yaw_deg", "delta_head_roll_deg",
)


class GeometryHolisticTrt:
    def __init__(
        self,
        fallback: HolisticTrt | None,
        fused_engines: list[Path],
        face_geometry_metadata: Path,
    ):
        self.fallback = fallback
        self.fused: dict[tuple[int, int], FusedHolisticTrt] = {}
        for path in fused_engines:
            pipeline = FusedHolisticTrt(path, face_geometry_metadata)
            geometry = (pipeline.height, pipeline.width)
            if geometry in self.fused:
                raise ValueError(f"Duplicate fused engine geometry: {geometry}")
            self.fused[geometry] = pipeline

    def infer_rgb_nchw(self, rgb_nchw: torch.Tensor) -> list[dict]:
        geometry = tuple(rgb_nchw.shape[2:])
        pipeline = self.fused.get(geometry)
        if pipeline is None or rgb_nchw.shape[0] > pipeline.batch_size:
            if self.fallback is None:
                available = ", ".join(f"{h}x{w}" for h, w in self.fused)
                raise RuntimeError(
                    f"No fused engine for batch={rgb_nchw.shape[0]} geometry="
                    f"{geometry}; available geometries: {available}"
                )
            return self.fallback.infer_rgb_nchw(rgb_nchw)
        return pipeline.infer_rgb_nchw(rgb_nchw)


def create(directory: Path, name: str, dtype: str, shape: tuple[int, ...], fill=None) -> np.memmap:
    data = np.lib.format.open_memmap(directory / f"{name}.npy", mode="w+", dtype=dtype, shape=shape)
    if fill is not None:
        data[...] = fill
    return data


def close_memmap(data: np.ndarray) -> None:
    if isinstance(data, np.memmap):
        data.flush()
        mmap = getattr(data, "_mmap", None)
        if mmap is not None:
            mmap.close()


def output_arrays(directory: Path, frames: int) -> dict[str, np.memmap]:
    arrays = {
        "frame": create(directory, "frame", "int32", (frames,)),
        "time_s": create(directory, "time_s", "float32", (frames,)),
        "source_time_s": create(directory, "source_time_s", "float32", (frames,)),
        "face_detected": create(directory, "face_detected", "bool", (frames,)),
        "pose_detected": create(directory, "pose_detected", "bool", (frames,)),
        "face_candidates": create(directory, "face_candidates", "int16", (frames,)),
        "left_hand_detected": create(directory, "left_hand_detected", "bool", (frames,)),
        "right_hand_detected": create(directory, "right_hand_detected", "bool", (frames,)),
        "face_landmarks": create(directory, "face_landmarks", "float32", (frames, 478, 3)),
        "pose_landmarks": create(directory, "pose_landmarks", "float32", (frames, 33, 5)),
        "pose_world_landmarks": create(directory, "pose_world_landmarks", "float32", (frames, 33, 3)),
        "left_hand_landmarks": create(directory, "left_hand_landmarks", "float32", (frames, 21, 3)),
        "left_hand_world_landmarks": create(directory, "left_hand_world_landmarks", "float32", (frames, 21, 3)),
        "right_hand_landmarks": create(directory, "right_hand_landmarks", "float32", (frames, 21, 3)),
        "right_hand_world_landmarks": create(directory, "right_hand_world_landmarks", "float32", (frames, 21, 3)),
        "head_rotation_matrix": create(directory, "head_rotation_matrix", "float32", (frames, 9)),
        "right_eye": create(directory, "right_eye", "float32", (frames, 5)),
        "left_eye": create(directory, "left_eye", "float32", (frames, 5)),
    }
    for field in SCALARS:
        arrays[field] = create(directory, field, "float32", (frames,))
    return arrays


def write_batch(
    arrays: dict[str, np.memmap],
    rows: list[dict],
    begin: int,
    source_indices: np.ndarray,
    source_fps: float,
    target_fps: float,
    baseline: dict,
) -> None:
    end = begin + len(rows)
    arrays["frame"][begin:end] = source_indices
    arrays["time_s"][begin:end] = np.arange(begin, end, dtype=np.float32) / target_fps
    arrays["source_time_s"][begin:end] = source_indices.astype(np.float32) / source_fps
    for name in (
        "face_detected", "pose_detected", "left_hand_detected", "right_hand_detected",
    ):
        arrays[name][begin:end] = False
    arrays["face_candidates"][begin:end] = 0
    for name in (
        "face_landmarks", "pose_landmarks", "pose_world_landmarks", "head_rotation_matrix",
        "left_hand_landmarks", "left_hand_world_landmarks", "right_hand_landmarks",
        "right_hand_world_landmarks", "right_eye", "left_eye", *SCALARS,
    ):
        arrays[name][begin:end] = np.nan
    eye_names = ("horizontal", "vertical", "openness", "iris_x", "iris_y")
    for offset, row in enumerate(rows):
        index = begin + offset
        arrays["face_detected"][index] = row["detected"]
        arrays["pose_detected"][index] = row["pose_detected"]
        arrays["face_candidates"][index] = row["face_candidates"]
        for name in (
            "face_landmarks", "pose_landmarks", "pose_world_landmarks", "head_rotation_matrix",
            "left_hand_landmarks", "left_hand_world_landmarks", "right_hand_landmarks",
            "right_hand_world_landmarks",
        ):
            if row.get(name) is not None:
                arrays[name][index] = np.asarray(row[name], dtype=np.float32).reshape(arrays[name].shape[1:])
        for side in ("left", "right"):
            hand_name = f"{side}_hand_landmarks"
            arrays[f"{side}_hand_detected"][index] = row.get(hand_name) is not None
            eye = row.get(side)
            if eye is not None:
                arrays[f"{side}_eye"][index] = [eye[name] for name in eye_names]
        if row.get("eye_horizontal") is not None:
            row["delta_horizontal"] = row["eye_horizontal"] - baseline["eye_horizontal"]
            row["delta_vertical"] = row["eye_vertical"] - baseline["eye_vertical"]
        for axis in ("pitch", "yaw", "roll"):
            key = f"head_{axis}_deg"
            if row.get(key) is not None:
                row[f"delta_head_{axis}_deg"] = row[key] - baseline[key]
        for field in SCALARS:
            value = row.get(field)
            if value is not None:
                arrays[field][index] = value


def percentile(sorted_values: list[float], q: float) -> float:
    position = (len(sorted_values) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = position - lower
    return sorted_values[lower] * (1 - fraction) + sorted_values[upper] * fraction


def centered_rolling(values: np.ndarray, window: int, quantiles: tuple[float, ...], minimum: int) -> list[np.ndarray]:
    half = window // 2
    outputs = [np.full(len(values), np.nan, dtype=np.float32) for _ in quantiles]
    ordered: list[float] = []
    left, right = 0, min(len(values), half + 1)
    for value in values[left:right]:
        if np.isfinite(value):
            bisect.insort(ordered, float(value))
    for index in range(len(values)):
        wanted_left = max(0, index - half)
        wanted_right = min(len(values), index + half + 1)
        while left < wanted_left:
            value = values[left]
            if np.isfinite(value):
                ordered.pop(bisect.bisect_left(ordered, float(value)))
            left += 1
        while right < wanted_right:
            value = values[right]
            if np.isfinite(value):
                bisect.insort(ordered, float(value))
            right += 1
        if len(ordered) >= minimum:
            for output, q in zip(outputs, quantiles):
                output[index] = percentile(ordered, q)
    return outputs


def add_temporal_columns(directory: Path, frames: int, fps: float) -> None:
    smooth_window = max(3, round(fps * 0.375)) | 1
    envelope_window = max(5, round(fps * 1.0)) | 1
    for field in TRACE_FIELDS:
        values = np.load(directory / f"{field}.npy", mmap_mode="r")
        smooth, = centered_rolling(values, smooth_window, (0.5,), 3)
        p10, p90 = centered_rolling(values, envelope_window, (0.1, 0.9), 5)
        for suffix, data in (("smooth", smooth), ("p10", p10), ("p90", p90)):
            target = create(directory, f"{field}_{suffix}", "float32", (frames,))
            target[:] = data
            close_memmap(target)
        close_memmap(values)
        del values, smooth, p10, p90, target
        gc.collect()


def rotate_tensor(frame: torch.Tensor, clockwise: int) -> torch.Tensor:
    clockwise %= 360
    if clockwise == 90:
        frame = torch.rot90(frame, -1, (1, 2))
    elif clockwise == 180:
        frame = torch.rot90(frame, 2, (1, 2))
    elif clockwise == 270:
        frame = torch.rot90(frame, 1, (1, 2))
    elif clockwise:
        raise ValueError(f"Unsupported rotation: {clockwise}")
    return frame


def probe_source_frames(path: str, fallback: int) -> int:
    command = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=nb_frames", "-of", "default=nw=1:nk=1", path,
    ]
    try:
        value = subprocess.check_output(command, text=True).strip()
        frames = int(value)
    except (OSError, subprocess.CalledProcessError, ValueError):
        frames = fallback
    if frames <= 0:
        raise RuntimeError(f"Could not determine source frame count for {path}")
    return frames


class FFmpegSoftwareDecoder:
    """CPU H.264 decode with exact frame filtering and pinned I420 output."""

    def __init__(
        self,
        path: str,
        width: int,
        height: int,
        source_fps: float,
        target_fps: float,
        source_frames: int,
        threads: int,
    ) -> None:
        self.width = width
        self.height = height
        self.frame_bytes = width * height * 3 // 2
        self.stderr = tempfile.TemporaryFile(mode="w+b")
        command = [
            "ffmpeg", "-hide_banner", "-nostdin", "-loglevel", "error",
            "-threads", str(threads), "-noautorotate", "-i", path,
            "-map", "0:v:0", "-an", "-sn", "-dn",
            "-vf", ffmpeg_select_filter(source_fps, target_fps, source_frames),
            "-vsync", "0", "-pix_fmt", "yuv420p", "-f", "rawvideo", "pipe:1",
        ]
        self.process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=self.stderr, bufsize=0
        )

    def get_batch_frames(self, count: int) -> torch.Tensor | None:
        if self.process.stdout is None:
            raise RuntimeError("FFmpeg stdout is unavailable")
        batch = torch.empty(
            (count, self.height * 3 // 2, self.width),
            dtype=torch.uint8,
            pin_memory=True,
        )
        target = memoryview(batch.numpy().reshape(-1))
        offset = 0
        while offset < len(target):
            got = self.process.stdout.readinto(target[offset:])
            if not got:
                break
            offset += got
        if offset == 0:
            return None
        if offset % self.frame_bytes:
            raise RuntimeError(
                f"FFmpeg returned a partial NV12 frame: {offset}/{self.frame_bytes} bytes"
            )
        frames = offset // self.frame_bytes
        return batch[:frames]

    def end(self) -> None:
        if self.process.stdout is not None:
            # The exact select filter should have no more output.  Reading to
            # EOF lets FFmpeg finish the final GOP without turning a normal
            # producer shutdown into EPIPE.
            self.process.stdout.read()
            self.process.stdout.close()
        returncode = self.process.wait()
        self.stderr.seek(0)
        message = self.stderr.read().decode("utf-8", errors="replace").strip()
        self.stderr.close()
        if returncode:
            raise RuntimeError(
                f"FFmpeg software decoder failed with code {returncode}: {message}"
            )


def find_windows_ffmpeg(requested: Path | None) -> Path:
    if requested is not None:
        if not requested.is_file():
            raise FileNotFoundError(requested)
        return requested
    candidates: list[Path] = []
    for pattern in (
        "/mnt/c/Users/*/AppData/Local/Microsoft/WinGet/Packages/"
        "Gyan.FFmpeg.Shared*/ffmpeg-*/bin/ffmpeg.exe",
        "/mnt/c/Users/*/AppData/Local/Microsoft/WinGet/Packages/"
        "Gyan.FFmpeg_*/ffmpeg-*/bin/ffmpeg.exe",
        "/mnt/c/Users/*/AppData/Local/Microsoft/WinGet/Packages/"
        "yt-dlp.FFmpeg_*/ffmpeg-*/bin/ffmpeg.exe",
    ):
        candidates.extend(Path("/").glob(pattern.removeprefix("/")))
    if not candidates:
        raise FileNotFoundError("Could not find a Windows ffmpeg.exe with D3D11VA")
    return sorted(candidates, key=lambda path: path.stat().st_mtime_ns)[-1]


class WindowsIGPUDecoder(FFmpegSoftwareDecoder):
    """Intel D3D11VA decode on Windows with selected NV12 frames piped to WSL."""

    def __init__(
        self,
        ffmpeg: Path,
        path: str,
        width: int,
        height: int,
        source_fps: float,
        target_fps: float,
        source_frames: int,
    ) -> None:
        self.width = width
        self.height = height
        self.frame_bytes = width * height * 3 // 2
        self.stderr = tempfile.TemporaryFile(mode="w+b")
        windows_path = subprocess.check_output(
            ["wslpath", "-w", path], text=True
        ).strip()
        filters = (
            ffmpeg_select_filter(source_fps, target_fps, source_frames)
            + ",hwdownload,format=nv12"
        )
        command = [
            str(ffmpeg), "-hide_banner", "-nostdin", "-loglevel", "error",
            "-init_hw_device", "d3d11va=igpu:,vendor_id=0x8086",
            "-filter_hw_device", "igpu", "-hwaccel", "d3d11va",
            "-hwaccel_device", "igpu", "-hwaccel_output_format", "d3d11",
            "-i", windows_path, "-map", "0:v:0", "-an", "-sn", "-dn",
            "-vf", filters, "-fps_mode", "passthrough", "-pix_fmt", "nv12",
            "-f", "rawvideo", "pipe:1",
        ]
        self.process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=self.stderr, bufsize=0
        )


def solve_source(
    pipeline: GeometryHolisticTrt,
    item: dict,
    baseline: dict,
    args: argparse.Namespace,
) -> dict:
    source_started = time.perf_counter()
    timing = {
        "setup": 0.0,
        "decode_wait": 0.0,
        "gpu_assembly_and_rotation": 0.0,
        "inference": 0.0,
        "writer_wait": 0.0,
        "finalize_arrays": 0.0,
        "temporal_columns": 0.0,
    }
    ident = item["id"]
    source_fps = float(Fraction(str(item["fps"])))
    target_fps = args.target_fps or source_fps
    host_decode = args.software_decode or args.igpu_decode
    if target_fps > source_fps + 1e-6:
        raise ValueError(
            f"Target FPS {target_fps:g} exceeds source FPS {source_fps:g} for {ident}"
        )
    final = args.output_dir / "sources" / ident
    marker = final / "summary.json"
    if marker.is_file():
        saved = json.loads(marker.read_text())
        if (
            saved.get("signature") == item["signature"]
            and math.isclose(
                float(saved.get("target_fps", saved.get("fps", -1))),
                target_fps,
                rel_tol=0,
                abs_tol=1e-6,
            )
        ):
            return {**saved, "state": "resume"}
    partial = args.output_dir / "sources" / f".{ident}.partial"
    shutil.rmtree(partial, ignore_errors=True)
    partial.mkdir(parents=True)

    setup_started = time.perf_counter()
    if host_decode:
        source_frames = probe_source_frames(
            item["path"], round(float(item["duration"]) * source_fps)
        )
        if args.igpu_decode:
            decoder = WindowsIGPUDecoder(
                args.windows_ffmpeg, item["path"], int(item["width"]),
                int(item["height"]), source_fps, target_fps, source_frames,
            )
        else:
            decoder = FFmpegSoftwareDecoder(
                item["path"], int(item["width"]), int(item["height"]),
                source_fps, target_fps, source_frames,
                args.software_decode_threads,
            )
    else:
        decoder = nvc.ThreadedDecoder(
            item["path"], buffer_size=args.decode_buffer, gpu_id=0,
            use_device_memory=True,
            output_color_type=(
                nvc.OutputColorType.NATIVE
                if args.native_nv12
                else nvc.OutputColorType.RGBP
            ),
        )
        source_frames = len(decoder)
    if source_fps <= 0 or source_frames <= 0:
        raise RuntimeError(f"Invalid video metadata for {ident}")
    selected_source_indices = target_source_indices(
        source_frames, source_fps, target_fps
    )
    selected_frames = len(selected_source_indices)
    arrays = output_arrays(partial, selected_frames)
    timing["setup"] = time.perf_counter() - setup_started
    decoded_source_frames, processed, infer_seconds = 0, 0, 0.0
    geometry = None
    pending_write = None
    writer = ThreadPoolExecutor(max_workers=1, thread_name_prefix="holistic-writer")
    try:
        while True:
            stage_started = time.perf_counter()
            if host_decode:
                next_output_end = min(processed + args.batch_size, selected_frames)
                decoded = decoder.get_batch_frames(next_output_end - processed)
                selected_indices = selected_source_indices[processed:next_output_end]
            else:
                next_output_end = min(processed + args.batch_size, selected_frames)
                if next_output_end > processed:
                    desired_decode_end = int(selected_source_indices[next_output_end - 1]) + 1
                    if next_output_end == selected_frames:
                        desired_decode_end = source_frames
                else:
                    desired_decode_end = source_frames
                decode_count = max(1, desired_decode_end - decoded_source_frames)
                decoded = decoder.get_batch_frames(decode_count)
            timing["decode_wait"] += time.perf_counter() - stage_started
            if decoded is None or len(decoded) == 0:
                break
            if host_decode:
                selected_decoded = decoded
                if len(selected_indices) != len(decoded):
                    raise RuntimeError(
                        f"Software decoder frame mismatch for {ident}: "
                        f"{len(decoded)}/{len(selected_indices)}"
                    )
            else:
                decoded_begin = decoded_source_frames
                decoded_source_frames += len(decoded)
                selection_begin = int(np.searchsorted(
                    selected_source_indices, decoded_begin, side="left"
                ))
                selection_end = int(np.searchsorted(
                    selected_source_indices, decoded_source_frames, side="left"
                ))
                selected_indices = selected_source_indices[selection_begin:selection_end]
                if not len(selected_indices):
                    del decoded, selected_indices
                    continue
                selected_decoded = [
                    decoded[int(index) - decoded_begin] for index in selected_indices
                ]
            if len(selected_indices) > args.batch_size:
                raise RuntimeError(
                    f"Sampling batch overflow for {ident}: "
                    f"{len(selected_indices)}/{args.batch_size}"
                )
            stage_started = time.perf_counter()
            if args.native_nv12 or host_decode:
                rgb_nchw = nv12_surfaces_to_rgb_nchw(
                    selected_decoded,
                    int(item.get("rotation_correction_clockwise", 0)),
                    chroma_layout="i420" if args.software_decode else "nv12",
                )
                tensors = []
            else:
                tensors = [
                    rotate_tensor(
                        torch.from_dlpack(frame),
                        int(item.get("rotation_correction_clockwise", 0)),
                    )
                    for frame in selected_decoded
                ]
                rgb_nchw = torch.stack(tensors)
            timing["gpu_assembly_and_rotation"] += time.perf_counter() - stage_started
            geometry = tuple(rgb_nchw.shape[2:])
            started = time.perf_counter()
            rows = pipeline.infer_rgb_nchw(rgb_nchw)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            infer_seconds += elapsed
            timing["inference"] += elapsed
            if processed + len(rows) > selected_frames:
                raise RuntimeError(f"Decoded more frames than declared for {ident}")
            if pending_write is not None:
                stage_started = time.perf_counter()
                pending_write.result()
                timing["writer_wait"] += time.perf_counter() - stage_started
            begin = processed
            processed += len(rows)
            pending_write = writer.submit(
                write_batch,
                arrays,
                rows,
                begin,
                selected_indices,
                source_fps,
                target_fps,
                baseline,
            )
            if processed % (args.batch_size * 100) == 0:
                print(f"    {ident} {processed}/{selected_frames}", flush=True)
            del rgb_nchw, tensors, decoded, selected_decoded
            del selected_indices
        if pending_write is not None:
            stage_started = time.perf_counter()
            pending_write.result()
            timing["writer_wait"] += time.perf_counter() - stage_started
    finally:
        writer.shutdown(wait=True)
    decoder.end()
    if host_decode:
        decoded_source_frames = source_frames
    if decoded_source_frames != source_frames:
        raise RuntimeError(
            f"Decoded source-frame mismatch for {ident}: "
            f"{decoded_source_frames}/{source_frames}"
        )
    if processed != selected_frames:
        raise RuntimeError(
            f"Selected-frame mismatch for {ident}: {processed}/{selected_frames}"
        )
    stage_started = time.perf_counter()
    for data in arrays.values():
        close_memmap(data)
    del arrays
    gc.collect()
    timing["finalize_arrays"] = time.perf_counter() - stage_started
    stage_started = time.perf_counter()
    add_temporal_columns(partial, processed, target_fps)
    timing["temporal_columns"] = time.perf_counter() - stage_started
    def count(name: str) -> int:
        data = np.load(partial / f"{name}.npy", mmap_mode="r")
        result = int(data.sum())
        close_memmap(data)
        return result
    summary = {
        "id": ident, "path": item["path"], "signature": item["signature"],
        "fps": target_fps, "source_fps": source_fps,
        "target_fps": target_fps,
        "sampling": "nearest source frame for each target timeline frame",
        "decode_backend": (
            "windows_intel_d3d11va"
            if args.igpu_decode
            else "ffmpeg_cpu_i420" if args.software_decode else "nvdec"
        ),
        "source_frames": source_frames,
        "decoded_source_frames": decoded_source_frames,
        "selected_source_frames": selected_frames,
        "processed_frames": processed,
        "all_source_frames_processed": True, "inference_geometry": list(geometry or (0, 0)),
        "face_detected_frames": count("face_detected"),
        "pose_detected_frames": count("pose_detected"),
        "left_hand_detected_frames": count("left_hand_detected"),
        "right_hand_detected_frames": count("right_hand_detected"),
        "inference_seconds": infer_seconds,
        "inference_fps": processed / infer_seconds,
        "timing_seconds": timing,
        "source_wall_seconds": time.perf_counter() - source_started,
    }
    (partial / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    gc.collect()
    shutil.rmtree(final, ignore_errors=True)
    partial.rename(final)
    return {**summary, "state": "done"}


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("catalog", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--calibration-image", type=Path, action="append", default=[])
    parser.add_argument(
        "--baseline-json",
        type=Path,
        help="Reuse an already solved calibration baseline instead of calibration images.",
    )
    parser.add_argument(
        "--fused-only",
        action="store_true",
        help="Load only fixed-geometry fused engines; fail on unsupported geometry.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--decode-buffer", type=int, default=48)
    parser.add_argument(
        "--target-fps",
        type=float,
        help="Sample source frames on this output timeline without video re-encoding.",
    )
    parser.add_argument("--native-nv12", action="store_true")
    parser.add_argument(
        "--software-decode", action="store_true",
        help="Decode on CPU with FFmpeg and upload only exact target frames.",
    )
    parser.add_argument(
        "--igpu-decode", action="store_true",
        help="Decode with the Windows Intel iGPU and pipe selected NV12 frames to WSL.",
    )
    parser.add_argument(
        "--windows-ffmpeg", type=Path,
        help="WSL path to a Windows ffmpeg.exe with D3D11VA support.",
    )
    parser.add_argument("--software-decode-threads", type=int, default=16)
    parser.add_argument(
        "--cpu-affinity",
        help="Linux CPU list for this worker and its FFmpeg child, e.g. 16-31.",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--face-threshold", type=float, default=0.5)
    parser.add_argument("--hand-threshold", type=float, default=0.5)
    model = Path("models/mediapipe")
    parser.add_argument("--face-detector-engine", type=Path, default=model / "trt/face_detector_fp16_b128.engine")
    parser.add_argument("--face-landmarker-engine", type=Path, default=model / "trt/face_landmarks_detector_fp16_b128.engine")
    parser.add_argument("--pose-landmarker-engine", type=Path, default=model / "trt/pose_landmarks_core_fp16_b64.engine")
    parser.add_argument("--hand-landmarker-engine", type=Path, default=model / "trt/hand_landmarks_detector_fp16_b128.engine")
    parser.add_argument("--face-geometry-metadata", type=Path, default=model / "face_geometry_procrustes.npz")
    parser.add_argument(
        "--fused-engine",
        type=Path,
        action="append",
        default=[],
        help="Fixed-batch fused engine; repeat for each source geometry.",
    )
    args = parser.parse_args()
    if args.target_fps is not None and args.target_fps <= 0:
        parser.error("--target-fps must be > 0")
    if args.batch_size > 64:
        parser.error("The pose engine profile supports batch <= 64")
    if args.software_decode_threads < 1:
        parser.error("--software-decode-threads must be >= 1")
    if args.software_decode and args.igpu_decode:
        parser.error("Choose only one of --software-decode and --igpu-decode")
    if (args.software_decode or args.igpu_decode) and not args.native_nv12:
        args.native_nv12 = True
    if args.igpu_decode:
        args.windows_ffmpeg = find_windows_ffmpeg(args.windows_ffmpeg)
    if args.cpu_affinity:
        cpus: set[int] = set()
        for part in args.cpu_affinity.split(","):
            bounds = [int(value) for value in part.split("-", 1)]
            cpus.update(range(bounds[0], bounds[-1] + 1))
        os.sched_setaffinity(0, cpus)
    items = json.loads(args.catalog.read_text())
    if args.limit:
        items = items[:args.limit]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "sources").mkdir(exist_ok=True)
    if args.baseline_json:
        baseline = json.loads(args.baseline_json.read_text())
    else:
        if not args.calibration_image:
            parser.error("Supply --baseline-json or at least one --calibration-image")
        baseline = None
    if args.fused_only:
        if not args.fused_engine:
            parser.error("--fused-only requires at least one --fused-engine")
        legacy_pipeline = None
        if baseline is None:
            parser.error("--fused-only requires --baseline-json")
    else:
        legacy_pipeline = HolisticTrt(args)
        if baseline is None:
            baseline = calibration_baseline(legacy_pipeline, args.calibration_image)
    if args.fused_engine and (
        args.face_threshold != 0.5 or args.hand_threshold != 0.5
    ):
        parser.error("Fused engines currently embed face/hand thresholds of 0.5")
    pipeline = GeometryHolisticTrt(
        legacy_pipeline, args.fused_engine, args.face_geometry_metadata
    )
    (args.output_dir / "calibration.json").write_text(json.dumps(baseline, indent=2) + "\n")
    started = time.perf_counter()
    summaries = []
    for index, item in enumerate(items, 1):
        summary = solve_source(pipeline, item, baseline, args)
        summaries.append(summary)
        print(
            f"[{index:03d}/{len(items):03d}] {item['id']} {summary['state']} "
            f"frames={summary['processed_frames']} infer_fps={summary['inference_fps']:.1f}", flush=True,
        )
    elapsed = time.perf_counter() - started
    total_frames = sum(item["processed_frames"] for item in summaries)
    aggregate = {
        "backend": "TensorRT 11 FP16",
        "fused_engines": [str(path) for path in args.fused_engine],
        "batch_size": args.batch_size,
        "target_fps": args.target_fps,
        "sampling": (
            "every decoded source frame"
            if args.target_fps is None
            else f"nearest source frame on a {args.target_fps:g} fps timeline"
        ),
        "cache_key": "source path + size + mtime + target_fps",
        "sources": len(summaries), "frames": total_frames, "wall_seconds": elapsed,
        "wall_fps": total_frames / elapsed if elapsed else None,
        "all_source_frames_processed": all(item["all_source_frames_processed"] for item in summaries),
    }
    (args.output_dir / "source_summaries.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2) + "\n")
    write_csv(args.output_dir / "source_summaries.csv", summaries)
    (args.output_dir / "aggregate.json").write_text(json.dumps(aggregate, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(aggregate, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
