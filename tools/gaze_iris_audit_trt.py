#!/usr/bin/env python3
"""Full-frame batched Holistic audit using TensorRT FP16 engines.

The detector and landmark weights come from the local MediaPipe ``.task``
archives.  Raw observations are retained for every decoded source frame.  A
centered robust trace and a one-second P10/P90 envelope are added afterwards;
neither replaces the raw measurements.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import tensorrt as trt
import torch
import torch.nn.functional as F


FACE_SIZE = 128
FACE_LANDMARK_SIZE = 256
POSE_SIZE = 256
HAND_SIZE = 224
FACE_LANDMARKS = 478
POSE_LANDMARKS = 33
RIGHT_EYE = (33, 133, (159, 158, 160), (145, 153, 144), (468, 469, 470, 471, 472))
LEFT_EYE = (362, 263, (386, 385, 387), (374, 380, 373), (473, 474, 475, 476, 477))


def probability(value: np.ndarray | float) -> np.ndarray | float:
    array = np.asarray(value)
    output = np.where(
        (array >= 0.0) & (array <= 1.0),
        array,
        1.0 / (1.0 + np.exp(-np.clip(array, -80.0, 80.0))),
    )
    return float(output) if output.ndim == 0 else output


class TrtModule:
    def __init__(self, engine_path: Path):
        self.logger = trt.Logger(trt.Logger.ERROR)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(engine_path.read_bytes())
        if self.engine is None:
            raise RuntimeError(f"Could not load TensorRT engine: {engine_path}")
        self.context = self.engine.create_execution_context()
        self.input_name = next(
            self.engine.get_tensor_name(index)
            for index in range(self.engine.num_io_tensors)
            if self.engine.get_tensor_mode(self.engine.get_tensor_name(index))
            == trt.TensorIOMode.INPUT
        )

    def __call__(self, tensor: torch.Tensor) -> dict[str, torch.Tensor]:
        tensor = tensor.contiguous()
        if not self.context.set_input_shape(self.input_name, tuple(tensor.shape)):
            raise RuntimeError(f"Rejected input shape {tuple(tensor.shape)}")
        self.context.set_tensor_address(self.input_name, tensor.data_ptr())
        outputs: dict[str, torch.Tensor] = {}
        for index in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(index)
            if name == self.input_name:
                continue
            shape = tuple(self.context.get_tensor_shape(name))
            dtype = {
                trt.DataType.FLOAT: torch.float32,
                trt.DataType.HALF: torch.float16,
                trt.DataType.INT32: torch.int32,
                trt.DataType.INT64: torch.int64,
                trt.DataType.INT8: torch.int8,
                trt.DataType.BOOL: torch.bool,
                trt.DataType.UINT8: torch.uint8,
            }[self.engine.get_tensor_dtype(name)]
            output = torch.empty(shape, dtype=dtype, device=tensor.device)
            outputs[name] = output
            self.context.set_tensor_address(name, output.data_ptr())
        stream = torch.cuda.current_stream()
        if not self.context.execute_async_v3(stream.cuda_stream):
            raise RuntimeError("TensorRT enqueue failed")
        return outputs


def generate_face_anchors() -> np.ndarray:
    anchors = []
    strides = [8, 16, 16, 16]
    index = 0
    while index < len(strides):
        last = index
        while last < len(strides) and strides[last] == strides[index]:
            last += 1
        repeats = 2 * (last - index)
        cells = FACE_SIZE // strides[index]
        for y in range(cells):
            for x in range(cells):
                anchors.extend([((x + 0.5) / cells, (y + 0.5) / cells)] * repeats)
        index = last
    return np.asarray(anchors, dtype=np.float64)


FACE_ANCHORS = generate_face_anchors()


def weighted_nms(detections: np.ndarray, threshold: float = 0.3) -> np.ndarray:
    remaining = detections[np.argsort(-detections[:, 0])]
    output = []
    while len(remaining):
        top = remaining[0]
        x1 = remaining[:, 1] - remaining[:, 3] / 2
        y1 = remaining[:, 2] - remaining[:, 4] / 2
        x2 = remaining[:, 1] + remaining[:, 3] / 2
        y2 = remaining[:, 2] + remaining[:, 4] / 2
        tx1, ty1 = top[1] - top[3] / 2, top[2] - top[4] / 2
        tx2, ty2 = top[1] + top[3] / 2, top[2] + top[4] / 2
        overlap = np.maximum(0, np.minimum(x2, tx2) - np.maximum(x1, tx1)) * np.maximum(
            0, np.minimum(y2, ty2) - np.maximum(y1, ty1)
        )
        union = (x2 - x1) * (y2 - y1) + (tx2 - tx1) * (ty2 - ty1) - overlap
        mask = overlap / np.maximum(union, 1e-9) > threshold
        group = remaining[mask]
        weights = group[:, :1]
        blended = top.copy()
        blended[1:] = (group[:, 1:] * weights).sum(axis=0) / weights.sum()
        output.append(blended)
        remaining = remaining[~mask]
    return np.asarray(output)


def letterbox(
    images: torch.Tensor, size: int, low: float, high: float
) -> tuple[torch.Tensor, float, float, float]:
    """Return NHWC FP16 model input and the pixel-to-model transform."""
    batch, height, width, _ = images.shape
    scale = size / max(height, width)
    resized_h = max(1, round(height * scale))
    resized_w = max(1, round(width * scale))
    nchw = images[..., [2, 1, 0]].permute(0, 3, 1, 2).to(torch.float16)
    resized = F.interpolate(nchw, (resized_h, resized_w), mode="bilinear", align_corners=False)
    canvas = torch.zeros((batch, 3, size, size), dtype=torch.float16, device=images.device)
    y0 = (size - resized_h) // 2
    x0 = (size - resized_w) // 2
    canvas[:, :, y0 : y0 + resized_h, x0 : x0 + resized_w] = resized
    canvas = canvas / 255.0
    if (low, high) == (-1.0, 1.0):
        canvas = canvas * 2.0 - 1.0
    return canvas.permute(0, 2, 3, 1).contiguous(), scale, float(x0), float(y0)


@dataclass(frozen=True)
class Roi:
    cx: float
    cy: float
    side: float
    angle: float


def roi_from_face(box: np.ndarray, keypoints: np.ndarray, margin: float = 0.25) -> Roi:
    x1, y1, x2, y2 = (float(value) for value in box)
    delta = keypoints[1] - keypoints[0]
    angle = math.degrees(math.atan2(float(delta[1]), float(delta[0])))
    return Roi((x1 + x2) / 2, (y1 + y2) / 2, (1 + 2 * margin) * max(x2 - x1, y2 - y1), angle)


def roi_matrices(rois: Iterable[Roi], size: int) -> tuple[np.ndarray, np.ndarray]:
    forward, inverse = [], []
    for roi in rois:
        matrix = cv2.getRotationMatrix2D((roi.cx, roi.cy), roi.angle, size / roi.side)
        matrix[0, 2] += size / 2 - roi.cx
        matrix[1, 2] += size / 2 - roi.cy
        forward.append(matrix)
        inverse.append(cv2.invertAffineTransform(matrix))
    return np.asarray(forward), np.asarray(inverse)


def crop_rois(images: torch.Tensor, rois: list[Roi], size: int) -> tuple[torch.Tensor, np.ndarray]:
    """Batched GPU equivalent of MediaPipe's one-pass affine ROI sampler."""
    _, _, height, width = images.shape
    _, inverses = roi_matrices(rois, size)
    norm_to_crop = np.array(
        [[(size - 1) / 2, 0, (size - 1) / 2], [0, (size - 1) / 2, (size - 1) / 2], [0, 0, 1]],
        dtype=np.float32,
    )
    image_to_norm = np.array(
        [[2 / max(width - 1, 1), 0, -1], [0, 2 / max(height - 1, 1), -1], [0, 0, 1]],
        dtype=np.float32,
    )
    theta = []
    for inverse in inverses:
        affine = np.vstack((inverse, (0, 0, 1))).astype(np.float32)
        theta.append((image_to_norm @ affine @ norm_to_crop)[:2])
    theta_tensor = torch.from_numpy(np.asarray(theta)).to(images.device, dtype=images.dtype)
    grid = F.affine_grid(theta_tensor, (len(rois), 3, size, size), align_corners=True)
    crops = F.grid_sample(images, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    return crops, inverses


def decode_face_detection(
    regressors: np.ndarray,
    logits: np.ndarray,
    height: int,
    width: int,
    scale: float,
    pad_x: float,
    pad_y: float,
    threshold: float,
) -> tuple[np.ndarray | None, np.ndarray | None, int]:
    scores = probability(logits.reshape(-1)).astype(np.float64)
    keep = scores >= threshold
    if not np.any(keep):
        return None, None, 0
    raw = regressors[keep].astype(np.float64)
    anchors = FACE_ANCHORS[keep]
    rows = np.empty((len(raw), 17), dtype=np.float64)
    rows[:, 0] = scores[keep]
    rows[:, 1:3] = raw[:, :2] / FACE_SIZE + anchors
    rows[:, 3:5] = raw[:, 2:4] / FACE_SIZE
    for keypoint in range(6):
        begin = 4 + keypoint * 2
        rows[:, 5 + keypoint * 2 : 7 + keypoint * 2] = raw[:, begin : begin + 2] / FACE_SIZE + anchors
    rows = weighted_nms(rows)
    centers = (rows[:, 1:3] * FACE_SIZE - (pad_x, pad_y)) / scale
    halves = rows[:, 3:5] * FACE_SIZE / scale / 2
    boxes = np.concatenate((centers - halves, centers + halves), axis=1)
    keypoints = (rows[:, 5:].reshape(-1, 6, 2) * FACE_SIZE - (pad_x, pad_y)) / scale
    areas = np.maximum(0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0, boxes[:, 3] - boxes[:, 1])
    primary = int(np.argmax(areas))
    box = boxes[primary]
    box[[0, 2]] = np.clip(box[[0, 2]], -0.25 * width, 1.25 * width)
    box[[1, 3]] = np.clip(box[[1, 3]], -0.25 * height, 1.25 * height)
    return box.astype(np.float32), keypoints[primary].astype(np.float32), len(rows)


def decode_face_detection_gpu(
    regressors: torch.Tensor,
    logits: torch.Tensor,
    height: int,
    width: int,
    scale: float,
    pad_x: float,
    pad_y: float,
    threshold: float,
    anchors: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Decode and suppress a batch of BlazeFace outputs without leaving CUDA.

    The audited library is single-person footage.  Build the primary
    MediaPipe weighted-suppression group around each frame's best anchor in one
    batched CUDA pass, including all six detector keypoints.
    """
    batch = regressors.shape[0]
    scores = logits.reshape(batch, -1).float()
    scores = torch.where(
        (scores >= 0.0) & (scores <= 1.0), scores, torch.sigmoid(scores)
    )
    raw = regressors.float()
    centers = raw[..., :2] / FACE_SIZE + anchors[None]
    sizes = raw[..., 2:4] / FACE_SIZE
    keypoints = raw[..., 4:16].reshape(batch, -1, 6, 2) / FACE_SIZE + anchors[None, :, None]
    payload = torch.cat((centers, sizes, keypoints.flatten(2)), dim=2)
    boxes = torch.cat((centers - sizes / 2, centers + sizes / 2), dim=2)
    # The audited library is explicitly single-person. For that case the first
    # MediaPipe weighted-NMS group is the only useful group, so form it for the
    # whole batch at once around each frame's highest-score anchor. This avoids
    # one CUDA NMS launch and one IoU launch per frame.
    mask = scores >= threshold
    valid = mask.any(1)
    primary_indices = scores.masked_fill(~mask, -torch.inf).argmax(1)
    primary_boxes = boxes[torch.arange(batch, device=raw.device), primary_indices]
    left_top = torch.maximum(boxes[:, :, :2], primary_boxes[:, None, :2])
    right_bottom = torch.minimum(boxes[:, :, 2:], primary_boxes[:, None, 2:])
    intersection = (right_bottom - left_top).clamp_min(0).prod(2)
    areas = (boxes[:, :, 2:] - boxes[:, :, :2]).clamp_min(0).prod(2)
    primary_areas = (primary_boxes[:, 2:] - primary_boxes[:, :2]).clamp_min(0).prod(1)
    iou = intersection / (areas + primary_areas[:, None] - intersection).clamp_min(1e-9)
    group = mask & (iou > 0.3)
    weights = scores * group
    blended = (payload * weights[:, :, None]).sum(1) / weights.sum(1, keepdim=True).clamp_min(1e-9)
    center_pixels = (blended[:, :2] * FACE_SIZE - blended.new_tensor((pad_x, pad_y))) / scale
    half_pixels = blended[:, 2:4] * FACE_SIZE / scale / 2
    output_boxes = torch.cat((center_pixels - half_pixels, center_pixels + half_pixels), dim=1)
    output_boxes[:, [0, 2]] = output_boxes[:, [0, 2]].clamp(-0.25 * width, 1.25 * width)
    output_boxes[:, [1, 3]] = output_boxes[:, [1, 3]].clamp(-0.25 * height, 1.25 * height)
    output_keypoints = (
        blended[:, 4:].reshape(batch, 6, 2) * FACE_SIZE
        - blended.new_tensor((pad_x, pad_y))
    ) / scale
    output_boxes = output_boxes.masked_fill(~valid[:, None], torch.nan)
    output_keypoints = output_keypoints.masked_fill(~valid[:, None, None], torch.nan)
    candidate_counts = valid.to(torch.int16)
    return output_boxes, output_keypoints, candidate_counts, valid


def face_rois_gpu(boxes: torch.Tensor, keypoints: torch.Tensor, margin: float = 0.25) -> torch.Tensor:
    delta = keypoints[:, 1] - keypoints[:, 0]
    angles = torch.atan2(delta[:, 1], delta[:, 0]) * (180.0 / math.pi)
    sides = (1 + 2 * margin) * torch.maximum(boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1])
    centers = (boxes[:, :2] + boxes[:, 2:]) / 2
    return torch.cat((centers, sides[:, None], angles[:, None]), dim=1)


def crop_rois_gpu(
    images: torch.Tensor, rois: torch.Tensor, size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotated ROI sampler with all affine construction on CUDA."""
    _, _, height, width = images.shape
    device = images.device
    rois = rois.float()
    cx, cy, side, angle = rois.unbind(1)
    radians = angle * (math.pi / 180.0)
    cosine, sine = torch.cos(radians), torch.sin(radians)
    roi_scale = side / size
    inverse = torch.zeros((len(rois), 2, 3), dtype=torch.float32, device=device)
    inverse[:, 0, 0] = cosine * roi_scale
    inverse[:, 0, 1] = -sine * roi_scale
    inverse[:, 1, 0] = sine * roi_scale
    inverse[:, 1, 1] = cosine * roi_scale
    half = size / 2.0
    inverse[:, 0, 2] = cx - inverse[:, 0, 0] * half - inverse[:, 0, 1] * half
    inverse[:, 1, 2] = cy - inverse[:, 1, 0] * half - inverse[:, 1, 1] * half

    norm_to_crop = torch.tensor(
        [[(size - 1) / 2, 0, (size - 1) / 2], [0, (size - 1) / 2, (size - 1) / 2], [0, 0, 1]],
        dtype=torch.float32, device=device,
    )
    image_to_norm = torch.tensor(
        [[2 / max(width - 1, 1), 0, -1], [0, 2 / max(height - 1, 1), -1], [0, 0, 1]],
        dtype=torch.float32, device=device,
    )
    affine = torch.cat(
        (inverse, torch.tensor([0, 0, 1], dtype=torch.float32, device=device).view(1, 1, 3).expand(len(rois), -1, -1)),
        dim=1,
    )
    theta = (image_to_norm[None] @ affine @ norm_to_crop[None])[:, :2].to(images.dtype)
    grid = F.affine_grid(theta, (len(rois), 3, size, size), align_corners=True)
    crops = F.grid_sample(images, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    return crops, inverse


def normalized_eye(landmarks: np.ndarray, spec: tuple) -> dict | None:
    left_index, right_index, upper_indices, lower_indices, iris_indices = spec
    screen_left, screen_right = landmarks[left_index, :2], landmarks[right_index, :2]
    upper = landmarks[list(upper_indices), :2].mean(axis=0)
    lower = landmarks[list(lower_indices), :2].mean(axis=0)
    iris = landmarks[list(iris_indices), :2].mean(axis=0)
    axis = screen_right - screen_left
    width = float(np.linalg.norm(axis))
    if width < 2:
        return None
    x_axis = axis / width
    y_axis = np.array((-x_axis[1], x_axis[0]))
    if float(np.dot(lower - upper, y_axis)) < 0:
        y_axis *= -1
    aperture = float(np.dot(lower - upper, y_axis))
    if aperture < 0.5:
        return None
    center = (screen_left + screen_right) / 2
    return {
        "horizontal": float(np.dot(iris - center, x_axis) / width),
        "vertical": float(np.dot(iris - center, y_axis) / width),
        "openness": aperture / width,
        "iris_x": float(iris[0]),
        "iris_y": float(iris[1]),
    }


def normalized_eye_batch(landmarks: np.ndarray, spec: tuple) -> dict[str, np.ndarray]:
    """Vectorized eye coordinates for a batch of pixel-space face landmarks."""
    left_index, right_index, upper_indices, lower_indices, iris_indices = spec
    screen_left = landmarks[:, left_index, :2]
    screen_right = landmarks[:, right_index, :2]
    upper = landmarks[:, list(upper_indices), :2].mean(axis=1)
    lower = landmarks[:, list(lower_indices), :2].mean(axis=1)
    iris = landmarks[:, list(iris_indices), :2].mean(axis=1)
    axis = screen_right - screen_left
    width = np.linalg.norm(axis, axis=1)
    safe_width = np.maximum(width, 1e-12)
    x_axis = axis / safe_width[:, None]
    y_axis = np.stack((-x_axis[:, 1], x_axis[:, 0]), axis=1)
    y_axis[np.sum((lower - upper) * y_axis, axis=1) < 0] *= -1
    aperture = np.sum((lower - upper) * y_axis, axis=1)
    center = (screen_left + screen_right) / 2
    return {
        "valid": (width >= 2) & (aperture >= 0.5),
        "horizontal": np.sum((iris - center) * x_axis, axis=1) / safe_width,
        "vertical": np.sum((iris - center) * y_axis, axis=1) / safe_width,
        "openness": aperture / safe_width,
        "iris_x": iris[:, 0],
        "iris_y": iris[:, 1],
    }


class RigidFaceFit:
    def __init__(self, metadata: Path):
        data = np.load(metadata)
        self.canonical = data["canonical_xyz"].astype(np.float64)
        self.ids = data["basis_ids"].astype(np.int64)
        self.weights = data["basis_weights"].astype(np.float64)

    def __call__(self, landmarks: np.ndarray, width: int) -> dict:
        source = self.canonical[self.ids]
        observed = landmarks[self.ids].astype(np.float64).copy()
        observed[:, 0] = (observed[:, 0] - width / 2) / width
        observed[:, 1] = -observed[:, 1] / width
        observed[:, 2] = -observed[:, 2] / width
        weights = self.weights / self.weights.sum()
        source_center = np.sum(source * weights[:, None], axis=0)
        observed_center = np.sum(observed * weights[:, None], axis=0)
        x = source - source_center
        y = observed - observed_center
        covariance = (x * weights[:, None]).T @ y
        u, singular, vt = np.linalg.svd(covariance)
        rotation = vt.T @ u.T
        if np.linalg.det(rotation) < 0:
            vt[-1] *= -1
            rotation = vt.T @ u.T
        source_energy = float(np.sum(weights[:, None] * x * x))
        scale = float(singular.sum() / max(source_energy, 1e-12))
        fitted = scale * (x @ rotation.T)
        residual = float(np.sqrt(np.sum(weights[:, None] * (fitted - y) ** 2)))
        angles = cv2.RQDecomp3x3(rotation)[0]
        return {
            "head_pitch_deg": float(angles[0]),
            "head_yaw_deg": float(angles[1]),
            "head_roll_deg": float(angles[2]),
            "head_pose_rigidity_error": residual,
            "head_rotation_matrix": rotation.reshape(-1).tolist(),
        }

    def batch(self, landmarks: np.ndarray, width: int) -> list[dict]:
        """Solve the same weighted similarity fit for a batch of faces."""
        if not len(landmarks):
            return []
        source = self.canonical[self.ids]
        observed = landmarks[:, self.ids].astype(np.float64, copy=True)
        observed[:, :, 0] = (observed[:, :, 0] - width / 2) / width
        observed[:, :, 1] = -observed[:, :, 1] / width
        observed[:, :, 2] = -observed[:, :, 2] / width
        weights = self.weights / self.weights.sum()
        source_center = np.sum(source * weights[:, None], axis=0)
        observed_center = np.sum(
            observed * weights[None, :, None], axis=1, keepdims=True
        )
        x = source - source_center
        y = observed - observed_center
        covariance = np.einsum("ki,k,bkj->bij", x, weights, y)
        u, singular, vt = np.linalg.svd(covariance)
        rotation = vt.transpose(0, 2, 1) @ u.transpose(0, 2, 1)
        reflected = np.linalg.det(rotation) < 0
        if np.any(reflected):
            vt[reflected, -1] *= -1
            rotation[reflected] = (
                vt[reflected].transpose(0, 2, 1)
                @ u[reflected].transpose(0, 2, 1)
            )
        source_energy = float(np.sum(weights[:, None] * x * x))
        scale = singular.sum(axis=1) / max(source_energy, 1e-12)
        fitted = scale[:, None, None] * np.einsum(
            "ki,bji->bkj", x, rotation
        )
        residual = np.sqrt(
            np.sum(weights[None, :, None] * (fitted - y) ** 2, axis=(1, 2))
        )
        results = []
        for index in range(len(landmarks)):
            angles = cv2.RQDecomp3x3(rotation[index])[0]
            results.append(
                {
                    "head_pitch_deg": float(angles[0]),
                    "head_yaw_deg": float(angles[1]),
                    "head_roll_deg": float(angles[2]),
                    "head_pose_rigidity_error": float(residual[index]),
                    "head_rotation_matrix": rotation[index].reshape(-1),
                }
            )
        return results


class HolisticTrt:
    def __init__(self, args: argparse.Namespace):
        self.face_detector = TrtModule(args.face_detector_engine)
        self.face_landmarker = TrtModule(args.face_landmarker_engine)
        self.pose_landmarker = TrtModule(args.pose_landmarker_engine)
        self.hand_landmarker = TrtModule(args.hand_landmarker_engine)
        self.rigid_fit = RigidFaceFit(args.face_geometry_metadata)
        self.face_threshold = args.face_threshold
        self.hand_threshold = args.hand_threshold
        self.face_anchors = torch.from_numpy(FACE_ANCHORS).to("cuda", dtype=torch.float32)
        self.face_landmark_stream = torch.cuda.Stream()
        self.pose_landmark_stream = torch.cuda.Stream()

    @torch.inference_mode()
    def infer(self, frames: list[np.ndarray]) -> list[dict]:
        height, width = frames[0].shape[:2]
        if any(frame.shape[:2] != (height, width) for frame in frames):
            raise ValueError("A batch must contain one video geometry")
        host = torch.from_numpy(np.stack(frames)).pin_memory()
        images = host.to("cuda", non_blocking=True)
        rgb_nchw = images[..., [2, 1, 0]].permute(0, 3, 1, 2).to(torch.float16)

        return self.infer_rgb_nchw(rgb_nchw)

    @torch.inference_mode()
    def infer_rgb_nchw(self, rgb_nchw: torch.Tensor) -> list[dict]:
        """Infer directly from a CUDA RGB NCHW tensor.

        This entry point accepts NVDEC/DLPack frames without a CPU round trip or
        an intermediate 1280-pixel resize.
        """
        if rgb_nchw.ndim != 4 or rgb_nchw.shape[1] != 3 or not rgb_nchw.is_cuda:
            raise ValueError("Expected a CUDA RGB NCHW tensor")
        batch_size, _, height, width = rgb_nchw.shape
        rgb_nchw = rgb_nchw.to(torch.float16)

        scale = FACE_SIZE / max(height, width)
        resized_h = max(1, round(height * scale))
        resized_w = max(1, round(width * scale))
        resized = F.interpolate(rgb_nchw, (resized_h, resized_w), mode="bilinear", align_corners=False)
        face_canvas = torch.zeros((batch_size, 3, FACE_SIZE, FACE_SIZE), dtype=torch.float16, device=rgb_nchw.device)
        face_pad_y = (FACE_SIZE - resized_h) // 2
        face_pad_x = (FACE_SIZE - resized_w) // 2
        face_canvas[:, :, face_pad_y : face_pad_y + resized_h, face_pad_x : face_pad_x + resized_w] = resized
        face_input = (face_canvas / 127.5 - 1.0).permute(0, 2, 3, 1).contiguous()
        face_scale = scale
        face_outputs = self.face_detector(face_input)
        boxes_gpu, keypoints_gpu, candidate_counts_gpu, face_valid_gpu = decode_face_detection_gpu(
            face_outputs["regressors"], face_outputs["classificators"],
            height, width, face_scale, face_pad_x, face_pad_y,
            self.face_threshold, self.face_anchors,
        )
        valid_indices_gpu = torch.where(face_valid_gpu)[0]
        valid_count = valid_indices_gpu.numel()
        face_landmarks: dict[int, np.ndarray] = {}
        face_scores: dict[int, float] = {}
        face_rois_tensor = face_rois_gpu(
            boxes_gpu[valid_indices_gpu], keypoints_gpu[valid_indices_gpu]
        ) if valid_count else torch.empty((0, 4), device=rgb_nchw.device)
        face_side_by_frame = torch.full(
            (batch_size,), min(height, width) * 0.25, device=rgb_nchw.device
        )
        if valid_count:
            face_side_by_frame[valid_indices_gpu] = face_rois_tensor[:, 2]

        # Talking-head footage is commonly portrait.  Feeding the whole portrait
        # frame through a square letterbox makes the person unnecessarily small.
        # Seed a square upper-body ROI from the already decoded face box, while
        # retaining a full-frame square fallback when the face detector misses.
        pose_rois_tensor = torch.zeros((batch_size, 4), dtype=torch.float32, device=rgb_nchw.device)
        pose_rois_tensor[:, 0] = width / 2
        pose_rois_tensor[:, 1] = height / 2
        pose_rois_tensor[:, 2] = max(width, height)
        if valid_count:
            valid_boxes = boxes_gpu[valid_indices_gpu]
            pose_sides = torch.maximum(
                torch.full_like(valid_boxes[:, 0], width * 0.95),
                (valid_boxes[:, 3] - valid_boxes[:, 1]) * 5.0,
            )
            pose_rois_tensor[valid_indices_gpu, 0] = (valid_boxes[:, 0] + valid_boxes[:, 2]) / 2
            pose_rois_tensor[valid_indices_gpu, 1] = (
                (valid_boxes[:, 1] + valid_boxes[:, 3]) / 2 + pose_sides * 0.25
            )
            pose_rois_tensor[valid_indices_gpu, 2] = pose_sides

        # Once face detection supplies the ROIs, face and body landmarks are
        # independent. Enqueue them on separate streams and join only before
        # their small outputs are copied to the CPU.
        ready = torch.cuda.Event()
        ready.record(torch.cuda.current_stream())
        face_outputs = face_inverses = None
        if valid_count:
            with torch.cuda.stream(self.face_landmark_stream):
                self.face_landmark_stream.wait_event(ready)
                face_crops, face_inverses = crop_rois_gpu(
                    rgb_nchw[valid_indices_gpu], face_rois_tensor, FACE_LANDMARK_SIZE
                )
                face_input = (face_crops / 255.0).permute(0, 2, 3, 1).contiguous()
                face_outputs = self.face_landmarker(face_input)
        with torch.cuda.stream(self.pose_landmark_stream):
            self.pose_landmark_stream.wait_event(ready)
            pose_crops, pose_inverses = crop_rois_gpu(rgb_nchw, pose_rois_tensor, POSE_SIZE)
            pose_input = (pose_crops / 255.0).permute(0, 2, 3, 1).contiguous()
            pose_outputs = self.pose_landmarker(pose_input)
        current_stream = torch.cuda.current_stream()
        if valid_count:
            current_stream.wait_stream(self.face_landmark_stream)
        current_stream.wait_stream(self.pose_landmark_stream)

        if valid_count:
            assert face_outputs is not None and face_inverses is not None
            raw_gpu = face_outputs["Identity"].float().reshape(-1, FACE_LANDMARKS, 3)
            transformed_gpu = raw_gpu.clone()
            transformed_gpu[:, :, :2] = (
                raw_gpu[:, :, :2] @ face_inverses[:, :, :2].transpose(1, 2)
                + face_inverses[:, :, 2].unsqueeze(1)
            )
            transformed_gpu[:, :, 2] *= face_rois_tensor[:, 2, None] / FACE_LANDMARK_SIZE
            raw = transformed_gpu.cpu().numpy()
            score_values = face_outputs["Identity_1"].float().cpu().numpy().reshape(-1)
            valid_indices = valid_indices_gpu.cpu().tolist()
            for local_index, frame_index in enumerate(valid_indices):
                face_landmarks[frame_index] = raw[local_index]
                face_scores[frame_index] = float(probability(score_values[local_index]))
        pose_raw_gpu = pose_outputs["Identity"].float().reshape(-1, 39, 5)
        transformed_pose_gpu = pose_raw_gpu.clone()
        transformed_pose_gpu[:, :, :2] = (
            pose_raw_gpu[:, :, :2] @ pose_inverses[:, :, :2].transpose(1, 2)
            + pose_inverses[:, :, 2].unsqueeze(1)
        )
        transformed_pose_gpu[:, :, 2] *= pose_rois_tensor[:, 2, None] / POSE_SIZE
        transformed_pose_gpu[:, :, 3:5] = torch.where(
            (transformed_pose_gpu[:, :, 3:5] >= 0) & (transformed_pose_gpu[:, :, 3:5] <= 1),
            transformed_pose_gpu[:, :, 3:5],
            torch.sigmoid(transformed_pose_gpu[:, :, 3:5]),
        )
        pose_raw = transformed_pose_gpu.cpu().numpy()
        pose_world = pose_outputs["Identity_4"].float().cpu().numpy().reshape(-1, 39, 3)
        pose_scores = probability(pose_outputs["Identity_1"].float().cpu().numpy().reshape(-1))
        candidate_counts = candidate_counts_gpu.cpu().numpy()
        face_side_cpu = face_side_by_frame.cpu().numpy()

        results = []
        hand_requests: list[tuple[int, str, Roi]] = []
        pose_points_by_frame: list[np.ndarray | None] = []
        for index in range(batch_size):
            pose_points = pose_raw[index].astype(np.float64)
            if pose_scores[index] < 0.5:
                pose_points_by_frame.append(None)
                continue
            pose_points_by_frame.append(pose_points)
            face_side = float(face_side_cpu[index])
            for side, ids in (("left", (13, 15, 17, 19)), ("right", (14, 16, 18, 20))):
                elbow, wrist, pinky, pointer = pose_points[list(ids)]
                if min(wrist[3], wrist[4], elbow[3], elbow[4]) < 0.35:
                    continue
                finger = (pinky[:2] + pointer[:2]) / 2
                direction = finger - wrist[:2]
                if np.linalg.norm(direction) < 4:
                    direction = wrist[:2] - elbow[:2]
                norm = float(np.linalg.norm(direction))
                if norm < 4:
                    continue
                unit = direction / norm
                side_length = max(float(np.linalg.norm(wrist[:2] - elbow[:2])) * 1.1, face_side * 0.32, 80.0)
                center = wrist[:2] + unit * side_length * 0.28
                angle = math.degrees(math.atan2(float(direction[1]), float(direction[0]))) + 90.0
                hand_requests.append((index, side, Roi(float(center[0]), float(center[1]), side_length, angle)))

        hands: dict[tuple[int, str], tuple[np.ndarray, np.ndarray, float]] = {}
        if hand_requests:
            source = rgb_nchw[[request[0] for request in hand_requests]]
            rois = [request[2] for request in hand_requests]
            hand_rois_tensor = torch.tensor(
                [[roi.cx, roi.cy, roi.side, roi.angle] for roi in rois],
                dtype=torch.float32,
                device=rgb_nchw.device,
            )
            crops, inverses_gpu = crop_rois_gpu(source, hand_rois_tensor, HAND_SIZE)
            hand_input = (crops / 255.0).permute(0, 2, 3, 1).contiguous()
            outputs = self.hand_landmarker(hand_input)
            raw_gpu = outputs["Identity"].float().reshape(-1, 21, 3)
            transformed_gpu = raw_gpu.clone()
            transformed_gpu[:, :, :2] = (
                raw_gpu[:, :, :2] @ inverses_gpu[:, :, :2].transpose(1, 2)
                + inverses_gpu[:, :, 2].unsqueeze(1)
            )
            transformed_gpu[:, :, 2] *= hand_rois_tensor[:, 2, None] / HAND_SIZE
            raw = transformed_gpu.cpu().numpy()
            world = outputs["Identity_3"].float().cpu().numpy().reshape(-1, 21, 3)
            scores = probability(outputs["Identity_1"].float().cpu().numpy().reshape(-1))
            for local_index, (frame_index, side, roi) in enumerate(hand_requests):
                if scores[local_index] < self.hand_threshold:
                    continue
                hands[(frame_index, side)] = (raw[local_index], world[local_index], float(scores[local_index]))

        for index in range(batch_size):
            item: dict = {
                "detected": index in face_landmarks,
                "face_candidates": int(candidate_counts[index]),
                "pose_detected": pose_points_by_frame[index] is not None,
            }
            landmarks = face_landmarks.get(index)
            if landmarks is not None:
                normalized = landmarks.copy()
                normalized[:, 0] /= width
                normalized[:, 1] /= height
                normalized[:, 2] /= width
                right = normalized_eye(landmarks, RIGHT_EYE)
                left = normalized_eye(landmarks, LEFT_EYE)
                item.update(
                    {
                        "face_score": face_scores[index],
                        "face_landmarks": normalized.tolist(),
                        "right": right,
                        "left": left,
                        **self.rigid_fit(landmarks, width),
                    }
                )
                if right is not None and left is not None:
                    item["eye_horizontal"] = float((right["horizontal"] + left["horizontal"]) / 2)
                    item["eye_vertical"] = float((right["vertical"] + left["vertical"]) / 2)
            pose_points = pose_points_by_frame[index]
            if pose_points is not None:
                normalized_pose = pose_points[:POSE_LANDMARKS].copy()
                normalized_pose[:, 0] /= width
                normalized_pose[:, 1] /= height
                normalized_pose[:, 2] /= width
                world = pose_world[index, :POSE_LANDMARKS]
                item["pose_score"] = float(pose_scores[index])
                item["pose_landmarks"] = normalized_pose.tolist()
                item["pose_world_landmarks"] = world.tolist()
            for side in ("left", "right"):
                hand = hands.get((index, side))
                if hand is None:
                    item[f"{side}_hand_landmarks"] = None
                    item[f"{side}_hand_world_landmarks"] = None
                    continue
                points, world, score = hand
                normalized_hand = points.copy()
                normalized_hand[:, 0] /= width
                normalized_hand[:, 1] /= height
                normalized_hand[:, 2] /= width
                item[f"{side}_hand_score"] = score
                item[f"{side}_hand_landmarks"] = normalized_hand.tolist()
                item[f"{side}_hand_world_landmarks"] = world.tolist()
            results.append(item)
        return results


class FusedHolisticTrt:
    """Fixed-batch fused TensorRT graph with the same row schema as HolisticTrt."""

    def __init__(self, engine_path: Path, face_geometry_metadata: Path):
        self.module = TrtModule(engine_path)
        shape = tuple(self.module.engine.get_tensor_shape(self.module.input_name))
        if len(shape) != 4 or any(value <= 0 for value in shape):
            raise ValueError(f"Fused engine must have one fixed NCHW input: {shape}")
        self.batch_size, channels, self.height, self.width = shape
        if channels != 3:
            raise ValueError(f"Fused engine must accept RGB NCHW input: {shape}")
        self.rigid_fit = RigidFaceFit(face_geometry_metadata)

    @torch.inference_mode()
    def infer_rgb_nchw(self, rgb_nchw: torch.Tensor) -> list[dict]:
        if rgb_nchw.ndim != 4 or rgb_nchw.shape[1] != 3 or not rgb_nchw.is_cuda:
            raise ValueError("Expected a CUDA RGB NCHW tensor")
        count, _, height, width = rgb_nchw.shape
        if (height, width) != (self.height, self.width):
            raise ValueError(
                f"Fused engine geometry is {(self.height, self.width)}, "
                f"received {(height, width)}"
            )
        if count > self.batch_size:
            raise ValueError(
                f"Fused engine batch is {self.batch_size}, received {count}"
            )
        tensor = rgb_nchw.to(torch.float16).contiguous()
        if count < self.batch_size:
            padding = tensor[-1:].expand(self.batch_size - count, -1, -1, -1)
            tensor = torch.cat((tensor, padding), dim=0)
        outputs = self.module(tensor)

        # Schedule every small result transfer before the one synchronization.
        host_outputs: dict[str, torch.Tensor] = {}
        for name, output in outputs.items():
            sliced = output[:count]
            host = torch.empty_like(sliced, device="cpu", pin_memory=True)
            host.copy_(sliced, non_blocking=True)
            host_outputs[name] = host
        torch.cuda.current_stream().synchronize()
        values = {name: tensor.numpy() for name, tensor in host_outputs.items()}

        face_valid = values["face_valid"]
        pose_valid = values["pose_valid"]
        hand_valid = values["hand_valid"]
        face_indices = np.flatnonzero(face_valid)
        rigid_fits = self.rigid_fit.batch(
            values["face_landmarks"][face_indices], width
        )
        rigid_by_index = dict(zip(face_indices.tolist(), rigid_fits))
        face_normalized = values["face_landmarks"].copy()
        face_normalized[:, :, 0] /= width
        face_normalized[:, :, 1] /= height
        face_normalized[:, :, 2] /= width
        right_eyes = normalized_eye_batch(values["face_landmarks"], RIGHT_EYE)
        left_eyes = normalized_eye_batch(values["face_landmarks"], LEFT_EYE)
        pose_normalized = values["pose_landmarks"].copy()
        pose_normalized[:, :, 0] /= width
        pose_normalized[:, :, 1] /= height
        pose_normalized[:, :, 2] /= width
        hand_normalized = values["hand_landmarks"].copy()
        hand_normalized[:, :, :, 0] /= width
        hand_normalized[:, :, :, 1] /= height
        hand_normalized[:, :, :, 2] /= width

        def eye_row(batch: dict[str, np.ndarray], index: int) -> dict | None:
            if not batch["valid"][index]:
                return None
            return {
                name: float(batch[name][index])
                for name in (
                    "horizontal",
                    "vertical",
                    "openness",
                    "iris_x",
                    "iris_y",
                )
            }

        results = []
        for index in range(count):
            item: dict = {
                "detected": bool(face_valid[index]),
                "face_candidates": int(face_valid[index]),
                "pose_detected": bool(pose_valid[index]),
            }
            if face_valid[index]:
                landmarks = values["face_landmarks"][index]
                right = eye_row(right_eyes, index)
                left = eye_row(left_eyes, index)
                item.update(
                    {
                        "face_score": float(values["face_score"][index]),
                        "face_landmarks": face_normalized[index],
                        "right": right,
                        "left": left,
                        **rigid_by_index[index],
                    }
                )
                if right is not None and left is not None:
                    item["eye_horizontal"] = float(
                        (right["horizontal"] + left["horizontal"]) / 2
                    )
                    item["eye_vertical"] = float(
                        (right["vertical"] + left["vertical"]) / 2
                    )
            if pose_valid[index]:
                item["pose_score"] = float(values["pose_score"][index])
                item["pose_landmarks"] = pose_normalized[index]
                item["pose_world_landmarks"] = values[
                    "pose_world_landmarks"
                ][index]
            for side_index, side in enumerate(("left", "right")):
                if not hand_valid[index, side_index]:
                    item[f"{side}_hand_landmarks"] = None
                    item[f"{side}_hand_world_landmarks"] = None
                    continue
                item[f"{side}_hand_score"] = float(
                    values["hand_score"][index, side_index]
                )
                item[f"{side}_hand_landmarks"] = hand_normalized[
                    index, side_index
                ]
                item[f"{side}_hand_world_landmarks"] = values[
                    "hand_world_landmarks"
                ][index, side_index]
            results.append(item)
        return results


def robust_trace(rows: list[dict], field: str, fps: float) -> None:
    values = np.asarray([row.get(field, np.nan) for row in rows], dtype=np.float64)
    smooth_window = max(3, round(fps * 0.375)) | 1
    envelope_window = max(5, round(fps * 1.0)) | 1
    smooth_half, envelope_half = smooth_window // 2, envelope_window // 2
    for index, row in enumerate(rows):
        smooth_slice = values[max(0, index - smooth_half) : index + smooth_half + 1]
        envelope_slice = values[max(0, index - envelope_half) : index + envelope_half + 1]
        smooth_valid = smooth_slice[np.isfinite(smooth_slice)]
        envelope_valid = envelope_slice[np.isfinite(envelope_slice)]
        row[f"{field}_smooth"] = float(np.median(smooth_valid)) if len(smooth_valid) >= 3 else None
        row[f"{field}_p10"] = float(np.percentile(envelope_valid, 10)) if len(envelope_valid) >= 5 else None
        row[f"{field}_p90"] = float(np.percentile(envelope_valid, 90)) if len(envelope_valid) >= 5 else None


def calibration_baseline(pipeline: HolisticTrt, paths: list[Path]) -> dict:
    frames = [cv2.imread(str(path)) for path in paths]
    if any(frame is None for frame in frames):
        raise RuntimeError("Could not read one or more calibration images")
    inferred = pipeline.infer(frames)
    fields = ("eye_horizontal", "eye_vertical", "head_pitch_deg", "head_yaw_deg", "head_roll_deg")
    baseline = {}
    for field in fields:
        values = [row[field] for row in inferred if row.get(field) is not None]
        if not values:
            raise RuntimeError(f"Calibration did not produce {field}")
        baseline[field] = float(np.median(values))
    baseline["samples"] = len(inferred)
    return baseline


def process_video(
    pipeline: HolisticTrt,
    video: Path,
    ident: str,
    baseline: dict,
    batch_size: int,
) -> tuple[list[dict], dict]:
    capture = cv2.VideoCapture(str(video))
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 24.0)
    source_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    rows: list[dict] = []
    batch: list[np.ndarray] = []
    first_index = 0
    infer_seconds = 0.0
    while True:
        ok, frame = capture.read()
        if ok:
            batch.append(frame)
        if batch and (len(batch) == batch_size or not ok):
            started = time.perf_counter()
            inferred = pipeline.infer(batch)
            torch.cuda.synchronize()
            infer_seconds += time.perf_counter() - started
            for offset, row in enumerate(inferred):
                frame_index = first_index + offset
                row["id"] = ident
                row["frame"] = frame_index
                row["time_s"] = frame_index / fps
                if row.get("eye_horizontal") is not None:
                    row["delta_horizontal"] = row["eye_horizontal"] - baseline["eye_horizontal"]
                    row["delta_vertical"] = row["eye_vertical"] - baseline["eye_vertical"]
                for axis in ("pitch", "yaw", "roll"):
                    key = f"head_{axis}_deg"
                    if row.get(key) is not None:
                        row[f"delta_head_{axis}_deg"] = row[key] - baseline[key]
                rows.append(row)
            first_index += len(batch)
            batch = []
        if not ok:
            break
    capture.release()
    for field in (
        "delta_horizontal", "delta_vertical", "delta_head_pitch_deg",
        "delta_head_yaw_deg", "delta_head_roll_deg",
    ):
        robust_trace(rows, field, fps)
    summary = {
        "id": ident,
        "video": str(video),
        "fps": fps,
        "source_frames": source_frames,
        "processed_frames": len(rows),
        "all_source_frames_processed": len(rows) == source_frames,
        "face_detected_frames": sum(bool(row["detected"]) for row in rows),
        "pose_detected_frames": sum(bool(row["pose_detected"]) for row in rows),
        "left_hand_detected_frames": sum(row.get("left_hand_landmarks") is not None for row in rows),
        "right_hand_detected_frames": sum(row.get("right_hand_landmarks") is not None for row in rows),
        "inference_seconds": infer_seconds,
        "end_to_end_inference_fps": len(rows) / infer_seconds if infer_seconds else None,
    }
    return rows, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("selection", type=Path)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--calibration-image", type=Path, action="append", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--face-threshold", type=float, default=0.5)
    parser.add_argument("--hand-threshold", type=float, default=0.5)
    model = Path("models/mediapipe")
    parser.add_argument("--face-detector-engine", type=Path, default=model / "trt/face_detector_fp16_b128.engine")
    parser.add_argument("--face-landmarker-engine", type=Path, default=model / "trt/face_landmarks_detector_fp16_b128.engine")
    parser.add_argument("--pose-landmarker-engine", type=Path, default=model / "trt/pose_landmarks_core_fp16_b64.engine")
    parser.add_argument("--hand-landmarker-engine", type=Path, default=model / "trt/hand_landmarks_detector_fp16_b128.engine")
    parser.add_argument("--face-geometry-metadata", type=Path, default=model / "face_geometry_procrustes.npz")
    args = parser.parse_args()
    if args.batch_size > 64:
        parser.error("The pose engine profile currently supports batch <= 64")

    selection = [row for row in json.loads(args.selection.read_text()) if row.get("split") == "train"]
    if args.limit:
        selection = selection[: args.limit]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pipeline = HolisticTrt(args)
    baseline = calibration_baseline(pipeline, args.calibration_image)
    (args.output_dir / "calibration.json").write_text(json.dumps(baseline, indent=2) + "\n")
    summaries = []
    output_path = args.output_dir / "gaze_frames.jsonl.gz"
    started = time.perf_counter()
    with gzip.open(output_path, "wt", encoding="utf-8", compresslevel=5) as output:
        for index, item in enumerate(selection, 1):
            video = args.dataset_root / "video" / f"{item['id']}.mp4"
            rows, summary = process_video(pipeline, video, item["id"], baseline, args.batch_size)
            for row in rows:
                output.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            summaries.append({**item, "holistic_summary": summary})
            print(
                f"[{index:03d}/{len(selection):03d}] {item['id']} "
                f"frames={summary['processed_frames']} fps={summary['end_to_end_inference_fps']:.1f}",
                flush=True,
            )
    elapsed = time.perf_counter() - started
    total_frames = sum(row["holistic_summary"]["processed_frames"] for row in summaries)
    aggregate = {
        "backend": "TensorRT 11 FP16",
        "batch_size": args.batch_size,
        "sampling": "every decoded source frame",
        "temporal_outputs": {
            "raw": "per-frame observation",
            "smooth": "centered 0.375 second rolling median",
            "envelope": "centered 1.0 second P10/P90",
        },
        "clips": len(summaries),
        "frames": total_frames,
        "wall_seconds": elapsed,
        "wall_fps": total_frames / elapsed if elapsed else None,
        "all_source_frames_processed": all(
            row["holistic_summary"]["all_source_frames_processed"] for row in summaries
        ),
    }
    (args.output_dir / "gaze_summary.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2) + "\n"
    )
    (args.output_dir / "aggregate.json").write_text(
        json.dumps(aggregate, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(aggregate, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
