#!/usr/bin/env python3
"""Export a fixed-geometry single-person Holistic graph as one ONNX model.

The graph keeps detector decode, MediaPipe-style weighted suppression, rotated
face/body/hand crops, and all landmark networks on the GPU.  It intentionally
uses fixed batch and image geometry so TensorRT can optimize the whole graph
without dynamic-shape synchronization points.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import onnx
from onnx import numpy_helper
from onnx2torch import convert
import torch
import torch.nn.functional as F

from gaze_iris_audit_trt import FACE_ANCHORS


FACE_SIZE = 128
FACE_LANDMARK_SIZE = 256
POSE_SIZE = 256
HAND_SIZE = 224


def probability_tensor(value: torch.Tensor) -> torch.Tensor:
    return torch.where(
        (value >= 0.0) & (value <= 1.0), value, torch.sigmoid(value)
    )


def crop_rois(
    images: torch.Tensor, rois: torch.Tensor, size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply fixed-size rotated crops and return crop-to-image transforms."""
    _, _, height, width = images.shape
    cx, cy, side, angle = rois.float().unbind(1)
    radians = angle * (math.pi / 180.0)
    cosine, sine = torch.cos(radians), torch.sin(radians)
    roi_scale = side / size
    a = cosine * roi_scale
    b = -sine * roi_scale
    c = sine * roi_scale
    d = cosine * roi_scale
    half = size / 2.0
    tx = cx - a * half - b * half
    ty = cy - c * half - d * half
    inverse = torch.stack(
        (torch.stack((a, b, tx), dim=1), torch.stack((c, d, ty), dim=1)),
        dim=1,
    )

    norm_to_crop = torch.tensor(
        [
            [(size - 1) / 2, 0, (size - 1) / 2],
            [0, (size - 1) / 2, (size - 1) / 2],
            [0, 0, 1],
        ],
        dtype=torch.float32,
        device=images.device,
    )
    image_to_norm = torch.tensor(
        [
            [2 / max(width - 1, 1), 0, -1],
            [0, 2 / max(height - 1, 1), -1],
            [0, 0, 1],
        ],
        dtype=torch.float32,
        device=images.device,
    )
    bottom = inverse.new_tensor((0, 0, 1)).view(1, 1, 3).expand(
        rois.shape[0], -1, -1
    )
    affine = torch.cat((inverse, bottom), dim=1)
    theta = (image_to_norm[None] @ affine @ norm_to_crop[None])[:, :2].to(
        images.dtype
    )

    # Reproduce affine_grid explicitly using normalized output coordinates.
    # Keeping theta and the coordinate grid in FP16 also preserves the exact
    # rounding behavior of the existing GPU path while avoiding ONNX AffineGrid.
    axis = torch.linspace(
        -1, 1, size, dtype=images.dtype, device=images.device
    )
    grid_y, grid_x = torch.meshgrid(axis, axis, indexing="ij")
    normalized_crop = torch.stack(
        (
            grid_x.reshape(-1),
            grid_y.reshape(-1),
            torch.ones(size * size, dtype=images.dtype, device=images.device),
        ),
        dim=1,
    )
    grid = (normalized_crop[None] @ theta.transpose(1, 2)).reshape(
        rois.shape[0], size, size, 2
    )
    crops = F.grid_sample(
        images,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    return crops, inverse


class FusedHolistic(torch.nn.Module):
    """Single-person full-frame Holistic graph with fixed batch/geometry."""

    def __init__(
        self, model_root: Path, height: int, width: int, include_hands: bool = True
    ):
        super().__init__()
        onnx_root = model_root / "onnx_dynamic"
        self.face_detector = convert(onnx.load(onnx_root / "face_detector_fp16.onnx"))
        self.face_landmarker = convert(
            onnx.load(onnx_root / "face_landmarks_detector_fp16.onnx")
        )
        self.pose_landmarker = convert(
            onnx.load(onnx_root / "pose_landmarks_core_fp16.onnx")
        )
        self.include_hands = include_hands
        if include_hands:
            self.hand_landmarker = convert(
                onnx.load(onnx_root / "hand_landmarks_detector_fp16.onnx")
            )
        self.height = height
        self.width = width
        self.register_buffer(
            "face_anchors", torch.from_numpy(FACE_ANCHORS).float(), persistent=True
        )

    def decode_face(
        self, regressors: torch.Tensor, logits: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = regressors.shape[0]
        scores = probability_tensor(logits.reshape(batch, -1).float())
        raw = regressors.float()
        centers = raw[..., :2] / FACE_SIZE + self.face_anchors[None]
        sizes = raw[..., 2:4] / FACE_SIZE
        keypoints = (
            raw[..., 4:16].reshape(batch, -1, 6, 2) / FACE_SIZE
            + self.face_anchors[None, :, None]
        )
        payload = torch.cat((centers, sizes, keypoints.flatten(2)), dim=2)
        boxes = torch.cat((centers - sizes / 2, centers + sizes / 2), dim=2)

        threshold_mask = scores >= 0.5
        valid = threshold_mask.any(1)
        primary_indices = scores.masked_fill(~threshold_mask, -torch.inf).argmax(1)
        batch_indices = torch.arange(batch, device=raw.device)
        primary_boxes = boxes[batch_indices, primary_indices]
        left_top = torch.maximum(boxes[:, :, :2], primary_boxes[:, None, :2])
        right_bottom = torch.minimum(boxes[:, :, 2:], primary_boxes[:, None, 2:])
        intersection = (right_bottom - left_top).clamp_min(0).prod(2)
        areas = (boxes[:, :, 2:] - boxes[:, :, :2]).clamp_min(0).prod(2)
        primary_areas = (
            primary_boxes[:, 2:] - primary_boxes[:, :2]
        ).clamp_min(0).prod(1)
        iou = intersection / (
            areas + primary_areas[:, None] - intersection
        ).clamp_min(1e-9)
        group = threshold_mask & (iou > 0.3)
        weights = scores * group
        blended = (payload * weights[:, :, None]).sum(1) / weights.sum(
            1, keepdim=True
        ).clamp_min(1e-9)

        scale = FACE_SIZE / max(self.height, self.width)
        resized_h = max(1, round(self.height * scale))
        resized_w = max(1, round(self.width * scale))
        pad_y = (FACE_SIZE - resized_h) // 2
        pad_x = (FACE_SIZE - resized_w) // 2
        center_pixels = (
            blended[:, :2] * FACE_SIZE
            - blended.new_tensor((pad_x, pad_y))
        ) / scale
        half_pixels = blended[:, 2:4] * FACE_SIZE / scale / 2
        output_boxes = torch.cat(
            (center_pixels - half_pixels, center_pixels + half_pixels), dim=1
        )
        output_keypoints = (
            blended[:, 4:].reshape(batch, 6, 2) * FACE_SIZE
            - blended.new_tensor((pad_x, pad_y))
        ) / scale

        # A fixed fallback keeps the downstream graph finite on detector misses;
        # the valid mask still tells the caller to discard those predictions.
        fallback_box = output_boxes.new_tensor(
            [
                self.width * 0.3,
                self.height * 0.2,
                self.width * 0.7,
                self.height * 0.6,
            ]
        )
        fallback_keypoints = output_keypoints.new_tensor(
            [
                [self.width * 0.43, self.height * 0.34],
                [self.width * 0.57, self.height * 0.34],
                [self.width * 0.50, self.height * 0.42],
                [self.width * 0.50, self.height * 0.50],
                [self.width * 0.37, self.height * 0.43],
                [self.width * 0.63, self.height * 0.43],
            ]
        )
        output_boxes = torch.where(valid[:, None], output_boxes, fallback_box)
        output_keypoints = torch.where(
            valid[:, None, None], output_keypoints, fallback_keypoints
        )
        return output_boxes, output_keypoints, valid

    def forward(self, rgb_nchw: torch.Tensor) -> tuple[torch.Tensor, ...]:
        batch = rgb_nchw.shape[0]
        scale = FACE_SIZE / max(self.height, self.width)
        resized_h = max(1, round(self.height * scale))
        resized_w = max(1, round(self.width * scale))
        face_pad_y = (FACE_SIZE - resized_h) // 2
        face_pad_x = (FACE_SIZE - resized_w) // 2
        resized = F.interpolate(
            rgb_nchw,
            (resized_h, resized_w),
            mode="bilinear",
            align_corners=False,
        )
        face_canvas = F.pad(
            resized,
            (
                face_pad_x,
                FACE_SIZE - resized_w - face_pad_x,
                face_pad_y,
                FACE_SIZE - resized_h - face_pad_y,
            ),
        )
        detector_input = (
            face_canvas / 127.5 - 1.0
        ).permute(0, 2, 3, 1).contiguous()
        detector_outputs = self.face_detector(detector_input)
        boxes, keypoints, face_valid = self.decode_face(
            detector_outputs[0], detector_outputs[1]
        )

        eye_delta = keypoints[:, 1] - keypoints[:, 0]
        face_angles = torch.atan2(eye_delta[:, 1], eye_delta[:, 0]) * (
            180.0 / math.pi
        )
        face_sides = 1.5 * torch.maximum(
            boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]
        )
        face_centers = (boxes[:, :2] + boxes[:, 2:]) / 2
        face_rois = torch.cat(
            (face_centers, face_sides[:, None], face_angles[:, None]), dim=1
        )

        pose_sides = torch.maximum(
            torch.full_like(boxes[:, 0], self.width * 0.95),
            (boxes[:, 3] - boxes[:, 1]) * 5.0,
        )
        pose_cx = (boxes[:, 0] + boxes[:, 2]) / 2
        pose_cy = (boxes[:, 1] + boxes[:, 3]) / 2 + pose_sides * 0.25
        pose_rois = torch.stack(
            (pose_cx, pose_cy, pose_sides, torch.zeros_like(pose_cx)), dim=1
        )
        fallback_pose = pose_rois.new_tensor(
            [self.width / 2, self.height / 2, max(self.width, self.height), 0]
        )
        pose_rois = torch.where(face_valid[:, None], pose_rois, fallback_pose)

        face_crops, face_inverse = crop_rois(
            rgb_nchw, face_rois, FACE_LANDMARK_SIZE
        )
        pose_crops, pose_inverse = crop_rois(rgb_nchw, pose_rois, POSE_SIZE)
        face_outputs = self.face_landmarker(
            (face_crops / 255.0).permute(0, 2, 3, 1).contiguous()
        )
        pose_outputs = self.pose_landmarker(
            (pose_crops / 255.0).permute(0, 2, 3, 1).contiguous()
        )

        face_raw = face_outputs[0].float().reshape(batch, 478, 3)
        face_xy = (
            face_raw[:, :, :2] @ face_inverse[:, :, :2].transpose(1, 2)
            + face_inverse[:, :, 2].unsqueeze(1)
        )
        face_z = face_raw[:, :, 2:3] * (
            face_rois[:, 2, None, None] / FACE_LANDMARK_SIZE
        )
        face_points = torch.cat((face_xy, face_z), dim=2)
        face_score = probability_tensor(face_outputs[1].float().reshape(batch))

        pose_raw = pose_outputs[0].float().reshape(batch, 39, 5)
        pose_xy = (
            pose_raw[:, :, :2] @ pose_inverse[:, :, :2].transpose(1, 2)
            + pose_inverse[:, :, 2].unsqueeze(1)
        )
        pose_z = pose_raw[:, :, 2:3] * (
            pose_rois[:, 2, None, None] / POSE_SIZE
        )
        pose_confidence = probability_tensor(pose_raw[:, :, 3:5])
        pose_points = torch.cat((pose_xy, pose_z, pose_confidence), dim=2)
        pose_world = pose_outputs[2].float().reshape(batch, 39, 3)
        pose_score = probability_tensor(pose_outputs[1].float().reshape(batch))
        pose_valid = pose_score >= 0.5

        if not self.include_hands:
            return (
                face_points,
                face_score,
                face_valid,
                pose_points[:, :33],
                pose_world[:, :33],
                pose_score,
                pose_valid,
            )

        elbow = pose_points[:, [13, 14]]
        wrist = pose_points[:, [15, 16]]
        pinky = pose_points[:, [17, 18]]
        pointer = pose_points[:, [19, 20]]
        direction = (pinky[:, :, :2] + pointer[:, :, :2]) / 2 - wrist[:, :, :2]
        direction_norm = torch.linalg.vector_norm(direction, dim=2)
        fallback_direction = wrist[:, :, :2] - elbow[:, :, :2]
        direction = torch.where(
            (direction_norm < 4)[:, :, None], fallback_direction, direction
        )
        direction_norm = torch.linalg.vector_norm(direction, dim=2)
        unit = direction / direction_norm.clamp_min(1e-6)[:, :, None]
        arm_length = torch.linalg.vector_norm(
            wrist[:, :, :2] - elbow[:, :, :2], dim=2
        )
        hand_sides = torch.maximum(
            torch.maximum(
                arm_length * 1.1, face_sides[:, None].expand(-1, 2) * 0.32
            ),
            torch.full_like(arm_length, 80.0),
        )
        hand_centers = wrist[:, :, :2] + unit * hand_sides[:, :, None] * 0.28
        hand_angles = torch.atan2(direction[:, :, 1], direction[:, :, 0]) * (
            180.0 / math.pi
        ) + 90.0
        hand_rois = torch.cat(
            (hand_centers, hand_sides[:, :, None], hand_angles[:, :, None]), dim=2
        )
        hand_confidence = torch.minimum(
            torch.amin(wrist[:, :, 3:5], dim=2),
            torch.amin(elbow[:, :, 3:5], dim=2),
        )
        hand_request_valid = (
            pose_valid[:, None] & (hand_confidence >= 0.35) & (direction_norm >= 4)
        )
        fallback_hand = hand_rois.new_tensor(
            [self.width / 2, self.height / 2, 80.0, 0.0]
        )
        hand_rois_by_side = torch.where(
            hand_request_valid[:, :, None], hand_rois, fallback_hand
        )
        left_crops, left_inverse = crop_rois(
            rgb_nchw, hand_rois_by_side[:, 0], HAND_SIZE
        )
        right_crops, right_inverse = crop_rois(
            rgb_nchw, hand_rois_by_side[:, 1], HAND_SIZE
        )
        # Interleave left/right crops per source frame without expanding and
        # materializing two copies of every full-resolution input frame.
        hand_crops = torch.stack((left_crops, right_crops), dim=1).reshape(
            batch * 2, 3, HAND_SIZE, HAND_SIZE
        )
        hand_inverse = torch.stack((left_inverse, right_inverse), dim=1).reshape(
            batch * 2, 2, 3
        )
        hand_rois = hand_rois_by_side.reshape(batch * 2, 4)
        hand_outputs = self.hand_landmarker(
            (hand_crops / 255.0).permute(0, 2, 3, 1).contiguous()
        )
        hand_raw = hand_outputs[0].float().reshape(batch * 2, 21, 3)
        hand_xy = (
            hand_raw[:, :, :2] @ hand_inverse[:, :, :2].transpose(1, 2)
            + hand_inverse[:, :, 2].unsqueeze(1)
        )
        hand_z = hand_raw[:, :, 2:3] * (
            hand_rois[:, 2, None, None] / HAND_SIZE
        )
        hand_points = torch.cat((hand_xy, hand_z), dim=2)
        hand_world = hand_outputs[3].float().reshape(batch, 2, 21, 3)
        hand_score = probability_tensor(
            hand_outputs[1].float().reshape(batch, 2)
        )
        hand_valid = hand_request_valid & (hand_score >= 0.5)

        return (
            face_points,
            face_score,
            face_valid,
            pose_points[:, :33],
            pose_world[:, :33],
            pose_score,
            pose_valid,
            hand_points.reshape(batch, 2, 21, 3),
            hand_world,
            hand_score,
            hand_valid,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, default=Path("models/mediapipe"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--height", type=int, default=1920)
    parser.add_argument("--width", type=int, default=1080)
    parser.add_argument("--without-hands", action="store_true")
    args = parser.parse_args()

    module = FusedHolistic(
        args.model_root,
        args.height,
        args.width,
        include_hands=not args.without_hands,
    ).cuda().eval()
    example = torch.zeros(
        (args.batch_size, 3, args.height, args.width),
        dtype=torch.float16,
        device="cuda",
    )
    output_names = [
        "face_landmarks",
        "face_score",
        "face_valid",
        "pose_landmarks",
        "pose_world_landmarks",
        "pose_score",
        "pose_valid",
    ]
    if not args.without_hands:
        output_names.extend(
            (
                "hand_landmarks",
                "hand_world_landmarks",
                "hand_score",
                "hand_valid",
            )
        )
    with torch.inference_mode():
        traced = torch.jit.trace(module, example, check_trace=False)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        torch.onnx.export(
            traced,
            example,
            str(args.output),
            opset_version=17,
            input_names=["rgb_nchw"],
            output_names=output_names,
            do_constant_folding=True,
            dynamo=False,
        )
    model = onnx.load(args.output)
    # onnx2torch implements a two-axis Squeeze as sort/unbind followed by two
    # scalar squeezes.  The legacy exporter preserves those scalar axes as
    # shape tensors, while TensorRT requires constant Squeeze axes.  The source
    # hand model's axes are [2, 3], applied in descending order.
    squeeze_axes = {
        "/hand_landmarker/model_1/model/global_average_pooling2d/Mean_Squeeze__591/Squeeze_2": 3,
        "/hand_landmarker/model_1/model/global_average_pooling2d/Mean_Squeeze__591/Squeeze_3": 2,
    }
    for node in model.graph.node:
        axis = squeeze_axes.get(node.name)
        if axis is None:
            continue
        name = f"{node.name}/axes_constant"
        model.graph.initializer.append(
            numpy_helper.from_array(
                __import__("numpy").asarray([axis], dtype="int64"), name
            )
        )
        if len(node.input) == 1:
            node.input.append(name)
        else:
            node.input[1] = name
    onnx.checker.check_model(model)
    onnx.save(model, args.output)
    print(
        f"wrote {args.output} nodes={len(model.graph.node)} "
        f"size_mib={args.output.stat().st_size / (1 << 20):.1f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
