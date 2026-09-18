#!/usr/bin/env python3
"""Stitch native MediaPipe ONNX networks into a fixed-geometry TRT graph.

Only the detector pre/post-processing, rotated crops, and landmark coordinate
mapping are exported from PyTorch.  The original model graphs are inserted
verbatim so native PRelu and convolution nodes are not expanded by an
ONNX -> PyTorch -> ONNX round trip.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import onnx
from onnx import compose
import onnx_graphsurgeon as gs
import torch
import torch.nn.functional as F

from export_fused_holistic_onnx import (
    FACE_LANDMARK_SIZE,
    FACE_SIZE,
    HAND_SIZE,
    POSE_SIZE,
    crop_rois,
    probability_tensor,
)
from gaze_iris_audit_trt import FACE_ANCHORS


class DetectorPreprocess(torch.nn.Module):
    def __init__(self, height: int, width: int):
        super().__init__()
        self.height = height
        self.width = width

    def forward(self, rgb_nchw: torch.Tensor) -> torch.Tensor:
        scale = FACE_SIZE / max(self.height, self.width)
        resized_h = max(1, round(self.height * scale))
        resized_w = max(1, round(self.width * scale))
        pad_y = (FACE_SIZE - resized_h) // 2
        pad_x = (FACE_SIZE - resized_w) // 2
        resized = F.interpolate(
            rgb_nchw,
            (resized_h, resized_w),
            mode="bilinear",
            align_corners=False,
        )
        canvas = F.pad(
            resized,
            (
                pad_x,
                FACE_SIZE - resized_w - pad_x,
                pad_y,
                FACE_SIZE - resized_h - pad_y,
            ),
        )
        return (canvas / 127.5 - 1.0).permute(0, 2, 3, 1).contiguous()


class DetectorDecodeAndRoi(torch.nn.Module):
    def __init__(self, height: int, width: int):
        super().__init__()
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
            blended[:, :2] * FACE_SIZE - blended.new_tensor((pad_x, pad_y))
        ) / scale
        half_pixels = blended[:, 2:4] * FACE_SIZE / scale / 2
        output_boxes = torch.cat(
            (center_pixels - half_pixels, center_pixels + half_pixels), dim=1
        )
        output_keypoints = (
            blended[:, 4:].reshape(batch, 6, 2) * FACE_SIZE
            - blended.new_tensor((pad_x, pad_y))
        ) / scale

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

    def forward(
        self,
        rgb_nchw: torch.Tensor,
        regressors: torch.Tensor,
        logits: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        boxes, keypoints, face_valid = self.decode_face(regressors, logits)
        eye_delta = keypoints[:, 1] - keypoints[:, 0]
        face_angles = torch.atan2(eye_delta[:, 1], eye_delta[:, 0]) * (
            180.0 / torch.pi
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
        face_input = (face_crops / 255.0).permute(0, 2, 3, 1).contiguous()
        pose_input = (pose_crops / 255.0).permute(0, 2, 3, 1).contiguous()
        return (
            face_input,
            pose_input,
            face_inverse,
            face_rois,
            pose_inverse,
            pose_rois,
            face_valid,
        )


class LandmarkPostprocess(torch.nn.Module):
    def forward(
        self,
        face_raw_input: torch.Tensor,
        face_score_input: torch.Tensor,
        pose_raw_input: torch.Tensor,
        pose_score_input: torch.Tensor,
        pose_world_input: torch.Tensor,
        face_inverse: torch.Tensor,
        face_rois: torch.Tensor,
        pose_inverse: torch.Tensor,
        pose_rois: torch.Tensor,
        face_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        batch = face_raw_input.shape[0]
        face_raw = face_raw_input.float().reshape(batch, 478, 3)
        face_xy = (
            face_raw[:, :, :2] @ face_inverse[:, :, :2].transpose(1, 2)
            + face_inverse[:, :, 2].unsqueeze(1)
        )
        face_z = face_raw[:, :, 2:3] * (
            face_rois[:, 2, None, None] / FACE_LANDMARK_SIZE
        )
        face_points = torch.cat((face_xy, face_z), dim=2)
        face_score = probability_tensor(face_score_input.float().reshape(batch))

        pose_raw = pose_raw_input.float().reshape(batch, 39, 5)
        pose_xy = (
            pose_raw[:, :, :2] @ pose_inverse[:, :, :2].transpose(1, 2)
            + pose_inverse[:, :, 2].unsqueeze(1)
        )
        pose_z = pose_raw[:, :, 2:3] * (
            pose_rois[:, 2, None, None] / POSE_SIZE
        )
        pose_confidence = probability_tensor(pose_raw[:, :, 3:5])
        pose_points = torch.cat((pose_xy, pose_z, pose_confidence), dim=2)
        pose_world = pose_world_input.float().reshape(batch, 39, 3)
        pose_score = probability_tensor(pose_score_input.float().reshape(batch))
        pose_valid = pose_score >= 0.5
        return (
            face_points,
            face_score,
            face_valid,
            pose_points[:, :33],
            pose_world[:, :33],
            pose_score,
            pose_valid,
        )


class HandRoiPreprocess(torch.nn.Module):
    def __init__(self, height: int, width: int):
        super().__init__()
        self.height = height
        self.width = width

    def forward(
        self,
        rgb_nchw: torch.Tensor,
        pose_points: torch.Tensor,
        pose_valid: torch.Tensor,
        face_rois: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        batch = rgb_nchw.shape[0]
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
                arm_length * 1.1,
                face_rois[:, 2, None].expand(-1, 2) * 0.32,
            ),
            torch.full_like(arm_length, 80.0),
        )
        hand_centers = wrist[:, :, :2] + unit * hand_sides[:, :, None] * 0.28
        hand_angles = torch.atan2(direction[:, :, 1], direction[:, :, 0]) * (
            180.0 / torch.pi
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
        hand_crops = torch.stack((left_crops, right_crops), dim=1).reshape(
            batch * 2, 3, HAND_SIZE, HAND_SIZE
        )
        hand_inverse = torch.stack((left_inverse, right_inverse), dim=1).reshape(
            batch * 2, 2, 3
        )
        hand_rois_flat = hand_rois_by_side.reshape(batch * 2, 4)
        hand_input = (hand_crops / 255.0).permute(0, 2, 3, 1).contiguous()
        return hand_input, hand_inverse, hand_rois_flat, hand_request_valid


class HandPostprocess(torch.nn.Module):
    def forward(
        self,
        hand_raw_input: torch.Tensor,
        hand_score_input: torch.Tensor,
        hand_world_input: torch.Tensor,
        hand_inverse: torch.Tensor,
        hand_rois: torch.Tensor,
        hand_request_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        batch = hand_request_valid.shape[0]
        hand_raw = hand_raw_input.float().reshape(batch * 2, 21, 3)
        hand_xy = (
            hand_raw[:, :, :2] @ hand_inverse[:, :, :2].transpose(1, 2)
            + hand_inverse[:, :, 2].unsqueeze(1)
        )
        hand_z = hand_raw[:, :, 2:3] * (
            hand_rois[:, 2, None, None] / HAND_SIZE
        )
        hand_points = torch.cat((hand_xy, hand_z), dim=2).reshape(
            batch, 2, 21, 3
        )
        hand_world = hand_world_input.float().reshape(batch, 2, 21, 3)
        hand_score = probability_tensor(hand_score_input.float().reshape(batch, 2))
        hand_valid = hand_request_valid & (hand_score >= 0.5)
        return hand_points, hand_world, hand_score, hand_valid


def export_glue(
    module: torch.nn.Module,
    examples: tuple[torch.Tensor, ...],
    path: Path,
    input_names: list[str],
    output_names: list[str],
) -> None:
    module = module.cuda().eval()
    path.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        torch.onnx.export(
            module,
            examples,
            str(path),
            opset_version=17,
            input_names=input_names,
            output_names=output_names,
            do_constant_folding=True,
            dynamo=False,
        )
    model = onnx.load(path)
    onnx.checker.check_model(model)


def prefixed_graph(path: Path, prefix: str) -> gs.Graph:
    model = compose.add_prefix(onnx.load(path), prefix)
    return gs.import_onnx(model)


def append_graph(
    destination: gs.Graph,
    path: Path,
    prefix: str,
    inputs: dict[str, gs.Tensor],
) -> dict[str, gs.Tensor]:
    source = prefixed_graph(path, prefix)
    replacements = {prefix + name: tensor for name, tensor in inputs.items()}
    for node in source.nodes:
        for index, tensor in enumerate(node.inputs):
            replacement = replacements.get(tensor.name)
            if replacement is not None:
                node.inputs[index] = replacement
    destination.nodes.extend(source.nodes)
    return {
        tensor.name.removeprefix(prefix): tensor for tensor in source.outputs
    }


def stitch(
    model_root: Path,
    pre_path: Path,
    roi_path: Path,
    post_path: Path,
    output_path: Path,
    hand_pre_path: Path | None = None,
    hand_post_path: Path | None = None,
) -> None:
    native_root = model_root / "onnx_dynamic"
    graph = prefixed_graph(pre_path, "pre/")
    rgb = graph.inputs[0]
    rgb.name = "rgb_nchw"
    pre_outputs = {
        tensor.name.removeprefix("pre/"): tensor for tensor in graph.outputs
    }
    detector = append_graph(
        graph,
        native_root / "face_detector_fp16.onnx",
        "detector/",
        {"input": pre_outputs["detector_input"]},
    )
    roi = append_graph(
        graph,
        roi_path,
        "roi/",
        {
            "rgb_nchw": rgb,
            "regressors": detector["regressors"],
            "classificators": detector["classificators"],
        },
    )
    face = append_graph(
        graph,
        native_root / "face_landmarks_detector_fp16.onnx",
        "face/",
        {"input_12": roi["face_input"]},
    )
    pose = append_graph(
        graph,
        native_root / "pose_landmarks_core_fp16.onnx",
        "pose/",
        {"input_1": roi["pose_input"]},
    )
    post = append_graph(
        graph,
        post_path,
        "post/",
        {
            "face_raw_input": face["Identity"],
            "face_score_input": face["Identity_1"],
            "pose_raw_input": pose["Identity"],
            "pose_score_input": pose["Identity_1"],
            "pose_world_input": pose["Identity_4"],
            "face_inverse": roi["face_inverse"],
            "face_rois": roi["face_rois"],
            "pose_inverse": roi["pose_inverse"],
            "pose_rois": roi["pose_rois"],
            "face_detected_valid": roi["face_valid"],
        },
    )
    main_names = (
        "face_landmarks",
        "face_score",
        "face_valid",
        "pose_landmarks",
        "pose_world_landmarks",
        "pose_score",
        "pose_valid",
    )
    outputs = [post[name] for name in main_names]
    final_names = list(main_names)
    if hand_pre_path is not None and hand_post_path is not None:
        hand_pre = append_graph(
            graph,
            hand_pre_path,
            "hand_pre/",
            {
                "rgb_nchw": rgb,
                "pose_landmarks_input": post["pose_landmarks"],
                "pose_valid_input": post["pose_valid"],
                "face_rois": roi["face_rois"],
            },
        )
        hand = append_graph(
            graph,
            native_root / "hand_landmarks_detector_fp16.onnx",
            "hand/",
            {"input_1": hand_pre["hand_input"]},
        )
        hand_post = append_graph(
            graph,
            hand_post_path,
            "hand_post/",
            {
                "hand_raw_input": hand["Identity"],
                "hand_score_input": hand["Identity_1"],
                "hand_world_input": hand["Identity_3"],
                "hand_inverse": hand_pre["hand_inverse"],
                "hand_rois": hand_pre["hand_rois"],
                "hand_request_valid": hand_pre["hand_request_valid"],
            },
        )
        hand_names = (
            "hand_landmarks",
            "hand_world_landmarks",
            "hand_score",
            "hand_valid",
        )
        outputs.extend(hand_post[name] for name in hand_names)
        final_names.extend(hand_names)
    graph.outputs = outputs
    for tensor, name in zip(graph.outputs, final_names, strict=True):
        tensor.name = name
    graph.cleanup(remove_unused_graph_inputs=True).toposort()
    model = gs.export_onnx(graph)
    model.ir_version = min(model.ir_version, 10)
    onnx.checker.check_model(model)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, output_path)
    print(
        f"wrote {output_path} nodes={len(model.graph.node)} "
        f"size_mib={output_path.stat().st_size / (1 << 20):.1f}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, default=Path("models/mediapipe"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--height", type=int, default=1920)
    parser.add_argument("--width", type=int, default=1080)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--with-hands", action="store_true")
    args = parser.parse_args()

    work_dir = args.work_dir or args.output.parent / ".stitch_glue"
    work_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    rgb = torch.zeros(
        (args.batch_size, 3, args.height, args.width),
        dtype=torch.float16,
        device=device,
    )
    regressors = torch.zeros(
        (args.batch_size, 896, 16), dtype=torch.float16, device=device
    )
    logits = torch.zeros(
        (args.batch_size, 896, 1), dtype=torch.float16, device=device
    )
    pre_path = work_dir / "detector_pre.onnx"
    roi_path = work_dir / "detector_roi.onnx"
    post_path = work_dir / "landmark_post.onnx"
    hand_pre_path = work_dir / "hand_pre.onnx"
    hand_post_path = work_dir / "hand_post.onnx"
    export_glue(
        DetectorPreprocess(args.height, args.width),
        (rgb,),
        pre_path,
        ["rgb_nchw"],
        ["detector_input"],
    )
    export_glue(
        DetectorDecodeAndRoi(args.height, args.width),
        (rgb, regressors, logits),
        roi_path,
        ["rgb_nchw", "regressors", "classificators"],
        [
            "face_input",
            "pose_input",
            "face_inverse",
            "face_rois",
            "pose_inverse",
            "pose_rois",
            "face_valid",
        ],
    )
    float_input = lambda *shape: torch.zeros(  # noqa: E731
        shape, dtype=torch.float16, device=device
    )
    float32_input = lambda *shape: torch.zeros(  # noqa: E731
        shape, dtype=torch.float32, device=device
    )
    bool_input = torch.zeros(args.batch_size, dtype=torch.bool, device=device)
    export_glue(
        LandmarkPostprocess(),
        (
            float_input(args.batch_size, 1, 1, 1434),
            float_input(args.batch_size, 1, 1, 1),
            float_input(args.batch_size, 195),
            float_input(args.batch_size, 1),
            float_input(args.batch_size, 117),
            float32_input(args.batch_size, 2, 3),
            float32_input(args.batch_size, 4),
            float32_input(args.batch_size, 2, 3),
            float32_input(args.batch_size, 4),
            bool_input,
        ),
        post_path,
        [
            "face_raw_input",
            "face_score_input",
            "pose_raw_input",
            "pose_score_input",
            "pose_world_input",
            "face_inverse",
            "face_rois",
            "pose_inverse",
            "pose_rois",
            "face_detected_valid",
        ],
        [
            "face_landmarks",
            "face_score",
            "face_valid",
            "pose_landmarks",
            "pose_world_landmarks",
            "pose_score",
            "pose_valid",
        ],
    )
    if args.with_hands:
        export_glue(
            HandRoiPreprocess(args.height, args.width),
            (
                rgb,
                float32_input(args.batch_size, 33, 5),
                bool_input,
                float32_input(args.batch_size, 4),
            ),
            hand_pre_path,
            [
                "rgb_nchw",
                "pose_landmarks_input",
                "pose_valid_input",
                "face_rois",
            ],
            [
                "hand_input",
                "hand_inverse",
                "hand_rois",
                "hand_request_valid",
            ],
        )
        export_glue(
            HandPostprocess(),
            (
                float_input(args.batch_size * 2, 63),
                float_input(args.batch_size * 2, 1),
                float_input(args.batch_size * 2, 63),
                float32_input(args.batch_size * 2, 2, 3),
                float32_input(args.batch_size * 2, 4),
                torch.zeros(
                    (args.batch_size, 2), dtype=torch.bool, device=device
                ),
            ),
            hand_post_path,
            [
                "hand_raw_input",
                "hand_score_input",
                "hand_world_input",
                "hand_inverse",
                "hand_rois",
                "hand_request_valid",
            ],
            [
                "hand_landmarks",
                "hand_world_landmarks",
                "hand_score",
                "hand_valid",
            ],
        )
    stitch(
        args.model_root,
        pre_path,
        roi_path,
        post_path,
        args.output,
        hand_pre_path if args.with_hands else None,
        hand_post_path if args.with_hands else None,
    )


if __name__ == "__main__":
    main()
