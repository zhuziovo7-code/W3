#!/usr/bin/env python3
"""Hybrid tracked video behavior test with RTMDet ROI heads.

This variant keeps the RTMDet + RTM-Pose + PoseMLP path and replaces the two
YOLO crop classifiers with ROI classification heads trained from RTMDet
features:

1. Person detection.
2. Duplicate/contained bbox removal.
3. Simple IoU tracking with smoothed boxes.
4. RTM-Pose on tracked boxes.
5. RTMDet ROI identity classification.
6. PoseMLP + rules behavior classification.
7. RTMDet ROI behavior classification for students.
8. Weighted fusion of pose/rule and image behavior predictions.
9. Temporal confirmation/recovery for abnormal classes.
10. Annotated video and JSON report output.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import timedelta
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from mmdet.apis import inference_detector, init_detector
from mmdet.utils import register_all_modules as register_mmdet_modules
from mmengine.registry import init_default_scope
from mmpose.apis import inference_topdown, init_model
from mmpose.utils import register_all_modules as register_mmpose_modules
from torchvision.ops import roi_align

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from exam_pose.behavior import build_pose_mlp, flatten_pose, normalize_keypoints_by_bbox


DEFAULT_DET_CONFIG = PROJECT_ROOT / "models/detector/rtmdet_tiny_8xb32-300e_coco.py"
DEFAULT_DET_CHECKPOINT = (
    PROJECT_ROOT
    / "models/detector/rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth"
)
DEFAULT_POSE_CONFIG = (
    PROJECT_ROOT / "models/rtmpose/rtmpose-m_8xb256-420e_coco-256x192.py"
)
DEFAULT_POSE_CHECKPOINT = (
    PROJECT_ROOT
    / "models/rtmpose/rtmpose-m_simcc-coco_pt-aic-coco_420e-256x192-d8dd5ca4_20230127.pth"
)
DEFAULT_BEHAVIOR_CHECKPOINT = PROJECT_ROOT / "models/behavior/pose_mlp_fifth_plus_exch_hardx3.pt"
DEFAULT_ROI_HEAD_CHECKPOINT = (
    PROJECT_ROOT / "models/behavior/rtm_roi_heads_fifth_original_plus_review_correctedx3.pt"
)
RTMDET_MEAN = torch.tensor([103.53, 116.28, 123.675], dtype=torch.float32).view(3, 1, 1)
RTMDET_STD = torch.tensor([57.375, 57.12, 58.395], dtype=torch.float32).view(3, 1, 1)

CLASS_NAMES = {
    "normal": "other",
    "raise_hand": "raise hand",
    "head_turn": "head turn",
    "pass_suspicious_item": "passing object",
    "hand_under_table_look_down": "hand under table",
    "pick_suspicious_item": "picking object",
    "carry_phone": "phone",
    "tear_paper": "tear paper",
    "student_standing": "standing",
    "invigilator": "invigilator",
}
ABNORMAL_LABELS = {
    "raise_hand",
    "head_turn",
    "pass_suspicious_item",
    "hand_under_table_look_down",
    "pick_suspicious_item",
    "carry_phone",
    "tear_paper",
    "student_standing",
}
BEHAVIOR_DECISION_CONFIG = {
    "raise_hand": {
        "temporal_abnormal_n": 10,
        "pose_weight": 1.0,
        "rule_weight": 0.0,
        "image_weight": 0.0,
        "fused_thr": 0.72,
    },
    "head_turn": {
        "temporal_abnormal_n": 6,
        "pose_weight": 1.0,
        "rule_weight": 0.0,
        "image_weight": 0.0,
        "fused_thr": 0.95,
    },
    "pass_suspicious_item": {
        "temporal_abnormal_n": 6,
        "pose_weight": 1.0,
        "rule_weight": 0.0,
        "image_weight": 0.0,
        "fused_thr": 0.60,
    },
    "hand_under_table_look_down": {
        "temporal_abnormal_n": 8,
        "pose_weight": 0.95,
        "rule_weight": 0.0,
        "image_weight": 0.05,
        "fused_thr": 0.75,
    },
    "pick_suspicious_item": {
        "temporal_abnormal_n": 6,
        "pose_weight": 0.9,
        "rule_weight": 0.0,
        "image_weight": 0.1,
        "fused_thr": 0.60,
    },
    "carry_phone": {
        "temporal_abnormal_n": 8,
        "pose_weight": 0.85,
        "rule_weight": 0.05,
        "image_weight": 0.1,
        "fused_thr": 0.85,
    },
    "tear_paper": {
        "temporal_abnormal_n": 8,
        "pose_weight": 0.95,
        "rule_weight": 0.0,
        "image_weight": 0.05,
        "fused_thr": 0.80,
    },
    "student_standing": {
        "temporal_abnormal_n": 6,
        "pose_weight": 1.0,
        "rule_weight": 0.0,
        "image_weight": 0.0,
        "fused_thr": 0.90,
    },
}
COLORS = {
    "normal": (80, 220, 80),
    "raise_hand": (80, 255, 255),
    "head_turn": (0, 220, 255),
    "pass_suspicious_item": (0, 165, 255),
    "hand_under_table_look_down": (0, 0, 255),
    "pick_suspicious_item": (255, 0, 255),
    "carry_phone": (255, 128, 0),
    "tear_paper": (180, 80, 255),
    "student_standing": (80, 160, 255),
    "invigilator": (255, 220, 80),
}

NORMAL_BEHAVIOR_NAMES = {
    "normal",
    "other",
    "others",
    "other_behavior",
    "background",
    "正常",
    "其他",
    "其它",
    "其他行为",
}
IMAGE_BEHAVIOR_LABEL_MAP = {
    "normal": "normal",
    "head_turn": "head_turn",
    "passing_object": "pass_suspicious_item",
    "pass_suspicious_item": "pass_suspicious_item",
    "hand_under_table": "hand_under_table_look_down",
    "hand_under_table_look_down": "hand_under_table_look_down",
    "picking_object": "pick_suspicious_item",
    "pick_suspicious_item": "pick_suspicious_item",
    "phone": "carry_phone",
    "carry_phone": "carry_phone",
    "raise_hand": "raise_hand",
    "standing": "student_standing",
    "student_standing": "student_standing",
    "tearing_paper": "tear_paper",
    "tear_paper": "tear_paper",
}

# Per-class single-frame fusion parameters. Edit this list to decide how much
# each abnormal behavior trusts the PoseMLP prediction versus the geometry rule.
BEHAVIOR_RULE_FUSION = [
    {
        "label": "raise_hand",
        "model_weight": 0.60,
        "rule_weight": 0.40,
        "fused_thr": 0.65,
        "model_override_thr": 0.95,
    },
    {
        "label": "head_turn",
        "model_weight": 0.75,
        "rule_weight": 0.25,
        "fused_thr": 0.70,
        "model_override_thr": 0.95,
    },
    {
        "label": "pass_suspicious_item",
        "model_weight": 0.80,
        "rule_weight": 0.20,
        "fused_thr": 0.72,
        "model_override_thr": 0.95,
    },
    {
        "label": "hand_under_table_look_down",
        "model_weight": 0.75,
        "rule_weight": 0.25,
        "fused_thr": 0.70,
        "model_override_thr": 0.95,
    },
    {
        "label": "pick_suspicious_item",
        "model_weight": 0.75,
        "rule_weight": 0.25,
        "fused_thr": 0.70,
        "model_override_thr": 0.95,
    },
    {
        "label": "carry_phone",
        "model_weight": 0.70,
        "rule_weight": 0.30,
        "fused_thr": 0.70,
        "model_override_thr": 0.95,
    },
    {
        "label": "tear_paper",
        "model_weight": 0.75,
        "rule_weight": 0.25,
        "fused_thr": 0.70,
        "model_override_thr": 0.95,
    },
    {
        "label": "student_standing",
        "model_weight": 0.45,
        "rule_weight": 0.55,
        "fused_thr": 0.75,
        "model_override_thr": 1.01,
    },
]
BEHAVIOR_RULE_FUSION_BY_LABEL = {item["label"]: item for item in BEHAVIOR_RULE_FUSION}

# Compatibility view for reports. The live fusion path reads
# BEHAVIOR_DECISION_CONFIG directly.
HYBRID_BEHAVIOR_FUSION = [
    {
        "label": label,
        "pose_weight": config["pose_weight"],
        "rule_weight": config["rule_weight"],
        "image_weight": config["image_weight"],
        "fused_thr": config["fused_thr"],
    }
    for label, config in BEHAVIOR_DECISION_CONFIG.items()
]

KP_NOSE = 0
KP_LEFT_EYE = 1
KP_RIGHT_EYE = 2
KP_LEFT_EAR = 3
KP_RIGHT_EAR = 4
KP_LEFT_SHOULDER = 5
KP_RIGHT_SHOULDER = 6
KP_LEFT_ELBOW = 7
KP_RIGHT_ELBOW = 8
KP_LEFT_WRIST = 9
KP_RIGHT_WRIST = 10
KP_LEFT_HIP = 11
KP_RIGHT_HIP = 12
KP_LEFT_KNEE = 13
KP_RIGHT_KNEE = 14
KP_LEFT_ANKLE = 15
KP_RIGHT_ANKLE = 16


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RTM-Pose behavior test with bbox de-dup, IoU tracking, and invigilator rules."
    )
    parser.add_argument("--video", type=Path, default=PROJECT_ROOT / "模拟考试1.mp4")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs/visualizations/video_behavior_tracked.mp4",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "outputs/reports/video_behavior_tracked_report.json",
    )
    parser.add_argument("--max-frames", type=int, default=3000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--det-config", type=Path, default=DEFAULT_DET_CONFIG)
    parser.add_argument("--det-checkpoint", type=Path, default=DEFAULT_DET_CHECKPOINT)
    parser.add_argument("--pose-config", type=Path, default=DEFAULT_POSE_CONFIG)
    parser.add_argument("--pose-checkpoint", type=Path, default=DEFAULT_POSE_CHECKPOINT)
    parser.add_argument("--behavior-checkpoint", type=Path, default=DEFAULT_BEHAVIOR_CHECKPOINT)
    parser.add_argument("--roi-head-checkpoint", type=Path, default=DEFAULT_ROI_HEAD_CHECKPOINT)
    parser.add_argument("--roi-size", type=int, default=4)
    parser.add_argument("--no-write-video", action="store_true", help="Run inference and report FPS without writing video.")

    parser.add_argument("--det-score-thr", type=float, default=0.35)
    parser.add_argument("--nms-iou-thr", type=float, default=0.50)
    parser.add_argument("--dedup-iou-thr", type=float, default=0.50)
    parser.add_argument("--dedup-containment-thr", type=float, default=0.75)
    parser.add_argument("--min-box-area", type=float, default=20.0)

    parser.add_argument("--head-turn-thr", type=float, default=0.75)
    parser.add_argument("--passing-object-thr", type=float, default=0.75)
    parser.add_argument("--hand-under-table-thr", type=float, default=0.75)
    parser.add_argument("--picking-object-thr", type=float, default=0.75)
    parser.add_argument("--raise-hand-thr", type=float, default=0.75)
    parser.add_argument("--carry-phone-thr", type=float, default=0.75)
    parser.add_argument("--tear-paper-thr", type=float, default=0.75)
    parser.add_argument("--student-standing-thr", type=float, default=0.75)

    parser.add_argument("--disable-single-frame-rules", action="store_true")
    parser.add_argument("--rule-min-score", type=float, default=0.35)
    parser.add_argument("--rule-keep-model-thr", type=float, default=0.90)
    parser.add_argument("--rule-kp-score-thr", type=float, default=0.25)
    parser.add_argument("--standing-leg-angle-thr", type=float, default=170.0)
    parser.add_argument("--standing-leg-length-ratio-thr", type=float, default=0.38)
    parser.add_argument("--standing-leg-vertical-ratio-thr", type=float, default=0.30)
    parser.add_argument("--standing-height-history", type=int, default=1500)
    parser.add_argument("--standing-height-rise-ratio-thr", type=float, default=1.35)
    parser.add_argument("--standing-height-rise-min-delta", type=float, default=0.08)
    parser.add_argument("--standing-height-min-ratio", type=float, default=0.34)
    parser.add_argument("--track-switch-center-thr", type=float, default=0.25)
    parser.add_argument("--track-switch-height-ratio-thr", type=float, default=1.80)
    parser.add_argument("--crop-padding-ratio", type=float, default=0.05)
    parser.add_argument("--crop-pad-value", type=int, default=114)
    parser.add_argument("--square-crop", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--square-crop-mode",
        choices=["expand_image", "gray_pad"],
        default="expand_image",
    )

    parser.add_argument("--track-iou-thr", type=float, default=0.20)
    parser.add_argument("--track-max-missing", type=int, default=8)
    parser.add_argument("--box-smooth-alpha", type=float, default=0.90)
    parser.add_argument("--id-pool-window", type=int, default=10)
    parser.add_argument("--student-anchor-min-frames", type=int, default=80)
    parser.add_argument("--student-anchor-center-thr", type=float, default=0.20)
    parser.add_argument("--student-anchor-update-alpha", type=float, default=0.98)
    parser.add_argument("--student-unanchored-jump-thr", type=float, default=0.32)
    parser.add_argument("--invigilator-jump-thr", type=float, default=0.38)
    parser.add_argument("--track-height-jump-ratio-thr", type=float, default=2.20)
    parser.add_argument("--temporal-abnormal-n", type=int, default=8)
    parser.add_argument("--temporal-recover-n", type=int, default=12)

    parser.add_argument("--max-invigilators", type=int, default=2)
    parser.add_argument("--inv-score-min", type=float, default=0.45)
    parser.add_argument("--inv-score-alpha", type=float, default=0.90)
    parser.add_argument("--inv-score-decay", type=float, default=0.98)
    parser.add_argument("--inv-slot-keep-bonus", type=float, default=0.15)
    parser.add_argument("--inv-history-bonus-max", type=float, default=0.25)
    parser.add_argument("--inv-history-bonus-frames", type=int, default=300)
    parser.add_argument("--student-suppress-conf", type=float, default=0.90)
    parser.add_argument("--student-suppress-factor", type=float, default=0.50)

    parser.add_argument("--progress-interval", type=int, default=100)
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    requested = torch.device(device)
    if requested.type == "cuda" and not torch.cuda.is_available():
        print(f"CUDA is not available in this process; falling back from {device} to cpu")
        return torch.device("cpu")
    return requested


def format_timestamp(frame_idx: int, fps: float) -> str:
    return str(timedelta(seconds=frame_idx / fps))[:-3]


def clip_box(xyxy, width: int, height: int):
    x1, y1, x2, y2 = [int(round(v)) for v in xyxy]
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(0, min(width - 1, x2))
    y2 = max(0, min(height - 1, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def box_area(box) -> float:
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)


def intersection_area(box1, box2) -> float:
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    return max(0, x2 - x1) * max(0, y2 - y1)


def iou(box1, box2) -> float:
    inter = intersection_area(box1, box2)
    union = box_area(box1) + box_area(box2) - inter
    return inter / union if union > 0 else 0.0


def smooth_box(old_box, new_box, alpha: float):
    alpha = max(0.0, min(1.0, alpha))
    return tuple(
        int(round(alpha * old_value + (1.0 - alpha) * new_value))
        for old_value, new_value in zip(old_box, new_box)
    )


def box_center(box) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return (float(x1 + x2) * 0.5, float(y1 + y2) * 0.5)


def normalized_center_distance(box1, box2, image_width: int = 1280, image_height: int = 720) -> float:
    c1 = box_center(box1)
    c2 = box_center(box2)
    dx = (c1[0] - c2[0]) / max(1.0, float(image_width))
    dy = (c1[1] - c2[1]) / max(1.0, float(image_height))
    return float(np.hypot(dx, dy))


def normalized_center_distance_to_anchor(box, anchor_center, image_width: int = 1280, image_height: int = 720) -> float:
    cx, cy = box_center(box)
    ax, ay = anchor_center
    dx = (cx - float(ax)) / max(1.0, float(image_width))
    dy = (cy - float(ay)) / max(1.0, float(image_height))
    return float(np.hypot(dx, dy))


def box_height_ratio(box1, box2) -> float:
    h1 = max(1.0, float(box1[3] - box1[1]))
    h2 = max(1.0, float(box2[3] - box2[1]))
    return max(h1 / h2, h2 / h1)


def is_duplicate(box, kept_box, args: argparse.Namespace) -> bool:
    area1 = box_area(box)
    area2 = box_area(kept_box)
    if area1 <= 0 or area2 <= 0:
        return False
    if iou(box, kept_box) > args.dedup_iou_thr:
        return True
    containment = intersection_area(box, kept_box) / min(area1, area2)
    return containment > args.dedup_containment_thr


def deduplicate_people(people: list[dict], args: argparse.Namespace) -> list[dict]:
    people = sorted(
        people,
        key=lambda item: (item.get("det_confidence", 0.0), box_area(item["bbox"])),
        reverse=True,
    )
    kept = []
    for person in people:
        if any(is_duplicate(person["bbox"], old["bbox"], args) for old in kept):
            continue
        kept.append(person)
    return kept


def pad_box(box, width: int, height: int, padding_ratio: float):
    x1, y1, x2, y2 = box
    box_w = x2 - x1
    box_h = y2 - y1
    pad_x = box_w * padding_ratio
    pad_y = box_h * padding_ratio
    return clip_box((x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y), width, height)


def crop_box_region(frame, box, square_crop: bool, pad_value: int, square_crop_mode: str):
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    clipped = clip_box((x1, y1, x2, y2), width, height)
    if clipped is None:
        return None
    x1, y1, x2, y2 = clipped
    raw_crop = frame[y1:y2, x1:x2]
    if raw_crop.size == 0:
        return None
    if not square_crop:
        return raw_crop

    if square_crop_mode == "gray_pad":
        crop_h, crop_w = raw_crop.shape[:2]
        side = max(1, max(crop_w, crop_h))
        crop = np.full((side, side, 3), pad_value, dtype=frame.dtype)
        dst_x1 = 0 if crop_w >= crop_h else (side - crop_w) // 2
        dst_y1 = 0 if crop_h >= crop_w else (side - crop_h) // 2
        crop[dst_y1:dst_y1 + crop_h, dst_x1:dst_x1 + crop_w] = raw_crop
        return crop

    box_w = x2 - x1
    box_h = y2 - y1
    side = max(1, max(box_w, box_h))
    if box_w > box_h:
        sx1, sy1 = x1, y1
    elif box_h > box_w:
        cx = (x1 + x2) / 2.0
        sx1, sy1 = int(round(cx - side / 2.0)), y1
    else:
        sx1, sy1 = x1, y1
    sx2, sy2 = sx1 + side, sy1 + side
    src_x1, src_y1 = max(0, sx1), max(0, sy1)
    src_x2, src_y2 = min(width, sx2), min(height, sy2)
    if src_x2 <= src_x1 or src_y2 <= src_y1:
        return None
    crop = np.full((side, side, 3), pad_value, dtype=frame.dtype)
    dst_x1, dst_y1 = src_x1 - sx1, src_y1 - sy1
    crop[dst_y1:dst_y1 + (src_y2 - src_y1), dst_x1:dst_x1 + (src_x2 - src_x1)] = frame[
        src_y1:src_y2, src_x1:src_x2
    ]
    return crop


def letterbox_crop(image, size: int, pad_value: int = 114):
    height, width = image.shape[:2]
    if height <= 0 or width <= 0:
        return None
    scale = min(size / width, size / height)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    top = (size - new_height) // 2
    bottom = size - new_height - top
    left = (size - new_width) // 2
    right = size - new_width - left
    return cv2.copyMakeBorder(
        resized,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=(pad_value, pad_value, pad_value),
    )


def new_track(track_id: int, box, frame_idx: int):
    return {
        "id": track_id,
        "box": box,
        "created_frame": frame_idx,
        "last_seen": frame_idx,
        "abn_streak_label": None,
        "abn_streak_count": 0,
        "confirmed_abn_label": None,
        "confirmed_conf": 0.0,
        "recover_count": 0,
        "inv_score": 0.0,
        "inv_confirmed_frames": 0,
        "height_history": [],
        "seat_anchor_center": None,
        "seat_anchor_box": None,
        "seat_anchor_frames": 0,
        "last_role": "unknown",
    }


class SimpleIoUTracker:
    def __init__(self, iou_thr: float, max_missing: int):
        self.iou_thr = iou_thr
        self.max_missing = max_missing
        self.tracks: dict[int, dict] = {}

    def prune_to_pool(self, pool_size: int) -> None:
        for track_id in [tid for tid in self.tracks if tid >= pool_size]:
            del self.tracks[track_id]

    def track_role_hint(self, track: dict, args: argparse.Namespace) -> str:
        if float(track.get("inv_score", 0.0)) >= args.inv_score_min or int(track.get("inv_confirmed_frames", 0)) > 0:
            return "invigilator"
        if track.get("last_role") == "student":
            return "student"
        return "unknown"

    def is_compatible_match(self, det_box, track: dict, args: argparse.Namespace) -> bool:
        center_jump = normalized_center_distance(det_box, track["box"])
        height_jump = box_height_ratio(det_box, track["box"])
        role_hint = self.track_role_hint(track, args)
        if role_hint == "invigilator":
            return center_jump <= args.invigilator_jump_thr and height_jump <= args.track_height_jump_ratio_thr

        anchor_center = track.get("seat_anchor_center")
        anchor_frames = int(track.get("seat_anchor_frames", 0))
        if role_hint == "student" and anchor_center is not None and anchor_frames >= args.student_anchor_min_frames:
            anchor_dist = normalized_center_distance_to_anchor(det_box, anchor_center)
            return anchor_dist <= args.student_anchor_center_thr

        return center_jump <= args.student_unanchored_jump_thr and height_jump <= args.track_height_jump_ratio_thr

    def match_score(self, det_box, track: dict, args: argparse.Namespace) -> float:
        base_iou = iou(det_box, track["box"])
        anchor_center = track.get("seat_anchor_center")
        anchor_frames = int(track.get("seat_anchor_frames", 0))
        if anchor_center is None or anchor_frames < args.student_anchor_min_frames:
            return base_iou
        anchor_dist = normalized_center_distance_to_anchor(det_box, anchor_center)
        anchor_score = max(0.0, 1.0 - anchor_dist / max(1e-6, args.student_anchor_center_thr))
        return 0.65 * base_iou + 0.35 * anchor_score

    def choose_pool_id(self, det: dict, assigned_tracks: set[int], pool_size: int, args: argparse.Namespace):
        available_ids = [tid for tid in range(pool_size) if tid not in assigned_tracks]
        if not available_ids:
            return None
        scored = []
        for track_id in available_ids:
            track = self.tracks.get(track_id)
            if track is None:
                scored.append((0.0, 1, track_id))
            elif not self.is_compatible_match(det["bbox"], track, args):
                continue
            else:
                scored.append((self.match_score(det["bbox"], track, args), 0, track_id))
        if not scored:
            return None
        scored.sort(key=lambda item: (-item[0], item[1], item[2]))
        return scored[0][2]

    def update_track_box(self, track: dict, new_box, frame_idx: int, args: argparse.Namespace) -> None:
        track["box"] = smooth_box(track["box"], new_box, args.box_smooth_alpha)
        track["last_seen"] = frame_idx

    def update(self, detections: list[dict], frame_idx: int, pool_size: int, args: argparse.Namespace):
        self.prune_to_pool(pool_size)
        assigned_tracks = set()
        assigned_dets = set()
        pairs = []

        for det_idx, det in enumerate(detections):
            for track_id, track in self.tracks.items():
                if track_id >= pool_size:
                    continue
                if not self.is_compatible_match(det["bbox"], track, args):
                    continue
                score = self.match_score(det["bbox"], track, args)
                if iou(det["bbox"], track["box"]) >= self.iou_thr or score >= self.iou_thr:
                    pairs.append((score, det_idx, track_id))

        pairs.sort(reverse=True)
        for _, det_idx, track_id in pairs:
            if det_idx in assigned_dets or track_id in assigned_tracks:
                continue
            det = detections[det_idx]
            track = self.tracks[track_id]
            self.update_track_box(track, det["bbox"], frame_idx, args)
            det["track_id"] = track_id
            det["bbox"] = track["box"]
            assigned_dets.add(det_idx)
            assigned_tracks.add(track_id)

        for det_idx, det in enumerate(detections):
            if det_idx in assigned_dets:
                continue
            track_id = self.choose_pool_id(det, assigned_tracks, pool_size, args)
            if track_id is None:
                continue
            if track_id not in self.tracks:
                self.tracks[track_id] = new_track(track_id, det["bbox"], frame_idx)
            else:
                self.update_track_box(self.tracks[track_id], det["bbox"], frame_idx, args)
            det["track_id"] = track_id
            det["bbox"] = self.tracks[track_id]["box"]
            assigned_tracks.add(track_id)

        for track_id in [
            tid for tid, track in self.tracks.items() if frame_idx - track["last_seen"] > self.max_missing
        ]:
            del self.tracks[track_id]

        return [det for det in detections if "track_id" in det]


def clear_abnormal_state(track: dict) -> None:
    track["abn_streak_label"] = None
    track["abn_streak_count"] = 0
    track["confirmed_abn_label"] = None
    track["confirmed_conf"] = 0.0
    track["recover_count"] = 0


def update_track_height_history(track: dict, bbox, image_height: int, args: argparse.Namespace) -> None:
    if args.standing_height_history <= 0:
        return
    x1, y1, x2, y2 = bbox
    height_ratio = max(0.0, float(y2 - y1) / max(1.0, float(image_height)))
    history = track.setdefault("height_history", [])
    history.append(height_ratio)
    if len(history) > args.standing_height_history:
        del history[: len(history) - args.standing_height_history]


def box_motion_is_track_switch(track: dict, bbox, image_width: int, image_height: int, args: argparse.Namespace) -> bool:
    last_bbox = track.get("last_bbox")
    if last_bbox is None:
        return False
    x1, y1, x2, y2 = bbox
    px1, py1, px2, py2 = last_bbox
    cx = (float(x1) + float(x2)) * 0.5 / max(1.0, float(image_width))
    cy = (float(y1) + float(y2)) * 0.5 / max(1.0, float(image_height))
    pcx = (float(px1) + float(px2)) * 0.5 / max(1.0, float(image_width))
    pcy = (float(py1) + float(py2)) * 0.5 / max(1.0, float(image_height))
    center_shift = ((cx - pcx) ** 2 + (cy - pcy) ** 2) ** 0.5

    height = max(1.0, float(y2) - float(y1))
    prev_height = max(1.0, float(py2) - float(py1))
    height_ratio = max(height / prev_height, prev_height / height)

    return (
        center_shift >= args.track_switch_center_thr
        or height_ratio >= args.track_switch_height_ratio_thr
    )


def reset_track_history_after_switch(track: dict) -> None:
    clear_abnormal_state(track)
    track["height_history"] = []


def standing_height_change_reason(track: dict, bbox, image_height: int, args: argparse.Namespace) -> str | None:
    if args.standing_height_history <= 0:
        return None
    history = track.get("height_history") or []
    if len(history) < 30:
        return None
    x1, y1, x2, y2 = bbox
    current = max(0.0, float(y2 - y1) / max(1.0, float(image_height)))
    baseline = sorted(history)[len(history) // 2]
    if baseline <= 1e-6:
        return None
    rise_ratio = current / baseline
    rise_delta = current - baseline
    if (
        current >= args.standing_height_min_ratio
        and rise_ratio >= args.standing_height_rise_ratio_thr
        and rise_delta >= args.standing_height_rise_min_delta
    ):
        return f"body_height_rise_{rise_ratio:.2f}_delta_{rise_delta:.2f}_base_{baseline:.2f}"
    return None


def apply_standing_height_change_rule(
    rules: dict[str, dict], track: dict, bbox, image_height: int, args: argparse.Namespace
) -> None:
    standing_rule = rules.get("student_standing")
    if not standing_rule:
        return
    reason = standing_height_change_reason(track, bbox, image_height, args)
    if reason is None:
        return
    reasons = standing_rule.setdefault("reasons", [])
    if reason not in reasons:
        reasons.append(reason)
    standing_rule["score"] = max(float(standing_rule.get("score", 0.0)), 1.0)
    standing_rule["passed"] = standing_rule["score"] >= args.rule_min_score


def apply_behavior_threshold(label: str, score: float, args: argparse.Namespace) -> str:
    thresholds = {
        "raise_hand": args.raise_hand_thr,
        "head_turn": args.head_turn_thr,
        "pass_suspicious_item": args.passing_object_thr,
        "hand_under_table_look_down": args.hand_under_table_thr,
        "pick_suspicious_item": args.picking_object_thr,
        "carry_phone": args.carry_phone_thr,
        "tear_paper": args.tear_paper_thr,
        "student_standing": args.student_standing_thr,
    }
    if label in thresholds and score < thresholds[label]:
        return "normal"
    return label


def kp_ok(pose: list[list[float]], idx: int, min_score: float) -> bool:
    return idx < len(pose) and len(pose[idx]) >= 3 and float(pose[idx][2]) >= min_score


def kp_xy(pose: list[list[float]], idx: int) -> np.ndarray:
    return np.asarray([float(pose[idx][0]), float(pose[idx][1])], dtype=np.float32)


def visible_points(pose: list[list[float]], indices: list[int], min_score: float) -> list[np.ndarray]:
    return [kp_xy(pose, idx) for idx in indices if kp_ok(pose, idx, min_score)]


def point_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def avg_point(points: list[np.ndarray]):
    if not points:
        return None
    return np.mean(np.stack(points, axis=0), axis=0)


def shoulder_stats(pose: list[list[float]], min_score: float):
    if not kp_ok(pose, KP_LEFT_SHOULDER, min_score) or not kp_ok(pose, KP_RIGHT_SHOULDER, min_score):
        return None, None
    left = kp_xy(pose, KP_LEFT_SHOULDER)
    right = kp_xy(pose, KP_RIGHT_SHOULDER)
    center = (left + right) / 2.0
    width = max(0.08, abs(float(left[0] - right[0])))
    return center, width


def rule_raise_hand(pose: list[list[float]], args: argparse.Namespace) -> tuple[float, list[str]]:
    min_score = args.rule_kp_score_thr
    reasons = []
    wrists = [KP_LEFT_WRIST, KP_RIGHT_WRIST]
    shoulders = [KP_LEFT_SHOULDER, KP_RIGHT_SHOULDER]
    for wrist_idx, shoulder_idx, side in zip(wrists, shoulders, ["left", "right"]):
        if kp_ok(pose, wrist_idx, min_score) and kp_ok(pose, shoulder_idx, min_score):
            if float(pose[wrist_idx][1]) < float(pose[shoulder_idx][1]) - 0.03:
                reasons.append(f"{side}_wrist_above_shoulder")
    if kp_ok(pose, KP_NOSE, min_score):
        nose_y = float(pose[KP_NOSE][1])
        for wrist_idx, side in zip(wrists, ["left", "right"]):
            if kp_ok(pose, wrist_idx, min_score) and float(pose[wrist_idx][1]) < nose_y + 0.04:
                reasons.append(f"{side}_wrist_near_head_height")
    return (1.0 if reasons else 0.0), reasons


def rule_head_turn(pose: list[list[float]], args: argparse.Namespace) -> tuple[float, list[str]]:
    min_score = args.rule_kp_score_thr
    reasons = []
    shoulder_center, shoulder_width = shoulder_stats(pose, min_score)
    if shoulder_center is not None and kp_ok(pose, KP_NOSE, min_score):
        nose = kp_xy(pose, KP_NOSE)
        offset = abs(float(nose[0] - shoulder_center[0])) / shoulder_width
        if offset > 0.22:
            reasons.append(f"nose_offset_{offset:.2f}")
    left_face = len(visible_points(pose, [KP_LEFT_EYE, KP_LEFT_EAR], min_score))
    right_face = len(visible_points(pose, [KP_RIGHT_EYE, KP_RIGHT_EAR], min_score))
    if abs(left_face - right_face) >= 2:
        reasons.append("face_visibility_asymmetry")
    return (1.0 if reasons else 0.0), reasons


def rule_hand_under_table(pose: list[list[float]], args: argparse.Namespace) -> tuple[float, list[str]]:
    min_score = args.rule_kp_score_thr
    reasons = []
    upper_indices = [
        KP_NOSE,
        KP_LEFT_EYE,
        KP_RIGHT_EYE,
        KP_LEFT_EAR,
        KP_RIGHT_EAR,
        KP_LEFT_SHOULDER,
        KP_RIGHT_SHOULDER,
        KP_LEFT_WRIST,
        KP_RIGHT_WRIST,
    ]
    upper_conf = sum(float(pose[idx][2]) for idx in upper_indices) / len(upper_indices)
    shoulder_conf = (float(pose[KP_LEFT_SHOULDER][2]) + float(pose[KP_RIGHT_SHOULDER][2])) * 0.5
    wrist_conf = (float(pose[KP_LEFT_WRIST][2]) + float(pose[KP_RIGHT_WRIST][2])) * 0.5
    if upper_conf < 0.55 and shoulder_conf < 0.55 and wrist_conf < 0.55:
        reasons.append(f"low_upper_kp_conf_{upper_conf:.2f}_sh_{shoulder_conf:.2f}_wr_{wrist_conf:.2f}")
    shoulders = visible_points(pose, [KP_LEFT_SHOULDER, KP_RIGHT_SHOULDER], min_score)
    shoulder_center = avg_point(shoulders)
    if shoulder_center is None:
        return (1.0 if reasons else 0.0), reasons
    if kp_ok(pose, KP_NOSE, min_score) and float(pose[KP_NOSE][1]) > float(shoulder_center[1]) + 0.12:
        reasons.append("head_low")
    low_wrists = 0
    for idx in [KP_LEFT_WRIST, KP_RIGHT_WRIST]:
        if kp_ok(pose, idx, min_score) and float(pose[idx][1]) > float(shoulder_center[1]) + 0.35:
            low_wrists += 1
    if low_wrists > 0:
        reasons.append("wrist_low")
    has_low_conf = any(reason.startswith("low_upper_kp_conf") for reason in reasons)
    score = 1.0 if has_low_conf or len(reasons) >= 2 else (0.5 if reasons else 0.0)
    return score, reasons


def rule_pick_suspicious_item(pose: list[list[float]], args: argparse.Namespace) -> tuple[float, list[str]]:
    min_score = args.rule_kp_score_thr
    reasons = []
    if kp_ok(pose, KP_NOSE, min_score) and float(pose[KP_NOSE][1]) > 0.45:
        reasons.append("head_near_lower_body")
    low_wrists = [
        idx for idx in [KP_LEFT_WRIST, KP_RIGHT_WRIST] if kp_ok(pose, idx, min_score) and float(pose[idx][1]) > 0.55
    ]
    if low_wrists:
        reasons.append("wrist_near_lower_body")
    score = 1.0 if len(reasons) >= 2 else (0.5 if reasons else 0.0)
    return score, reasons


def rule_pass_suspicious_item(pose: list[list[float]], args: argparse.Namespace) -> tuple[float, list[str]]:
    min_score = args.rule_kp_score_thr
    reasons = []
    shoulder_center, shoulder_width = shoulder_stats(pose, min_score)
    if shoulder_center is None:
        return 0.0, reasons
    for wrist_idx, side in [(KP_LEFT_WRIST, "left"), (KP_RIGHT_WRIST, "right")]:
        if not kp_ok(pose, wrist_idx, min_score):
            continue
        wrist = kp_xy(pose, wrist_idx)
        x_offset = abs(float(wrist[0] - shoulder_center[0])) / shoulder_width
        if x_offset > 0.85 and 0.15 < float(wrist[1]) < 0.80:
            reasons.append(f"{side}_arm_reaches_side")
    return (1.0 if reasons else 0.0), reasons


def rule_carry_phone(pose: list[list[float]], args: argparse.Namespace) -> tuple[float, list[str]]:
    min_score = args.rule_kp_score_thr
    reasons = []
    face_points = visible_points(
        pose,
        [KP_NOSE, KP_LEFT_EYE, KP_RIGHT_EYE, KP_LEFT_EAR, KP_RIGHT_EAR],
        min_score,
    )
    if not face_points:
        return 0.0, reasons
    for wrist_idx, side in [(KP_LEFT_WRIST, "left"), (KP_RIGHT_WRIST, "right")]:
        if not kp_ok(pose, wrist_idx, min_score):
            continue
        wrist = kp_xy(pose, wrist_idx)
        nearest = min(point_distance(wrist, face) for face in face_points)
        if nearest < 0.18:
            reasons.append(f"{side}_wrist_near_face")
    return (1.0 if reasons else 0.0), reasons


def rule_tear_paper(pose: list[list[float]], args: argparse.Namespace) -> tuple[float, list[str]]:
    min_score = args.rule_kp_score_thr
    reasons = []
    if not kp_ok(pose, KP_LEFT_WRIST, min_score) or not kp_ok(pose, KP_RIGHT_WRIST, min_score):
        return 0.0, reasons
    left = kp_xy(pose, KP_LEFT_WRIST)
    right = kp_xy(pose, KP_RIGHT_WRIST)
    if point_distance(left, right) < 0.30:
        reasons.append("wrists_close")
    if 0.20 < float(left[1]) < 0.75 and 0.20 < float(right[1]) < 0.75:
        reasons.append("wrists_in_front_body")
    score = 1.0 if len(reasons) >= 2 else (0.5 if reasons else 0.0)
    return score, reasons


def knee_angle_degrees(hip: np.ndarray, knee: np.ndarray, ankle: np.ndarray) -> float:
    thigh = hip - knee
    calf = ankle - knee
    denom = max(1e-6, float(np.linalg.norm(thigh) * np.linalg.norm(calf)))
    cosine = float(np.dot(thigh, calf)) / denom
    cosine = max(-1.0, min(1.0, cosine))
    return float(np.degrees(np.arccos(cosine)))


def standing_leg_reasons(pose: list[list[float]], side: str, args: argparse.Namespace) -> list[str]:
    min_score = args.rule_kp_score_thr
    if side == "left":
        hip_idx, knee_idx, ankle_idx = KP_LEFT_HIP, KP_LEFT_KNEE, KP_LEFT_ANKLE
    else:
        hip_idx, knee_idx, ankle_idx = KP_RIGHT_HIP, KP_RIGHT_KNEE, KP_RIGHT_ANKLE
    if not all(kp_ok(pose, idx, min_score) for idx in [hip_idx, knee_idx, ankle_idx]):
        return []

    hip = kp_xy(pose, hip_idx)
    knee = kp_xy(pose, knee_idx)
    ankle = kp_xy(pose, ankle_idx)
    angle = knee_angle_degrees(hip, knee, ankle)
    leg_length = point_distance(hip, knee) + point_distance(knee, ankle)
    vertical_span = abs(float(ankle[1] - hip[1]))

    reasons = []
    if (
        angle >= args.standing_leg_angle_thr
        and leg_length >= args.standing_leg_length_ratio_thr
        and vertical_span >= args.standing_leg_vertical_ratio_thr
    ):
        reasons.append(
            f"{side}_leg_straight_len_{leg_length:.2f}_vspan_{vertical_span:.2f}_angle_{angle:.0f}"
        )
    return reasons


def rule_student_standing(
    pose: list[list[float]], box, image_width: int, image_height: int, args: argparse.Namespace
) -> tuple[float, list[str]]:
    reasons = []
    x1, y1, x2, y2 = box
    box_w = max(1.0, float(x2 - x1))
    box_h = max(1.0, float(y2 - y1))
    has_large_box = False
    has_upright_box = False
    if box_h / max(1.0, float(image_height)) > 0.42:
        has_large_box = True
        reasons.append("large_box_height")
    if box_h / box_w > 1.45:
        has_upright_box = True
        reasons.append("upright_box_ratio")
    leg_reasons = standing_leg_reasons(pose, "left", args) + standing_leg_reasons(pose, "right", args)
    if leg_reasons:
        reasons.extend(leg_reasons)
    strong_leg_support = has_large_box or len(leg_reasons) >= 2
    strong_body_support = has_large_box or has_upright_box
    score = 1.0 if leg_reasons and strong_leg_support and strong_body_support else (0.5 if reasons else 0.0)
    return score, reasons


def evaluate_single_frame_rule(
    label: str, pose: list[list[float]], box, image_width: int, image_height: int, args: argparse.Namespace
) -> dict:
    evaluators = {
        "raise_hand": lambda: rule_raise_hand(pose, args),
        "head_turn": lambda: rule_head_turn(pose, args),
        "pass_suspicious_item": lambda: rule_pass_suspicious_item(pose, args),
        "hand_under_table_look_down": lambda: rule_hand_under_table(pose, args),
        "pick_suspicious_item": lambda: rule_pick_suspicious_item(pose, args),
        "carry_phone": lambda: rule_carry_phone(pose, args),
        "tear_paper": lambda: rule_tear_paper(pose, args),
        "student_standing": lambda: rule_student_standing(pose, box, image_width, image_height, args),
    }
    if label not in evaluators:
        return {"label": label, "score": 1.0, "passed": True, "reasons": []}
    score, reasons = evaluators[label]()
    return {
        "label": label,
        "score": float(score),
        "passed": score >= args.rule_min_score,
        "reasons": reasons,
    }


def evaluate_all_single_frame_rules(
    pose: list[list[float]], box, image_width: int, image_height: int, args: argparse.Namespace
) -> dict[str, dict]:
    if args.disable_single_frame_rules:
        return {}
    return {
        label: evaluate_single_frame_rule(label, pose, box, image_width, image_height, args)
        for label in sorted(ABNORMAL_LABELS)
    }


def apply_single_frame_rules(
    behavior: dict,
    thresholded_label: str,
    pose: list[list[float]],
    box,
    image_width: int,
    image_height: int,
    args: argparse.Namespace,
) -> str:
    if args.disable_single_frame_rules or thresholded_label not in ABNORMAL_LABELS:
        behavior["rule"] = {"label": thresholded_label, "score": 1.0, "passed": True, "reasons": []}
        behavior["rule_fusion"] = None
        return thresholded_label

    rule = evaluate_single_frame_rule(thresholded_label, pose, box, image_width, image_height, args)
    behavior["rule"] = rule
    fusion = BEHAVIOR_RULE_FUSION_BY_LABEL.get(
        thresholded_label,
        {
            "label": thresholded_label,
            "model_weight": 0.75,
            "rule_weight": 0.25,
            "fused_thr": 0.70,
            "model_override_thr": args.rule_keep_model_thr,
        },
    )
    model_weight = float(fusion["model_weight"])
    rule_weight = float(fusion["rule_weight"])
    weight_sum = max(1e-6, model_weight + rule_weight)
    model_weight /= weight_sum
    rule_weight /= weight_sum
    fused_score = model_weight * float(behavior["raw_score"]) + rule_weight * float(rule["score"])
    fused_thr = float(fusion["fused_thr"])
    model_override_thr = float(fusion.get("model_override_thr", args.rule_keep_model_thr))
    behavior["rule_fusion"] = {
        "label": thresholded_label,
        "model_weight": model_weight,
        "rule_weight": rule_weight,
        "model_score": float(behavior["raw_score"]),
        "rule_score": float(rule["score"]),
        "fused_score": float(fused_score),
        "fused_thr": fused_thr,
        "model_override_thr": model_override_thr,
        "passed": bool(fused_score >= fused_thr),
        "model_override": bool(behavior["raw_score"] >= model_override_thr),
    }
    if fused_score >= fused_thr or behavior["raw_score"] >= model_override_thr:
        return thresholded_label
    return "normal"


def temporal_abnormal_n_for_label(label: str, args: argparse.Namespace) -> int:
    value = BEHAVIOR_DECISION_CONFIG.get(label, {}).get("temporal_abnormal_n", args.temporal_abnormal_n)
    return max(1, int(value))


def apply_temporal_rule(track: dict, raw_label: str, raw_conf: float, args: argparse.Namespace):
    if raw_label in ABNORMAL_LABELS:
        needed_frames = temporal_abnormal_n_for_label(raw_label, args)
        if track["abn_streak_label"] == raw_label:
            track["abn_streak_count"] += 1
        else:
            track["abn_streak_label"] = raw_label
            track["abn_streak_count"] = 1
        track["recover_count"] = 0

        if track["abn_streak_count"] >= needed_frames:
            track["confirmed_abn_label"] = raw_label
            track["confirmed_conf"] = raw_conf

        if track["confirmed_abn_label"] is not None:
            return track["confirmed_abn_label"], raw_conf, True
        return "normal", raw_conf, False

    track["abn_streak_label"] = None
    track["abn_streak_count"] = 0
    if track["confirmed_abn_label"] is not None:
        track["recover_count"] += 1
        if track["recover_count"] >= args.temporal_recover_n:
            track["confirmed_abn_label"] = None
            track["confirmed_conf"] = 0.0
            track["recover_count"] = 0
        else:
            return track["confirmed_abn_label"], track["confirmed_conf"], True
    return raw_label, raw_conf, False


def update_invigilator_scores(
    detections: list[dict], identity_predictions: list[dict], tracker: SimpleIoUTracker, args: argparse.Namespace
) -> None:
    for det, identity in zip(detections, identity_predictions):
        track = tracker.tracks[det["track_id"]]
        old_score = float(track.get("inv_score", 0.0))
        if identity["role"] == "invigilator":
            track["inv_score"] = args.inv_score_alpha * old_score + (1.0 - args.inv_score_alpha) * identity["confidence"]
        elif identity["role"] == "student" and identity["confidence"] >= args.student_suppress_conf:
            track["inv_score"] = args.student_suppress_factor * old_score
        else:
            track["inv_score"] = args.inv_score_decay * old_score


def invigilator_history_bonus(track: dict, args: argparse.Namespace) -> float:
    if args.inv_history_bonus_frames <= 0:
        return 0.0
    confirmed_frames = max(0, int(track.get("inv_confirmed_frames", 0)))
    progress = min(confirmed_frames / args.inv_history_bonus_frames, 1.0)
    return progress * args.inv_history_bonus_max


def select_invigilator_ids(
    detections: list[dict],
    tracker: SimpleIoUTracker,
    previous_invigilator_ids: set[int],
    args: argparse.Namespace,
) -> set[int]:
    candidates = []
    for det in detections:
        track_id = det["track_id"]
        track = tracker.tracks[track_id]
        inv_score = float(track.get("inv_score", 0.0))
        if inv_score < args.inv_score_min:
            continue
        history_bonus = invigilator_history_bonus(track, args)
        priority = inv_score + history_bonus
        if track_id in previous_invigilator_ids:
            priority += args.inv_slot_keep_bonus
        age = det.get("track_age", 0)
        candidates.append((priority, history_bonus, inv_score, age, -track_id, track_id))
    candidates.sort(reverse=True)
    return {item[-1] for item in candidates[: max(0, args.max_invigilators)]}


def update_invigilator_history(
    detections: list[dict], tracker: SimpleIoUTracker, frame_invigilator_ids: set[int]
) -> None:
    visible_ids = {det["track_id"] for det in detections}
    for track_id in visible_ids:
        track = tracker.tracks.get(track_id)
        if track is None:
            continue
        if track_id in frame_invigilator_ids:
            track["inv_confirmed_frames"] = int(track.get("inv_confirmed_frames", 0)) + 1


def update_position_anchors(
    detections: list[dict], tracker: SimpleIoUTracker, frame_invigilator_ids: set[int], args: argparse.Namespace
) -> None:
    alpha = max(0.0, min(1.0, args.student_anchor_update_alpha))
    for det in detections:
        track = tracker.tracks.get(det["track_id"])
        if track is None:
            continue
        box = tuple(float(v) for v in det["bbox"])
        role = "invigilator" if det["track_id"] in frame_invigilator_ids else "student"
        track["last_role"] = role
        if role == "invigilator":
            continue

        center = box_center(box)
        if track.get("seat_anchor_center") is None:
            track["seat_anchor_center"] = center
            track["seat_anchor_box"] = box
            track["seat_anchor_frames"] = 1
            continue

        anchor_center = track["seat_anchor_center"]
        anchor_dist = normalized_center_distance_to_anchor(box, anchor_center)
        if (
            int(track.get("seat_anchor_frames", 0)) < args.student_anchor_min_frames
            or anchor_dist <= args.student_anchor_center_thr
        ):
            old_box = track.get("seat_anchor_box") or box
            track["seat_anchor_center"] = (
                alpha * float(anchor_center[0]) + (1.0 - alpha) * center[0],
                alpha * float(anchor_center[1]) + (1.0 - alpha) * center[1],
            )
            track["seat_anchor_box"] = tuple(
                alpha * float(old_value) + (1.0 - alpha) * float(new_value)
                for old_value, new_value in zip(old_box, box)
            )
            track["seat_anchor_frames"] = int(track.get("seat_anchor_frames", 0)) + 1


def get_person_detections(det_result, width: int, height: int, args: argparse.Namespace) -> list[dict]:
    instances = det_result.pred_instances.cpu().numpy()
    detections = []
    for idx, (bbox, score, label) in enumerate(zip(instances.bboxes, instances.scores, instances.labels)):
        if int(label) != 0 or float(score) < args.det_score_thr:
            continue
        box = clip_box(bbox.tolist(), width, height)
        if box is None or box_area(box) <= args.min_box_area:
            continue
        detections.append({"index": idx, "bbox": box, "det_confidence": float(score)})
    return detections


def normalize_image_behavior_name(name: str) -> str:
    normalized = str(name).strip().lower()
    if normalized in NORMAL_BEHAVIOR_NAMES:
        return "normal"
    return IMAGE_BEHAVIOR_LABEL_MAP.get(normalized, normalized)


class RoiHead(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def pad_tensor_to_divisor(tensor: torch.Tensor, divisor: int = 32) -> torch.Tensor:
    _, height, width = tensor.shape
    pad_h = (divisor - height % divisor) % divisor
    pad_w = (divisor - width % divisor) % divisor
    if pad_h or pad_w:
        tensor = F.pad(tensor, (0, pad_w, 0, pad_h), value=0)
    return tensor


def preprocess_rtm_frame(frame_bgr: np.ndarray, device: torch.device) -> torch.Tensor:
    mean = RTMDET_MEAN.to(device)
    std = RTMDET_STD.to(device)
    image = torch.from_numpy(frame_bgr).permute(2, 0, 1).float().to(device)
    image = (image - mean) / std
    image = pad_tensor_to_divisor(image)
    return image.unsqueeze(0)


def load_roi_heads(checkpoint_path: Path, device: torch.device) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    hidden_dim = int(checkpoint.get("hidden_dim", 256))
    dropout = float(checkpoint.get("dropout", 0.25))
    behavior_info = checkpoint["behavior"]
    identity_info = checkpoint["identity"]
    behavior_labels = list(behavior_info["labels"])
    identity_labels = list(identity_info["labels"])
    behavior_head = RoiHead(
        int(behavior_info["input_dim"]),
        len(behavior_labels),
        hidden_dim,
        dropout,
    )
    identity_head = RoiHead(
        int(identity_info["input_dim"]),
        len(identity_labels),
        hidden_dim,
        dropout,
    )
    behavior_head.load_state_dict(behavior_info["state_dict"])
    identity_head.load_state_dict(identity_info["state_dict"])
    behavior_head.to(device).eval()
    identity_head.to(device).eval()
    return {
        "behavior": behavior_head,
        "identity": identity_head,
        "behavior_labels": behavior_labels,
        "identity_labels": identity_labels,
    }


@torch.no_grad()
def extract_frame_roi_features(
    det_model,
    frame_bgr: np.ndarray,
    boxes: list[tuple[int, int, int, int]],
    device: torch.device,
    roi_size: int,
) -> torch.Tensor:
    if not boxes:
        return torch.empty((0, 0), dtype=torch.float32, device=device)
    image = preprocess_rtm_frame(frame_bgr, device)
    feats = det_model.extract_feat(image)
    feat = feats[-1]
    spatial_scale = feat.shape[-1] / float(image.shape[-1])
    roi_tensor = torch.tensor(
        [[0.0, float(x1), float(y1), float(x2), float(y2)] for x1, y1, x2, y2 in boxes],
        dtype=torch.float32,
        device=device,
    )
    pooled = roi_align(feat, roi_tensor, output_size=(roi_size, roi_size), spatial_scale=spatial_scale, aligned=True)
    return pooled.flatten(1)


def classify_roi_identity(roi_heads: dict, features: torch.Tensor) -> list[dict]:
    if features.numel() == 0:
        return []
    model = roi_heads["identity"]
    labels = roi_heads["identity_labels"]
    with torch.no_grad():
        probs = torch.softmax(model(features), dim=1)
    predictions = []
    for row in probs.detach().cpu():
        class_idx = int(torch.argmax(row).item())
        label = labels[class_idx]
        conf = float(row[class_idx].item())
        predictions.append(
            {
                "raw_class_id": class_idx,
                "raw_class_name": label,
                "role": label,
                "role_id": 1 if label == "invigilator" else 0,
                "confidence": conf,
                "class_probs": {labels[idx]: float(row[idx].item()) for idx in range(len(labels))},
                "source": "rtmdet_roi_head",
            }
        )
    return predictions


def classify_roi_behavior(roi_heads: dict, features: torch.Tensor, indices: list[int]) -> list[dict]:
    if not indices:
        return []
    model = roi_heads["behavior"]
    labels = roi_heads["behavior_labels"]
    selected_features = features[torch.tensor(indices, dtype=torch.long, device=features.device)]
    with torch.no_grad():
        probs = torch.softmax(model(selected_features), dim=1)
    predictions = []
    for row in probs.detach().cpu():
        class_idx = int(torch.argmax(row).item())
        label = labels[class_idx]
        conf = float(row[class_idx].item())
        predictions.append(
            {
                "raw_class_id": class_idx,
                "raw_class_name": label,
                "label": label,
                "confidence": conf,
                "class_probs": {labels[idx]: float(row[idx].item()) for idx in range(len(labels))},
                "source": "rtmdet_roi_head",
            }
        )
    return predictions


def load_behavior_model(checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    idx_to_label = {int(idx): label for idx, label in checkpoint["idx_to_label"].items()}
    model = build_pose_mlp(
        num_classes=len(idx_to_label),
        hidden_dim=int(checkpoint.get("hidden_dim", 192)),
        num_layers=int(checkpoint.get("num_layers", 3)),
        dropout=float(checkpoint.get("dropout", 0.3)),
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model, idx_to_label


def iter_pose_instances(pose_results: Iterable):
    for pose in pose_results:
        pred = pose.pred_instances.cpu().numpy()
        yield {
            "bbox": pred.bboxes[0].tolist(),
            "keypoints": pred.keypoints[0].tolist(),
            "keypoint_scores": pred.keypoint_scores[0].tolist(),
        }


def classify_pose(model, idx_to_label: dict[int, str], pose: list[list[float]], device: torch.device) -> dict:
    features = torch.tensor(flatten_pose(pose), dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(model(features), dim=1)[0]
    score, class_idx = torch.max(probs, dim=0)
    label = idx_to_label[int(class_idx.item())]
    return {
        "raw_label": label,
        "raw_score": float(score.item()),
        "class_probs": {idx_to_label[idx]: float(probs[idx].item()) for idx in range(len(idx_to_label))},
    }


def pose_component_score(behavior: dict, label: str) -> float:
    return float(behavior.get("class_probs", {}).get(label, 0.0))


def rule_component_score(behavior: dict, label: str) -> float:
    rule_scores = behavior.get("all_rule_scores") or {}
    return float(rule_scores.get(label, {}).get("score", 0.0))


def adjusted_rule_component_score(behavior: dict, label: str, pose_score: float) -> tuple[float, list[str]]:
    rule = (behavior.get("all_rule_scores") or {}).get(label, {}) or {}
    score = float(rule.get("score", 0.0))
    adjustments: list[str] = []
    reasons = rule.get("reasons", []) or []
    if (
        label == "tear_paper"
        and score >= 0.5
        and pose_score >= 0.75
        and "wrists_close" in reasons
        and "wrists_in_front_body" in reasons
    ):
        score = max(score, 1.0)
        adjustments.append("tear_pose_high_wrists_front")
    return score, adjustments


def fuse_pose_and_image_behavior(pose_behavior: dict, image_behavior: dict | None) -> dict:
    image_behavior = image_behavior or {
        "label": "normal",
        "confidence": 0.0,
        "class_probs": {"normal": 1.0},
        "raw_class_id": None,
        "raw_class_name": None,
    }
    labels = set(ABNORMAL_LABELS) | {"normal"}
    candidates = []
    for label in sorted(labels):
        if label == "normal":
            pose_score = pose_component_score(pose_behavior, label)
            rule_score = 0.0
            image_score = float(image_behavior.get("class_probs", {}).get(label, 0.0))
            fused_score = max(pose_score, image_score)
            candidates.append(
                {
                    "label": "normal",
                    "pose_score": pose_score,
                    "rule_score": rule_score,
                    "image_score": image_score,
                    "fused_score": fused_score,
                    "fused_thr": 0.0,
                    "pose_weight": 0.5,
                    "rule_weight": 0.0,
                    "image_weight": 0.5,
                    "passed": True,
                }
            )
            continue
        params = BEHAVIOR_DECISION_CONFIG[label]
        pose_weight = float(params["pose_weight"])
        rule_weight = float(params["rule_weight"])
        image_weight = float(params["image_weight"])
        weight_sum = max(1e-6, pose_weight + rule_weight + image_weight)
        pose_weight /= weight_sum
        rule_weight /= weight_sum
        image_weight /= weight_sum
        pose_score = pose_component_score(pose_behavior, label)
        rule_score, rule_adjustments = adjusted_rule_component_score(pose_behavior, label, pose_score)
        image_score = float(image_behavior.get("class_probs", {}).get(label, 0.0))
        fused_score = pose_weight * pose_score + rule_weight * rule_score + image_weight * image_score
        fused_thr = float(params["fused_thr"])
        candidates.append(
            {
                "label": label,
                "pose_score": pose_score,
                "rule_score": rule_score,
                "image_score": image_score,
                "fused_score": fused_score,
                "fused_thr": fused_thr,
                "pose_weight": pose_weight,
                "rule_weight": rule_weight,
                "image_weight": image_weight,
                "passed": fused_score >= fused_thr,
                "rule_adjustments": rule_adjustments,
            }
        )

    passed_abnormal = [item for item in candidates if item["label"] in ABNORMAL_LABELS and item["passed"]]
    if passed_abnormal:
        best = max(passed_abnormal, key=lambda item: item["fused_score"])
    else:
        best = next(item for item in candidates if item["label"] == "normal")
    return {
        "label": best["label"],
        "confidence": float(best["fused_score"]),
        "best": best,
        "candidates": candidates,
        "image_behavior": image_behavior,
    }


def draw_box(frame, box, label: str, final_label: str) -> None:
    color = COLORS.get(final_label, (255, 255, 255))
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thickness = 2
    (tw, th), baseline = cv2.getTextSize(label, font, scale, thickness)
    y0 = max(0, y1 - th - baseline - 6)
    cv2.rectangle(frame, (x1, y0), (min(frame.shape[1] - 1, x1 + tw + 8), y1), color, -1)
    cv2.putText(frame, label, (x1 + 4, y1 - 6), font, scale, (0, 0, 0), thickness, cv2.LINE_AA)


def draw_status(frame, text: str) -> None:
    cv2.rectangle(frame, (10, 10), (900, 45), (0, 0, 0), -1)
    cv2.putText(frame, text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    if not args.video.is_file():
        raise FileNotFoundError(args.video)
    if not args.roi_head_checkpoint.is_file():
        raise FileNotFoundError(args.roi_head_checkpoint)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)

    register_mmdet_modules(init_default_scope=True)
    register_mmpose_modules(init_default_scope=True)
    det_model = init_detector(str(args.det_config), str(args.det_checkpoint), device=str(device))
    pose_model = init_model(str(args.pose_config), str(args.pose_checkpoint), device=str(device))
    behavior_model, idx_to_label = load_behavior_model(args.behavior_checkpoint, device)
    roi_heads = load_roi_heads(args.roi_head_checkpoint, device)
    tracker = SimpleIoUTracker(args.track_iou_thr, args.track_max_missing)

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_to_process = min(args.max_frames, total_frames) if total_frames > 0 else args.max_frames

    writer = None
    if not args.no_write_video:
        writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"Could not create output video: {args.output}")

    report = {
        "video": str(args.video),
        "detector": str(args.det_checkpoint),
        "pose": str(args.pose_checkpoint),
        "behavior": str(args.behavior_checkpoint),
        "roi_head_checkpoint": str(args.roi_head_checkpoint),
        "roi_head_behavior_labels": roi_heads["behavior_labels"],
        "roi_head_identity_labels": roi_heads["identity_labels"],
        "tracker": "SimpleIoUTracker",
        "roi_classification": {
            "roi_size": args.roi_size,
            "feature_source": "det_model.extract_feat(frame) + roi_align(boxes)",
            "note": "This test version computes ROI features after detection; a lower-level detector forward can reuse detector features for higher FPS.",
        },
        "processed_frames": 0,
        "events": [],
        "summary": {
            "person_boxes": 0,
            "confirmed_abnormal_boxes": 0,
            "confirmed_abnormal_frames": 0,
            "max_id_pool_size": 0,
            "invigilator_boxes": 0,
            "student_boxes": 0,
            "max_invigilators_in_frame": 0,
            "by_class": {},
            "by_pose_label": {},
            "by_image_behavior": {},
            "by_hybrid_behavior": {},
        },
        "rules": {
            "temporal_abnormal_n": args.temporal_abnormal_n,
            "temporal_abnormal_n_by_label": {
                label: temporal_abnormal_n_for_label(label, args) for label in sorted(ABNORMAL_LABELS)
            },
            "behavior_decision_config": BEHAVIOR_DECISION_CONFIG,
            "temporal_recover_n": args.temporal_recover_n,
            "max_invigilators": args.max_invigilators,
            "inv_score_min": args.inv_score_min,
            "inv_score_alpha": args.inv_score_alpha,
            "inv_score_decay": args.inv_score_decay,
            "inv_slot_keep_bonus": args.inv_slot_keep_bonus,
            "inv_history_bonus_max": args.inv_history_bonus_max,
            "inv_history_bonus_frames": args.inv_history_bonus_frames,
            "student_anchor_min_frames": args.student_anchor_min_frames,
            "student_anchor_center_thr": args.student_anchor_center_thr,
            "student_anchor_update_alpha": args.student_anchor_update_alpha,
            "student_unanchored_jump_thr": args.student_unanchored_jump_thr,
            "invigilator_jump_thr": args.invigilator_jump_thr,
            "track_height_jump_ratio_thr": args.track_height_jump_ratio_thr,
            "single_frame_rules": not args.disable_single_frame_rules,
            "rule_min_score": args.rule_min_score,
            "rule_keep_model_thr": args.rule_keep_model_thr,
            "rule_kp_score_thr": args.rule_kp_score_thr,
            "standing_height_history": args.standing_height_history,
            "standing_height_rise_ratio_thr": args.standing_height_rise_ratio_thr,
            "standing_height_rise_min_delta": args.standing_height_rise_min_delta,
            "standing_height_min_ratio": args.standing_height_min_ratio,
            "track_switch_center_thr": args.track_switch_center_thr,
            "track_switch_height_ratio_thr": args.track_switch_height_ratio_thr,
            "behavior_rule_fusion": BEHAVIOR_RULE_FUSION,
            "hybrid_behavior_fusion": HYBRID_BEHAVIOR_FUSION,
            "student_suppress_conf": args.student_suppress_conf,
            "student_suppress_factor": args.student_suppress_factor,
        },
    }
    timing = {
        "read": 0.0,
        "detect": 0.0,
        "track": 0.0,
        "roi_heads": 0.0,
        "pose": 0.0,
        "postprocess_draw": 0.0,
        "write": 0.0,
    }

    start_time = time.time()
    frame_idx = 0
    recent_counts: list[int] = []
    previous_invigilator_ids: set[int] = set()

    try:
        while frame_idx < frames_to_process:
            t0 = time.perf_counter()
            ok, frame = cap.read()
            timing["read"] += time.perf_counter() - t0
            if not ok:
                break

            t0 = time.perf_counter()
            init_default_scope("mmdet")
            det_result = inference_detector(det_model, frame)
            detections = get_person_detections(det_result, width, height, args)
            detections = deduplicate_people(detections, args)
            timing["detect"] += time.perf_counter() - t0

            t0 = time.perf_counter()
            recent_counts.append(len(detections))
            if len(recent_counts) > args.id_pool_window:
                recent_counts.pop(0)
            id_pool_size = max(recent_counts) if recent_counts else 0
            report["summary"]["max_id_pool_size"] = max(report["summary"]["max_id_pool_size"], id_pool_size)
            detections = tracker.update(detections, frame_idx, id_pool_size, args)
            timing["track"] += time.perf_counter() - t0

            valid_detections = []
            for det in detections:
                valid_detections.append(det)

            t0 = time.perf_counter()
            roi_features = extract_frame_roi_features(
                det_model,
                frame,
                [det["bbox"] for det in valid_detections],
                device,
                args.roi_size,
            )
            identity_predictions = classify_roi_identity(roi_heads, roi_features)
            for det, identity in zip(valid_detections, identity_predictions):
                det["identity"] = identity
                track = tracker.tracks[det["track_id"]]
                det["track_age"] = frame_idx - int(track.get("created_frame", frame_idx))
            timing["roi_heads"] += time.perf_counter() - t0

            update_invigilator_scores(valid_detections, identity_predictions, tracker, args)
            frame_invigilator_ids = select_invigilator_ids(
                valid_detections, tracker, previous_invigilator_ids, args
            )
            update_invigilator_history(valid_detections, tracker, frame_invigilator_ids)
            update_position_anchors(valid_detections, tracker, frame_invigilator_ids, args)
            report["summary"]["max_invigilators_in_frame"] = max(
                report["summary"]["max_invigilators_in_frame"], len(frame_invigilator_ids)
            )

            student_indices = [
                idx for idx, det in enumerate(valid_detections) if det["track_id"] not in frame_invigilator_ids
            ]
            student_detections = [valid_detections[idx] for idx in student_indices]

            if student_detections:
                t0 = time.perf_counter()
                boxes = np.asarray([det["bbox"] for det in student_detections], dtype=np.float32)
                init_default_scope("mmpose")
                pose_results = inference_topdown(pose_model, frame, bboxes=boxes, bbox_format="xyxy")
                for det, pose_instance in zip(student_detections, iter_pose_instances(pose_results)):
                    pose = normalize_keypoints_by_bbox(
                        pose_instance["keypoints"],
                        pose_instance["keypoint_scores"],
                        pose_instance["bbox"],
                    ).tolist()
                    pose_behavior = classify_pose(behavior_model, idx_to_label, pose, device)
                    det.update(pose_instance)
                    det["pose"] = pose
                    model_thresholded_label = apply_behavior_threshold(
                        pose_behavior["raw_label"], pose_behavior["raw_score"], args
                    )
                    rule_adjusted_label = apply_single_frame_rules(
                        pose_behavior,
                        model_thresholded_label,
                        pose,
                        det["bbox"],
                        width,
                        height,
                        args,
                    )
                    pose_behavior["model_thresholded_label"] = model_thresholded_label
                    pose_behavior["thresholded_label"] = rule_adjusted_label
                    track = tracker.tracks[det["track_id"]]
                    track_switched = box_motion_is_track_switch(track, det["bbox"], width, height, args)
                    if track_switched:
                        reset_track_history_after_switch(track)
                    all_rule_scores = evaluate_all_single_frame_rules(
                        pose,
                        det["bbox"],
                        width,
                        height,
                        args,
                    )
                    apply_standing_height_change_rule(all_rule_scores, track, det["bbox"], height, args)
                    update_track_height_history(track, det["bbox"], height, args)
                    track["last_bbox"] = tuple(float(v) for v in det["bbox"])
                    pose_behavior["all_rule_scores"] = all_rule_scores
                    pose_behavior["track_switch_reset"] = bool(track_switched)
                    det["pose_behavior"] = pose_behavior
                timing["pose"] += time.perf_counter() - t0

            t0 = time.perf_counter()
            image_behavior_predictions = classify_roi_behavior(roi_heads, roi_features, student_indices)
            for idx, image_behavior in zip(student_indices, image_behavior_predictions):
                valid_detections[idx]["image_behavior"] = image_behavior
            timing["roi_heads"] += time.perf_counter() - t0

            t0 = time.perf_counter()
            frame_has_abnormal = False
            for det in valid_detections:
                track = tracker.tracks[det["track_id"]]
                pose_behavior = det.get("pose_behavior")
                identity = det.get("identity", {})
                inv_score = float(track.get("inv_score", 0.0))
                inv_history_frames = int(track.get("inv_confirmed_frames", 0))
                inv_history_bonus = invigilator_history_bonus(track, args)

                if det["track_id"] in frame_invigilator_ids:
                    clear_abnormal_state(track)
                    final_label = "invigilator"
                    final_conf = max(inv_score, float(identity.get("confidence", 0.0)))
                    confirmed = False
                    role = "invigilator"
                    hybrid_behavior = None
                    raw_label = "invigilator"
                    raw_conf = final_conf
                else:
                    if pose_behavior is None:
                        continue
                    role = "student"
                    hybrid_behavior = fuse_pose_and_image_behavior(pose_behavior, det.get("image_behavior"))
                    det["hybrid_behavior"] = hybrid_behavior
                    raw_label = hybrid_behavior["label"]
                    raw_conf = hybrid_behavior["confidence"]
                    final_label, final_conf, confirmed = apply_temporal_rule(track, raw_label, raw_conf, args)

                final_name = CLASS_NAMES.get(final_label, final_label)
                if final_label in ABNORMAL_LABELS:
                    report["summary"]["confirmed_abnormal_boxes"] += 1
                    frame_has_abnormal = True
                if final_label == "invigilator":
                    report["summary"]["invigilator_boxes"] += 1
                if role == "student":
                    report["summary"]["student_boxes"] += 1

                label = f"ID{det['track_id']} {final_name} {final_conf:.2f}"
                if confirmed and final_label in ABNORMAL_LABELS:
                    label += " CONFIRMED"
                draw_box(frame, det["bbox"], label, final_label)

                report["summary"]["person_boxes"] += 1
                report["summary"]["by_class"][final_name] = report["summary"]["by_class"].get(final_name, 0) + 1
                if pose_behavior is not None:
                    report["summary"]["by_pose_label"][pose_behavior["thresholded_label"]] = (
                        report["summary"]["by_pose_label"].get(pose_behavior["thresholded_label"], 0) + 1
                    )
                image_label = (det.get("image_behavior") or {}).get("label")
                if image_label is not None:
                    report["summary"]["by_image_behavior"][image_label] = (
                        report["summary"]["by_image_behavior"].get(image_label, 0) + 1
                    )
                report["summary"]["by_hybrid_behavior"][raw_label] = (
                    report["summary"]["by_hybrid_behavior"].get(raw_label, 0) + 1
                )
                report["events"].append(
                    {
                        "frame": frame_idx,
                        "time": format_timestamp(frame_idx, fps),
                        "track_id": det["track_id"],
                        "track_age": det.get("track_age", 0),
                        "track_role": role,
                        "identity": identity,
                        "raw_label": pose_behavior["raw_label"] if pose_behavior else None,
                        "model_thresholded_label": pose_behavior.get("model_thresholded_label") if pose_behavior else None,
                        "thresholded_label": raw_label,
                        "pose_thresholded_label": pose_behavior.get("thresholded_label") if pose_behavior else None,
                        "single_frame_rule": pose_behavior.get("rule") if pose_behavior else None,
                        "all_single_frame_rules": pose_behavior.get("all_rule_scores") if pose_behavior else None,
                        "rule_fusion": pose_behavior.get("rule_fusion") if pose_behavior else None,
                        "image_behavior": det.get("image_behavior"),
                        "hybrid_behavior": hybrid_behavior,
                        "raw_confidence": raw_conf,
                        "invigilator_score": inv_score,
                        "invigilator_history_frames": inv_history_frames,
                        "invigilator_history_bonus": inv_history_bonus,
                        "track_switch_reset": bool(pose_behavior.get("track_switch_reset", False))
                        if pose_behavior
                        else False,
                        "final_label": final_label,
                        "final_name": final_name,
                        "final_confidence": final_conf,
                        "confirmed_abnormal": bool(confirmed and final_label in ABNORMAL_LABELS),
                        "det_confidence": det["det_confidence"],
                        "bbox": {
                            "x1": det["bbox"][0],
                            "y1": det["bbox"][1],
                            "x2": det["bbox"][2],
                            "y2": det["bbox"][3],
                        },
                    }
                )

            if frame_has_abnormal:
                report["summary"]["confirmed_abnormal_frames"] += 1
            previous_invigilator_ids = frame_invigilator_ids

            draw_status(
                frame,
                f"Frame: {frame_idx + 1}/{frames_to_process} | IDs: {len(valid_detections)}/{id_pool_size} | "
                f"Inv: {len(frame_invigilator_ids)}/{args.max_invigilators} | ROI Heads",
            )
            timing["postprocess_draw"] += time.perf_counter() - t0

            if writer is not None:
                t0 = time.perf_counter()
                writer.write(frame)
                timing["write"] += time.perf_counter() - t0

            if args.show:
                cv2.imshow("Tracked RTM-Pose behavior", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_idx += 1
            if args.progress_interval > 0 and frame_idx % args.progress_interval == 0:
                elapsed = time.time() - start_time
                speed = frame_idx / elapsed if elapsed > 0 else 0
                print(f"Processed {frame_idx}/{frames_to_process} frames, {speed:.2f} FPS")

    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if args.show:
            cv2.destroyAllWindows()

    report["processed_frames"] = frame_idx
    report["elapsed_seconds"] = round(time.time() - start_time, 3)
    elapsed = time.time() - start_time
    timed_total = sum(timing.values())
    no_write_elapsed = max(1e-6, elapsed - timing["write"])
    model_elapsed = max(
        1e-6,
        timing["detect"] + timing["roi_heads"] + timing["pose"] + timing["track"] + timing["postprocess_draw"],
    )
    report["timing"] = {
        "seconds": {key: round(value, 4) for key, value in timing.items()},
        "seconds_per_frame": {
            key: round(value / max(frame_idx, 1), 6) for key, value in timing.items()
        },
        "timed_total_seconds": round(timed_total, 4),
        "wall_seconds": round(elapsed, 4),
        "actual_fps": round(frame_idx / max(elapsed, 1e-6), 3),
        "potential_fps_without_video_write": round(frame_idx / no_write_elapsed, 3),
        "potential_fps_model_pipeline_only": round(frame_idx / model_elapsed, 3),
    }
    with args.report.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)

    if writer is not None:
        print(f"Annotated video saved to: {args.output}")
    else:
        print("Annotated video writing disabled.")
    print(f"Report saved to: {args.report}")
    print(f"Processed frames: {frame_idx}")
    print(f"Average speed: {frame_idx / elapsed:.2f} FPS" if elapsed > 0 else "Average speed: unknown")
    print(f"Potential FPS without video write: {frame_idx / no_write_elapsed:.2f}")
    print(f"Potential FPS model pipeline only: {frame_idx / model_elapsed:.2f}")
    print(f"Person boxes: {report['summary']['person_boxes']}")
    print(f"Student boxes: {report['summary']['student_boxes']}")
    print(f"Confirmed abnormal boxes: {report['summary']['confirmed_abnormal_boxes']}")
    print(f"Confirmed abnormal frames: {report['summary']['confirmed_abnormal_frames']}")
    print(f"Invigilator boxes: {report['summary']['invigilator_boxes']}")


if __name__ == "__main__":
    main()
