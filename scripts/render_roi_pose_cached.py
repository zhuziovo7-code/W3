#!/usr/bin/env python3
"""Render ROI+Pose behavior video from cached model outputs and a fusion config."""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter, defaultdict
from datetime import timedelta
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[1]
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
CHINESE_FONT_CANDIDATES = [
    Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
    Path("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf"),
    Path("/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc"),
]
CHINESE_FONT_PATH = next((path for path in CHINESE_FONT_CANDIDATES if path.exists()), None)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay cached ROI+Pose scores with new fusion parameters.")
    parser.add_argument("--video", type=Path, default=PROJECT_ROOT / "模拟考试1.mp4")
    parser.add_argument(
        "--cache-report",
        type=Path,
        default=PROJECT_ROOT / "outputs/reports/roi_pose_behavior_first3000_cache.json",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs/behavior_decision/roi_pose_behavior_acc_tuned.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs/visualizations/roi_pose_cached_tuned_first3000.mp4",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "outputs/reports/roi_pose_cached_tuned_first3000.json",
    )
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--progress-interval", type=int, default=500)
    parser.add_argument("--enable-warnings", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--warning-init-frames", type=int, default=300)
    parser.add_argument("--warning-tail-ignore-frames", type=int, default=0)
    parser.add_argument("--student-count-min-drop", type=int, default=1)
    parser.add_argument("--invigilator-count-min-drop", type=int, default=1)
    parser.add_argument("--count-alert-frames", type=int, default=150)
    parser.add_argument("--student-count-alert-frames", type=int)
    parser.add_argument("--invigilator-count-alert-frames", type=int)
    parser.add_argument("--gather-min-count", type=int, default=2)
    parser.add_argument("--gather-distance-ratio", type=float, default=0.12)
    parser.add_argument("--gather-alert-frames", type=int, default=150)
    parser.add_argument("--group-head-turn-min-count", type=int, default=3)
    parser.add_argument("--group-head-turn-alert-frames", type=int, default=60)
    parser.add_argument("--long-head-turn-alert-frames", type=int, default=150)
    parser.add_argument("--repeated-head-turn-alert-count", type=int, default=5)
    parser.add_argument("--group-raise-hand-min-count", type=int, default=3)
    parser.add_argument("--group-raise-hand-alert-frames", type=int, default=60)
    parser.add_argument("--group-standing-min-count", type=int, default=3)
    parser.add_argument("--group-standing-alert-frames", type=int, default=60)
    parser.add_argument("--whisper-neighbor-ratio", type=float, default=0.18)
    parser.add_argument("--whisper-alert-frames", type=int, default=60)
    parser.add_argument(
        "--allowed-behavior-track-ids",
        type=str,
        default="",
        help="Semicolon separated label:id list filters, e.g. carry_phone:4;hand_under_table_look_down:1,3.",
    )
    parser.add_argument("--draw-status", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--draw-abnormal-only",
        action="store_true",
        help="Only draw confirmed abnormal boxes and warning panel; keep report statistics.",
    )
    parser.add_argument("--invigilator-seat-alert-frames", type=int, default=5)
    parser.add_argument("--invigilator-seat-distance-ratio", type=float, default=0.075)
    parser.add_argument("--invigilator-seat-score-thr", type=float, default=0.40)
    parser.add_argument("--invigilator-seat-track-ids", type=str, default="")
    parser.add_argument("--invigilator-seat-start-frame", type=int, default=0)
    parser.add_argument("--invigilator-seat-recover-frames", type=int, default=120)
    parser.add_argument("--sitting-height-ratio-thr", type=float, default=0.24)
    parser.add_argument("--invigilator-seat-min-height-ratio", type=float, default=0.0)
    parser.add_argument("--invigilator-seat-height-ratio-thr", type=float, default=0.32)
    parser.add_argument("--invigilator-seat-min-history-frames", type=int, default=0)
    return parser.parse_args()


def format_timestamp(frame_idx: int, fps: float) -> str:
    return str(timedelta(seconds=frame_idx / fps))[:-3]


def parse_allowed_behavior_track_ids(value: str | None) -> dict[str, set[int]]:
    mapping: dict[str, set[int]] = {}
    if not value:
        return mapping
    for chunk in str(value).split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ":" not in chunk:
            raise ValueError(f"Invalid --allowed-behavior-track-ids item: {chunk}")
        label, ids_text = chunk.split(":", 1)
        label = label.strip()
        ids = {int(item.strip()) for item in ids_text.split(",") if item.strip()}
        if label and ids:
            mapping[label] = ids
    return mapping


def parse_id_set(value: str | None) -> set[int]:
    if not value:
        return set()
    return {int(item.strip()) for item in str(value).split(",") if item.strip()}


def behavior_allowed_for_track(label: str, track_id: int, allowed: dict[str, set[int]]) -> bool:
    ids = allowed.get(label)
    return ids is None or int(track_id) in ids


def wrap_message(text: str, max_chars: int = 24) -> list[str]:
    text = str(text)
    if len(text) <= max_chars:
        return [text]
    return [text[i : i + max_chars] for i in range(0, len(text), max_chars)]


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


def put_text(frame, text: str, pos: tuple[int, int], font_size: int, color: tuple[int, int, int]) -> None:
    if CHINESE_FONT_PATH is None:
        cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, font_size / 32.0, color, 2, cv2.LINE_AA)
        return
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(str(CHINESE_FONT_PATH), font_size)
    rgb = (int(color[2]), int(color[1]), int(color[0]))
    draw.text(pos, text, fill=rgb, font=font)
    frame[:] = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def draw_warning_panel(frame, warnings: list[dict]) -> None:
    if not warnings:
        return
    shown = warnings[:5]
    panel_w = min(660, frame.shape[1] - 30)
    line_h = 30
    wrapped = []
    for idx, warning in enumerate(shown, start=1):
        lines = wrap_message(f"{idx}. {warning['message']}", max_chars=28)
        wrapped.append(lines)
    panel_h = 48 + sum(max(1, len(lines)) * line_h for lines in wrapped)
    x1 = max(10, frame.shape[1] - panel_w - 20)
    y1 = 18
    x2 = x1 + panel_w
    y2 = y1 + panel_h
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.58, frame, 0.42, 0, frame)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 255), 2)
    put_text(frame, "预警信息", (x1 + 14, y1 + 8), 25, (0, 220, 255))
    y = y1 + 46
    for lines in wrapped:
        for line in lines:
            put_text(frame, line, (x1 + 14, y), 20, (80, 80, 255))
            y += line_h


def component_score(event: dict, label: str, key: str) -> float:
    for candidate in (event.get("hybrid_behavior") or {}).get("candidates") or []:
        if candidate.get("label") == label:
            value = candidate.get(key, 0.0)
            if value is None:
                return 0.0
            return float(value)
    return 0.0


def fused_label(event: dict, config: dict) -> tuple[str, float, dict | None]:
    normal_score = component_score(event, "normal", "model_score")
    if normal_score <= 0:
        normal_score = component_score(event, "normal", "pose_score")
    best_label = "normal"
    best_score = normal_score
    best_detail = None
    for label in sorted(ABNORMAL_LABELS):
        params = config[label]
        model_score = component_score(event, label, "model_score")
        if model_score <= 0:
            model_score = component_score(event, label, "pose_score")
        rule_score = component_score(event, label, "rule_score")
        score = (
            float(params["model_weight"]) * model_score
            + float(params["rule_weight"]) * rule_score
        )
        if score >= float(params["fused_thr"]) and score > best_score:
            best_label = label
            best_score = score
            best_detail = {
                "label": label,
                "model_score": model_score,
                "pose_score": model_score,
                "rule_score": rule_score,
                "image_score": None,
                "fused_score": score,
                "fused_thr": float(params["fused_thr"]),
                "model_weight": float(params["model_weight"]),
                "pose_weight": float(params["model_weight"]),
                "rule_weight": float(params["rule_weight"]),
                "image_weight": 0.0,
            }
    return best_label, best_score, best_detail


def has_pick_low_body_evidence(event: dict) -> bool:
    rule = (event.get("all_single_frame_rules") or {}).get("pick_suspicious_item") or {}
    reasons = rule.get("reasons", []) or []
    return float(rule.get("score", 0.0) or 0.0) >= 0.5 and "wrist_near_lower_body" in reasons


def bbox_center(box: list[float]) -> tuple[float, float]:
    return ((float(box[0]) + float(box[2])) * 0.5, (float(box[1]) + float(box[3])) * 0.5)


def is_sitting_pose(event: dict, box: list[float], frame_height: int, args: argparse.Namespace) -> bool:
    height_ratio = max(0.0, (float(box[3]) - float(box[1])) / max(1.0, float(frame_height)))
    if height_ratio > float(args.sitting_height_ratio_thr):
        return False
    keypoints = event.get("pose_keypoints") or []
    scores = event.get("pose_keypoint_scores") or []
    if len(keypoints) < 17 or len(scores) < 17:
        return False
    lower_body = [11, 12, 13, 14, 15, 16]
    visible_lower = sum(1 for idx in lower_body if float(scores[idx] or 0.0) >= 0.25)
    visible_upper = sum(1 for idx in [5, 6, 11, 12] if float(scores[idx] or 0.0) >= 0.25)
    return visible_upper >= 3 and visible_lower >= 4


def estimate_head_direction(event: dict, box: list[float]) -> str | None:
    keypoints = event.get("pose_keypoints") or []
    scores = event.get("pose_keypoint_scores") or []
    if len(keypoints) < 7 or len(scores) < 7:
        return None
    if float(scores[0] or 0.0) < 0.25:
        return None

    nose_x = float(keypoints[0][0])
    left_shoulder_ok = float(scores[5] or 0.0) >= 0.2
    right_shoulder_ok = float(scores[6] or 0.0) >= 0.2
    if left_shoulder_ok and right_shoulder_ok:
        left_shoulder_x = float(keypoints[5][0])
        right_shoulder_x = float(keypoints[6][0])
        center_x = (left_shoulder_x + right_shoulder_x) * 0.5
        ref_width = abs(left_shoulder_x - right_shoulder_x)
    else:
        center_x = (float(box[0]) + float(box[2])) * 0.5
        ref_width = float(box[2]) - float(box[0])

    if ref_width <= 1:
        return None
    offset = (nose_x - center_x) / ref_width
    if offset >= 0.12:
        return "right"
    if offset <= -0.12:
        return "left"
    return None


class WarningState:
    def __init__(self, args: argparse.Namespace, frame_width: int, fps: float) -> None:
        self.enabled = bool(args.enable_warnings)
        self.fps = float(fps)
        self.init_frames = max(1, int(args.warning_init_frames))
        self.student_min_drop = max(1, int(args.student_count_min_drop))
        self.invigilator_min_drop = max(1, int(args.invigilator_count_min_drop))
        self.count_alert_frames = max(1, int(args.count_alert_frames))
        self.student_count_alert_frames = max(
            1, int(args.student_count_alert_frames or args.count_alert_frames)
        )
        self.invigilator_count_alert_frames = max(
            1, int(args.invigilator_count_alert_frames or args.count_alert_frames)
        )
        self.gather_min_count = max(2, int(args.gather_min_count))
        self.gather_distance_px = max(1.0, float(args.gather_distance_ratio) * float(frame_width))
        self.gather_alert_frames = max(1, int(args.gather_alert_frames))
        self.group_head_turn_min_count = max(2, int(args.group_head_turn_min_count))
        self.group_head_turn_alert_frames = max(1, int(args.group_head_turn_alert_frames))
        self.long_head_turn_alert_frames = max(1, int(args.long_head_turn_alert_frames))
        self.repeated_head_turn_alert_count = max(1, int(args.repeated_head_turn_alert_count))
        self.group_raise_hand_min_count = max(2, int(args.group_raise_hand_min_count))
        self.group_raise_hand_alert_frames = max(1, int(args.group_raise_hand_alert_frames))
        self.group_standing_min_count = max(2, int(args.group_standing_min_count))
        self.group_standing_alert_frames = max(1, int(args.group_standing_alert_frames))
        self.whisper_neighbor_px = max(1.0, float(args.whisper_neighbor_ratio) * float(frame_width))
        self.whisper_alert_frames = max(1, int(args.whisper_alert_frames))
        self.invigilator_seat_alert_frames = max(1, int(args.invigilator_seat_alert_frames))
        self.invigilator_seat_distance_px = max(1.0, float(args.invigilator_seat_distance_ratio) * float(frame_width))
        self.invigilator_seat_score_thr = max(0.0, float(args.invigilator_seat_score_thr))
        self.invigilator_seat_track_ids = parse_id_set(args.invigilator_seat_track_ids)
        self.invigilator_seat_start_frame = max(0, int(args.invigilator_seat_start_frame))
        self.invigilator_seat_recover_frames = max(0, int(args.invigilator_seat_recover_frames))
        self.invigilator_seat_min_height_ratio = max(0.0, float(args.invigilator_seat_min_height_ratio))
        self.invigilator_seat_height_ratio_thr = max(0.0, float(args.invigilator_seat_height_ratio_thr))
        self.invigilator_seat_min_history_frames = max(0, int(args.invigilator_seat_min_history_frames))

        self.student_hist: Counter[int] = Counter()
        self.invigilator_hist: Counter[int] = Counter()
        self.student_anchor_samples: dict[int, list[tuple[float, float]]] = defaultdict(list)
        self.student_seat_anchors: list[tuple[int, float, float]] = []
        self.student_base_count: int | None = None
        self.invigilator_base_count: int | None = None
        self.student_drop_frames = 0
        self.invigilator_drop_frames = 0
        self.gather_frames = 0
        self.group_head_turn_frames = {"left": 0, "right": 0}
        self.head_turn_streaks: dict[int, int] = {}
        self.head_turn_episode_counts: Counter[int] = Counter()
        self.head_turn_active_ids: set[int] = set()
        self.group_raise_hand_frames = 0
        self.group_standing_frames = 0
        self.whisper_frames = 0
        self.invigilator_seat_frames = 0
        self.invigilator_seat_recover_count = 0
        self.last_invigilator_seat_match: list[dict] = []
        self.current_student_track_ids: set[int] = set()

    def update(
        self,
        frame_idx: int,
        students: list[dict],
        invigilators: list[dict],
        invigilator_seat_candidates: list[dict] | None = None,
    ) -> list[dict]:
        if not self.enabled:
            return []
        student_count = len(students)
        invigilator_count = len(invigilators)
        if frame_idx < self.init_frames:
            self.student_hist[student_count] += 1
            self.invigilator_hist[invigilator_count] += 1
            for student in students:
                cx, cy = bbox_center(student["box"])
                self.student_anchor_samples[int(student["track_id"])].append((cx, cy))
            if frame_idx == self.init_frames - 1:
                self.student_base_count = self.student_hist.most_common(1)[0][0]
                self.invigilator_base_count = self.invigilator_hist.most_common(1)[0][0]
                self.student_seat_anchors = self._build_student_seat_anchors()
            return []

        if self.student_base_count is None:
            self.student_base_count = self.student_hist.most_common(1)[0][0] if self.student_hist else student_count
        if self.invigilator_base_count is None:
            self.invigilator_base_count = (
                self.invigilator_hist.most_common(1)[0][0] if self.invigilator_hist else invigilator_count
            )
        if not self.student_seat_anchors and self.student_anchor_samples:
            self.student_seat_anchors = self._build_student_seat_anchors()
        self.current_student_track_ids = {int(student["track_id"]) for student in students}

        warnings = []
        if student_count <= self.student_base_count - self.student_min_drop:
            self.student_drop_frames += 1
        else:
            self.student_drop_frames = 0
        if self.student_drop_frames >= self.student_count_alert_frames:
            warning = {
                "type": "student_count_drop",
                "message": f"考生数量减少: {student_count}/{self.student_base_count}",
                "level": "orange",
                "duration_frames": self.student_drop_frames,
                "base_count": self.student_base_count,
                "current_count": student_count,
            }
            warnings.append(warning)

        if invigilator_count <= self.invigilator_base_count - self.invigilator_min_drop:
            self.invigilator_drop_frames += 1
        else:
            self.invigilator_drop_frames = 0
        if self.invigilator_drop_frames >= self.invigilator_count_alert_frames:
            warning = {
                "type": "invigilator_count_drop",
                "message": f"监考员数量减少: {invigilator_count}/{self.invigilator_base_count}",
                "level": "red",
                "duration_frames": self.invigilator_drop_frames,
                "base_count": self.invigilator_base_count,
                "current_count": invigilator_count,
            }
            warnings.append(warning)

        gather_pair = self._closest_invigilator_pair(invigilators)
        if gather_pair is not None:
            self.gather_frames += 1
        else:
            self.gather_frames = 0
        if self.gather_frames >= self.gather_alert_frames:
            warning = {
                "type": "invigilator_gather",
                "message": f"监考员聚集: ID{','.join(map(str, gather_pair['track_ids'])) if gather_pair else '-'}",
                "level": "orange",
                "duration_frames": self.gather_frames,
            }
            if gather_pair is not None:
                warning["distance_px"] = round(gather_pair["distance_px"], 2)
                warning["track_ids"] = gather_pair["track_ids"]
            warnings.append(warning)

        group_head_turn_warning = self._group_head_turn_warning(students)
        if group_head_turn_warning is not None:
            warnings.append(group_head_turn_warning)

        head_turn_warnings = self._individual_head_turn_warnings(students)
        warnings.extend(head_turn_warnings)

        raise_hand_warning = self._group_behavior_warning(
            students,
            "raise_hand",
            self.group_raise_hand_min_count,
            self.group_raise_hand_alert_frames,
            "group_raise_hand_frames",
            "group_raise_hand",
            "多人举手",
        )
        if raise_hand_warning is not None:
            warnings.append(raise_hand_warning)

        standing_warning = self._group_behavior_warning(
            students,
            "student_standing",
            self.group_standing_min_count,
            self.group_standing_alert_frames,
            "group_standing_frames",
            "group_student_standing",
            "多人起立",
        )
        if standing_warning is not None:
            warnings.append(standing_warning)

        whisper_warning = self._whisper_warning(students)
        if whisper_warning is not None:
            warnings.append(whisper_warning)

        seat_candidates = list(invigilator_seat_candidates or [])
        invigilator_seat_warning = self._invigilator_in_student_seat_warning(frame_idx, seat_candidates)
        if invigilator_seat_warning is not None:
            warnings.append(invigilator_seat_warning)
        return warnings

    def _build_student_seat_anchors(self) -> list[tuple[int, float, float]]:
        anchors = []
        for track_id, samples in self.student_anchor_samples.items():
            if len(samples) < max(5, self.init_frames // 6):
                continue
            xs = sorted(point[0] for point in samples)
            ys = sorted(point[1] for point in samples)
            mid = len(samples) // 2
            anchors.append((int(track_id), float(xs[mid]), float(ys[mid])))
        return anchors

    def _closest_invigilator_pair(self, invigilators: list[dict]) -> dict | None:
        if len(invigilators) < self.gather_min_count:
            return None
        closest = None
        for i, left in enumerate(invigilators):
            for right in invigilators[i + 1 :]:
                lx, ly = bbox_center(left["box"])
                rx, ry = bbox_center(right["box"])
                distance = math.hypot(lx - rx, ly - ry)
                if closest is None or distance < closest["distance_px"]:
                    closest = {
                        "distance_px": distance,
                    "track_ids": [left["track_id"], right["track_id"]],
                }
        if closest is not None and closest["distance_px"] <= self.gather_distance_px:
            return closest
        return None

    def _group_behavior_warning(
        self,
        students: list[dict],
        label: str,
        min_count: int,
        alert_frames: int,
        counter_name: str,
        warning_type: str,
        message_prefix: str,
    ) -> dict | None:
        matched = [student for student in students if student.get("final_label") == label]
        if len(matched) >= min_count:
            setattr(self, counter_name, int(getattr(self, counter_name)) + 1)
        else:
            setattr(self, counter_name, 0)
        duration = int(getattr(self, counter_name))
        if duration < alert_frames:
            return None
        track_ids = [int(student["track_id"]) for student in matched[:min_count]]
        return {
            "type": warning_type,
            "message": f"{message_prefix}: {len(matched)}人 ID{','.join(map(str, track_ids))}",
            "level": "red",
            "count": len(matched),
            "track_ids": track_ids,
            "duration_frames": duration,
        }

    def _group_head_turn_warning(self, students: list[dict]) -> dict | None:
        matched = [student for student in students if student.get("final_label") == "head_turn"]
        if len(matched) >= self.group_head_turn_min_count:
            self.group_head_turn_frames["left"] += 1
        else:
            self.group_head_turn_frames["left"] = 0
        duration = self.group_head_turn_frames["left"]
        if duration < self.group_head_turn_alert_frames:
            return None
        track_ids = [int(student["track_id"]) for student in matched[: self.group_head_turn_min_count]]
        return {
            "type": "group_head_turn",
            "message": f"多人左右偏头: {len(matched)}人 ID{','.join(map(str, track_ids))}",
            "level": "red",
            "count": len(matched),
            "track_ids": track_ids,
            "duration_frames": duration,
        }

    def _individual_head_turn_warnings(self, students: list[dict]) -> list[dict]:
        warnings = []
        current_head_turn_ids = {
            int(student["track_id"]) for student in students if student.get("final_label") == "head_turn"
        }
        known_student_ids = {int(student["track_id"]) for student in students}

        for track_id in current_head_turn_ids:
            previous = int(self.head_turn_streaks.get(track_id, 0))
            self.head_turn_streaks[track_id] = previous + 1
            if track_id not in self.head_turn_active_ids:
                self.head_turn_episode_counts[track_id] += 1
                self.head_turn_active_ids.add(track_id)

        for track_id in list(self.head_turn_streaks):
            if track_id not in current_head_turn_ids:
                self.head_turn_streaks[track_id] = 0
        self.head_turn_active_ids &= current_head_turn_ids

        for track_id in current_head_turn_ids:
            duration_frames = int(self.head_turn_streaks.get(track_id, 0))
            if duration_frames >= self.long_head_turn_alert_frames:
                duration_seconds = duration_frames / max(self.fps, 1e-6)
                warnings.append(
                    {
                        "type": "long_head_turn",
                        "message": f"长时间左右偏头: ID{track_id} 持续{duration_seconds:.1f}秒",
                        "level": "orange",
                        "track_ids": [track_id],
                        "duration_frames": duration_frames,
                        "duration_seconds": round(duration_seconds, 2),
                    }
                )

            episode_count = int(self.head_turn_episode_counts.get(track_id, 0))
            if episode_count >= self.repeated_head_turn_alert_count:
                warnings.append(
                    {
                        "type": "repeated_head_turn",
                        "message": f"多次左右偏头: ID{track_id} 累计{episode_count}次",
                        "level": "orange",
                        "track_ids": [track_id],
                        "count": episode_count,
                        "duration_frames": duration_frames,
                    }
                )

        for track_id in list(self.head_turn_episode_counts):
            if track_id not in known_student_ids:
                self.head_turn_streaks.pop(track_id, None)
                self.head_turn_episode_counts.pop(track_id, None)
                self.head_turn_active_ids.discard(track_id)
        return warnings

    def _whisper_warning(self, students: list[dict]) -> dict | None:
        head_turn_students = [
            student
            for student in students
            if student.get("final_label") == "head_turn" and student.get("head_direction") in {"left", "right"}
        ]
        matched_pair = None
        for i, left in enumerate(head_turn_students):
            for right in head_turn_students[i + 1 :]:
                lx, ly = bbox_center(left["box"])
                rx, ry = bbox_center(right["box"])
                if lx > rx:
                    left, right = right, left
                    lx, ly, rx, ry = rx, ry, lx, ly
                distance = math.hypot(lx - rx, ly - ry)
                if distance > self.whisper_neighbor_px:
                    continue
                if left.get("head_direction") == "right" and right.get("head_direction") == "left":
                    matched_pair = {
                        "track_ids": [int(left["track_id"]), int(right["track_id"])],
                        "distance_px": round(distance, 2),
                    }
                    break
            if matched_pair is not None:
                break

        if matched_pair is not None:
            self.whisper_frames += 1
        else:
            self.whisper_frames = 0
        if self.whisper_frames < self.whisper_alert_frames:
            return None
        return {
            "type": "student_whisper",
            "message": f"交头接耳: ID{','.join(map(str, matched_pair['track_ids']))}",
            "level": "red",
            "track_ids": matched_pair["track_ids"],
            "distance_px": matched_pair["distance_px"],
            "duration_frames": self.whisper_frames,
        }

    def _invigilator_in_student_seat_warning(self, frame_idx: int, invigilators: list[dict]) -> dict | None:
        if frame_idx < self.invigilator_seat_start_frame:
            self.invigilator_seat_frames = 0
            self.invigilator_seat_recover_count = 0
            self.last_invigilator_seat_match = []
            return None
        matched = []
        for invigilator in invigilators:
            track_id = int(invigilator["track_id"])
            if self.invigilator_seat_track_ids and track_id not in self.invigilator_seat_track_ids:
                continue
            seat_posture_ok = bool(invigilator.get("is_sitting") or invigilator.get("seat_height_ok"))
            if not self.invigilator_seat_track_ids and not seat_posture_ok:
                continue
            inv_score = float(invigilator.get("invigilator_score", 0.0) or 0.0)
            if inv_score < self.invigilator_seat_score_thr:
                continue
            height_ratio = float(invigilator.get("height_ratio", 0.0) or 0.0)
            if height_ratio < self.invigilator_seat_min_height_ratio:
                continue
            history_frames = int(invigilator.get("invigilator_history_frames", 0) or 0)
            if history_frames < self.invigilator_seat_min_history_frames:
                continue
            cx, cy = bbox_center(invigilator["box"])
            nearest = None
            for seat_id, sx, sy in self.student_seat_anchors:
                distance = math.hypot(cx - sx, cy - sy)
                if nearest is None or distance < nearest[1]:
                    nearest = (seat_id, distance)
            if (
                nearest is not None
                and nearest[1] <= self.invigilator_seat_distance_px
                and int(nearest[0]) not in self.current_student_track_ids
                and int(nearest[0]) != track_id
            ):
                matched.append(
                    {
                        "track_id": int(invigilator["track_id"]),
                        "seat_id": int(nearest[0]),
                        "distance_px": round(nearest[1], 2),
                    }
                )
        if matched:
            self.invigilator_seat_frames += 1
            self.invigilator_seat_recover_count = 0
            self.last_invigilator_seat_match = matched
        else:
            if self.invigilator_seat_frames > 0 and self.invigilator_seat_recover_count < self.invigilator_seat_recover_frames:
                self.invigilator_seat_recover_count += 1
                matched = self.last_invigilator_seat_match
            else:
                self.invigilator_seat_frames = 0
                self.invigilator_seat_recover_count = 0
                self.last_invigilator_seat_match = []
        if self.invigilator_seat_frames < self.invigilator_seat_alert_frames:
            return None
        track_ids = [item["track_id"] for item in matched]
        seat_ids = [item["seat_id"] for item in matched]
        return {
            "type": "invigilator_in_student_seat",
            "message": f"监考员坐在考生位置: 人数{len(matched)} ID{','.join(map(str, track_ids))} 座位ID{','.join(map(str, seat_ids))}",
            "level": "red",
            "count": len(matched),
            "track_ids": track_ids,
            "seat_ids": seat_ids,
            "duration_frames": self.invigilator_seat_frames,
        }

    def report_config(self) -> dict:
        return {
            "enabled": self.enabled,
            "init_frames": self.init_frames,
            "student_count_min_drop": self.student_min_drop,
            "invigilator_count_min_drop": self.invigilator_min_drop,
            "count_alert_frames": self.count_alert_frames,
            "student_count_alert_frames": self.student_count_alert_frames,
            "invigilator_count_alert_frames": self.invigilator_count_alert_frames,
            "gather_min_count": self.gather_min_count,
            "gather_distance_px": round(self.gather_distance_px, 2),
            "gather_alert_frames": self.gather_alert_frames,
            "group_head_turn_min_count": self.group_head_turn_min_count,
            "group_head_turn_alert_frames": self.group_head_turn_alert_frames,
            "long_head_turn_alert_frames": self.long_head_turn_alert_frames,
            "repeated_head_turn_alert_count": self.repeated_head_turn_alert_count,
            "group_raise_hand_min_count": self.group_raise_hand_min_count,
            "group_raise_hand_alert_frames": self.group_raise_hand_alert_frames,
            "group_standing_min_count": self.group_standing_min_count,
            "group_standing_alert_frames": self.group_standing_alert_frames,
            "whisper_neighbor_px": round(self.whisper_neighbor_px, 2),
            "whisper_alert_frames": self.whisper_alert_frames,
            "invigilator_seat_alert_frames": self.invigilator_seat_alert_frames,
            "invigilator_seat_distance_px": round(self.invigilator_seat_distance_px, 2),
            "invigilator_seat_score_thr": self.invigilator_seat_score_thr,
            "invigilator_seat_min_height_ratio": self.invigilator_seat_min_height_ratio,
            "invigilator_seat_height_ratio_thr": float(getattr(self, "invigilator_seat_height_ratio_thr", 0.0)),
            "invigilator_seat_min_history_frames": self.invigilator_seat_min_history_frames,
            "invigilator_seat_track_ids": sorted(self.invigilator_seat_track_ids),
            "invigilator_seat_start_frame": self.invigilator_seat_start_frame,
            "invigilator_seat_recover_frames": self.invigilator_seat_recover_frames,
            "student_seat_anchor_count": len(self.student_seat_anchors),
            "student_base_count": self.student_base_count,
            "invigilator_base_count": self.invigilator_base_count,
        }


def apply_temporal(
    track: dict, raw_label: str, raw_conf: float, config: dict, recover_n: int, event: dict | None = None
) -> tuple[str, float, bool]:
    if (
        track.get("confirmed_label") == "pick_suspicious_item"
        and raw_label in {"normal", "carry_phone", "pass_suspicious_item"}
        and event is not None
        and has_pick_low_body_evidence(event)
    ):
        raw_label = "pick_suspicious_item"

    if raw_label in ABNORMAL_LABELS:
        needed = int(config[raw_label].get("temporal_abnormal_n", 8))
        if track.get("streak_label") == raw_label:
            track["streak_count"] = int(track.get("streak_count", 0)) + 1
        else:
            track["streak_label"] = raw_label
            track["streak_count"] = 1
        track["recover_count"] = 0
        if track["streak_count"] >= needed:
            track["confirmed_label"] = raw_label
            track["confirmed_conf"] = raw_conf
        if track.get("confirmed_label"):
            return track["confirmed_label"], raw_conf, True
        return "normal", raw_conf, False

    track["streak_label"] = None
    track["streak_count"] = 0
    if track.get("confirmed_label"):
        track["recover_count"] = int(track.get("recover_count", 0)) + 1
        if track["recover_count"] >= recover_n:
            track["confirmed_label"] = None
            track["confirmed_conf"] = 0.0
            track["recover_count"] = 0
            return "normal", raw_conf, False
        return track["confirmed_label"], float(track.get("confirmed_conf", raw_conf)), True
    return "normal", raw_conf, False


def main() -> None:
    args = parse_args()
    cache = json.loads(args.cache_report.read_text(encoding="utf-8"))
    config_payload = json.loads(args.config.read_text(encoding="utf-8"))
    behavior_config = config_payload.get("behavior_decision_config", config_payload)
    recover_n = int(config_payload.get("temporal_recover_n", 12))
    allowed_behavior_track_ids = parse_allowed_behavior_track_ids(args.allowed_behavior_track_ids)
    seat_track_ids = parse_id_set(args.invigilator_seat_track_ids)

    events_by_frame: dict[int, list[dict]] = defaultdict(list)
    for event in cache["events"]:
        events_by_frame[int(event["frame"])].append(event)

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_to_process = int(cache.get("processed_frames") or total_frames)
    if args.max_frames:
        frames_to_process = min(frames_to_process, args.max_frames)
    if total_frames > 0:
        frames_to_process = min(frames_to_process, total_frames)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not create output video: {args.output}")

    states: dict[int, dict] = {}
    sitting_holds: dict[int, int] = {}
    warning_state = WarningState(args, width, fps)
    report = {
        "video": str(args.video),
        "cache_report": str(args.cache_report),
        "config": str(args.config),
        "warning_config": {},
        "processed_frames": 0,
        "events": [],
        "warning_events": [],
        "summary": {
            "person_boxes": 0,
            "student_boxes": 0,
            "invigilator_boxes": 0,
            "confirmed_abnormal_boxes": 0,
            "confirmed_abnormal_frames": 0,
            "warning_frames": 0,
            "by_class": {},
            "by_hybrid_behavior": {},
            "by_warning_type": {},
        },
    }
    start = time.time()
    try:
        for frame_idx in range(frames_to_process):
            ok, frame = cap.read()
            if not ok:
                break
            frame_has_abnormal = False
            frame_students: list[dict] = []
            frame_invigilators: list[dict] = []
            invigilator_seat_candidates: list[dict] = []
            for event in events_by_frame.get(frame_idx, []):
                track_id = int(event["track_id"])
                track = states.setdefault(track_id, {})
                bbox = event["bbox"]
                box = [bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]]
                if is_sitting_pose(event, box, height, args):
                    sitting_holds[track_id] = 120
                else:
                    sitting_holds[track_id] = max(0, int(sitting_holds.get(track_id, 0)) - 1)
                is_sitting = sitting_holds.get(track_id, 0) > 0
                role = event.get("track_role", "student")
                if role == "invigilator":
                    track.clear()
                    final_label = "invigilator"
                    raw_label = "invigilator"
                    final_conf = float(event.get("final_confidence", 0.0))
                    confirmed = False
                else:
                    raw_label, raw_conf, detail = fused_label(event, behavior_config)
                    final_label, final_conf, confirmed = apply_temporal(
                        track, raw_label, raw_conf, behavior_config, recover_n, event
                    )
                if final_label in ABNORMAL_LABELS and not behavior_allowed_for_track(
                    final_label, track_id, allowed_behavior_track_ids
                ):
                    final_label = "normal"
                    final_conf = 0.0
                    confirmed = False

                final_name = CLASS_NAMES.get(final_label, final_label)
                if final_label in ABNORMAL_LABELS:
                    report["summary"]["confirmed_abnormal_boxes"] += 1
                    frame_has_abnormal = True
                if final_label == "invigilator":
                    report["summary"]["invigilator_boxes"] += 1
                else:
                    report["summary"]["student_boxes"] += 1
                report["summary"]["person_boxes"] += 1
                report["summary"]["by_class"][final_name] = report["summary"]["by_class"].get(final_name, 0) + 1
                report["summary"]["by_hybrid_behavior"][raw_label] = (
                    report["summary"]["by_hybrid_behavior"].get(raw_label, 0) + 1
                )
                label_text = f"ID{track_id} {final_name} {final_conf:.2f}"
                if confirmed and final_label in ABNORMAL_LABELS:
                    label_text += " CONFIRMED"
                if not args.draw_abnormal_only or final_label in ABNORMAL_LABELS:
                    draw_box(frame, box, label_text, final_label)
                target_for_warning = {
                    "track_id": track_id,
                    "box": box,
                    "final_label": final_label,
                    "is_sitting": is_sitting,
                    "invigilator_score": event.get("invigilator_score", 0.0),
                    "invigilator_history_frames": event.get("invigilator_history_frames", 0),
                    "head_direction": estimate_head_direction(event, box) if final_label == "head_turn" else None,
                }
                height_ratio = max(0.0, (float(box[3]) - float(box[1])) / max(1.0, float(height)))
                target_for_warning["height_ratio"] = height_ratio
                target_for_warning["seat_height_ok"] = height_ratio <= args.invigilator_seat_height_ratio_thr
                if final_label == "invigilator":
                    frame_invigilators.append(target_for_warning)
                else:
                    frame_students.append(target_for_warning)
                seat_candidate_by_id = bool(seat_track_ids and track_id in seat_track_ids)
                invigilator_like_for_seat = final_label == "invigilator" or seat_candidate_by_id
                seat_posture_ok = bool(is_sitting or target_for_warning["seat_height_ok"])
                if (
                    invigilator_like_for_seat
                    and seat_posture_ok
                    and float(event.get("invigilator_score", 0.0) or 0.0) >= args.invigilator_seat_score_thr
                ):
                    invigilator_seat_candidates.append(target_for_warning)
                report["events"].append(
                    {
                        "frame": frame_idx,
                        "time": format_timestamp(frame_idx, fps),
                        "track_id": track_id,
                        "track_role": "invigilator" if final_label == "invigilator" else "student",
                        "thresholded_label": raw_label,
                        "final_label": final_label,
                        "final_name": final_name,
                        "final_confidence": final_conf,
                        "confirmed_abnormal": bool(confirmed and final_label in ABNORMAL_LABELS),
                        "det_confidence": event.get("det_confidence"),
                        "bbox": bbox,
                    }
                )
            if frame_has_abnormal:
                report["summary"]["confirmed_abnormal_frames"] += 1
            active_warnings = warning_state.update(
                frame_idx, frame_students, frame_invigilators, invigilator_seat_candidates
            )
            if args.warning_tail_ignore_frames > 0 and frame_idx >= frames_to_process - args.warning_tail_ignore_frames:
                active_warnings = []
            if active_warnings:
                report["summary"]["warning_frames"] += 1
                for warning in active_warnings:
                    report["summary"]["by_warning_type"][warning["type"]] = (
                        report["summary"]["by_warning_type"].get(warning["type"], 0) + 1
                    )
                report["warning_events"].append(
                    {
                        "frame": frame_idx,
                        "time": format_timestamp(frame_idx, fps),
                        "student_count": len(frame_students),
                        "invigilator_count": len(frame_invigilators),
                        "warnings": active_warnings,
                    }
                )
            draw_warning_panel(frame, active_warnings)
            if args.draw_status and not args.draw_abnormal_only:
                draw_status(frame, f"Frame: {frame_idx + 1}/{frames_to_process} | Cached ROI+Pose replay")
            writer.write(frame)
            report["processed_frames"] = frame_idx + 1
            if args.progress_interval > 0 and (frame_idx + 1) % args.progress_interval == 0:
                elapsed = time.time() - start
                print(f"rendered {frame_idx + 1}/{frames_to_process}, {((frame_idx + 1) / elapsed):.2f} FPS")
    finally:
        cap.release()
        writer.release()

    report["elapsed_seconds"] = round(time.time() - start, 3)
    report["warning_config"] = warning_state.report_config()
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved video: {args.output}")
    print(f"saved report: {args.report}")
    print(f"render FPS: {report['processed_frames'] / max(report['elapsed_seconds'], 1e-6):.2f}")


if __name__ == "__main__":
    main()
