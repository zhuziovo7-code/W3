#!/usr/bin/env python3
"""Sample a video stream and append abnormal behavior / warning events as JSONL."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import cv2
import numpy as np
import torch
from mmdet.apis import inference_detector, init_detector
from mmengine.registry import init_default_scope
from mmpose.apis import inference_topdown, init_model
from mmpose.utils import register_all_modules as register_mmpose_modules
from mmdet.utils import register_all_modules as register_mmdet_modules

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import infer_video_behavior_roi_pose as roi
import render_roi_pose_cached as cached_render


DEFAULT_RUNTIME_CONFIG = PROJECT_ROOT / "configs/runtime/stream_runtime.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run sampled ROI+Pose inference on a video stream.")
    parser.add_argument("--source", default=str(PROJECT_ROOT / "模拟考试1.mp4"), help="RTSP URL or video file path.")
    parser.add_argument("--stream-id", default=None)
    parser.add_argument("--runtime-config", type=Path, default=DEFAULT_RUNTIME_CONFIG)
    parser.add_argument("--output-jsonl", type=Path, default=PROJECT_ROOT / "outputs/stream/exam_room_001_events.jsonl")
    parser.add_argument("--latest-json", type=Path, default=PROJECT_ROOT / "outputs/stream/exam_room_001_latest.json")
    parser.add_argument("--sample-interval-frames", type=int, help="Override config sampleIntervalFrames.")
    parser.add_argument("--source-fps", type=float, help="Override FPS reported by the stream.")
    parser.add_argument("--max-samples", type=int, help="Stop after processing this many sampled frames.")
    parser.add_argument("--max-source-frames", type=int, help="Stop after reading this many source frames.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--progress-interval", type=int, default=20)
    parser.add_argument("--dry-run-config", action="store_true", help="Print resolved config without loading models.")
    return parser.parse_args()


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def seconds_to_sampled_frames(seconds: float, source_fps: float, sample_interval: int) -> int:
    return max(1, int(math.ceil(float(seconds) * float(source_fps) / max(1, int(sample_interval)))))


def default_roi_args() -> argparse.Namespace:
    old_argv = sys.argv[:]
    try:
        sys.argv = [old_argv[0]]
        return roi.parse_args()
    finally:
        sys.argv = old_argv


def resolve_stream_id(args: argparse.Namespace, config: dict) -> str:
    return str(args.stream_id or config.get("streamId") or "stream")


def apply_runtime_config(
    roi_args: argparse.Namespace,
    config: dict,
    source_fps: float,
    sample_interval: int,
    device: str,
) -> argparse.Namespace:
    roi_args.device = device
    roi_args.no_write_video = True
    roi_args.draw_abnormal_only = str(config.get("drawMode", "NONE")).upper() != "FULL"

    behavior_seconds = config.get("behaviorDurationsSec") or {}
    for label in sorted(roi.ABNORMAL_LABELS):
        if label in behavior_seconds:
            roi.BEHAVIOR_DECISION_CONFIG[label]["temporal_abnormal_n"] = seconds_to_sampled_frames(
                behavior_seconds[label], source_fps, sample_interval
            )
    if "temporal_recover" in behavior_seconds:
        roi_args.temporal_recover_n = seconds_to_sampled_frames(
            behavior_seconds["temporal_recover"], source_fps, sample_interval
        )

    thresholds = config.get("warningThresholds") or {}
    roi_args.student_count_min_drop = int(thresholds.get("student_count_min_drop", 1))
    roi_args.invigilator_count_min_drop = int(thresholds.get("invigilator_count_min_drop", 1))
    roi_args.gather_min_count = int(thresholds.get("gather_min_count", 2))
    roi_args.gather_distance_ratio = float(thresholds.get("gather_distance_ratio", 0.12))
    roi_args.group_head_turn_min_count = int(thresholds.get("group_head_turn_min_count", 3))
    roi_args.repeated_head_turn_alert_count = int(thresholds.get("repeated_head_turn_alert_count", 5))
    roi_args.group_raise_hand_min_count = int(thresholds.get("group_raise_hand_min_count", 2))
    roi_args.group_standing_min_count = int(thresholds.get("group_standing_min_count", 3))
    roi_args.whisper_neighbor_ratio = float(thresholds.get("whisper_neighbor_ratio", 0.18))
    roi_args.invigilator_seat_distance_ratio = float(thresholds.get("invigilator_seat_distance_ratio", 0.06))
    roi_args.invigilator_seat_score_thr = float(thresholds.get("invigilator_seat_score_thr", 0.4))
    roi_args.invigilator_seat_min_height_ratio = float(thresholds.get("invigilator_seat_min_height_ratio", 0.23))
    roi_args.invigilator_seat_height_ratio_thr = float(thresholds.get("invigilator_seat_height_ratio_thr", 0.32))
    roi_args.invigilator_seat_track_ids = ",".join(
        str(item) for item in thresholds.get("invigilator_seat_track_ids", [])
    )
    roi_args.invigilator_seat_start_frame = 0

    warning_seconds = config.get("warningDurationsSec") or {}
    roi_args.enable_warnings = bool(config.get("enableWarnings", True))
    roi_args.warning_init_frames = seconds_to_sampled_frames(
        warning_seconds.get("warning_init", 10.0), source_fps, sample_interval
    )
    roi_args.warning_tail_ignore_frames = seconds_to_sampled_frames(
        warning_seconds.get("warning_tail_ignore", 0.0), source_fps, sample_interval
    )
    roi_args.count_alert_frames = seconds_to_sampled_frames(
        warning_seconds.get("student_count_drop", 3.0), source_fps, sample_interval
    )
    roi_args.student_count_alert_frames = seconds_to_sampled_frames(
        warning_seconds.get("student_count_drop", 3.0), source_fps, sample_interval
    )
    roi_args.invigilator_count_alert_frames = seconds_to_sampled_frames(
        warning_seconds.get("invigilator_count_drop", 3.0), source_fps, sample_interval
    )
    roi_args.gather_alert_frames = seconds_to_sampled_frames(
        warning_seconds.get("invigilator_gather", 1.5), source_fps, sample_interval
    )
    roi_args.group_head_turn_alert_frames = seconds_to_sampled_frames(
        warning_seconds.get("group_head_turn", 0.6), source_fps, sample_interval
    )
    roi_args.long_head_turn_alert_frames = seconds_to_sampled_frames(
        warning_seconds.get("long_head_turn", 5.0), source_fps, sample_interval
    )
    roi_args.group_raise_hand_alert_frames = seconds_to_sampled_frames(
        warning_seconds.get("group_raise_hand", 0.6), source_fps, sample_interval
    )
    roi_args.group_standing_alert_frames = seconds_to_sampled_frames(
        warning_seconds.get("group_student_standing", 0.6), source_fps, sample_interval
    )
    roi_args.whisper_alert_frames = seconds_to_sampled_frames(
        warning_seconds.get("student_whisper", 2.0), source_fps, sample_interval
    )
    roi_args.invigilator_seat_alert_frames = seconds_to_sampled_frames(
        warning_seconds.get("invigilator_in_student_seat", 0.5), source_fps, sample_interval
    )
    roi_args.invigilator_seat_min_history_frames = seconds_to_sampled_frames(
        thresholds.get("invigilator_seat_min_history_sec", 10.0), source_fps, sample_interval
    )
    roi_args.invigilator_seat_recover_frames = seconds_to_sampled_frames(
        thresholds.get("invigilator_seat_recover_sec", 5.0), source_fps, sample_interval
    )

    # Histories are counted in processed samples in stream mode.
    roi_args.standing_height_history = seconds_to_sampled_frames(50.0, source_fps, sample_interval)
    roi_args.student_anchor_min_frames = seconds_to_sampled_frames(2.7, source_fps, sample_interval)
    roi_args.inv_history_bonus_frames = seconds_to_sampled_frames(10.0, source_fps, sample_interval)
    roi_args.track_max_missing = max(2, seconds_to_sampled_frames(0.5, source_fps, sample_interval))
    return roi_args


def serialize_bbox(box) -> dict:
    x1, y1, x2, y2 = [float(v) for v in box]
    return {"x1": round(x1, 2), "y1": round(y1, 2), "x2": round(x2, 2), "y2": round(y2, 2)}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def append_jsonl(handle, item: dict, state: dict) -> None:
    handle.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n")
    state["pending"] += 1
    if state["pending"] >= state["flush_every"]:
        handle.flush()
        state["pending"] = 0


def compact_object(det: dict, role: str, final_label: str, final_name: str, final_conf: float, confirmed: bool) -> dict:
    return {
        "trackId": int(det["track_id"]),
        "role": role,
        "category": final_label,
        "categoryName": final_name,
        "confidence": round(float(final_conf), 4),
        "isAbnormal": bool(final_label in roi.ABNORMAL_LABELS),
        "isConfirmed": bool(confirmed and final_label in roi.ABNORMAL_LABELS),
        "bbox": serialize_bbox(det["bbox"]),
        "detConfidence": round(float(det.get("det_confidence", 0.0)), 4),
    }


def warning_to_output(warning: dict, stream_id: str, source_frame_idx: int, sample_idx: int, timestamp_sec: float) -> dict:
    duration_frames = int(warning.get("duration_frames", 0) or 0)
    return {
        "type": "warning_event",
        "streamId": stream_id,
        "warningType": warning.get("type"),
        "level": warning.get("level"),
        "message": warning.get("message"),
        "sourceFrameIndex": source_frame_idx,
        "sampleIndex": sample_idx,
        "timestampSec": round(timestamp_sec, 3),
        "durationSamples": duration_frames,
        "trackIds": warning.get("track_ids", []),
        "details": {k: v for k, v in warning.items() if k not in {"type", "level", "message"}},
        "createdAt": now_iso(),
    }


def build_latest(
    stream_id: str,
    source: str,
    source_frame_idx: int,
    sample_idx: int,
    timestamp_sec: float,
    objects: list[dict],
    warnings: list[dict],
    source_fps: float,
    sample_interval: int,
) -> dict:
    return {
        "type": "latest_frame",
        "streamId": stream_id,
        "source": source,
        "sourceFrameIndex": source_frame_idx,
        "sampleIndex": sample_idx,
        "timestampSec": round(timestamp_sec, 3),
        "sourceFps": round(source_fps, 3),
        "sampleIntervalFrames": sample_interval,
        "processedFps": round(source_fps / max(1, sample_interval), 3),
        "objects": objects,
        "warnings": warnings,
        "updatedAt": now_iso(),
    }


def write_json_atomic(path: Path, payload: dict) -> None:
    tmp_path = path.with_name(f".{path.name}.tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp_path, path)


def print_resolved_config(stream_id: str, source_fps: float, sample_interval: int, roi_args: argparse.Namespace) -> None:
    payload = {
        "streamId": stream_id,
        "sourceFps": source_fps,
        "sampleIntervalFrames": sample_interval,
        "processedFps": source_fps / max(1, sample_interval),
        "temporalAbnormalNByLabel": {
            label: roi.temporal_abnormal_n_for_label(label, roi_args) for label in sorted(roi.ABNORMAL_LABELS)
        },
        "temporalRecoverN": roi_args.temporal_recover_n,
        "warningFrames": {
            "warningInit": roi_args.warning_init_frames,
            "studentCountDrop": roi_args.student_count_alert_frames,
            "invigilatorCountDrop": roi_args.invigilator_count_alert_frames,
            "invigilatorGather": roi_args.gather_alert_frames,
            "groupHeadTurn": roi_args.group_head_turn_alert_frames,
            "longHeadTurn": roi_args.long_head_turn_alert_frames,
            "repeatedHeadTurnCount": roi_args.repeated_head_turn_alert_count,
            "groupRaiseHand": roi_args.group_raise_hand_alert_frames,
            "groupStanding": roi_args.group_standing_alert_frames,
            "studentWhisper": roi_args.whisper_alert_frames,
            "invigilatorInStudentSeat": roi_args.invigilator_seat_alert_frames,
        },
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def main() -> None:
    args = parse_args()
    config = load_json(args.runtime_config)
    stream_id = resolve_stream_id(args, config)
    sample_interval = int(args.sample_interval_frames or config.get("sampleIntervalFrames", 1))
    if sample_interval <= 0:
        raise ValueError("--sample-interval-frames must be positive")

    cap = cv2.VideoCapture(str(args.source))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source: {args.source}")
    source_fps = float(args.source_fps or cap.get(cv2.CAP_PROP_FPS) or config.get("sourceFps") or 25.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    roi_args = apply_runtime_config(default_roi_args(), config, source_fps, sample_interval, args.device)
    if args.dry_run_config:
        print_resolved_config(stream_id, source_fps, sample_interval, roi_args)
        cap.release()
        return

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.latest_json.parent.mkdir(parents=True, exist_ok=True)

    device = roi.resolve_device(roi_args.device)
    register_mmdet_modules(init_default_scope=True)
    register_mmpose_modules(init_default_scope=True)
    det_model = init_detector(str(roi_args.det_config), str(roi_args.det_checkpoint), device=str(device))
    pose_model = init_model(str(roi_args.pose_config), str(roi_args.pose_checkpoint), device=str(device))
    roi_heads = roi.load_roi_heads(roi_args.roi_head_checkpoint, device)
    roi_pose_behavior = roi.load_roi_pose_behavior_head(roi_args.roi_pose_behavior_checkpoint, device)
    tracker = roi.SimpleIoUTracker(roi_args.track_iou_thr, roi_args.track_max_missing)
    warning_state = cached_render.WarningState(roi_args, width, source_fps / sample_interval)

    write_frame_results = bool(config.get("writeFrameResults", True))
    write_normal_objects = bool(config.get("writeNormalObjects", False))
    allowed_behavior_track_ids = {
        str(label): {int(item) for item in ids}
        for label, ids in (config.get("allowedBehaviorTrackIds") or {}).items()
    }
    flush_state = {"pending": 0, "flush_every": max(1, int(config.get("flushEveryEvents", 1)))}

    recent_counts: list[int] = []
    previous_invigilator_ids: set[int] = set()
    source_frame_idx = 0
    sample_idx = 0
    start_time = time.time()

    with args.output_jsonl.open("a", encoding="utf-8") as out:
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if args.max_source_frames is not None and source_frame_idx >= args.max_source_frames:
                    break
                if source_frame_idx % sample_interval != 0:
                    source_frame_idx += 1
                    continue
                if args.max_samples is not None and sample_idx >= args.max_samples:
                    break

                timestamp_sec = source_frame_idx / max(source_fps, 1e-6)
                init_default_scope("mmdet")
                det_result = inference_detector(det_model, frame)
                detections = roi.get_person_detections(det_result, width, height, roi_args)
                detections = roi.deduplicate_people(detections, roi_args)

                recent_counts.append(len(detections))
                if len(recent_counts) > roi_args.id_pool_window:
                    recent_counts.pop(0)
                id_pool_size = max(recent_counts) if recent_counts else 0
                detections = tracker.update(detections, sample_idx, id_pool_size, roi_args)
                valid_detections = list(detections)

                roi_features = roi.extract_frame_roi_features(
                    det_model,
                    frame,
                    [det["bbox"] for det in valid_detections],
                    device,
                    roi_args.roi_size,
                )
                identity_predictions = roi.classify_roi_identity(roi_heads, roi_features)
                for det, identity in zip(valid_detections, identity_predictions):
                    det["identity"] = identity
                    track = tracker.tracks[det["track_id"]]
                    det["track_age"] = sample_idx - int(track.get("created_frame", sample_idx))

                roi.update_invigilator_scores(valid_detections, identity_predictions, tracker, roi_args)
                frame_invigilator_ids = roi.select_invigilator_ids(
                    valid_detections, tracker, previous_invigilator_ids, roi_args
                )
                roi.update_invigilator_history(valid_detections, tracker, frame_invigilator_ids)
                roi.update_position_anchors(valid_detections, tracker, frame_invigilator_ids, roi_args)

                student_indices = [
                    idx for idx, det in enumerate(valid_detections) if det["track_id"] not in frame_invigilator_ids
                ]
                student_detections = [valid_detections[idx] for idx in student_indices]
                pose_by_index: dict[int, list[list[float]]] = {}

                if student_detections:
                    boxes = np.asarray([det["bbox"] for det in student_detections], dtype=np.float32)
                    init_default_scope("mmpose")
                    pose_results = inference_topdown(pose_model, frame, bboxes=boxes, bbox_format="xyxy")
                    for det_idx, det, pose_instance in zip(
                        student_indices,
                        student_detections,
                        roi.iter_pose_instances(pose_results),
                    ):
                        pose = roi.normalize_keypoints_by_bbox(
                            pose_instance["keypoints"],
                            pose_instance["keypoint_scores"],
                            pose_instance["bbox"],
                        ).tolist()
                        det.update(pose_instance)
                        det["pose"] = pose
                        pose_by_index[det_idx] = pose
                        track = tracker.tracks[det["track_id"]]
                        track_switched = roi.box_motion_is_track_switch(track, det["bbox"], width, height, roi_args)
                        if track_switched:
                            roi.reset_track_history_after_switch(track)
                        all_rule_scores = roi.evaluate_all_single_frame_rules(pose, det["bbox"], width, height, roi_args)
                        roi.apply_standing_height_change_rule(all_rule_scores, track, det["bbox"], height, roi_args)
                        roi.update_track_height_history(track, det["bbox"], height, roi_args)
                        track["last_bbox"] = tuple(float(v) for v in det["bbox"])
                        det["all_rule_scores"] = all_rule_scores
                        det["track_switch_reset"] = bool(track_switched)

                behavior_predictions = roi.classify_roi_pose_behavior(
                    roi_pose_behavior,
                    roi_features,
                    pose_by_index,
                    student_indices,
                    device,
                )
                for idx, model_behavior in behavior_predictions:
                    model_behavior["all_rule_scores"] = valid_detections[idx].get("all_rule_scores", {})
                    model_behavior["track_switch_reset"] = bool(valid_detections[idx].get("track_switch_reset", False))
                    model_behavior["model_thresholded_label"] = roi.apply_behavior_threshold(
                        model_behavior["raw_label"], model_behavior["raw_score"], roi_args
                    )
                    model_behavior["thresholded_label"] = model_behavior["model_thresholded_label"]
                    valid_detections[idx]["pose_behavior"] = model_behavior
                    valid_detections[idx]["image_behavior"] = model_behavior

                objects: list[dict] = []
                abnormal_objects: list[dict] = []
                frame_students: list[dict] = []
                frame_invigilators: list[dict] = []
                invigilator_seat_candidates: list[dict] = []

                for det in valid_detections:
                    track = tracker.tracks[det["track_id"]]
                    pose_behavior = det.get("pose_behavior")
                    identity = det.get("identity", {})
                    inv_score = float(track.get("inv_score", 0.0))
                    inv_history_frames = int(track.get("inv_confirmed_frames", 0))

                    if det["track_id"] in frame_invigilator_ids:
                        roi.clear_abnormal_state(track)
                        final_label = "invigilator"
                        final_conf = max(inv_score, float(identity.get("confidence", 0.0)))
                        confirmed = False
                        role = "invigilator"
                    else:
                        if pose_behavior is None:
                            continue
                        role = "student"
                        hybrid_behavior = roi.fuse_model_and_rule_behavior(pose_behavior)
                        raw_label = hybrid_behavior["label"]
                        raw_conf = hybrid_behavior["confidence"]
                        final_label, final_conf, confirmed = roi.apply_temporal_rule(
                            track, raw_label, raw_conf, roi_args, pose_behavior.get("all_rule_scores")
                        )
                        allowed_ids = allowed_behavior_track_ids.get(final_label)
                        if final_label in roi.ABNORMAL_LABELS and allowed_ids is not None and int(det["track_id"]) not in allowed_ids:
                            final_label = "normal"
                            final_conf = 0.0
                            confirmed = False

                    final_name = roi.CLASS_NAMES.get(final_label, final_label)
                    output_object = compact_object(det, role, final_label, final_name, final_conf, confirmed)
                    if write_normal_objects or output_object["isAbnormal"]:
                        objects.append(output_object)
                    if output_object["isAbnormal"]:
                        abnormal_objects.append(output_object)
                        append_jsonl(
                            out,
                            {
                                "type": "abnormal_event",
                                "streamId": stream_id,
                                "sourceFrameIndex": source_frame_idx,
                                "sampleIndex": sample_idx,
                                "timestampSec": round(timestamp_sec, 3),
                                "trackId": output_object["trackId"],
                                "category": output_object["category"],
                                "categoryName": output_object["categoryName"],
                                "confidence": output_object["confidence"],
                                "bbox": output_object["bbox"],
                                "createdAt": now_iso(),
                            },
                            flush_state,
                        )

                    target_for_warning = {
                        "track_id": int(det["track_id"]),
                        "box": det["bbox"],
                        "final_label": final_label,
                        "is_sitting": False,
                        "invigilator_score": inv_score,
                        "invigilator_history_frames": inv_history_frames,
                        "head_direction": cached_render.estimate_head_direction(
                            {
                                "pose_keypoints": det.get("keypoints"),
                                "pose_keypoint_scores": det.get("keypoint_scores"),
                            },
                            det["bbox"],
                        )
                        if final_label == "head_turn"
                        else None,
                    }
                    height_ratio = max(0.0, (float(det["bbox"][3]) - float(det["bbox"][1])) / max(1.0, float(height)))
                    target_for_warning["height_ratio"] = height_ratio
                    target_for_warning["seat_height_ok"] = height_ratio <= roi_args.invigilator_seat_height_ratio_thr
                    if final_label == "invigilator":
                        frame_invigilators.append(target_for_warning)
                    else:
                        frame_students.append(target_for_warning)
                    if (
                        final_label == "invigilator"
                        and target_for_warning["seat_height_ok"]
                        and inv_score >= roi_args.invigilator_seat_score_thr
                        and height_ratio >= roi_args.invigilator_seat_min_height_ratio
                    ):
                        invigilator_seat_candidates.append(target_for_warning)

                active_warnings = warning_state.update(
                    sample_idx,
                    frame_students,
                    frame_invigilators,
                    invigilator_seat_candidates,
                )
                warnings_out = [
                    warning_to_output(warning, stream_id, source_frame_idx, sample_idx, timestamp_sec)
                    for warning in active_warnings
                ]
                for warning in warnings_out:
                    append_jsonl(out, warning, flush_state)

                if write_frame_results:
                    append_jsonl(
                        out,
                        {
                            "type": "frame_result",
                            "streamId": stream_id,
                            "sourceFrameIndex": source_frame_idx,
                            "sampleIndex": sample_idx,
                            "timestampSec": round(timestamp_sec, 3),
                            "objects": objects,
                            "warnings": warnings_out,
                            "stats": {
                                "abnormalCount": len(abnormal_objects),
                                "studentCount": len(frame_students),
                                "invigilatorCount": len(frame_invigilators),
                            },
                            "createdAt": now_iso(),
                        },
                        flush_state,
                    )

                latest = build_latest(
                    stream_id,
                    str(args.source),
                    source_frame_idx,
                    sample_idx,
                    timestamp_sec,
                    objects,
                    warnings_out,
                    source_fps,
                    sample_interval,
                )
                write_json_atomic(args.latest_json, latest)

                previous_invigilator_ids = frame_invigilator_ids
                sample_idx += 1
                source_frame_idx += 1
                if args.progress_interval > 0 and sample_idx % args.progress_interval == 0:
                    elapsed = time.time() - start_time
                    print(
                        f"processed {sample_idx} samples at source frame {source_frame_idx}, "
                        f"{sample_idx / max(elapsed, 1e-6):.2f} sampled FPS"
                    )
        finally:
            cap.release()
            out.flush()

    print(f"stream events: {args.output_jsonl}")
    print(f"latest snapshot: {args.latest_json}")
    print(f"processed samples: {sample_idx}")


if __name__ == "__main__":
    main()
