"""Pose feature helpers for behavior recognition."""

from __future__ import annotations

from typing import Sequence

import numpy as np


NUM_COCO_KEYPOINTS = 17
POSE_FEATURE_DIM = NUM_COCO_KEYPOINTS * 3


def normalize_keypoints_by_bbox(
    keypoints: Sequence[Sequence[float]],
    keypoint_scores: Sequence[float],
    bbox: Sequence[float],
    clip: bool = True,
) -> np.ndarray:
    """Convert image-space keypoints to bbox-normalized pose features.

    Args:
        keypoints: COCO keypoints in image coordinates, shaped [17, 2].
        keypoint_scores: Per-keypoint confidence, shaped [17].
        bbox: Person box in xyxy format.
        clip: Whether to clip normalized x/y coordinates into [0, 1].

    Returns:
        Array shaped [17, 3], each row is [x_norm, y_norm, score].
    """
    points = np.asarray(keypoints, dtype=np.float32)
    scores = np.asarray(keypoint_scores, dtype=np.float32).reshape(-1, 1)
    box = np.asarray(bbox, dtype=np.float32)

    if points.shape != (NUM_COCO_KEYPOINTS, 2):
        raise ValueError(f"Expected keypoints shape (17, 2), got {points.shape}")
    if scores.shape != (NUM_COCO_KEYPOINTS, 1):
        raise ValueError(f"Expected 17 keypoint scores, got {scores.shape}")
    if box.shape[0] != 4:
        raise ValueError(f"Expected bbox xyxy with 4 values, got {box.shape}")

    x1, y1, x2, y2 = box[:4]
    width = max(float(x2 - x1), 1e-6)
    height = max(float(y2 - y1), 1e-6)

    normalized = points.copy()
    normalized[:, 0] = (normalized[:, 0] - x1) / width
    normalized[:, 1] = (normalized[:, 1] - y1) / height
    if clip:
        normalized[:, :2] = np.clip(normalized[:, :2], 0.0, 1.0)

    return np.concatenate([normalized, scores], axis=1).astype(np.float32)


def flatten_pose(pose: Sequence[Sequence[float]]) -> np.ndarray:
    """Flatten a normalized pose [17, 3] into a 51-D vector."""
    pose_array = np.asarray(pose, dtype=np.float32)
    if pose_array.shape != (NUM_COCO_KEYPOINTS, 3):
        raise ValueError(f"Expected pose shape (17, 3), got {pose_array.shape}")
    return pose_array.reshape(POSE_FEATURE_DIM)


def visible_keypoint_count(pose: Sequence[Sequence[float]], score_thr: float) -> int:
    """Count keypoints whose confidence is above a threshold."""
    pose_array = np.asarray(pose, dtype=np.float32)
    if pose_array.shape != (NUM_COCO_KEYPOINTS, 3):
        raise ValueError(f"Expected pose shape (17, 3), got {pose_array.shape}")
    return int((pose_array[:, 2] >= score_thr).sum())
