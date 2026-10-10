"""Dataset for normalized pose pseudo labels stored as JSONL."""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from exam_pose.behavior.pose_features import (
    flatten_pose,
    normalize_keypoints_by_bbox,
    visible_keypoint_count,
)


class PoseJsonlDataset(Dataset):
    """Read pose samples from JSONL.

    Supported sample shapes:

    1. Final training format:
       {"label": "look_down", "pose": [[x_norm, y_norm, score], ...]}

    2. Raw RTM-Pose person format:
       {"label": "look_down", "bbox": [...], "keypoints": [...],
        "keypoint_scores": [...]}

    3. Frame format with multiple persons:
       {"label": "look_down", "persons": [{"bbox": ..., "keypoints": ...}, ...]}
       This expands each person into one training sample with the same label.
    """

    def __init__(
        self,
        jsonl_path: str | Path,
        label_to_idx: dict[str, int] | None = None,
        min_visible_keypoints: int = 6,
        keypoint_score_thr: float = 0.2,
        augment: bool = False,
        noise_std: float = 0.015,
        drop_keypoint_prob: float = 0.05,
        flip_prob: float = 0.5,
        affine_prob: float = 0.5,
        rotate_deg: float = 10.0,
        scale_range: tuple[float, float] = (0.9, 1.1),
        translate: float = 0.05,
        shear: float = 0.08,
        occlusion_prob: float = 0.2,
        score_jitter_prob: float = 0.3,
        score_jitter: float = 0.15,
    ) -> None:
        self.jsonl_path = Path(jsonl_path)
        self.min_visible_keypoints = min_visible_keypoints
        self.keypoint_score_thr = keypoint_score_thr
        self.augment = augment
        self.noise_std = noise_std
        self.drop_keypoint_prob = drop_keypoint_prob
        self.flip_prob = flip_prob
        self.affine_prob = affine_prob
        self.rotate_deg = rotate_deg
        self.scale_range = scale_range
        self.translate = translate
        self.shear = shear
        self.occlusion_prob = occlusion_prob
        self.score_jitter_prob = score_jitter_prob
        self.score_jitter = score_jitter

        raw_records = self._load_records(self.jsonl_path)
        labels = sorted({record["label"] for record in raw_records})
        self.label_to_idx = label_to_idx or {label: idx for idx, label in enumerate(labels)}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}

        self.samples: list[dict[str, Any]] = []
        for record in raw_records:
            if record["label"] not in self.label_to_idx:
                continue
            pose = record["pose"]
            if visible_keypoint_count(pose, self.keypoint_score_thr) < min_visible_keypoints:
                continue
            self.samples.append(record)

        if not self.samples:
            raise ValueError(f"No usable pose samples found in {self.jsonl_path}")

    @staticmethod
    def _load_records(jsonl_path: Path) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        with jsonl_path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                item = json.loads(line)
                label = item.get("label")
                if label is None:
                    raise ValueError(f"Missing label at {jsonl_path}:{line_no}")

                if "pose" in item:
                    records.append({**item, "pose": item["pose"]})
                    continue

                if "persons" in item:
                    for person_idx, person in enumerate(item["persons"]):
                        records.append(
                            {
                                "label": label,
                                "pose": normalize_keypoints_by_bbox(
                                    person["keypoints"],
                                    person["keypoint_scores"],
                                    person["bbox"],
                                ).tolist(),
                                "source": item.get("image_path") or item.get("video"),
                                "frame_idx": item.get("frame_idx"),
                                "person_idx": person_idx,
                            }
                        )
                    continue

                if {"bbox", "keypoints", "keypoint_scores"}.issubset(item):
                    records.append(
                        {
                            **item,
                            "pose": normalize_keypoints_by_bbox(
                                item["keypoints"],
                                item["keypoint_scores"],
                                item["bbox"],
                            ).tolist(),
                        }
                    )
                    continue

                raise ValueError(f"Unsupported record format at {jsonl_path}:{line_no}")
        return records

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[index]
        pose = torch.tensor(flatten_pose(sample["pose"]), dtype=torch.float32)

        if self.augment:
            pose = self._augment_pose(pose)

        label = torch.tensor(self.label_to_idx[sample["label"]], dtype=torch.long)
        return pose, label

    def _augment_pose(self, pose: torch.Tensor) -> torch.Tensor:
        augmented = pose.clone().view(17, 3)

        if random.random() < self.flip_prob:
            augmented = self._horizontal_flip(augmented)

        if random.random() < self.affine_prob:
            augmented = self._random_affine(augmented)

        if self.noise_std > 0:
            coord_noise = torch.randn_like(augmented[:, :2]) * self.noise_std
            augmented[:, :2] = torch.clamp(augmented[:, :2] + coord_noise, 0.0, 1.0)

        if self.drop_keypoint_prob > 0:
            mask = torch.rand(17) < self.drop_keypoint_prob
            augmented[mask, 2] = 0.0

        if random.random() < self.occlusion_prob:
            augmented = self._random_part_occlusion(augmented)

        if random.random() < self.score_jitter_prob:
            score_noise = torch.randn(17) * self.score_jitter
            score_scale = 1.0 + torch.empty(17).uniform_(-self.score_jitter, self.score_jitter)
            augmented[:, 2] = torch.clamp(augmented[:, 2] * score_scale + score_noise, 0.0, 1.0)
        return augmented.reshape(-1)

    @staticmethod
    def _horizontal_flip(pose: torch.Tensor) -> torch.Tensor:
        flipped = pose.clone()
        flipped[:, 0] = 1.0 - flipped[:, 0]
        pairs = (
            (1, 2),
            (3, 4),
            (5, 6),
            (7, 8),
            (9, 10),
            (11, 12),
            (13, 14),
            (15, 16),
        )
        for left, right in pairs:
            tmp = flipped[left].clone()
            flipped[left] = flipped[right]
            flipped[right] = tmp
        return flipped

    def _random_affine(self, pose: torch.Tensor) -> torch.Tensor:
        transformed = pose.clone()
        coords = transformed[:, :2] - 0.5

        angle = math.radians(random.uniform(-self.rotate_deg, self.rotate_deg))
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        rotation = torch.tensor(
            [[cos_a, -sin_a], [sin_a, cos_a]], dtype=coords.dtype, device=coords.device
        )

        scale_x = random.uniform(*self.scale_range)
        scale_y = random.uniform(*self.scale_range)
        scale = torch.tensor(
            [[scale_x, 0.0], [0.0, scale_y]], dtype=coords.dtype, device=coords.device
        )

        shear_x = random.uniform(-self.shear, self.shear)
        shear_y = random.uniform(-self.shear, self.shear)
        shear = torch.tensor(
            [[1.0, shear_x], [shear_y, 1.0]], dtype=coords.dtype, device=coords.device
        )

        matrix = shear @ rotation @ scale
        coords = coords @ matrix.T
        offset = torch.tensor(
            [
                random.uniform(-self.translate, self.translate),
                random.uniform(-self.translate, self.translate),
            ],
            dtype=coords.dtype,
            device=coords.device,
        )
        transformed[:, :2] = torch.clamp(coords + 0.5 + offset, 0.0, 1.0)
        return transformed

    @staticmethod
    def _random_part_occlusion(pose: torch.Tensor) -> torch.Tensor:
        occluded = pose.clone()
        groups = (
            (5, 7, 9),
            (6, 8, 10),
            (11, 13, 15),
            (12, 14, 16),
            (11, 12, 13, 14, 15, 16),
            (0, 1, 2, 3, 4),
        )
        group = random.choice(groups)
        keep_scale = random.choice((0.0, 0.2, 0.4))
        occluded[list(group), 2] *= keep_scale
        return occluded


def split_indices(
    size: int,
    val_ratio: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    indices = list(range(size))
    random.Random(seed).shuffle(indices)
    val_size = max(1, int(size * val_ratio)) if size > 1 else 0
    return indices[val_size:], indices[:val_size]
