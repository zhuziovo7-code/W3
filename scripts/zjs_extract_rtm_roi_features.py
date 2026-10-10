#!/usr/bin/env python3
"""Extract RTMDet ROI features with optional training-image augmentation."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from mmdet.apis import inference_detector, init_detector
from mmengine.registry import init_default_scope
from torchvision.ops import roi_align


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DET_CONFIG = PROJECT_ROOT / "models/detector/rtmdet_tiny_8xb32-300e_coco.py"
DEFAULT_DET_CHECKPOINT = (
    PROJECT_ROOT / "models/detector/rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth"
)
MEAN = torch.tensor([103.53, 116.28, 123.675], dtype=torch.float32).view(3, 1, 1)
STD = torch.tensor([57.375, 57.12, 58.395], dtype=torch.float32).view(3, 1, 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cache RTMDet ROI features, optionally augmenting behavior-training images."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--failed-jsonl", type=Path)
    parser.add_argument("--det-config", type=Path, default=DEFAULT_DET_CONFIG)
    parser.add_argument("--det-checkpoint", type=Path, default=DEFAULT_DET_CHECKPOINT)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--det-score-thr", type=float, default=0.25)
    parser.add_argument("--roi-size", type=int, default=4)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--progress-interval", type=int, default=200)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument(
        "--augment-copies",
        type=int,
        default=0,
        help="Number of random augmented copies per source image. Use 0 for validation/test data.",
    )
    parser.add_argument("--augment-seed", type=int, default=42)
    parser.add_argument("--flip-prob", type=float, default=0.5)
    parser.add_argument("--blur-prob", type=float, default=0.2)
    parser.add_argument("--noise-prob", type=float, default=0.2)
    parser.add_argument("--max-rotate-deg", type=float, default=4.0)
    parser.add_argument("--max-translate-frac", type=float, default=0.03)
    parser.add_argument("--min-scale", type=float, default=0.95)
    parser.add_argument("--max-scale", type=float, default=1.05)
    parser.add_argument("--min-brightness", type=float, default=0.85)
    parser.add_argument("--max-brightness", type=float, default=1.15)
    parser.add_argument("--min-contrast", type=float, default=0.85)
    parser.add_argument("--max-contrast", type=float, default=1.15)
    parser.add_argument("--min-saturation", type=float, default=0.85)
    parser.add_argument("--max-saturation", type=float, default=1.15)
    parser.add_argument(
        "--path-map",
        action="append",
        default=[],
        metavar="OLD=NEW",
        help="Rewrite an old manifest path prefix. May be passed more than once.",
    )
    args = parser.parse_args()
    if args.augment_copies < 0:
        parser.error("--augment-copies must be >= 0")
    if args.num_shards < 1:
        parser.error("--num-shards must be >= 1")
    if not 0 <= args.shard_index < args.num_shards:
        parser.error("--shard-index must satisfy 0 <= index < num-shards")
    if not 0.0 <= args.flip_prob <= 1.0:
        parser.error("--flip-prob must be between 0 and 1")
    if not 0.0 <= args.blur_prob <= 1.0 or not 0.0 <= args.noise_prob <= 1.0:
        parser.error("augmentation probabilities must be between 0 and 1")
    if args.min_scale <= 0 or args.min_scale > args.max_scale:
        parser.error("invalid scale range")
    return args


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def read_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def parse_path_maps(values: list[str]) -> list[tuple[str, str]]:
    mappings = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"Invalid --path-map {value!r}; expected OLD=NEW")
        old, new = value.split("=", 1)
        mappings.append((old.rstrip("/"), new.rstrip("/")))
    return mappings


def resolve_image_path(raw_path: str, mappings: list[tuple[str, str]]) -> Path:
    for old, new in mappings:
        if raw_path == old or raw_path.startswith(old + "/"):
            candidate = Path(new + raw_path[len(old) :])
            try:
                if candidate.is_file():
                    return candidate
            except OSError:
                pass
    original = Path(raw_path)
    try:
        if original.is_file():
            return original
    except OSError:
        pass
    return original


def select_main_person(result, width: int, height: int, score_thr: float):
    pred = result.pred_instances
    bboxes = pred.bboxes.detach().cpu().numpy()
    scores = pred.scores.detach().cpu().numpy()
    labels = pred.labels.detach().cpu().numpy()
    candidates = []
    image_area = max(float(width * height), 1.0)
    image_center = np.asarray([width * 0.5, height * 0.5], dtype=np.float32)
    diag = max(float(np.hypot(width, height)), 1.0)
    for bbox, score, label in zip(bboxes, scores, labels):
        if int(label) != 0 or float(score) < score_thr:
            continue
        x1, y1, x2, y2 = [float(v) for v in bbox]
        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        center = np.asarray([(x1 + x2) * 0.5, (y1 + y2) * 0.5], dtype=np.float32)
        center_score = 1.0 - min(float(np.linalg.norm(center - image_center) / diag), 1.0)
        area_score = min(area / image_area, 1.0)
        rank = 0.45 * float(score) + 0.30 * center_score + 0.25 * area_score
        candidates.append((rank, [x1, y1, x2, y2], float(score)))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1], candidates[0][2], len(candidates)


def pad_to_divisor(tensor: torch.Tensor, divisor: int = 32) -> torch.Tensor:
    _, height, width = tensor.shape
    pad_h = (divisor - height % divisor) % divisor
    pad_w = (divisor - width % divisor) % divisor
    if pad_h or pad_w:
        tensor = F.pad(tensor, (0, pad_w, 0, pad_h), value=0)
    return tensor


@torch.no_grad()
def extract_roi_feature(
    model, image_bgr: np.ndarray, bbox: list[float], device: torch.device, roi_size: int
) -> torch.Tensor:
    image = torch.from_numpy(image_bgr).permute(2, 0, 1).float()
    image = (image - MEAN) / STD
    image = pad_to_divisor(image)
    image = image.unsqueeze(0).to(device)
    feats = model.extract_feat(image)
    feat = feats[-1]
    spatial_scale = feat.shape[-1] / float(image.shape[-1])
    roi = torch.tensor([[0.0, *bbox]], dtype=torch.float32, device=device)
    pooled = roi_align(
        feat,
        roi,
        output_size=(roi_size, roi_size),
        spatial_scale=spatial_scale,
        aligned=True,
    )
    return pooled.flatten(1).squeeze(0).cpu()


def augment_image(image: np.ndarray, rng: np.random.Generator, args: argparse.Namespace):
    height, width = image.shape[:2]
    output = image.copy()
    operations = []

    if rng.random() < args.flip_prob:
        output = cv2.flip(output, 1)
        operations.append("horizontal_flip")

    angle = float(rng.uniform(-args.max_rotate_deg, args.max_rotate_deg))
    scale = float(rng.uniform(args.min_scale, args.max_scale))
    tx = float(rng.uniform(-args.max_translate_frac, args.max_translate_frac) * width)
    ty = float(rng.uniform(-args.max_translate_frac, args.max_translate_frac) * height)
    matrix = cv2.getRotationMatrix2D((width * 0.5, height * 0.5), angle, scale)
    matrix[0, 2] += tx
    matrix[1, 2] += ty
    output = cv2.warpAffine(
        output,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    operations.append(f"affine(angle={angle:.2f},scale={scale:.3f},tx={tx:.1f},ty={ty:.1f})")

    contrast = float(rng.uniform(args.min_contrast, args.max_contrast))
    brightness = float(rng.uniform(args.min_brightness, args.max_brightness))
    output = np.clip((output.astype(np.float32) - 127.5) * contrast + 127.5, 0, 255)
    output = np.clip(output * brightness, 0, 255).astype(np.uint8)
    operations.append(f"color(brightness={brightness:.3f},contrast={contrast:.3f})")

    saturation = float(rng.uniform(args.min_saturation, args.max_saturation))
    hsv = cv2.cvtColor(output, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation, 0, 255)
    output = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    operations.append(f"saturation({saturation:.3f})")

    if rng.random() < args.blur_prob:
        output = cv2.GaussianBlur(output, (3, 3), 0)
        operations.append("gaussian_blur_3x3")

    if rng.random() < args.noise_prob:
        sigma = float(rng.uniform(1.0, 6.0))
        noise = rng.normal(0.0, sigma, output.shape).astype(np.float32)
        output = np.clip(output.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        operations.append(f"gaussian_noise(sigma={sigma:.2f})")

    return output, operations


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    records = read_jsonl(args.manifest)
    if args.max_samples:
        records = records[: args.max_samples]
    indexed_records = [
        (index, item) for index, item in enumerate(records) if index % args.num_shards == args.shard_index
    ]
    path_maps = parse_path_maps(args.path_map)

    init_default_scope("mmdet")
    model = init_detector(str(args.det_config), str(args.det_checkpoint), device=str(device))
    model.eval()

    features = []
    kept_records = []
    failures = []
    variants_per_image = 1 + args.augment_copies
    total_variants = len(indexed_records) * variants_per_image

    for local_index, (record_index, item) in enumerate(indexed_records):
        image_path = resolve_image_path(item["image_path"], path_maps)
        image = cv2.imread(str(image_path))
        if image is None:
            failures.append({**item, "resolved_image_path": str(image_path), "reason": "read_failed"})
            continue

        for copy_index in range(variants_per_image):
            if copy_index == 0:
                variant = image
                operations = ["original"]
            else:
                seed = args.augment_seed + record_index * max(args.augment_copies, 1) + copy_index - 1
                rng = np.random.default_rng(seed)
                variant, operations = augment_image(image, rng, args)

            height, width = variant.shape[:2]
            try:
                result = inference_detector(model, variant)
                selected = select_main_person(result, width, height, args.det_score_thr)
                if selected is None:
                    failures.append(
                        {
                            **item,
                            "resolved_image_path": str(image_path),
                            "augmentation_copy": copy_index,
                            "augmentation_operations": operations,
                            "reason": "no_person",
                        }
                    )
                    continue
                bbox, det_score, num_candidates = selected
                feature = extract_roi_feature(model, variant, bbox, device, args.roi_size)
            except Exception as exc:  # noqa: BLE001
                failures.append(
                    {
                        **item,
                        "resolved_image_path": str(image_path),
                        "augmentation_copy": copy_index,
                        "augmentation_operations": operations,
                        "reason": type(exc).__name__,
                        "error": str(exc),
                    }
                )
                continue

            enriched = dict(item)
            enriched.update(
                {
                    "resolved_image_path": str(image_path),
                    "bbox": bbox,
                    "det_score": det_score,
                    "num_person_candidates": num_candidates,
                    "image_width": width,
                    "image_height": height,
                    "feature_index": len(features),
                    "is_augmented": copy_index > 0,
                    "augmentation_copy": copy_index,
                    "augmentation_operations": operations,
                }
            )
            features.append(feature)
            kept_records.append(enriched)

        processed_variants = (local_index + 1) * variants_per_image
        if args.progress_interval and (local_index + 1) % args.progress_interval == 0:
            print(
                f"shard={args.shard_index + 1}/{args.num_shards} "
                f"source_images={local_index + 1}/{len(indexed_records)} "
                f"variants={processed_variants}/{total_variants} "
                f"kept={len(kept_records)} failed={len(failures)}"
            )
            sys.stdout.flush()

    if not features:
        raise RuntimeError("No ROI features were extracted.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "features": torch.stack(features, dim=0),
        "records": kept_records,
        "manifest": str(args.manifest),
        "det_config": str(args.det_config),
        "det_checkpoint": str(args.det_checkpoint),
        "roi_size": args.roi_size,
        "augmentation": {
            "copies": args.augment_copies,
            "seed": args.augment_seed,
            "flip_prob": args.flip_prob,
            "blur_prob": args.blur_prob,
            "noise_prob": args.noise_prob,
            "max_rotate_deg": args.max_rotate_deg,
            "max_translate_frac": args.max_translate_frac,
            "scale": [args.min_scale, args.max_scale],
            "brightness": [args.min_brightness, args.max_brightness],
            "contrast": [args.min_contrast, args.max_contrast],
            "saturation": [args.min_saturation, args.max_saturation],
        },
        "path_maps": args.path_map,
        "shard": {"index": args.shard_index, "count": args.num_shards},
    }
    torch.save(payload, args.output)

    if args.failed_jsonl:
        args.failed_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with args.failed_jsonl.open("w", encoding="utf-8") as handle:
            for item in failures:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(
        f"saved={args.output} shard={args.shard_index + 1}/{args.num_shards} "
        f"source_images={len(indexed_records)} "
        f"kept_features={len(kept_records)} failed_variants={len(failures)}"
    )


if __name__ == "__main__":
    main()
