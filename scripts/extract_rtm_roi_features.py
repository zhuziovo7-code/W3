#!/usr/bin/env python3
"""Extract main-person RTMDet ROI features for ROI classification heads."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

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
    parser = argparse.ArgumentParser(description="Cache RTMDet main-person ROI features from image manifests.")
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
    return parser.parse_args()


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
def extract_roi_feature(model, image_bgr: np.ndarray, bbox: list[float], device: torch.device, roi_size: int) -> torch.Tensor:
    image = torch.from_numpy(image_bgr).permute(2, 0, 1).float()
    image = (image - MEAN) / STD
    image = pad_to_divisor(image)
    image = image.unsqueeze(0).to(device)
    feats = model.extract_feat(image)
    feat = feats[-1]
    spatial_scale = feat.shape[-1] / float(image.shape[-1])
    roi = torch.tensor([[0.0, *bbox]], dtype=torch.float32, device=device)
    pooled = roi_align(feat, roi, output_size=(roi_size, roi_size), spatial_scale=spatial_scale, aligned=True)
    return pooled.flatten(1).squeeze(0).cpu()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    records = read_jsonl(args.manifest)
    if args.max_samples:
        records = records[: args.max_samples]

    init_default_scope("mmdet")
    model = init_detector(str(args.det_config), str(args.det_checkpoint), device=str(device))
    model.eval()

    features = []
    kept_records = []
    failures = []
    for index, item in enumerate(records, start=1):
        image_path = Path(item["image_path"])
        image = cv2.imread(str(image_path))
        if image is None:
            failures.append({**item, "reason": "read_failed"})
            continue
        height, width = image.shape[:2]
        try:
            result = inference_detector(model, image)
            selected = select_main_person(result, width, height, args.det_score_thr)
            if selected is None:
                failures.append({**item, "reason": "no_person"})
                continue
            bbox, det_score, num_candidates = selected
            feature = extract_roi_feature(model, image, bbox, device, args.roi_size)
        except Exception as exc:  # noqa: BLE001
            failures.append({**item, "reason": type(exc).__name__, "error": str(exc)})
            continue
        enriched = dict(item)
        enriched.update(
            {
                "bbox": bbox,
                "det_score": det_score,
                "num_person_candidates": num_candidates,
                "image_width": width,
                "image_height": height,
                "feature_index": len(features),
            }
        )
        features.append(feature)
        kept_records.append(enriched)
        if args.progress_interval and index % args.progress_interval == 0:
            print(f"processed={index}/{len(records)} kept={len(kept_records)} failed={len(failures)}")
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
    }
    torch.save(payload, args.output)
    if args.failed_jsonl:
        args.failed_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with args.failed_jsonl.open("w", encoding="utf-8") as handle:
            for item in failures:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"saved={args.output} kept={len(kept_records)} failed={len(failures)}")


if __name__ == "__main__":
    main()
