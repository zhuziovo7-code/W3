#!/usr/bin/env python3
"""Train only the ROI behavior head and preserve the existing identity head."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_CHECKPOINT = (
    PROJECT_ROOT / "models/behavior/rtm_roi_heads_fifth_original_plus_review_correctedx3.pt"
)
BEHAVIOR_LABELS = [
    "normal",
    "raise_hand",
    "head_turn",
    "pass_suspicious_item",
    "hand_under_table_look_down",
    "pick_suspicious_item",
    "carry_phone",
    "tear_paper",
    "student_standing",
]


class FeatureDataset(Dataset):
    def __init__(self, path: Path, labels: list[str]):
        self.payload = torch.load(path, map_location="cpu")
        self.features = self.payload["features"].float()
        self.records = self.payload["records"]
        self.label_to_idx = {label: idx for idx, label in enumerate(labels)}
        self.labels = []
        self.sample_types = []
        for item in self.records:
            label = item["label"]
            if label not in self.label_to_idx:
                raise ValueError(f"Unknown label {label} in {path}")
            self.labels.append(self.label_to_idx[label])
            self.sample_types.append(item.get("sample_type", "base"))

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        return (
            self.features[index],
            torch.tensor(self.labels[index], dtype=torch.long),
            self.sample_types[index],
        )


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the RTMDet ROI behavior head only.")
    parser.add_argument("--behavior-train", type=Path, required=True)
    parser.add_argument("--behavior-val", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, default=DEFAULT_BASE_CHECKPOINT)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs(zjs)/rtm_roi_heads_behavior_augmented.pt",
    )
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--hard-weight", type=float, default=1.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def make_sampler(dataset: FeatureDataset, hard_weight: float) -> WeightedRandomSampler:
    counts = Counter(dataset.labels)
    weights = []
    for label, sample_type in zip(dataset.labels, dataset.sample_types):
        weight = 1.0 / max(counts[label], 1)
        if sample_type == "hard":
            weight *= hard_weight
        weights.append(weight)
    return WeightedRandomSampler(torch.tensor(weights, dtype=torch.double), len(weights), replacement=True)


def class_weights(dataset: FeatureDataset, device: torch.device) -> torch.Tensor:
    counts = Counter(dataset.labels)
    total = sum(counts.values())
    weights = torch.ones(len(dataset.label_to_idx), dtype=torch.float32)
    for label_idx in range(len(weights)):
        weights[label_idx] = total / (len(weights) * max(counts[label_idx], 1))
    return weights.to(device)


def run_epoch(model, loader, criterion, device, num_classes: int, optimizer=None):
    train = optimizer is not None
    model.train(train)
    total_loss = 0.0
    total = 0
    correct = 0
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)
    with torch.set_grad_enabled(train):
        for features, labels, _sample_types in loader:
            features = features.to(device)
            labels = labels.to(device)
            logits = model(features)
            loss = criterion(logits, labels)
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            preds = logits.argmax(dim=1)
            for target, pred in zip(labels.cpu(), preds.cpu()):
                confusion[int(target), int(pred)] += 1
            batch = labels.size(0)
            total += batch
            correct += int((preds == labels).sum().item())
            total_loss += float(loss.item()) * batch

    recalls = [
        float(confusion[idx, idx]) / max(float(confusion[idx, :].sum()), 1.0)
        for idx in range(num_classes)
    ]
    return {
        "loss": total_loss / max(total, 1),
        "acc": correct / max(total, 1),
        "balanced_acc": sum(recalls) / len(recalls),
        "per_class_recall": {
            BEHAVIOR_LABELS[idx]: recalls[idx] for idx in range(len(BEHAVIOR_LABELS))
        },
    }


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    torch.manual_seed(args.seed)

    base_checkpoint = torch.load(args.base_checkpoint, map_location="cpu")
    if "identity" not in base_checkpoint:
        raise ValueError("The base checkpoint does not contain an identity head to preserve.")
    hidden_dim = int(base_checkpoint.get("hidden_dim", 256))
    dropout = float(base_checkpoint.get("dropout", 0.25))

    train_ds = FeatureDataset(args.behavior_train, BEHAVIOR_LABELS)
    val_ds = FeatureDataset(args.behavior_val, BEHAVIOR_LABELS)
    if train_ds.features.shape[1] != val_ds.features.shape[1]:
        raise ValueError("behavior train/val feature dimensions differ")

    model = RoiHead(train_ds.features.shape[1], len(BEHAVIOR_LABELS), hidden_dim, dropout).to(device)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=make_sampler(train_ds, args.hard_weight),
        num_workers=0,
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    criterion = nn.CrossEntropyLoss(weight=class_weights(train_ds, device), label_smoothing=0.03)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_state = None
    best_score = -1.0
    best_epoch = 0
    stale = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model, train_loader, criterion, device, len(BEHAVIOR_LABELS), optimizer
        )
        val_metrics = run_epoch(model, val_loader, criterion, device, len(BEHAVIOR_LABELS))
        scheduler.step()
        score = val_metrics["balanced_acc"]
        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})
        print(
            f"behavior epoch={epoch:03d} train_acc={train_metrics['acc']:.4f} "
            f"val_acc={val_metrics['acc']:.4f} val_bal={score:.4f}"
        )
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                print(f"early stopping at epoch={epoch}; best_epoch={best_epoch}")
                break

    assert best_state is not None
    behavior = {
        "state_dict": best_state,
        "labels": BEHAVIOR_LABELS,
        "input_dim": train_ds.features.shape[1],
        "history": history,
        "best_balanced_acc": best_score,
        "best_epoch": best_epoch,
        "train_counts": dict(Counter(train_ds.labels)),
        "val_counts": dict(Counter(val_ds.labels)),
        "train_augmentation": train_ds.payload.get("augmentation", {}),
    }

    output_checkpoint = dict(base_checkpoint)
    output_checkpoint["behavior"] = behavior
    output_checkpoint["hidden_dim"] = hidden_dim
    output_checkpoint["dropout"] = dropout
    output_checkpoint["zjs_training"] = {
        "mode": "behavior_only_image_augmentation",
        "base_checkpoint": str(args.base_checkpoint),
        "identity_head_preserved": True,
        "pose_used": False,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output_checkpoint, args.output)
    summary_path = args.output.with_suffix(".summary.json")
    summary = {
        "output": str(args.output),
        "base_checkpoint": str(args.base_checkpoint),
        "training_mode": "ROI image behavior head only",
        "pose_used": False,
        "identity_head_preserved": True,
        "behavior_best_epoch": best_epoch,
        "behavior_best_balanced_acc": best_score,
        "behavior_labels": BEHAVIOR_LABELS,
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "train_augmentation": train_ds.payload.get("augmentation", {}),
        "best_val_per_class_recall": history[best_epoch - 1]["val"]["per_class_recall"],
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved checkpoint: {args.output}")
    print(f"saved summary: {summary_path}")


if __name__ == "__main__":
    main()
