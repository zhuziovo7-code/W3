#!/usr/bin/env python3
"""Train identity and behavior heads on cached RTMDet ROI features."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
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
IDENTITY_LABELS = ["student", "invigilator"]


class FeatureDataset(Dataset):
    def __init__(self, path: Path, labels: list[str]):
        payload = torch.load(path, map_location="cpu")
        self.features = payload["features"].float()
        self.records = payload["records"]
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
        return self.features[index], torch.tensor(self.labels[index], dtype=torch.long), self.sample_types[index]


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
    parser = argparse.ArgumentParser(description="Train RTMDet ROI identity and behavior heads.")
    parser.add_argument("--behavior-train", type=Path, required=True)
    parser.add_argument("--behavior-val", type=Path, required=True)
    parser.add_argument("--identity-train", type=Path, required=True)
    parser.add_argument("--identity-val", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "models/behavior/rtm_roi_heads.pt")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--hard-weight", type=float, default=3.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--patience", type=int, default=15)
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


def run_epoch(model, loader, criterion, device, optimizer=None):
    train = optimizer is not None
    model.train(train)
    total_loss = 0.0
    total = 0
    correct = 0
    confusion = None
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
            if confusion is None:
                confusion = torch.zeros(logits.size(1), logits.size(1), dtype=torch.long)
            for target, pred in zip(labels.cpu(), preds.cpu()):
                confusion[int(target), int(pred)] += 1
            batch = labels.size(0)
            total += batch
            correct += int((preds == labels).sum().item())
            total_loss += float(loss.item()) * batch
    macro_recall = 0.0
    if confusion is not None:
        recalls = []
        for idx in range(confusion.size(0)):
            recalls.append(float(confusion[idx, idx]) / max(float(confusion[idx, :].sum()), 1.0))
        macro_recall = sum(recalls) / len(recalls)
    return {
        "loss": total_loss / max(total, 1),
        "acc": correct / max(total, 1),
        "balanced_acc": macro_recall,
    }


def train_one(name: str, labels: list[str], train_path: Path, val_path: Path, args, device):
    train_ds = FeatureDataset(train_path, labels)
    val_ds = FeatureDataset(val_path, labels)
    if train_ds.features.shape[1] != val_ds.features.shape[1]:
        raise ValueError(f"{name} train/val feature dims differ")
    model = RoiHead(train_ds.features.shape[1], len(labels), args.hidden_dim, args.dropout).to(device)
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
    stale = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, criterion, device, optimizer)
        val_metrics = run_epoch(model, val_loader, criterion, device)
        scheduler.step()
        score = val_metrics["balanced_acc"]
        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})
        print(
            f"{name} epoch={epoch:03d} "
            f"train_acc={train_metrics['acc']:.4f} val_acc={val_metrics['acc']:.4f} "
            f"val_bal={val_metrics['balanced_acc']:.4f}"
        )
        if score > best_score:
            best_score = score
            best_state = {key: value.cpu() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break
    assert best_state is not None
    return {
        "state_dict": best_state,
        "labels": labels,
        "input_dim": train_ds.features.shape[1],
        "history": history,
        "best_balanced_acc": best_score,
        "train_counts": dict(Counter(train_ds.labels)),
        "val_counts": dict(Counter(val_ds.labels)),
    }


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    torch.manual_seed(42)
    behavior = train_one("behavior", BEHAVIOR_LABELS, args.behavior_train, args.behavior_val, args, device)
    identity = train_one("identity", IDENTITY_LABELS, args.identity_train, args.identity_val, args, device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "behavior": behavior,
        "identity": identity,
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
        "hard_weight": args.hard_weight,
    }
    torch.save(checkpoint, args.output)
    summary_path = args.output.with_suffix(".summary.json")
    summary = {
        "output": str(args.output),
        "behavior_best_balanced_acc": behavior["best_balanced_acc"],
        "identity_best_balanced_acc": identity["best_balanced_acc"],
        "behavior_labels": BEHAVIOR_LABELS,
        "identity_labels": IDENTITY_LABELS,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved checkpoint: {args.output}")
    print(f"saved summary: {summary_path}")


if __name__ == "__main__":
    main()
