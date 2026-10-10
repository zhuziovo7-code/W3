#!/usr/bin/env python3
"""Merge independently extracted RTMDet ROI feature shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge ROI feature .pt shard files.")
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--failed-inputs", type=Path, nargs="*", default=[])
    parser.add_argument("--failed-output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payloads = [torch.load(path, map_location="cpu") for path in args.inputs]
    if not payloads:
        raise ValueError("No feature shards were provided.")

    feature_dims = {int(payload["features"].shape[1]) for payload in payloads}
    if len(feature_dims) != 1:
        raise ValueError(f"Feature dimensions differ across shards: {sorted(feature_dims)}")

    merged_features = torch.cat([payload["features"] for payload in payloads], dim=0)
    merged_records = []
    for payload in payloads:
        merged_records.extend(payload["records"])
    for feature_index, record in enumerate(merged_records):
        record["feature_index"] = feature_index

    merged = dict(payloads[0])
    merged["features"] = merged_features
    merged["records"] = merged_records
    merged["merged_shards"] = [str(path) for path in args.inputs]
    merged.pop("shard", None)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(merged, args.output)

    failure_count = 0
    if args.failed_output:
        args.failed_output.parent.mkdir(parents=True, exist_ok=True)
        with args.failed_output.open("w", encoding="utf-8") as output_handle:
            for path in args.failed_inputs:
                if not path.is_file():
                    continue
                with path.open("r", encoding="utf-8") as input_handle:
                    for line in input_handle:
                        if not line.strip():
                            continue
                        json.loads(line)
                        output_handle.write(line)
                        failure_count += 1

    print(
        f"saved={args.output} shards={len(payloads)} "
        f"features={len(merged_records)} failures={failure_count}"
    )


if __name__ == "__main__":
    main()
