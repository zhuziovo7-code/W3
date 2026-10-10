#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/roi_image_augmentation}"
CACHE_DIR="${OUTPUT_ROOT}/cache"
LOG_DIR="${OUTPUT_ROOT}/logs"
mkdir -p "${CACHE_DIR}" "${LOG_DIR}"

PYTHON_CMD=(python)
PATH_MAP="${PATH_MAP:-}"
NUM_SHARDS="${NUM_SHARDS:-4}"
TRAIN_MANIFEST="${TRAIN_MANIFEST:-data/annotations/rtm_roi_heads/behavior_train.jsonl}"
VAL_MANIFEST="${VAL_MANIFEST:-data/annotations/rtm_roi_heads/behavior_val.jsonl}"
BASE_CHECKPOINT="${BASE_CHECKPOINT:-models/behavior/rtm_roi_heads_base.pt}"

echo "[$(date '+%F %T')] start ZJS ROI image-augmentation pipeline"

extract_shards() {
  local split_name="$1"
  local manifest="$2"
  local augment_copies="$3"
  local pids=()

  echo "[$(date '+%F %T')] extract ${split_name} with ${NUM_SHARDS} GPU shards"
  for ((shard = 0; shard < NUM_SHARDS; shard++)); do
    local path_map_args=()
    if [[ -n "${PATH_MAP}" ]]; then
      path_map_args=(--path-map "${PATH_MAP}")
    fi
    CUDA_VISIBLE_DEVICES="${shard}" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
      "${PYTHON_CMD[@]}" "${SCRIPT_DIR}/zjs_extract_rtm_roi_features.py" \
        --manifest "${manifest}" \
        --output "${CACHE_DIR}/${split_name}.part${shard}.pt" \
        --failed-jsonl "${CACHE_DIR}/${split_name}.part${shard}.failed.jsonl" \
        "${path_map_args[@]}" \
        --augment-copies "${augment_copies}" \
        --augment-seed 42 \
        --num-shards "${NUM_SHARDS}" \
        --shard-index "${shard}" \
        --device auto \
        --progress-interval 250 \
        > "${LOG_DIR}/extract_${split_name}.part${shard}.log" 2>&1 &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do
    wait "${pid}"
  done

  local part_files=()
  local failed_files=()
  for ((shard = 0; shard < NUM_SHARDS; shard++)); do
    part_files+=("${CACHE_DIR}/${split_name}.part${shard}.pt")
    failed_files+=("${CACHE_DIR}/${split_name}.part${shard}.failed.jsonl")
  done
  "${PYTHON_CMD[@]}" "${SCRIPT_DIR}/zjs_merge_rtm_roi_features.py" \
    --inputs "${part_files[@]}" \
    --output "${CACHE_DIR}/${split_name}.pt" \
    --failed-inputs "${failed_files[@]}" \
    --failed-output "${CACHE_DIR}/${split_name}.failed.jsonl"
}

extract_shards \
  behavior_train_aug \
  "${TRAIN_MANIFEST}" \
  1

extract_shards \
  behavior_val \
  "${VAL_MANIFEST}" \
  0

echo "[$(date '+%F %T')] train ROI behavior head only"
"${PYTHON_CMD[@]}" "${SCRIPT_DIR}/zjs_train_rtm_roi_heads.py" \
  --behavior-train "${CACHE_DIR}/behavior_train_aug.pt" \
  --behavior-val "${CACHE_DIR}/behavior_val.pt" \
  --base-checkpoint "${BASE_CHECKPOINT}" \
  --output "${OUTPUT_ROOT}/rtm_roi_heads_behavior_augmented.pt" \
  --epochs 80 \
  --batch-size 256 \
  --hard-weight 1.0 \
  --lr 0.001 \
  --weight-decay 0.0005 \
  --device auto \
  > "${LOG_DIR}/train_behavior_head.log" 2>&1

echo "[$(date '+%F %T')] done ZJS ROI image-augmentation pipeline"
