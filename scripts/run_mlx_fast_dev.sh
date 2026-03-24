#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export RUN_ID="${RUN_ID:-mlx_fast_dev}"
export FAST_DEV_RUN="${FAST_DEV_RUN:-1}"
export ITERATIONS="${ITERATIONS:-50}"
export TRAIN_BATCH_TOKENS="${TRAIN_BATCH_TOKENS:-8192}"
export VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-0}"
export VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-8192}"
export TRAIN_LOG_EVERY="${TRAIN_LOG_EVERY:-10}"
export WARMUP_STEPS="${WARMUP_STEPS:-5}"

exec python3 train_gpt_mlx.py "$@"
