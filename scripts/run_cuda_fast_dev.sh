#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export RUN_ID="${RUN_ID:-cuda_fast_dev}"
export DATA_PATH="${DATA_PATH:-./data/datasets/fineweb10B_sp1024/}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-./data/tokenizers/fineweb_1024_bpe.model}"
export VOCAB_SIZE="${VOCAB_SIZE:-1024}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
export FAST_DEV_RUN="${FAST_DEV_RUN:-1}"
export ITERATIONS="${ITERATIONS:-100}"
export TRAIN_BATCH_TOKENS="${TRAIN_BATCH_TOKENS:-65536}"
export VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-0}"
export TRAIN_LOG_EVERY="${TRAIN_LOG_EVERY:-10}"
export WARMUP_STEPS="${WARMUP_STEPS:-5}"

exec torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" train_gpt.py "$@"
