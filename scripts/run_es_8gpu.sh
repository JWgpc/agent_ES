#!/usr/bin/env bash
# 8×GPU layout: 4 SGLang groups × TP=2, each group runs 8 ClawGym tasks per generation.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

BASE_DIR="${BASE_DIR:-/dev/gpc_code/model/Qwen3_5_9B/Qwen3___5-9B}"
RUNS_ROOT="${RUNS_ROOT:-/data/gpc/agentic_es/runs}"
RUN_ID="${RUN_ID:-clawgym_es_8gpu_$(date -u +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-${RUNS_ROOT}/${RUN_ID}}"

NUM_GPUS="${NUM_GPUS:-8}"
NUM_GROUPS="${NUM_GROUPS:-4}"
TP_SIZE="${TP_SIZE:-2}"
TASKS_PER_GROUP="${TASKS_PER_GROUP:-8}"
CONCURRENCY="${CONCURRENCY:-8}"
GENERATIONS="${GENERATIONS:-2}"
GPU_OFFSET="${GPU_OFFSET:-0}"

export SLIME_ROOT="${SLIME_ROOT:-/dev/gpc_code/slime_0427/slime}"
export PYTHONPATH="${SLIME_ROOT}:${ROOT}/../clawGym/ClawGym-Agents/RL:${ROOT}:${PYTHONPATH:-}"

python train_es_clawgym.py \
  --run-id "$RUN_ID" \
  --run-root "$RUN_ROOT" \
  --base-dir "$BASE_DIR" \
  --generations "$GENERATIONS" \
  --num-gpus "$NUM_GPUS" \
  --num-groups "$NUM_GROUPS" \
  --tp-size "$TP_SIZE" \
  --gpu-offset "$GPU_OFFSET" \
  --tasks-per-group "$TASKS_PER_GROUP" \
  --concurrency "$CONCURRENCY" \
  --sandbox "${SANDBOX:-docker}" \
  ${GPUS:+--gpus "$GPUS"} \
  "$@"
