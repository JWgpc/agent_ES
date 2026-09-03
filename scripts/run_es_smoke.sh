#!/usr/bin/env bash
# Full ES smoke: N parallel SGLang instances + ClawGym batch rollout.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

BASE_DIR="${BASE_DIR:-/dev/gpc_code/model/Qwen3_5_9B/Qwen3___5-9B}"
RUN_ID="${RUN_ID:-clawgym_es_smoke_$(date -u +%Y%m%d_%H%M%S)}"
GPUS="${GPUS:-0,1,2,3}"
GENERATIONS="${GENERATIONS:-2}"
POPULATION="${POPULATION:-2}"
TASK_LIMIT="${TASK_LIMIT:-8}"

export SLIME_ROOT="${SLIME_ROOT:-/dev/gpc_code/slime_0427/slime}"
export PYTHONPATH="${SLIME_ROOT}:${ROOT}/../clawGym/ClawGym-Agents/RL:${ROOT}:${PYTHONPATH:-}"

python train_es_clawgym.py \
  --run-id "$RUN_ID" \
  --base-dir "$BASE_DIR" \
  --generations "$GENERATIONS" \
  --population "$POPULATION" \
  --task-limit "$TASK_LIMIT" \
  --gpus "$GPUS" \
  --sandbox "${SANDBOX:-docker}" \
  --concurrency "${CONCURRENCY:-2}" \
  "$@"
