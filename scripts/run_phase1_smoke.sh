#!/usr/bin/env bash
# Phase 1 smoke: merge base+delta, start one SGLang, run one ClawGym task.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

BASE_DIR="${BASE_DIR:-/dev/gpc_code/model/Qwen3_5_9B/Qwen3___5-9B}"
RUN_ID="${RUN_ID:-phase1_smoke_$(date -u +%Y%m%d_%H%M%S)}"
GPUS="${GPUS:-0}"
PORT="${PORT:-12000}"
SIGMA="${SIGMA:-0.001}"
SEED="${SEED:-12345}"

if [[ ! -d "$BASE_DIR" ]]; then
  echo "Base model not found: $BASE_DIR" >&2
  echo "Set BASE_DIR to a local HF checkpoint." >&2
  exit 1
fi

export SLIME_ROOT="${SLIME_ROOT:-/dev/gpc_code/slime_0427/slime}"
export PYTHONPATH="${SLIME_ROOT}:${ROOT}/../clawGym/ClawGym-Agents/RL:${ROOT}:${PYTHONPATH:-}"

python train_es_clawgym.py \
  --phase1-smoke \
  --run-id "$RUN_ID" \
  --base-dir "$BASE_DIR" \
  --gpus "$GPUS" \
  --base-port "$PORT" \
  --sigma-start "$SIGMA" \
  --sigma-end "$SIGMA" \
  --es-seed "$SEED" \
  --sandbox "${SANDBOX:-docker}" \
  --no-cleanup-merged \
  "$@"
