#!/usr/bin/env bash
# Ornith-9B ES smoke on 8×GPU (4 groups × TP=2).
#
# Needs on host only:
#   - conda qwen3_5_use (sglang 0.5.17 + ES hooks)
#   - docker CLI + clawgym-rl:v0.1 (per-task ClawGym sandboxes)
#
#   ./scripts/run_ornith9b_es_smoke.sh
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-/datas/miniconda3/envs/qwen3_5_use/bin/python}"
export SGLANG_PYTHON="${SGLANG_PYTHON:-$PYTHON}"

CONDA_SITE="$("$PYTHON" -c 'import site; print(site.getsitepackages()[0])')"
for _nv_lib in \
  "${CONDA_SITE}/nvidia/cu13/lib" \
  "${CONDA_SITE}/nvidia/cuda_nvrtc/lib"; do
  if [[ -d "$_nv_lib" ]]; then
    export LD_LIBRARY_PATH="${_nv_lib}:${LD_LIBRARY_PATH:-}"
  fi
done
unset _nv_lib CONDA_SITE

MODEL_HOST="${MODEL_HOST:-/data/models/Ornith-9B-merged-mixrl02-ornith08}"
MODEL_GPC="${MODEL_GPC:-/data/gpc/models/Ornith-9B-merged-mixrl02-ornith08}"
if [[ ! -d "${MODEL_GPC}" && -d "${MODEL_HOST}" ]]; then
  echo "[ornith9b_es_smoke] hardlink copy ${MODEL_HOST} -> ${MODEL_GPC}"
  mkdir -p "$(dirname "${MODEL_GPC}")"
  cp -al "${MODEL_HOST}" "${MODEL_GPC}"
fi

BASE_DIR="${BASE_DIR:-${MODEL_GPC}}"
TRAIN_DATASET="${TRAIN_DATASET:-/dev/gpc_code/clawGym/ClawGym-Agents/RL/data/clawgym_train}"
EVAL_DATASET="${EVAL_DATASET:-/dev/gpc_code/clawGym/ClawGym-Agents/RL/data/clawgym_eval}"
RUNS_ROOT="${RUNS_ROOT:-/data/gpc/agentic_es/runs}"
RUN_ID="${RUN_ID:-ornith9b_es_smoke_$(date -u +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-${RUNS_ROOT}/${RUN_ID}}"

export PYTHONPATH="${ROOT}/../clawGym/ClawGym-Agents/RL:${ROOT}:${PYTHONPATH:-}"
export OPENCLAW_TASK_TIMEOUT_SECONDS="${OPENCLAW_TASK_TIMEOUT_SECONDS:-900}"
export CLAWGYM_TOOL_EXEC_TIMEOUT_MAX="${CLAWGYM_TOOL_EXEC_TIMEOUT_MAX:-600}"

if [[ ! -d "${BASE_DIR}" ]]; then
  echo "[ornith9b_es_smoke] ERROR: base model not found: ${BASE_DIR}" >&2
  exit 1
fi
if ! command -v docker >/dev/null 2>&1; then
  echo "[ornith9b_es_smoke] ERROR: docker CLI required for --sandbox docker." >&2
  exit 1
fi

mkdir -p "$RUN_ROOT"

echo "[ornith9b_es_smoke] python=$("$PYTHON" --version 2>&1)"
echo "[ornith9b_es_smoke] base=${BASE_DIR} run=${RUN_ROOT}"

exec "$PYTHON" train_es_clawgym.py \
  --run-id "$RUN_ID" \
  --run-root "$RUN_ROOT" \
  --base-dir "$BASE_DIR" \
  --dataset-dir "$TRAIN_DATASET" \
  --eval-dataset-dir "$EVAL_DATASET" \
  --eval-suite all \
  --eval-limit 0 \
  --generations 3 \
  --num-gpus 8 \
  --num-groups 4 \
  --tp-size 2 \
  --tasks-per-group 8 \
  --concurrency 8 \
  --max-turns 40 \
  --max-tokens 8192 \
  --max-total-tokens 80000 \
  --context-length 81920 \
  --task-timeout 900 \
  --save-step 3 \
  --eval-step 3 \
  --inter-gen-cooldown 20 \
  --sandbox docker \
  "$@"
