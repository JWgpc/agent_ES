#!/usr/bin/env bash
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

BASE_DIR="${BASE_DIR:-/data/models/Ornith-9B-merged-mixrl02-ornith08}"
PORT="${PORT:-12100}"
GPUS="${GPUS:-0,1}"
TP_SIZE="${TP_SIZE:-2}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-81920}"
LOG="${LOG:-/tmp/sglang_es_hook_smoke.log}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

if [[ ! -d "$BASE_DIR" ]]; then
  echo "base model not found: $BASE_DIR" >&2
  exit 1
fi

cleanup() {
  if [[ -n "${SGLANG_PID:-}" ]] && kill -0 "$SGLANG_PID" 2>/dev/null; then
    kill "$SGLANG_PID" 2>/dev/null || true
    wait "$SGLANG_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

echo "[hook-smoke] starting SGLang ES server on port=$PORT gpus=$GPUS tp=$TP_SIZE"
CUDA_VISIBLE_DEVICES="$GPUS" "$PYTHON" -m sglang_es_launch_server \
  --model-path "$BASE_DIR" \
  --served-model-name clawgym-es-hook-smoke \
  --port "$PORT" \
  --tp-size "$TP_SIZE" \
  --mem-fraction-static 0.85 \
  --context-length "$CONTEXT_LENGTH" \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  --host 127.0.0.1 \
  >"$LOG" 2>&1 &
SGLANG_PID=$!

echo "[hook-smoke] server pid=$SGLANG_PID log=$LOG"
"$PYTHON" scripts/smoke_es_hook.py --base-url "http://127.0.0.1:${PORT}" --wait-timeout 900
echo "[hook-smoke] all ES endpoints passed"
