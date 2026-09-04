#!/usr/bin/env bash
# Eval merged base+noise_weight on ClawGym eval split using one SGLang instance.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

BASE_DIR="${BASE_DIR:-/dev/gpc_code/model/Qwen3_5_9B/Qwen3___5-9B}"
NOISE_WEIGHT_DIR="${1:-}"
RUN_ROOT="${RUN_ROOT:-/data/gpc/agentic_es/runs/eval_only}"
GPUS="${GPUS:-0}"

if [[ -z "$NOISE_WEIGHT_DIR" ]]; then
  echo "Usage: $0 NOISE_WEIGHT_DIR" >&2
  exit 1
fi

export SLIME_ROOT="${SLIME_ROOT:-/dev/gpc_code/slime_0427/slime}"
export PYTHONPATH="${SLIME_ROOT}:${ROOT}/../clawGym/ClawGym-Agents/RL:${ROOT}:${PYTHONPATH:-}"

python - <<PY
from pathlib import Path
import argparse

# Reuse train_es_clawgym eval path via CLI wrapper logic
from merge_hf import merge_base_delta
from noise_weight import NoiseWeightStore
from clawgym_es_rollout import batch_rollout, load_tasks
from sglang_pool import SGLangInstance

base = Path("$BASE_DIR").resolve()
noise_dir = Path("$NOISE_WEIGHT_DIR").resolve()
run_root = Path("$RUN_ROOT").resolve()
merged = run_root / "eval_merged"
noise = NoiseWeightStore.load(noise_dir, base_dir=base)
noise.save(merged / "delta")
merge_base_delta(base, merged / "delta", merged)

port = 12100
served = "clawgym-es-eval"
gpus = [int(x) for x in "$GPUS".split(",") if x.strip()]
inst = SGLangInstance(model_path=merged, port=port, gpu_ids=gpus, served_model_name=served)
inst.start()
try:
    tasks = load_tasks(Path("/dev/gpc_code/clawGym/ClawGym-Agents/RL/data/clawgym_eval"))
    summary = batch_rollout(
        tasks=tasks,
        model_url=inst.base_url,
        model_id=served,
        run_dir=run_root / "rollout",
        concurrency=int("${CONCURRENCY:-4}"),
        sandbox="${SANDBOX:-docker}",
    )
    print(summary)
finally:
    inst.stop()
PY
