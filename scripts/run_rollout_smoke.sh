#!/usr/bin/env bash
# Rollout smoke against an already-running SGLang server (no ES weight merge).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODEL_URL="${OPENCLAW_MODEL_URL:-http://127.0.0.1:8001}"
MODEL_ID="${OPENCLAW_MODEL_ID:-ornith-1.5-35b}"
DATASET="${CLAWGYM_BENCH_DATASET:-/dev/gpc_code/clawGym/ClawGym-Agents/RL/data/clawgym_eval}"
SUITE="${SUITE:-task_0008}"
SANDBOX="${SANDBOX:-docker}"
RUN_DIR="${RUN_DIR:-/dev/gpc_code/agentic_es/runs/rollout_smoke}"

export SLIME_ROOT="${SLIME_ROOT:-/dev/gpc_code/slime_0427/slime}"
export PYTHONPATH="${SLIME_ROOT}:${ROOT}/../clawGym/ClawGym-Agents/RL:${ROOT}:${PYTHONPATH:-}"
export CLAWGYM_TOOL_LOOP=1
export CLAWGYM_TOOL_LOOP_DOCKER=$([[ "$SANDBOX" == "docker" ]] && echo 1 || echo 0)

python - <<PY
from pathlib import Path
from clawgym_es_rollout import batch_rollout, load_tasks

tasks = load_tasks(Path("$DATASET"), suite="$SUITE", limit=1)
summary = batch_rollout(
    tasks=tasks,
    model_url="$MODEL_URL",
    model_id="$MODEL_ID",
    run_dir=Path("$RUN_DIR"),
    concurrency=1,
    sandbox="$SANDBOX",
    quiet=False,
)
print("avg_reward=", summary["avg_reward"])
print("n_ok=", summary["n_ok"], "n_error=", summary["n_error"])
PY
