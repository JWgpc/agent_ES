#!/usr/bin/env bash
# Resume ES training from an existing history.json (replay updates into fresh noise_weight).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 /path/to/history.json [extra train_es_clawgym.py args...]" >&2
  exit 1
fi

HISTORY="$1"
shift

BASE_DIR="${BASE_DIR:-/dev/gpc_code/model/Qwen3_5_9B/Qwen3___5-9B}"
RUN_ID="${RUN_ID:-clawgym_es_resume_$(date -u +%Y%m%d_%H%M%S)}"

export SLIME_ROOT="${SLIME_ROOT:-/dev/gpc_code/slime_0427/slime}"
export PYTHONPATH="${SLIME_ROOT}:${ROOT}/../clawGym/ClawGym-Agents/RL:${ROOT}:${PYTHONPATH:-}"

python train_es_clawgym.py \
  --run-id "$RUN_ID" \
  --base-dir "$BASE_DIR" \
  --resume-history "$HISTORY" \
  "$@"
