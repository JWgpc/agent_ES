#!/usr/bin/env bash
# Validate base + noise_weight merge without starting SGLang or Docker.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

BASE_DIR="${BASE_DIR:-/dev/gpc_code/model/Qwen3_5_9B/Qwen3___5-9B}"
OUT_ROOT="${OUT_ROOT:-/dev/gpc_code/agentic_es/runs/merge_smoke}"
SIGMA="${SIGMA:-0.001}"
SEED="${SEED:-12345}"

export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

export BASE_DIR OUT_ROOT SIGMA SEED
python - <<'PY'
import os
from pathlib import Path

from merge_hf import merge_base_delta
from noise_weight import NoiseWeightStore

base = Path(os.environ["BASE_DIR"]).resolve()
out_root = Path(os.environ["OUT_ROOT"]).resolve()
sigma = float(os.environ["SIGMA"])
seed = int(os.environ["SEED"])

out_root.mkdir(parents=True, exist_ok=True)
noise_dir = out_root / "noise_weight"
merged_dir = out_root / "merged_perturbed"

store = NoiseWeightStore.zeros_from_base(base)
store.add_perturbation(seed=seed, sigma=sigma)
store.save(noise_dir)
merge_base_delta(base, noise_dir, merged_dir)
print(f"noise_weight -> {noise_dir}")
print(f"merged       -> {merged_dir}")
PY
