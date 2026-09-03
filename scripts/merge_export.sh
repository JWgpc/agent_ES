#!/usr/bin/env bash
# Merge base + noise_weight into a deployable HF checkpoint.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 BASE_DIR NOISE_WEIGHT_DIR OUTPUT_DIR" >&2
  exit 1
fi

export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
python merge_hf.py "$1" "$2" "$3"
