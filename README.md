# ClawGym + noise_weight ES

Self-contained Agentic-ESOpt-style training for ClawGym tasks using a frozen base model and an accumulated `noise_weight` delta.

**完整设计文档（中文）**：[`docs/CLAWGYM_NOISE_WEIGHT_ES.md`](docs/CLAWGYM_NOISE_WEIGHT_ES.md)

## Layout

- `es_core.py` — seeded Gaussian perturbation, z-score, sigma schedule, ES update
- `noise_weight.py` — zero-init delta store (safetensors)
- `merge_hf.py` — `merged = base + delta`
- `sglang_pool.py` — parallel SGLang servers for population evaluation
- `clawgym_es_rollout.py` — ClawGym batch rollout + mean reward
- `train_es_clawgym.py` — main ES loop + `history.json` + resume
- `configs/smoke.yaml` — default smoke hyperparameters

## Quick start

```bash
cd /dev/gpc_code/agentic_es
python -m unittest discover -s tests -v

# Phase 1: one candidate, one task (needs base model + Docker + SGLang)
BASE_DIR=/path/to/Qwen3.5-9B GPUS=0 ./scripts/run_phase1_smoke.sh

# Phase 2 smoke: 2 candidates x 8 tasks x 2 generations
GPUS=0,1,2,3 ./scripts/run_es_smoke.sh

# Resume from history
./scripts/run_es_resume.sh runs/clawgym_es_smoke/history.json --generations 5

# Export merged checkpoint
./scripts/merge_export.sh $BASE_DIR runs/clawgym_es_smoke/noise_weight runs/exported
```

## Algorithm

Each generation:

1. Sample `population` seeds
2. For each candidate: `delta = noise_weight + σ·ε(seed)`, merge with base, start SGLang
3. Run ClawGym rollouts, average reward per candidate
4. z-score rewards → ES update on `noise_weight`
5. Save `history.json` and `noise_weight/`

Effective model: `base + noise_weight`.

## Requirements

- PyTorch, safetensors, requests
- SGLang with Qwen tool/reasoning parsers
- ClawGym Docker image (`clawgym-rl:v0.1`) for docker sandbox mode
- Local HF base checkpoint (not committed)
