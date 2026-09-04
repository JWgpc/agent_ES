# ClawGym + noise_weight ES

Self-contained Agentic-ESOpt-style training for ClawGym tasks using a frozen base model and an accumulated `noise_weight` delta.

**完整设计文档（中文）**：[`docs/CLAWGYM_NOISE_WEIGHT_ES.md`](docs/CLAWGYM_NOISE_WEIGHT_ES.md)

## Layout

| Module | Role |
|--------|------|
| `es_core.py` | Seeded Gaussian perturbation, z-score, sigma schedule, ES update |
| `noise_weight.py` | Zero-init delta store (safetensors) |
| `merge_hf.py` | `merged = base + delta` |
| `sglang_pool.py` | Parallel SGLang servers + port/PID cleanup |
| `sglang_es_*.py` | In-place `/es/*` hooks (apply / revert / update on GPU) |
| `clawgym_es_rollout.py` | ClawGym batch rollout + task sampling |
| `train_es_clawgym.py` | Main ES loop, `history.json`, resume, eval, inter-gen docker cleanup |
| `run_state.py` | Atomic history I/O, resume helpers |
| `configs/smoke.yaml` | Default smoke hyperparameters |

## Quick start

```bash
cd agentic_es
python -m unittest discover -s tests -q

# Verify ES hooks only
./scripts/run_es_hook_smoke.sh

# Ornith-9B full training (host SGLang + docker sandboxes)
./scripts/run_ornith9b_es_smoke.sh \
  --run-root /data/gpc/agentic_es/runs/my_run \
  --generations 100 --num-groups 4 --tp-size 2 --num-gpus 8 \
  --tasks-per-group 8 --concurrency 8 \
  --max-turns 40 --task-timeout 900 \
  --save-step 10 --eval-step 5 --inter-gen-cooldown 20

# Resume same run directory (loads noise_weight + history automatically)
./scripts/run_ornith9b_es_smoke.sh \
  --run-root /data/gpc/agentic_es/runs/my_run \
  --generations 100 ...

# Export merged checkpoint
./scripts/merge_export.sh $BASE_DIR /data/gpc/agentic_es/runs/my_run/noise_weight runs/exported
```

## Algorithm

Each generation (default **`--use-es-hook`**, no per-candidate HF merge):

1. Sample `num_groups` seeds
2. Start SGLang on **base model**; load accumulated `noise_weight` via `/es/load_delta`
3. For each candidate: `/es/apply` → ClawGym rollout → `/es/revert`
4. z-score rewards → `/es/update` on GPU + ES update on CPU `noise_weight`
5. Optional periodic **eval** on holdout set (`--eval-step N`)
6. **Inter-gen cleanup**: pause → remove this generation's docker sandboxes → pause
7. Save `history.json` and `noise_weight/`

**GPU layout**: `num_groups × tp_size == num_gpus` (default 4 × 2 = 8).

**Eval schedule**: runs when `(generation + 1) % eval_step == 0` (e.g. `--eval-step 5` → gen 4, 9, 14, …).

**Run outputs** (not committed): use `--run-root` pointing outside the repo, e.g. `/data/gpc/agentic_es/runs/`.

## Requirements

- PyTorch, safetensors, requests
- Host conda with **sglang 0.5.17+** (ES `/es/*` hooks)
- **docker** CLI + image `clawgym-rl:v0.1` for per-task ClawGym sandboxes
- ClawGym task data: `clawGym/ClawGym-Agents/RL/data/clawgym_train` (2000) / `clawgym_eval` (80)
- Local HF base checkpoint (not committed)

ClawGym bench/eval only needs HTTP → SGLang and docker sandboxes; slime/ray are lazy-loaded for RL-only paths.

## Push to GitHub

```bash
git add -A && git commit -m "..."
./scripts/push_github.sh
```

Remote: https://github.com/JWgpc/agent_ES.git
