# ClawGym + noise_weight ES 训练方案

本文档描述 `/dev/gpc_code/agentic_es/` 中自研的 **Agentic-ESOpt 风格** 训练框架：在 **冻结 base 模型** 上维护可累积的 **`noise_weight` delta**，通过 ClawGym 任务 rollout 得到标量 reward，用进化策略（ES）更新 delta，最终部署模型为 `base + noise_weight`。

---

## 1. 背景与动机

### 1.1 要解决的问题

长时程 Agent（多轮 tool-call、Docker sandbox）微调传统上依赖 GRPO/PPO 等 RL 方法，存在：

- 反向传播显存开销大
- 长轨迹 credit assignment 困难
- 与现有 SGLang 独立部署栈集成成本高

**ES（Evolution Strategies）** 把训练当成黑盒优化：对参数加扰动 → 跑完整轨迹 → 用标量 reward 加权更新，**不需要梯度**。

### 1.2 与 Agentic-ESOpt 原 repo 的关系

| 复用 | 不复用 |
|------|--------|
| seeded Gaussian 扰动 + seed-replay | vLLM worker ES 接口 |
| z-score 归一化 + 加权 ES update | Sudoku / WebArena / Math 环境 |
| σ 余弦/线性调度 | VERL GRPO baseline |

我们采用 **ClawGym 任务数据** + **SGLang 部署** + **noise_weight 双权重** 的工程方案，算法核心与 [Agentic-ESOpt](https://github.com/zz1358m/Agentic-ESOpt) 一致。

---

## 2. 总体架构

```
┌─────────────────────────────────────────────────────────────┐
│  base（冻结）          Qwen3.5-9B 等 HF checkpoint，只读       │
│  noise_weight（训练）  与 base 同结构的零初始化 delta，逐代累积 │
└─────────────────────────────────────────────────────────────┘
                              │
         有效模型 θ = base + noise_weight
                              │
    ┌─────────────────────────┴─────────────────────────┐
    │              每一代（generation）                  │
    │  1. 采样 N 个 seed（population）                  │
    │  2. 对每个候选 i：                                │
    │     delta_i = copy(noise_weight) + σ·ε(seed_i)  │
    │     checkpoint_i = merge(base, delta_i)           │
    │     启动 SGLang_i                                 │
    │  3. ClawGym batch rollout → r_i（平均 reward）    │
    │  4. z-score(r) → w                                │
    │  5. noise_weight += (α/N) Σ w_i·ε(seed_i)        │
    └───────────────────────────────────────────────────┘
                              │
         导出：merge(base, noise_weight) → 可部署 HF 模型
```

### 2.1 为何用 noise_weight 而不是直接改 base

1. **base 始终不变**，便于对比实验、回滚、多轮 resume
2. **扰动只加在 delta 拷贝上**，评估候选 `base + (noise_weight + σ·ε)`，永久更新只写入 master `noise_weight`
3. **与 SGLang 兼容**：每次评估需 materialize 完整 HF checkpoint（SGLang 不支持运行时 base+delta 热加载）

---

## 3. 算法细节

### 3.1 符号

| 符号 | 含义 |
|------|------|
| θ | 当前有效参数 = base + noise_weight |
| N | population 大小（每代候选数） |
| σ | 扰动幅度（可调度） |
| α | ES 更新步长 |
| ε(seed) | 由 seed 确定性生成的标准高斯噪声（每个权重元素独立） |
| r_i | 候选 i 在 case_batch 上的平均 reward |

### 3.2 单代流程

```text
输入：base（冻结）, noise_weight, case_batch 任务列表, 超参

1. σ ← sigma_at_step(generation)
2. seeds ← 随机采样 N 个整数
3. for i in 0..N-1:
       delta_i ← copy(noise_weight)
       delta_i += σ · ε(seeds[i])          # 只在拷贝上加噪
       ckpt_i ← merge(base, delta_i)
       启动 SGLang(ckpt_i)
       r_i ← mean( ClawGym_rollout(task) for task in case_batch )
       停止 SGLang
4. w ← zscore([r_0, ..., r_{N-1}])
5. noise_weight += (α/N) · Σ_i w_i · ε(seeds[i])
6. 写入 history.json，保存 noise_weight/
```

### 3.3 扰动如何加

- **不是**在 `[θ-δ, θ+δ]` 范围内均匀采样
- **是**对每个浮点元素独立：`θ'[j] = θ[j] + σ × N(0,1)`
- 同一 `seed` 永远生成同一组 ε（seed-replay）
- 每个参数张量用 `seed XOR hash(参数名)` 混合，保证层间独立且可复现

### 3.4 z-score

对一代内 N 个 reward：

```text
w_i = (r_i - mean(r)) / (std(r) + 1e-8)
```

表现高于平均 → w > 0，更新时沿该候选噪声方向**加强**；低于平均 → w < 0，沿反方向**削弱**。

### 3.5 σ 调度

支持 `constant` / `linear` / `cosine`。Smoke 默认 constant `1e-3`；正式训练可用 cosine `1e-3 → 5e-4`，早期探索、后期微调。

---

## 4. 代码结构

```
/dev/gpc_code/agentic_es/
├── es_core.py              # 扰动、ES update、z-score、σ 调度
├── noise_weight.py         # delta 存储（safetensors）
├── merge_hf.py             # merged = base + delta
├── sglang_pool.py          # N 路 SGLang 并行启停
├── clawgym_es_rollout.py   # ClawGym 任务加载 + batch rollout
├── train_es_clawgym.py     # ES 主循环 + history + resume + eval
├── run_state.py            # history.json 原子读写
├── configs/smoke.yaml      # smoke 超参
├── scripts/                # 一键脚本
├── tests/                  # 单元测试 + mock 一代训练
└── docs/
    └── CLAWGYM_NOISE_WEIGHT_ES.md   # 本文档
```

### 4.1 模块职责

| 模块 | 职责 |
|------|------|
| `es_core.py` | `apply_seeded_noise_tensors`、`es_update_tensors`、`normalize_rewards`、`sigma_at_step` |
| `noise_weight.py` | 从 base 创建全零 delta；`add_perturbation` / `apply_es_update`；save/load |
| `merge_hf.py` | 按 key 做 float32 加法后写 bf16 safetensors，复制 tokenizer/config |
| `sglang_pool.py` | 每候选一个 `SGLangInstance`（端口、GPU、日志） |
| `clawgym_es_rollout.py` | 读 `data_entry.json`；调用 `run_clawgym_bench.run_one_task` |
| `train_es_clawgym.py` | 完整训练、resume replay、周期性 eval |

---

## 5. 数据与环境

### 5.1 ClawGym 任务

- **训练 rollout**：`data/clawgym_train/`（2000 题，每代抽 N 题且不重复）
- **周期性 eval**：`data/clawgym_eval/`（80 题 holdout）
- **单任务目录**：
  ```text
  task_XXXX/
    data_entry.json    # user_query, task_id, metadata
    input_files/       # 可选，初始 workspace
    reward/
      reward.sh        # 执行后 stdout 最后一行为 [0,1] float
      check.py         # 实际判分逻辑
  ```

### 5.2 Rollout 流程

1. Docker sandbox 启动（`CLAWGYM_TOOL_LOOP_DOCKER=1`）
2. 准备 workspace，挂载 input_files
3. ReAct tool loop：调用 SGLang `/v1/chat/completions`（tools + reasoning）
4. 执行 `reward.sh` → 标量 reward

**Agent loop 长度限制**（CLI 可配）：

| 参数 | 默认 | 含义 |
|------|------|------|
| `--max-turns` | 32 | 单题最多 ReAct 轮数 |
| `--max-tokens` | 8192 | 单轮 LLM 最大生成 token |
| `--max-total-tokens` | 65536 | 单题整条轨迹总 token 预算（prompt+completion）；0=不限制 |
| `--context-length` | 65536 | SGLang 服务端 KV 窗口（应 ≥ max_total_tokens） |

每轮会根据当前 prompt 估算 token，动态 cap 本轮 `max_tokens`，避免超出总预算。

### 5.3 依赖

- Python 3.10+，PyTorch，safetensors，requests
- **SGLang**（`python -m sglang.launch_server`，需 qwen3 tool/reasoning parser）
- **Docker** + 镜像 `clawgym-rl:v0.1`
- **Slime 路径**（ClawGym rollout 懒加载）：`SLIME_ROOT=/dev/gpc_code/slime_0427/slime`
- 本地 **HF base checkpoint**（默认 ` /dev/gpc_code/model/Qwen3_5_9B/Qwen3___5-9B`）

---

## 6. 使用指南

### 6.1 安装与测试

```bash
cd /dev/gpc_code/agentic_es
pip install -r requirements.txt
python -m unittest discover -s tests -v
```

### 6.2 分阶段验证

**Phase 0 — 纯算法（无需 GPU/Docker）**

```bash
python -m unittest discover -s tests -v
```

**Phase 1 — merge 验证（已在 9B 上跑通）**

```bash
BASE_DIR=/dev/gpc_code/model/Qwen3_5_9B/Qwen3___5-9B \
  ./scripts/run_merge_only_smoke.sh
# 产出：runs/merge_smoke_real/merged_perturbed
```

**Phase 1 — 单候选 + 单题 + SGLang**

```bash
GPUS=0 BASE_DIR=/path/to/Qwen3.5-9B ./scripts/run_phase1_smoke.sh
```

**Phase 2 — 完整 ES smoke**

```bash
GPUS=0,1,2,3 POPULATION=2 TASK_LIMIT=8 GENERATIONS=2 \
  ./scripts/run_es_smoke.sh
```

**已有 SGLang 时仅测 rollout**

```bash
OPENCLAW_MODEL_URL=http://127.0.0.1:8001 \
OPENCLAW_MODEL_ID=your-model-id \
  ./scripts/run_rollout_smoke.sh
```

### 6.3 Resume 与导出

```bash
# 从 history replay 后继续训练
./scripts/run_es_resume.sh runs/xxx/history.json --generations 10

# 导出可部署模型
./scripts/merge_export.sh \
  /dev/gpc_code/model/Qwen3_5_9B/Qwen3___5-9B \
  runs/xxx/noise_weight \
  runs/xxx/final_export

# 单独 eval（merge + 单 SGLang + 全 eval 集）
./scripts/run_es_eval.sh runs/xxx/noise_weight
```

---

## 7. 超参数

| 参数 | 默认 | 说明 |
|------|------|------|
| `num_gpus` | 8 | 总 GPU 数 |
| `num_groups` / `population` | 4 | ES 候选数 = 并行 SGLang 组数 |
| `tp_size` | 2 | 每组 SGLang 的 tensor parallel |
| `tasks_per_group` | 8 | 每组每代跑多少 ClawGym 题 |
| `concurrency` | 8 | 组内并行 rollout 数（通常 = tasks_per_group） |
| `generations` | 2 | ES 代数 |
| `alpha` | 5e-4 | 永久更新步长 |
| `sigma_start/end` | 1e-3 | 评估扰动幅度 |
| `sigma_schedule` | constant | 可改 cosine |
| `reward_normalization` | zscore | 可选 centered_rank |
| `es_seed` | 20260627 | 控制每代 seed 序列 |
| `gpu_offset` | 0 | 未指定 `--gpus` 时从该 id 起连续分配 |
| `max_turns` / `max_steps` | 32 | 单题最大 ReAct 轮数 |
| `max_tokens` | 8192 | 单轮最大生成长度 |
| `max_total_tokens` | 65536 | 单题轨迹总 token 上限 |
| `context_length` | 65536 | SGLang `--context-length` |
| `turn_timeout` | 300 | 单轮 HTTP 超时（秒） |

**约束**：`num_groups × tp_size == num_gpus`。未传 `--gpus` 时自动使用 `gpu_offset .. gpu_offset+num_gpus-1`。

配置文件：`configs/smoke.yaml`（参考用，脚本以 CLI/env 为准）。

---

## 8. GPU 与资源规划（8×A800）

**推荐默认布局**（已实现为 CLI 默认值）：

```text
8 GPU = 4 groups × TP=2
每组 SGLang 每 step 跑 8 题（concurrency=8）
→ 一代共 4 个 ES 候选 × 8 题 = 32 次 rollout
```

```bash
./scripts/run_es_8gpu.sh
# 或
python train_es_clawgym.py \
  --num-gpus 8 --num-groups 4 --tp-size 2 \
  --tasks-per-group 8 --concurrency 8
```

| 方案 | num_groups | tp_size | num_gpus | 说明 |
|------|------------|---------|----------|------|
| **默认** | 4 | 2 | 8 | 8 卡全用，4 路候选并行 |
| 小显存 smoke | 2 | 1 | 2 | 2 卡调试 |
| 单组大 TP | 1 | 4 | 4 | 仅 1 候选，TP=4 |

**磁盘**：每个候选 merge 约 18GB（9B bf16），N=4 一代临时占用 ~72GB；训练完可 `--cleanup-merged` 删除。

**时间**：单题 ClawGym Docker rollout 约数分钟；smoke 8 题 × 2 候选 ≈ 数十分钟/代。

---

## 9. 产出物说明

一次训练 run 目录（如 `runs/clawgym_es_smoke/`）：

```text
runs/<run_id>/
├── history.json                    # 每代 seeds/rewards/weights/σ/α + checkpoint/eval 记录
├── noise_weight/                   # 最新累积 delta（每代覆盖）
├── checkpoints/
│   └── checkpoint_gen0004/         # --save-step N 时额外快照
├── gen_0000/
│   ├── candidate_00_rollout/       # 每组 rollout 落盘
│   │   ├── summary.json
│   │   ├── rollout_meta.json
│   │   └── tasks/<task_id>/
│   │       ├── result.json
│   │       ├── transcript.json     # OpenAI messages 原始格式
│   │       ├── trajectory.json     # 结构化轨迹：user_query + 每轮 assistant/tool
│   │       └── workspace/ ...
│   └── generation_rollout_summary.json
├── eval_gen0004/                   # --eval-step N 时的 eval rollout
├── logs/                           # SGLang 日志
└── final_merged/                   # 训练结束 base + noise_weight
```

### 周期性保存

| 参数 | 默认 | 行为 |
|------|------|------|
| `--save-step N` | 0（关） | 每 N 代额外保存 `checkpoints/checkpoint_genXXXX/noise_weight` |
| `--eval-step N` | 0（关） | 每 N 代在 eval 集上跑一轮 eval（`eval_genXXXX/`） |

触发条件：第 5、10、15… 代（即 `(generation+1) % N == 0`）。`--eval-interval` 为 `--eval-step` 别名。

每代仍照常更新 `noise_weight/` 与 `history.json`；`save-step` 是**额外**快照，不替代默认保存。

### history.json 结构（摘要）

```json
[
  {"config": {"base_dir": "...", "population": 2, ...}},
  {
    "generation": 0,
    "seeds": [123, 456],
    "rewards": [0.62, 0.41],
    "weights": [0.71, -0.71],
    "sigma": 0.001,
    "alpha": 0.0005,
    "case_batch": ["task_0008", "task_0012"]
  }
]
```

Resume 时：从 `history` 中 replay 所有 `es_update` 到新的零初始化 `noise_weight`，再从 `start_generation` 继续。

---

## 10. 常见问题

**Q：seed 和 rollout 的关系？**  
A：每个**候选**一个 seed；同一候选下 batch 内多题 rollout **共用**同一扰动模型。

**Q：为什么不用 Slime GRPO 那条链？**  
A：ES 只需标量 reward，不需要 token/logprob；直接用 `run_clawgym_bench.run_one_task` 即可。

**Q：SGLang 不在当前 Python 环境？**  
A：训练脚本会 `subprocess` 启动 `python -m sglang.launch_server`，需使用安装了 sglang 的 Python；或先用 `run_rollout_smoke.sh` 对接已有服务。

**Q：merge 很慢？**  
A：9B 全量 merge 约 2 分钟/次；Smoke 保持 N≤4，正式训练可考虑 NVMe 临时目录。

**Q：σ 太小看不出 reward 差异？**  
A：从 `1e-3` 试起，观察一代内 reward 方差；过大则模型输出可能崩坏。

---

## 11. 后续扩展

- [ ] 正式规模：`generations=20~50`，`task_limit=32~80`，σ cosine
- [ ] 固定 task 子集 vs 每代 resample 对比
- [ ] 与 QwenClawBench 100 题打通（当前默认 clawgym_eval 80）
- [x] 评估 hook：`--eval-step N`（`--eval-interval` 别名）跑 eval
- [x] 周期性 checkpoint：`--save-step N` 额外保存 noise_weight
- [x] 每代 rollout 落盘：`summary.json` + `tasks/*/result.json` + `generation_rollout_summary.json`
- [ ] 若 merge 成为瓶颈：探索 LoRA 化 delta 或顺序候选评估

---

## 12. 参考

- Agentic-ESOpt 论文与代码：https://github.com/zz1358m/Agentic-ESOpt
- ClawGym 数据格式：`/dev/gpc_code/clawGym/ClawGym-Agents/RL/data/DATASET_README.md`
- ClawGym Bench：`/dev/gpc_code/clawGym/ClawGym-Agents/RL/run_clawgym_bench.py`
