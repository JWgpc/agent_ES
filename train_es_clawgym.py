#!/usr/bin/env python3
"""ES training loop: base + noise_weight with ClawGym rollouts."""

from __future__ import annotations

import argparse
import json
import random
import shutil
import statistics
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from clawgym_es_rollout import (
    DEFAULT_EVAL_DATASET,
    DEFAULT_TRAIN_DATASET,
    batch_rollout,
    load_task_pool,
    load_tasks,
    sample_tasks,
)
from es_core import normalize_rewards, sigma_at_step
from merge_hf import merge_base_delta
from noise_weight import NoiseWeightStore
from run_state import (
    atomic_write_json,
    completed_update_records,
    read_history,
    should_run_periodic,
    used_task_ids_from_history,
)
from sglang_es_client import SGLangESClient
from sglang_pool import SGLangInstance, SGLangPool

DEFAULT_RUNS_ROOT = Path("/data/gpc/agentic_es/runs")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_gpu_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _allocate_gpu_groups(all_gpus: list[int], num_groups: int, tp_size: int) -> list[list[int]]:
    need = num_groups * tp_size
    if len(all_gpus) != need:
        raise ValueError(
            f"GPU layout mismatch: num_groups({num_groups}) * tp_size({tp_size}) = {need}, "
            f"but got {len(all_gpus)} GPU id(s): {all_gpus}"
        )
    groups = []
    cursor = 0
    for _ in range(num_groups):
        groups.append(all_gpus[cursor : cursor + tp_size])
        cursor += tp_size
    return groups


def resolve_gpu_layout(args: argparse.Namespace) -> None:
    """Normalize num_groups / num_gpus / tp / gpus; enforce num_groups * tp_size == num_gpus."""
    if args.task_limit > 0:
        args.tasks_per_group = args.task_limit
    elif args.tasks_per_group > 0:
        args.task_limit = args.tasks_per_group
    else:
        args.task_limit = 8
        args.tasks_per_group = 8
    if args.concurrency <= 0:
        args.concurrency = max(1, args.task_limit)

    if args.num_groups <= 0:
        raise ValueError("--num-groups must be positive")
    if args.tp_size <= 0:
        raise ValueError("--tp-size must be positive")
    if args.num_gpus <= 0:
        raise ValueError("--num-gpus must be positive")

    expected = args.num_groups * args.tp_size
    if expected != args.num_gpus:
        raise ValueError(
            f"GPU layout invalid: num_groups({args.num_groups}) * tp_size({args.tp_size}) "
            f"= {expected}, but num_gpus={args.num_gpus}. "
            "Require num_groups * tp_size == num_gpus."
        )

    if args.gpus:
        gpu_list = _parse_gpu_list(args.gpus)
        if len(gpu_list) != args.num_gpus:
            raise ValueError(
                f"--gpus lists {len(gpu_list)} id(s) but --num-gpus is {args.num_gpus}"
            )
    else:
        start = int(args.gpu_offset)
        gpu_list = list(range(start, start + args.num_gpus))
        args.gpus = ",".join(str(gpu_id) for gpu_id in gpu_list)

    # Backward-compatible alias used in logs / history.
    args.population = args.num_groups


def replay_noise_weight_updates(
    noise_weight: NoiseWeightStore,
    records: list[dict[str, Any]],
    *,
    default_alpha: float,
) -> int:
    replayed = 0
    for record in records:
        seeds = [int(seed) for seed in record["seeds"]]
        weights = [float(weight) for weight in record["weights"]]
        alpha = float(record.get("alpha", default_alpha))
        noise_weight.apply_es_update(seeds=seeds, weights=weights, alpha=alpha)
        replayed += 1
    return replayed


def noise_weight_is_nonzero(noise_weight: NoiseWeightStore) -> bool:
    for tensor in noise_weight.tensors.values():
        if float(tensor.abs().max()) > 0.0:
            return True
    return False


def setup_es_instances(
    instances: Sequence[SGLangInstance],
    *,
    noise_weight: NoiseWeightStore,
    noise_weight_dir: Path,
) -> None:
    delta_dir: Path | None = None
    if noise_weight_is_nonzero(noise_weight):
        delta_dir = noise_weight_dir
        if not (delta_dir / "noise_weight_meta.json").is_file():
            noise_weight.save(delta_dir)
    for instance in instances:
        client = SGLangESClient(instance.base_url)
        client.init()
        if delta_dir is not None:
            client.load_delta(str(delta_dir.resolve()))


def build_candidate_checkpoint(
    *,
    base_dir: Path,
    noise_weight: NoiseWeightStore,
    seed: int,
    sigma: float,
    output_dir: Path,
) -> Path:
    candidate = noise_weight.copy()
    candidate.add_perturbation(seed=seed, sigma=sigma)
    delta_dir = output_dir / "delta"
    candidate.save(delta_dir)
    merge_base_delta(base_dir, delta_dir, output_dir)
    return output_dir


def generation_docker_rollout_ids(generation: int, num_groups: int) -> set[int]:
    """Rollout ids used for train candidates and eval in one generation."""
    ids = {generation * 100 + index for index in range(num_groups)}
    ids.add(generation)
    return ids


def cleanup_generation_docker_containers(generation: int, num_groups: int) -> list[str]:
    """Remove clawgym-rl sandboxes for this generation (train + eval rollout ids)."""
    rollout_ids = generation_docker_rollout_ids(generation, num_groups)
    prefixes = tuple(f"clawgym-rl-{rollout_id}-" for rollout_id in sorted(rollout_ids))
    result = subprocess.run(
        ["docker", "ps", "-a", "--format", "{{.Names}}"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return []
    removed: list[str] = []
    for name in result.stdout.splitlines():
        name = name.strip()
        if not name or not name.startswith(prefixes):
            continue
        rm = subprocess.run(
            ["docker", "rm", "-f", name],
            capture_output=True,
            text=True,
            check=False,
        )
        if rm.returncode == 0:
            removed.append(name)
    return removed


def inter_generation_cleanup(
    *,
    generation: int,
    num_groups: int,
    cooldown_sec: float,
) -> None:
    """Pause, sweep this generation's docker sandboxes, pause again before next gen."""
    if cooldown_sec > 0:
        print(
            f"[cleanup] gen={generation} cooldown {cooldown_sec:.0f}s before docker sweep",
            flush=True,
        )
        time.sleep(cooldown_sec)
    removed = cleanup_generation_docker_containers(generation, num_groups)
    if removed:
        preview = ", ".join(removed[:8])
        suffix = " ..." if len(removed) > 8 else ""
        print(
            f"[cleanup] gen={generation} removed {len(removed)} docker container(s): "
            f"{preview}{suffix}",
            flush=True,
        )
    else:
        print(f"[cleanup] gen={generation} no stale docker containers", flush=True)
    if cooldown_sec > 0:
        print(
            f"[cleanup] gen={generation} cooldown {cooldown_sec:.0f}s before next generation",
            flush=True,
        )
        time.sleep(cooldown_sec)


def save_noise_weight_checkpoint(
    *,
    noise_weight: NoiseWeightStore,
    run_root: Path,
    generation: int,
) -> Path:
    checkpoint_dir = run_root / "checkpoints" / f"checkpoint_gen{generation:04d}"
    noise_weight.save(checkpoint_dir)
    return checkpoint_dir


def run_eval_generation(
    *,
    base_dir: Path,
    noise_weight: NoiseWeightStore,
    run_root: Path,
    generation: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    eval_tasks = load_tasks(
        Path(args.eval_dataset_dir),
        suite=args.eval_suite,
        limit=args.eval_limit,
        seed=args.task_seed + 999,
    )
    gpu_ids = _parse_gpu_list(args.gpus)[: max(1, args.tp_size)]
    port = args.base_port + 900
    served_name = f"{args.served_model_name}-eval"
    log_path = run_root / "logs" / f"sglang_eval_gen{generation:04d}.log"
    if args.use_es_hook:
        model_path = base_dir
    else:
        merged_dir = run_root / f"eval_gen{generation:04d}_merged"
        noise_weight.save(merged_dir / "delta")
        merge_base_delta(base_dir, merged_dir / "delta", merged_dir)
        model_path = merged_dir
    instance = SGLangInstance(
        model_path=model_path,
        port=port,
        gpu_ids=gpu_ids,
        served_model_name=served_name,
        tp_size=args.tp_size,
        mem_fraction=args.mem_fraction,
        context_length=args.context_length,
        log_path=log_path,
        use_es_hook=args.use_es_hook,
    )
    try:
        instance.start(wait_timeout=args.sglang_wait_timeout)
        if args.use_es_hook:
            setup_es_instances(
                [instance],
                noise_weight=noise_weight,
                noise_weight_dir=run_root / "noise_weight",
            )
        summary = batch_rollout(
            tasks=eval_tasks,
            model_url=instance.base_url,
            model_id=served_name,
            run_dir=run_root / f"eval_gen{generation:04d}",
            concurrency=args.concurrency,
            max_steps=args.max_turns,
            max_tokens=args.max_tokens,
            max_total_tokens=args.max_total_tokens,
            turn_timeout=args.turn_timeout,
            task_timeout_seconds=args.task_timeout,
            sandbox=args.sandbox,
            rollout_id=generation,
            quiet=not args.verbose,
        )
        summary["rollout_dir"] = str((run_root / f"eval_gen{generation:04d}").resolve())
    finally:
        instance.stop()
    if not args.use_es_hook and args.cleanup_merged:
        merged_dir = run_root / f"eval_gen{generation:04d}_merged"
        if merged_dir.exists():
            shutil.rmtree(merged_dir, ignore_errors=True)
    return summary


def run_one_generation(
    *,
    generation: int,
    noise_weight: NoiseWeightStore,
    tasks: list,
    args: argparse.Namespace,
    run_root: Path,
) -> dict[str, Any]:
    sigma = sigma_at_step(
        sigma_start=args.sigma_start,
        sigma_end=args.sigma_end,
        step=generation,
        total_steps=args.generations,
        schedule=args.sigma_schedule,
        warmup_steps=args.sigma_warmup_steps,
    )
    rng = random.Random(args.es_seed + generation)
    seeds = [rng.randrange(1, 2**31 - 1) for _ in range(args.num_groups)]
    gpu_groups = _allocate_gpu_groups(_parse_gpu_list(args.gpus), args.num_groups, args.tp_size)

    gen_dir = run_root / f"gen_{generation:04d}"
    gen_dir.mkdir(parents=True, exist_ok=True)
    candidate_dirs: list[Path] = []
    instances: list[SGLangInstance] = []
    es_clients: list[SGLangESClient] = []

    base_model_path = Path(args.base_dir)
    for index, (seed, gpu_ids) in enumerate(zip(seeds, gpu_groups)):
        if args.use_es_hook:
            model_path = base_model_path
            merged_dir = None
        else:
            merged_dir = gen_dir / f"candidate_{index:02d}_merged"
            build_candidate_checkpoint(
                base_dir=base_model_path,
                noise_weight=noise_weight,
                seed=seed,
                sigma=sigma,
                output_dir=merged_dir,
            )
            model_path = merged_dir
            candidate_dirs.append(merged_dir)
        port = args.base_port + index
        served_name = f"{args.served_model_name}-c{index}"
        log_path = run_root / "logs" / f"sglang_gen{generation:04d}_c{index}.log"
        instances.append(
            SGLangInstance(
                model_path=model_path,
                port=port,
                gpu_ids=gpu_ids,
                served_model_name=served_name,
                tp_size=args.tp_size,
                mem_fraction=args.mem_fraction,
                context_length=args.context_length,
                log_path=log_path,
                use_es_hook=args.use_es_hook,
            )
        )

    pool = SGLangPool(instances)
    sample_records: list[dict[str, Any]] = []
    try:
        pool.start_all(wait_timeout=args.sglang_wait_timeout)
        if args.use_es_hook:
            setup_es_instances(
                instances,
                noise_weight=noise_weight,
                noise_weight_dir=run_root / "noise_weight",
            )
            es_clients = [SGLangESClient(instance.base_url) for instance in instances]

        candidate_parallelism = args.candidate_parallelism or args.num_groups
        candidate_parallelism = max(1, min(candidate_parallelism, args.num_groups))

        def _run_candidate(index: int) -> dict[str, Any]:
            instance = instances[index]
            if args.use_es_hook:
                es_clients[index].apply(seed=seeds[index], sigma=sigma)
            rollout_dir = gen_dir / f"candidate_{index:02d}_rollout"
            summary = batch_rollout(
                tasks=tasks,
                model_url=instance.base_url,
                model_id=instance.served_model_name,
                run_dir=rollout_dir,
                concurrency=args.concurrency,
                max_steps=args.max_turns,
                max_tokens=args.max_tokens,
                max_total_tokens=args.max_total_tokens,
                turn_timeout=args.turn_timeout,
                task_timeout_seconds=args.task_timeout,
                sandbox=args.sandbox,
                rollout_id=generation * 100 + index,
                quiet=not args.verbose,
            )
            if args.use_es_hook:
                es_clients[index].revert(seed=seeds[index], sigma=sigma)
            print(
                f"[gen={generation} cand={index}] seed={seeds[index]} "
                f"reward={summary['avg_reward']:.6f}",
                flush=True,
            )
            return {
                "index": index,
                "seed": seeds[index],
                "sigma": sigma,
                "reward": summary["avg_reward"],
                "rollout_dir": str(rollout_dir.resolve()),
                "summary_path": str((rollout_dir / "summary.json").resolve()),
                "summary": {
                    "n_ok": summary["n_ok"],
                    "n_error": summary["n_error"],
                    "accuracy": summary["accuracy"],
                    "n_correct": summary.get("n_correct", 0),
                },
                "merged_dir": str(candidate_dirs[index]) if index < len(candidate_dirs) else None,
                "es_hook": args.use_es_hook,
            }

        if candidate_parallelism <= 1:
            for index in range(len(instances)):
                sample_records.append(_run_candidate(index))
        else:
            with ThreadPoolExecutor(max_workers=candidate_parallelism) as executor:
                futures = {
                    executor.submit(_run_candidate, index): index
                    for index in range(len(instances))
                }
                for future in as_completed(futures):
                    sample_records.append(future.result())
            sample_records.sort(key=lambda row: row["index"])

        rewards = [float(row["reward"]) for row in sample_records]
        weights = normalize_rewards(rewards, mode=args.reward_normalization)
        if args.use_es_hook:
            for client in es_clients:
                client.update(seeds=seeds, weights=weights, alpha=args.alpha)
        noise_weight.apply_es_update(seeds=seeds, weights=weights, alpha=args.alpha)
    finally:
        pool.stop_all()
        if args.cleanup_merged and not args.use_es_hook:
            for path in candidate_dirs:
                shutil.rmtree(path, ignore_errors=True)

    record: dict[str, Any] = {
        "generation": generation,
        "timestamp": _utc_now(),
        "sigma": sigma,
        "sigma_start": args.sigma_start,
        "sigma_end": args.sigma_end,
        "sigma_schedule": args.sigma_schedule,
        "alpha": args.alpha,
        "num_groups": args.num_groups,
        "num_gpus": args.num_gpus,
        "tp_size": args.tp_size,
        "tasks_per_group": args.task_limit,
        "population": args.num_groups,
        "case_batch": [task.task_id for task in tasks],
        "seeds": seeds,
        "rewards": rewards,
        "weights": weights,
        "reward_mean": float(statistics.mean(rewards)) if rewards else 0.0,
        "reward_std": float(statistics.pstdev(rewards)) if len(rewards) > 1 else 0.0,
        "samples": sample_records,
        "generation_rollout_summary": str((gen_dir / "generation_rollout_summary.json").resolve()),
    }
    atomic_write_json(
        gen_dir / "generation_rollout_summary.json",
        {
            "generation": generation,
            "timestamp": record["timestamp"],
            "sigma": sigma,
            "candidates": sample_records,
            "reward_mean": record["reward_mean"],
        },
    )
    return record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ClawGym ES training with noise_weight delta")
    parser.add_argument("--run-id", default="clawgym_es_smoke")
    parser.add_argument("--run-root", default="", help="Run output root (default: /data/gpc/agentic_es/runs/<run-id>)")
    parser.add_argument("--base-dir", default="/dev/gpc_code/model/Qwen3_5_9B/Qwen3___5-9B")
    parser.add_argument("--noise-weight-dir", default="")
    parser.add_argument("--resume-history", default="")
    parser.add_argument("--dataset-dir", default=str(DEFAULT_TRAIN_DATASET))
    parser.add_argument("--eval-dataset-dir", default=str(DEFAULT_EVAL_DATASET))
    parser.add_argument("--suite", default="all")
    parser.add_argument("--eval-suite", default="all")
    parser.add_argument("--task-limit", type=int, default=0, help="Tasks per group per generation (alias: --tasks-per-group)")
    parser.add_argument("--tasks-per-group", type=int, default=8, help="Each SGLang group runs this many ClawGym tasks")
    parser.add_argument("--eval-limit", type=int, default=0)
    parser.add_argument("--task-seed", type=int, default=42)
    parser.add_argument(
        "--fixed-tasks",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Reuse the same task batch every generation (default: sample fresh unused tasks each gen)",
    )
    parser.add_argument("--generations", type=int, default=2)
    parser.add_argument("--num-groups", "--population", dest="num_groups", type=int, default=4,
                        help="ES population = parallel SGLang groups (default 4)")
    parser.add_argument("--num-gpus", type=int, default=8, help="Total GPUs; must equal num_groups * tp_size")
    parser.add_argument("--gpu-offset", type=int, default=0, help="First GPU index when --gpus is omitted")
    parser.add_argument("--alpha", type=float, default=5e-4)
    parser.add_argument("--sigma-start", type=float, default=1e-3)
    parser.add_argument("--sigma-end", type=float, default=1e-3)
    parser.add_argument("--sigma-schedule", default="constant", choices=["constant", "linear", "cosine"])
    parser.add_argument("--sigma-warmup-steps", type=int, default=0)
    parser.add_argument("--reward-normalization", default="zscore")
    parser.add_argument("--es-seed", type=int, default=20260627)
    parser.add_argument("--gpus", default="", help="Explicit GPU ids (comma-separated). Default: gpu_offset .. gpu_offset+num_gpus-1")
    parser.add_argument("--base-port", type=int, default=12000)
    parser.add_argument("--tp-size", type=int, default=2, help="Tensor parallel size per SGLang group")
    parser.add_argument("--mem-fraction", type=float, default=0.85)
    parser.add_argument("--context-length", type=int, default=65536)
    parser.add_argument("--concurrency", type=int, default=8, help="Parallel ClawGym tasks per group")
    parser.add_argument(
        "--candidate-parallelism",
        type=int,
        default=0,
        help="Parallel ES candidate rollouts across groups (0=all num_groups, 1=sequential)",
    )
    parser.add_argument("--max-turns", "--max-steps", dest="max_turns", type=int, default=32,
                        help="Max ReAct turns per task (agent loop steps)")
    parser.add_argument("--max-tokens", type=int, default=8192,
                        help="Max generated tokens per turn (single LLM call)")
    parser.add_argument("--max-total-tokens", type=int, default=65536,
                        help="Max total tokens per task trajectory (prompt+completion budget; 0=disable)")
    parser.add_argument("--turn-timeout", type=int, default=300, help="HTTP timeout per LLM turn (seconds)")
    parser.add_argument(
        "--task-timeout",
        type=int,
        default=900,
        help="Wall-clock cap per ClawGym task in seconds (0=disable)",
    )
    parser.add_argument("--sandbox", default="docker", choices=["docker", "local"])
    parser.add_argument("--served-model-name", default="clawgym-es")
    parser.add_argument("--sglang-wait-timeout", type=float, default=900.0)
    parser.add_argument("--save-step", type=int, default=0,
                        help="Every N generations, also save noise_weight under checkpoints/checkpoint_genXXXX/")
    parser.add_argument("--eval-step", "--eval-interval", dest="eval_step", type=int, default=0,
                        help="Every N generations, run eval on eval dataset (0=disabled)")
    parser.add_argument(
        "--inter-gen-cooldown",
        type=float,
        default=20.0,
        help="Seconds to pause before/after docker cleanup between generations (0=disable)",
    )
    parser.add_argument("--cleanup-merged", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--use-es-hook",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use in-place SGLang /es/* hooks instead of per-candidate HF merge (default: on)",
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--phase1-smoke", action="store_true", help="Single candidate, one task, one generation")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.phase1_smoke:
        args.num_groups = 1
        args.num_gpus = args.tp_size
        args.tasks_per_group = 1
        args.task_limit = 1
        args.concurrency = 1
        args.max_turns = min(args.max_turns, 4)
        args.generations = 1
        args.cleanup_merged = False
    resolve_gpu_layout(args)
    run_root = (
        Path(args.run_root).expanduser().resolve()
        if args.run_root
        else DEFAULT_RUNS_ROOT / args.run_id
    )
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "logs").mkdir(parents=True, exist_ok=True)

    base_dir = Path(args.base_dir).expanduser().resolve()
    if not base_dir.is_dir():
        raise FileNotFoundError(f"base model not found: {base_dir}")

    noise_weight_dir = (
        Path(args.noise_weight_dir).expanduser().resolve()
        if args.noise_weight_dir
        else run_root / "noise_weight"
    )

    history_path = run_root / "history.json"
    start_generation = 0
    history: list[dict[str, Any]] = [
        {
            "config": {
                "run_id": args.run_id,
                "base_dir": str(base_dir),
                "dataset_dir": args.dataset_dir,
                "task_limit": args.task_limit,
                "tasks_per_group": args.task_limit,
                "num_groups": args.num_groups,
                "num_gpus": args.num_gpus,
                "population": args.num_groups,
                "generations": args.generations,
                "alpha": args.alpha,
                "sigma_start": args.sigma_start,
                "sigma_end": args.sigma_end,
                "sigma_schedule": args.sigma_schedule,
                "gpus": args.gpus,
                "tp_size": args.tp_size,
                "concurrency": args.concurrency,
                "max_turns": args.max_turns,
                "max_tokens": args.max_tokens,
                "max_total_tokens": args.max_total_tokens,
                "turn_timeout": args.turn_timeout,
                "task_timeout": args.task_timeout,
                "context_length": args.context_length,
                "save_step": args.save_step,
                "eval_step": args.eval_step,
                "inter_gen_cooldown": args.inter_gen_cooldown,
                "phase1_smoke": args.phase1_smoke,
                "use_es_hook": args.use_es_hook,
                "fixed_tasks": args.fixed_tasks,
                "task_seed": args.task_seed,
            }
        }
    ]

    used_task_ids: set[str] = set()
    if history_path.is_file():
        used_task_ids |= used_task_ids_from_history(read_history(history_path))

    if args.resume_history:
        source_path = Path(args.resume_history).expanduser().resolve()
        source_history = read_history(source_path)
        used_task_ids |= used_task_ids_from_history(source_history)
        records = completed_update_records(source_history)
        noise_weight = NoiseWeightStore.zeros_from_base(base_dir)
        replayed = replay_noise_weight_updates(noise_weight, records, default_alpha=args.alpha)
        start_generation = replayed
        if source_path == history_path.resolve() and history_path.is_file():
            history = read_history(history_path)
        else:
            history = list(source_history)
        history.append(
            {
                "resume": {
                    "source": str(source_path),
                    "replayed_generations": replayed,
                    "timestamp": _utc_now(),
                }
            }
        )
        print(f"[resume] replayed {replayed} generations from {source_path}", flush=True)
    elif noise_weight_dir.is_dir() and (noise_weight_dir / "noise_weight_meta.json").is_file():
        noise_weight = NoiseWeightStore.load(noise_weight_dir, base_dir=base_dir)
        if history_path.is_file():
            history = read_history(history_path)
            records = completed_update_records(history)
            start_generation = len(records)
            print(
                f"[resume] loaded noise_weight from {noise_weight_dir} "
                f"start_generation={start_generation}",
                flush=True,
            )
    else:
        noise_weight = NoiseWeightStore.zeros_from_base(base_dir)

    task_pool = load_task_pool(Path(args.dataset_dir), suite=args.suite)
    tasks_per_gen = args.task_limit
    fixed_tasks = None
    if args.fixed_tasks:
        fixed_tasks = sample_tasks(task_pool, limit=tasks_per_gen, seed=args.task_seed)
    else:
        remaining = args.generations - start_generation
        unused = len(task_pool) - len(used_task_ids)
        need = remaining * tasks_per_gen
        if need > unused:
            max_gens = unused // tasks_per_gen if tasks_per_gen else 0
            raise ValueError(
                f"not enough unused tasks: pool={len(task_pool)} already_used={len(used_task_ids)} "
                f"need {need} for {remaining} generation(s) × {tasks_per_gen} tasks, "
                f"at most {max_gens} more generation(s) without reuse"
            )

    cand_par = args.candidate_parallelism or args.num_groups
    cand_par = max(1, min(cand_par, args.num_groups))
    print(
        f"[setup] run_root={run_root} task_pool={len(task_pool)} tasks_per_gen={tasks_per_gen} "
        f"used_tasks={len(used_task_ids)} fixed_tasks={args.fixed_tasks} "
        f"num_groups={args.num_groups} tp_size={args.tp_size} num_gpus={args.num_gpus} "
        f"gpus={args.gpus} concurrency={args.concurrency} candidate_parallelism={cand_par} "
        f"task_timeout={args.task_timeout}s eval_step={args.eval_step} "
        f"inter_gen_cooldown={args.inter_gen_cooldown}s "
        f"generations={args.generations} use_es_hook={args.use_es_hook}",
        flush=True,
    )

    for generation in range(start_generation, args.generations):
        if args.fixed_tasks:
            tasks = fixed_tasks
        else:
            tasks = sample_tasks(
                task_pool,
                limit=tasks_per_gen,
                seed=args.task_seed + generation,
                exclude=used_task_ids,
            )
            used_task_ids.update(task.task_id for task in tasks)
        task_ids = [task.task_id for task in tasks]
        print(f"[train] starting generation {generation} tasks={task_ids}", flush=True)
        record = run_one_generation(
            generation=generation,
            noise_weight=noise_weight,
            tasks=tasks,
            args=args,
            run_root=run_root,
        )
        history.append(record)
        noise_weight.save(noise_weight_dir)
        atomic_write_json(history_path, history)
        print(
            f"[train] generation {generation} done "
            f"reward_mean={record['reward_mean']:.6f} "
            f"noise_weight={noise_weight_dir}",
            flush=True,
        )

        if should_run_periodic(step=generation, interval=args.save_step):
            checkpoint_dir = save_noise_weight_checkpoint(
                noise_weight=noise_weight,
                run_root=run_root,
                generation=generation,
            )
            checkpoint_record = {
                "generation": generation,
                "checkpoint": {
                    "noise_weight_dir": str(checkpoint_dir),
                    "timestamp": _utc_now(),
                },
            }
            history.append(checkpoint_record)
            atomic_write_json(history_path, history)
            print(f"[checkpoint] saved noise_weight -> {checkpoint_dir}", flush=True)

        if should_run_periodic(step=generation, interval=args.eval_step):
            eval_limit_label = str(args.eval_limit) if args.eval_limit > 0 else "all"
            print(
                f"[eval] starting generation {generation} eval_limit={eval_limit_label} "
                f"dataset={args.eval_dataset_dir}",
                flush=True,
            )
            eval_summary = run_eval_generation(
                base_dir=base_dir,
                noise_weight=noise_weight,
                run_root=run_root,
                generation=generation,
                args=args,
            )
            history.append(
                {
                    "generation": generation,
                    "eval": eval_summary,
                    "timestamp": _utc_now(),
                }
            )
            atomic_write_json(history_path, history)
            print(
                f"[eval] generation {generation} done avg_reward={eval_summary['avg_reward']:.6f} "
                f"rollout_dir={eval_summary.get('rollout_dir')}",
                flush=True,
            )

        if generation + 1 < args.generations and (
            args.inter_gen_cooldown > 0 or args.sandbox == "docker"
        ):
            inter_generation_cleanup(
                generation=generation,
                num_groups=args.num_groups,
                cooldown_sec=args.inter_gen_cooldown,
            )

    final_merged = run_root / "final_merged"
    noise_weight.save(noise_weight_dir)
    merge_base_delta(base_dir, noise_weight_dir, final_merged)
    history.append(
        {
            "final": {
                "noise_weight_dir": str(noise_weight_dir),
                "merged_model_dir": str(final_merged),
                "timestamp": _utc_now(),
            }
        }
    )
    atomic_write_json(history_path, history)
    print(f"[done] final merged model -> {final_merged}", flush=True)


if __name__ == "__main__":
    main()
