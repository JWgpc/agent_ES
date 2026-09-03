#!/usr/bin/env python3
"""ES training loop: base + noise_weight with ClawGym rollouts."""

from __future__ import annotations

import argparse
import json
import random
import shutil
import statistics
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from clawgym_es_rollout import DEFAULT_DATASET, batch_rollout, load_tasks
from es_core import normalize_rewards, sigma_at_step
from merge_hf import merge_base_delta
from noise_weight import NoiseWeightStore
from run_state import atomic_write_json, completed_update_records, read_history
from sglang_pool import SGLangInstance, SGLangPool


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_gpu_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _allocate_gpu_groups(all_gpus: list[int], population: int, tp_size: int) -> list[list[int]]:
    need = population * tp_size
    if len(all_gpus) < need:
        raise ValueError(
            f"Not enough GPUs: need population*tp_size={need}, have {len(all_gpus)} ({all_gpus})"
        )
    groups = []
    cursor = 0
    for _ in range(population):
        groups.append(all_gpus[cursor : cursor + tp_size])
        cursor += tp_size
    return groups


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
    merged_dir = run_root / f"eval_gen{generation:04d}_merged"
    noise_weight.save(merged_dir / "delta")
    merge_base_delta(base_dir, merged_dir / "delta", merged_dir)
    gpu_ids = _parse_gpu_list(args.gpus)[: max(1, args.tp_size)]
    port = args.base_port + 900
    served_name = f"{args.served_model_name}-eval"
    log_path = run_root / "logs" / f"sglang_eval_gen{generation:04d}.log"
    instance = SGLangInstance(
        model_path=merged_dir,
        port=port,
        gpu_ids=gpu_ids,
        served_model_name=served_name,
        tp_size=args.tp_size,
        mem_fraction=args.mem_fraction,
        context_length=args.context_length,
        log_path=log_path,
    )
    try:
        instance.start(wait_timeout=args.sglang_wait_timeout)
        summary = batch_rollout(
            tasks=eval_tasks,
            model_url=instance.base_url,
            model_id=served_name,
            run_dir=run_root / f"eval_gen{generation:04d}",
            concurrency=args.concurrency,
            max_steps=args.max_steps,
            turn_timeout=args.turn_timeout,
            sandbox=args.sandbox,
            rollout_id=generation,
            quiet=not args.verbose,
        )
    finally:
        instance.stop()
    if args.cleanup_merged and merged_dir.exists():
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
    seeds = [rng.randrange(1, 2**31 - 1) for _ in range(args.population)]
    gpu_groups = _allocate_gpu_groups(_parse_gpu_list(args.gpus), args.population, args.tp_size)

    gen_dir = run_root / f"gen_{generation:04d}"
    gen_dir.mkdir(parents=True, exist_ok=True)
    candidate_dirs: list[Path] = []
    instances: list[SGLangInstance] = []

    for index, (seed, gpu_ids) in enumerate(zip(seeds, gpu_groups)):
        merged_dir = gen_dir / f"candidate_{index:02d}_merged"
        build_candidate_checkpoint(
            base_dir=Path(args.base_dir),
            noise_weight=noise_weight,
            seed=seed,
            sigma=sigma,
            output_dir=merged_dir,
        )
        candidate_dirs.append(merged_dir)
        port = args.base_port + index
        served_name = f"{args.served_model_name}-c{index}"
        log_path = run_root / "logs" / f"sglang_gen{generation:04d}_c{index}.log"
        instances.append(
            SGLangInstance(
                model_path=merged_dir,
                port=port,
                gpu_ids=gpu_ids,
                served_model_name=served_name,
                tp_size=args.tp_size,
                mem_fraction=args.mem_fraction,
                context_length=args.context_length,
                log_path=log_path,
            )
        )

    pool = SGLangPool(instances)
    sample_records: list[dict[str, Any]] = []
    try:
        pool.start_all(wait_timeout=args.sglang_wait_timeout)
        for index, instance in enumerate(instances):
            rollout_dir = gen_dir / f"candidate_{index:02d}_rollout"
            summary = batch_rollout(
                tasks=tasks,
                model_url=instance.base_url,
                model_id=instance.served_model_name,
                run_dir=rollout_dir,
                concurrency=args.concurrency,
                max_steps=args.max_steps,
                turn_timeout=args.turn_timeout,
                sandbox=args.sandbox,
                rollout_id=generation * 100 + index,
                quiet=not args.verbose,
            )
            sample_records.append(
                {
                    "index": index,
                    "seed": seeds[index],
                    "sigma": sigma,
                    "reward": summary["avg_reward"],
                    "summary": {
                        "n_ok": summary["n_ok"],
                        "n_error": summary["n_error"],
                        "accuracy": summary["accuracy"],
                    },
                    "merged_dir": str(candidate_dirs[index]),
                }
            )
            print(
                f"[gen={generation} cand={index}] seed={seeds[index]} "
                f"reward={summary['avg_reward']:.6f}",
                flush=True,
            )
    finally:
        pool.stop_all()
        if args.cleanup_merged:
            for path in candidate_dirs:
                shutil.rmtree(path, ignore_errors=True)

    rewards = [float(row["reward"]) for row in sample_records]
    weights = normalize_rewards(rewards, mode=args.reward_normalization)
    noise_weight.apply_es_update(seeds=seeds, weights=weights, alpha=args.alpha)

    record: dict[str, Any] = {
        "generation": generation,
        "timestamp": _utc_now(),
        "sigma": sigma,
        "sigma_start": args.sigma_start,
        "sigma_end": args.sigma_end,
        "sigma_schedule": args.sigma_schedule,
        "alpha": args.alpha,
        "population": args.population,
        "case_batch": [task.task_id for task in tasks],
        "seeds": seeds,
        "rewards": rewards,
        "weights": weights,
        "reward_mean": float(statistics.mean(rewards)) if rewards else 0.0,
        "reward_std": float(statistics.pstdev(rewards)) if len(rewards) > 1 else 0.0,
        "samples": sample_records,
    }
    return record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ClawGym ES training with noise_weight delta")
    parser.add_argument("--run-id", default="clawgym_es_smoke")
    parser.add_argument("--run-root", default="")
    parser.add_argument("--base-dir", default="/dev/gpc_code/model/Qwen3_5_9B/Qwen3___5-9B")
    parser.add_argument("--noise-weight-dir", default="")
    parser.add_argument("--resume-history", default="")
    parser.add_argument("--dataset-dir", default=str(DEFAULT_DATASET))
    parser.add_argument("--eval-dataset-dir", default=str(DEFAULT_DATASET))
    parser.add_argument("--suite", default="all")
    parser.add_argument("--eval-suite", default="all")
    parser.add_argument("--task-limit", type=int, default=8)
    parser.add_argument("--eval-limit", type=int, default=0)
    parser.add_argument("--task-seed", type=int, default=42)
    parser.add_argument("--generations", type=int, default=2)
    parser.add_argument("--population", type=int, default=2)
    parser.add_argument("--alpha", type=float, default=5e-4)
    parser.add_argument("--sigma-start", type=float, default=1e-3)
    parser.add_argument("--sigma-end", type=float, default=1e-3)
    parser.add_argument("--sigma-schedule", default="constant", choices=["constant", "linear", "cosine"])
    parser.add_argument("--sigma-warmup-steps", type=int, default=0)
    parser.add_argument("--reward-normalization", default="zscore")
    parser.add_argument("--es-seed", type=int, default=20260627)
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--base-port", type=int, default=12000)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--mem-fraction", type=float, default=0.85)
    parser.add_argument("--context-length", type=int, default=65536)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=32)
    parser.add_argument("--turn-timeout", type=int, default=300)
    parser.add_argument("--sandbox", default="docker", choices=["docker", "local"])
    parser.add_argument("--served-model-name", default="clawgym-es")
    parser.add_argument("--sglang-wait-timeout", type=float, default=900.0)
    parser.add_argument("--eval-interval", type=int, default=0)
    parser.add_argument("--cleanup-merged", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--phase1-smoke", action="store_true", help="Single candidate, one task, one generation")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_root = (
        Path(args.run_root).expanduser().resolve()
        if args.run_root
        else Path("/dev/gpc_code/agentic_es/runs") / args.run_id
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
                "population": args.population,
                "generations": args.generations,
                "alpha": args.alpha,
                "sigma_start": args.sigma_start,
                "sigma_end": args.sigma_end,
                "sigma_schedule": args.sigma_schedule,
                "gpus": args.gpus,
                "tp_size": args.tp_size,
                "phase1_smoke": args.phase1_smoke,
            }
        }
    ]

    if args.resume_history:
        source_history = read_history(Path(args.resume_history))
        records = completed_update_records(source_history)
        noise_weight = NoiseWeightStore.zeros_from_base(base_dir)
        replayed = replay_noise_weight_updates(noise_weight, records, default_alpha=args.alpha)
        start_generation = replayed
        history.append(
            {
                "resume": {
                    "source": str(Path(args.resume_history).resolve()),
                    "replayed_generations": replayed,
                }
            }
        )
        print(f"[resume] replayed {replayed} generations from {args.resume_history}", flush=True)
    elif noise_weight_dir.is_dir() and (noise_weight_dir / "noise_weight_meta.json").is_file():
        noise_weight = NoiseWeightStore.load(noise_weight_dir, base_dir=base_dir)
        if history_path.is_file():
            records = completed_update_records(read_history(history_path))
            start_generation = len(records)
    else:
        noise_weight = NoiseWeightStore.zeros_from_base(base_dir)

    if args.phase1_smoke:
        args.population = 1
        args.generations = 1
        args.task_limit = 1
        args.gpus = args.gpus.split(",")[0]
        args.cleanup_merged = False

    tasks = load_tasks(
        Path(args.dataset_dir),
        suite=args.suite,
        limit=args.task_limit,
        seed=args.task_seed,
    )
    print(
        f"[setup] run_root={run_root} tasks={len(tasks)} "
        f"population={args.population} generations={args.generations}",
        flush=True,
    )

    for generation in range(start_generation, args.generations):
        print(f"[train] starting generation {generation}", flush=True)
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

        if args.eval_interval > 0 and ((generation + 1) % args.eval_interval == 0):
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
                f"[eval] generation {generation} avg_reward={eval_summary['avg_reward']:.6f}",
                flush=True,
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
