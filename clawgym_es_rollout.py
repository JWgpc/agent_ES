"""ClawGym batch rollout wrapper for ES fitness evaluation."""

from __future__ import annotations

import json
import os
import random
import statistics
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

RL_DIR = Path(__file__).resolve().parents[1] / "clawGym" / "ClawGym-Agents" / "RL"
DEFAULT_DATASET = RL_DIR / "data" / "clawgym_eval"
CORRECT_THRESHOLD = 0.5


@dataclass(frozen=True)
class TaskSpec:
    task_dir: Path
    task_id: str
    user_query: str
    input_mount_dir: str | None
    metadata: dict


def _require_dict(value: object, label: str) -> dict:
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be a dict, got {type(value).__name__}")
    return value


def _require_string(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{label} must be a string, got {type(value).__name__}")
    return value


def _discover_task_entries(source_path: Path) -> list[Path]:
    if not source_path.is_dir():
        raise ValueError(f"task source must be a dataset directory, got: {source_path}")
    entries: list[Path] = []
    for child in sorted(source_path.iterdir()):
        if not child.is_dir():
            continue
        entry_path = child / "data_entry.json"
        if entry_path.is_file():
            entries.append(entry_path)
    if not entries:
        raise ValueError(f"no task folders with data_entry.json found under: {source_path}")
    return entries


def _load_task_spec(entry_path: Path) -> TaskSpec:
    entry = _require_dict(json.loads(entry_path.read_text(encoding="utf-8")), str(entry_path))
    task_dir = entry_path.parent
    metadata = _require_dict(entry["metadata"], "metadata")
    input_mount_dir = None
    if "input_mount_dir" in entry:
        input_mount_dir = _require_string(entry["input_mount_dir"], "input_mount_dir")
        if not (task_dir / "input_files").exists():
            raise FileNotFoundError(f"input_files/ not found for {task_dir}")
    reward_sh = task_dir / "reward" / "reward.sh"
    if not reward_sh.is_file():
        raise FileNotFoundError(f"reward/reward.sh not found: {reward_sh}")
    return TaskSpec(
        task_dir=task_dir,
        task_id=_require_string(entry["task_id"], "task_id"),
        user_query=_require_string(entry["user_query"], "user_query"),
        input_mount_dir=input_mount_dir,
        metadata=metadata,
    )


def _ensure_clawgym_runtime_paths() -> None:
    slime_root = Path(os.environ.get("SLIME_ROOT", "/dev/gpc_code/slime_0427/slime"))
    for path in (slime_root, RL_DIR):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)


def load_tasks(
    dataset_dir: Path,
    *,
    suite: str = "all",
    limit: int = 0,
    seed: int = 42,
) -> list[TaskSpec]:
    entries = _discover_task_entries(dataset_dir)
    tasks = [_load_task_spec(path) for path in entries]
    if suite.strip().lower() not in {"", "all"}:
        wanted = {item.strip() for item in suite.split(",") if item.strip()}
        tasks = [task for task in tasks if task.task_id in wanted]
        missing = wanted - {task.task_id for task in tasks}
        if missing:
            raise ValueError(f"unknown task id(s): {sorted(missing)}")
    if limit > 0 and len(tasks) > limit:
        rng = random.Random(seed)
        tasks = rng.sample(tasks, limit)
        tasks.sort(key=lambda task: task.task_id)
    return tasks


def mean_reward(results: Sequence[dict]) -> float:
    rewards = [
        float(row["reward"])
        for row in results
        if row.get("status") == "ok" and row.get("reward") is not None
    ]
    if not rewards:
        return 0.0
    return float(statistics.mean(rewards))


def batch_rollout(
    *,
    tasks: Sequence[TaskSpec],
    model_url: str,
    model_id: str,
    run_dir: Path,
    concurrency: int = 4,
    max_steps: int | None = None,
    turn_timeout: int | None = None,
    sandbox: str = "docker",
    rollout_id: int = 0,
    quiet: bool = True,
) -> dict:
    _ensure_clawgym_runtime_paths()
    from run_clawgym_bench import _configure_sandbox_env, run_one_task  # noqa: WPS433

    _configure_sandbox_env(sandbox)
    run_dir.mkdir(parents=True, exist_ok=True)
    max_steps = max_steps or int(os.environ.get("OPENCLAW_MAX_REACT_STEPS", "32"))
    turn_timeout = turn_timeout or int(os.environ.get("OPENCLAW_CHAT_TURN_TIMEOUT", "300"))

    results: list[dict] = []
    errors: list[str] = []

    def _run(index: int, task: TaskSpec) -> dict:
        return run_one_task(
            task=task,
            task_index=index,
            rollout_id=rollout_id,
            run_dir=run_dir,
            model_url=model_url,
            model_id=model_id,
            max_steps=max_steps,
            turn_timeout=turn_timeout,
            sandbox=sandbox,
            keep_workspace=False,
            quiet=quiet,
        )

    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        futures = {
            pool.submit(_run, index, task): task.task_id
            for index, task in enumerate(tasks)
        }
        for future in as_completed(futures):
            task_id = futures[future]
            try:
                results.append(future.result())
            except Exception as exc:
                errors.append(f"{task_id}: {exc}\n{traceback.format_exc()}")
                results.append(
                    {
                        "task_id": task_id,
                        "status": "error",
                        "reward": 0.0,
                        "correct": False,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )

    avg = mean_reward(results)
    n_ok = sum(1 for row in results if row.get("status") == "ok")
    n_correct = sum(
        1
        for row in results
        if row.get("status") == "ok" and float(row.get("reward", 0)) > CORRECT_THRESHOLD
    )
    return {
        "model_url": model_url,
        "model_id": model_id,
        "n_tasks": len(tasks),
        "n_ok": n_ok,
        "n_error": len(tasks) - n_ok,
        "n_correct": n_correct,
        "avg_reward": avg,
        "accuracy": (n_correct / n_ok) if n_ok else 0.0,
        "results": sorted(results, key=lambda row: row.get("task_id", "")),
        "errors": errors,
    }


def rollout_single_task(
    *,
    task: TaskSpec,
    model_url: str,
    model_id: str,
    run_dir: Path,
    sandbox: str = "docker",
    max_steps: int | None = None,
    turn_timeout: int | None = None,
) -> float:
    summary = batch_rollout(
        tasks=[task],
        model_url=model_url,
        model_id=model_id,
        run_dir=run_dir,
        concurrency=1,
        max_steps=max_steps,
        turn_timeout=turn_timeout,
        sandbox=sandbox,
        quiet=False,
    )
    if summary["results"]:
        return float(summary["results"][0].get("reward", 0.0))
    return 0.0
