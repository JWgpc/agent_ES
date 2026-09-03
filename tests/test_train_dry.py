#!/usr/bin/env python3
"""Dry-run ES generation with mocked rollouts (no SGLang/Docker)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from safetensors.torch import save_file

from noise_weight import NoiseWeightStore
from train_es_clawgym import run_one_generation


class TestTrainDryRun(unittest.TestCase):
    def test_one_generation_mocked(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / "base"
            base.mkdir()
            save_file({"w": torch.zeros(16, dtype=torch.bfloat16)}, base / "model.safetensors")
            (base / "config.json").write_text("{}", encoding="utf-8")

            store = NoiseWeightStore.zeros_from_base(base)
            tasks = [
                type("T", (), {"task_id": "task_0001"})(),
                type("T", (), {"task_id": "task_0002"})(),
            ]

            class Args:
                base_dir = str(base)
                sigma_start = 1e-3
                sigma_end = 1e-3
                generations = 1
                sigma_schedule = "constant"
                sigma_warmup_steps = 0
                population = 2
                alpha = 5e-4
                reward_normalization = "zscore"
                gpus = "0,1"
                base_port = 13000
                tp_size = 1
                mem_fraction = 0.5
                context_length = 4096
                concurrency = 1
                max_steps = 4
                turn_timeout = 60
                sandbox = "local"
                served_model_name = "dry"
                sglang_wait_timeout = 10.0
                cleanup_merged = True
                verbose = False
                es_seed = 42

            rewards = iter([0.8, 0.2])

            def fake_batch_rollout(**kwargs):
                return {
                    "avg_reward": next(rewards),
                    "n_ok": 2,
                    "n_error": 0,
                    "accuracy": 0.5,
                    "results": [],
                }

            class FakeInstance:
                def __init__(self, *a, **k):
                    self.served_model_name = k.get("served_model_name", "dry")
                    self.base_url = f"http://127.0.0.1:{k.get('port', 0)}"

                def start(self, **k):
                    return None

                def stop(self, **k):
                    return None

            class FakePool:
                def __init__(self, instances):
                    self.instances = instances

                def start_all(self, **k):
                    return None

                def stop_all(self):
                    return None

            with patch("train_es_clawgym.build_candidate_checkpoint", side_effect=lambda **kw: kw["output_dir"].mkdir(parents=True)), patch(
                "train_es_clawgym.batch_rollout", side_effect=fake_batch_rollout
            ), patch("train_es_clawgym.SGLangPool", FakePool), patch(
                "train_es_clawgym.SGLangInstance", FakeInstance
            ):
                record = run_one_generation(
                    generation=0,
                    noise_weight=store,
                    tasks=tasks,
                    args=Args(),
                    run_root=root / "run",
                )

            self.assertEqual(len(record["seeds"]), 2)
            self.assertTrue(any(v != 0 for t in store.tensors.values() for v in t.flatten()))


if __name__ == "__main__":
    unittest.main()
