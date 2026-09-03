#!/usr/bin/env python3
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from es_core import normalize_rewards
from noise_weight import NoiseWeightStore
from run_state import atomic_write_json, completed_update_records, read_history
from train_es_clawgym import replay_noise_weight_updates


class TestRunState(unittest.TestCase):
    def test_history_replay(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "history.json"
            records = [
                {
                    "generation": 0,
                    "seeds": [1, 2],
                    "weights": [1.0, -1.0],
                    "alpha": 0.001,
                }
            ]
            atomic_write_json(path, [{"config": {}}, *records])
            loaded = completed_update_records(read_history(path))
            self.assertEqual(len(loaded), 1)

            root = Path(tmp)
            base = root / "base"
            base.mkdir()
            from safetensors.torch import save_file
            import torch

            save_file({"w": torch.zeros(4, dtype=torch.bfloat16)}, base / "model.safetensors")
            (base / "config.json").write_text("{}", encoding="utf-8")
            store = NoiseWeightStore.zeros_from_base(base)
            replayed = replay_noise_weight_updates(store, loaded, default_alpha=0.001)
            self.assertEqual(replayed, 1)
            self.assertTrue(any(torch.any(t != 0) for t in store.tensors.values()))


if __name__ == "__main__":
    unittest.main()
