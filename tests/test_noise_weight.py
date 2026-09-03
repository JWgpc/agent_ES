#!/usr/bin/env python3
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from es_core import apply_seeded_noise_tensors, normalize_rewards
from noise_weight import NoiseWeightStore


class TinyBaseFixture:
    def __init__(self, root: Path) -> None:
        from safetensors.torch import save_file

        root.mkdir(parents=True, exist_ok=True)
        self.base_dir = root / "base"
        self.base_dir.mkdir()
        tensors = {
            "layer.weight": torch.randn(8, 8, dtype=torch.bfloat16),
            "layer.bias": torch.randn(8, dtype=torch.bfloat16),
        }
        save_file(tensors, self.base_dir / "model.safetensors")
        (self.base_dir / "config.json").write_text("{}", encoding="utf-8")


class TestNoiseWeight(unittest.TestCase):
    def test_zeros_and_perturb(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture = TinyBaseFixture(Path(tmp))
            store = NoiseWeightStore.zeros_from_base(fixture.base_dir)
            self.assertTrue(all(torch.all(t == 0) for t in store.tensors.values()))
            store.add_perturbation(seed=99, sigma=0.001)
            self.assertTrue(any(torch.any(t != 0) for t in store.tensors.values()))

    def test_save_load_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = TinyBaseFixture(root)
            store = NoiseWeightStore.zeros_from_base(fixture.base_dir)
            store.add_perturbation(seed=7, sigma=0.002)
            out = root / "delta"
            store.save(out)
            loaded = NoiseWeightStore.load(out)
            for key in store.tensors:
                self.assertTrue(torch.allclose(store.tensors[key].float(), loaded.tensors[key].float()))

    def test_es_update_accumulates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture = TinyBaseFixture(Path(tmp))
            store = NoiseWeightStore.zeros_from_base(fixture.base_dir)
            seeds = [1, 2, 3, 4]
            rewards = [0.9, 0.1, 0.6, 0.2]
            weights = normalize_rewards(rewards)
            store.apply_es_update(seeds=seeds, weights=weights, alpha=5e-4)
            self.assertTrue(any(torch.any(t != 0) for t in store.tensors.values()))


if __name__ == "__main__":
    unittest.main()
