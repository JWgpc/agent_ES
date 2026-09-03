#!/usr/bin/env python3
from __future__ import annotations

import copy
import unittest

import torch

from es_core import (
    apply_seeded_noise_tensors,
    es_update_tensors,
    mix_seed,
    normalize_rewards,
    sigma_at_step,
    stable_tensor_id,
)


class TestESCore(unittest.TestCase):
    def _make_tensors(self) -> list[tuple[str, int, torch.Tensor]]:
        t1 = torch.zeros(4, dtype=torch.float32)
        t2 = torch.ones(3, dtype=torch.float32)
        return [
            ("layer.a", stable_tensor_id("layer.a"), t1),
            ("layer.b", stable_tensor_id("layer.b"), t2),
        ]

    def test_seed_replay_deterministic(self) -> None:
        a = self._make_tensors()
        b = copy.deepcopy([(n, i, t.clone()) for n, i, t in a])
        apply_seeded_noise_tensors(a, seed=123, sigma=0.01)
        apply_seeded_noise_tensors(b, seed=123, sigma=0.01)
        for (_, _, ta), (_, _, tb) in zip(a, b):
            self.assertTrue(torch.allclose(ta, tb))

    def test_apply_revert_restores(self) -> None:
        original = self._make_tensors()
        backup = [(n, i, t.clone()) for n, i, t in original]
        apply_seeded_noise_tensors(original, seed=456, sigma=0.02)
        apply_seeded_noise_tensors(original, seed=456, sigma=-0.02)
        for (_, _, ta), (_, _, tb) in zip(original, backup):
            self.assertTrue(torch.allclose(ta, tb, atol=1e-6))

    def test_zscore_example(self) -> None:
        rewards = [0.9, 0.1, 0.7, 0.2]
        weights = normalize_rewards(rewards, mode="zscore")
        self.assertAlmostEqual(sum(weights) / len(weights), 0.0, places=5)
        self.assertGreater(weights[0], weights[1])
        self.assertGreater(weights[2], weights[3])

    def test_es_update_direction(self) -> None:
        tensors = self._make_tensors()
        before = [(n, i, t.clone()) for n, i, t in tensors]
        seeds = [111, 222]
        weights = [1.0, -1.0]
        es_update_tensors(tensors, seeds=seeds, weights=weights, alpha=0.1)
        changed = any(not torch.allclose(a, b) for (_, _, a), (_, _, b) in zip(tensors, before))
        self.assertTrue(changed)

    def test_sigma_cosine_endpoints(self) -> None:
        start = sigma_at_step(
            sigma_start=1e-3,
            sigma_end=5e-4,
            step=0,
            total_steps=25,
            schedule="cosine",
        )
        end = sigma_at_step(
            sigma_start=1e-3,
            sigma_end=5e-4,
            step=24,
            total_steps=25,
            schedule="cosine",
        )
        self.assertAlmostEqual(start, 1e-3, places=9)
        self.assertAlmostEqual(end, 5e-4, places=9)

    def test_mix_seed_differs_by_name(self) -> None:
        id_a = stable_tensor_id("a")
        id_b = stable_tensor_id("b")
        self.assertNotEqual(mix_seed(42, id_a), mix_seed(42, id_b))


if __name__ == "__main__":
    unittest.main()
