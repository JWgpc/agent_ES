#!/usr/bin/env python3
"""Unit tests for SGLang ES hook helpers."""

from __future__ import annotations

import unittest

import torch

from es_core import apply_seeded_noise_tensors, parse_csv_floats, parse_csv_ints, stable_tensor_id


class TestEsHookHelpers(unittest.TestCase):
    def test_csv_parsers(self) -> None:
        self.assertEqual(parse_csv_ints("1, 2,3"), [1, 2, 3])
        self.assertEqual(parse_csv_floats("0.5,-1.0"), [0.5, -1.0])

    def test_flat_offset_advances_rng(self) -> None:
        full = torch.zeros(8, dtype=torch.float32)
        left = torch.zeros(4, dtype=torch.float32)
        right = torch.zeros(4, dtype=torch.float32)
        seed = 12345
        name = "w"
        tid = stable_tensor_id(name)
        apply_seeded_noise_tensors([(name, tid, full)], seed=seed, sigma=1.0)
        apply_seeded_noise_tensors(
            [(name, tid, left)],
            seed=seed,
            sigma=1.0,
            flat_offsets={name: 0},
        )
        apply_seeded_noise_tensors(
            [(name, tid, right)],
            seed=seed,
            sigma=1.0,
            flat_offsets={name: 4},
        )
        self.assertTrue(torch.allclose(full[:4], left))
        self.assertTrue(torch.allclose(full[4:], right))


if __name__ == "__main__":
    unittest.main()
