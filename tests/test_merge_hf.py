#!/usr/bin/env python3
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file

from merge_hf import merge_base_delta
from noise_weight import NoiseWeightStore


class TestMergeHF(unittest.TestCase):
    def test_merge_add(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / "base"
            base.mkdir()
            delta = root / "delta"
            delta.mkdir()
            out = root / "merged"
            tensors = {"w": torch.tensor([1.0, 2.0], dtype=torch.bfloat16)}
            save_file(tensors, base / "model.safetensors")
            save_file({"w": torch.tensor([0.5, -1.0], dtype=torch.bfloat16)}, delta / "model.safetensors")
            (base / "config.json").write_text("{}", encoding="utf-8")
            merge_base_delta(base, delta, out)
            from noise_weight import get_tensor_from_dir, load_weight_map

            merged = get_tensor_from_dir(out, load_weight_map(out), "w").float()
            self.assertTrue(torch.allclose(merged, torch.tensor([1.5, 1.0])))

    def test_build_candidate_from_store(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / "base"
            base.mkdir()
            save_file({"w": torch.zeros(4, dtype=torch.bfloat16)}, base / "model.safetensors")
            (base / "config.json").write_text("{}", encoding="utf-8")
            store = NoiseWeightStore.zeros_from_base(base)
            store.add_perturbation(seed=11, sigma=0.01)
            delta_dir = root / "delta"
            store.save(delta_dir)
            merge_base_delta(base, delta_dir, root / "merged")
            self.assertTrue((root / "merged" / "model.safetensors").is_file())


if __name__ == "__main__":
    unittest.main()
