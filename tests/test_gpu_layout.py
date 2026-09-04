#!/usr/bin/env python3
"""Unit tests for GPU layout resolution."""

from __future__ import annotations

import argparse
import unittest

from train_es_clawgym import resolve_gpu_layout


def _make_args(**overrides):
    defaults = dict(
        task_limit=0,
        tasks_per_group=8,
        concurrency=8,
        num_groups=4,
        num_gpus=8,
        tp_size=2,
        gpu_offset=0,
        gpus="",
        population=4,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


class TestGpuLayout(unittest.TestCase):
    def test_default_8gpu_layout(self) -> None:
        args = _make_args()
        resolve_gpu_layout(args)
        self.assertEqual(args.gpus, "0,1,2,3,4,5,6,7")
        self.assertEqual(args.task_limit, 8)
        self.assertEqual(args.population, 4)

    def test_explicit_gpus(self) -> None:
        args = _make_args(gpus="2,3,4,5", num_groups=2, num_gpus=4, tp_size=2)
        resolve_gpu_layout(args)
        self.assertEqual(args.gpus, "2,3,4,5")

    def test_rejects_mismatch(self) -> None:
        args = _make_args(num_groups=4, num_gpus=6, tp_size=2)
        with self.assertRaises(ValueError):
            resolve_gpu_layout(args)

    def test_task_limit_overrides_tasks_per_group(self) -> None:
        args = _make_args(task_limit=3, tasks_per_group=8)
        resolve_gpu_layout(args)
        self.assertEqual(args.task_limit, 3)


if __name__ == "__main__":
    unittest.main()
