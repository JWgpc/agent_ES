#!/usr/bin/env python3
"""Tests for periodic step helper and rollout persistence."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from clawgym_es_rollout import persist_rollout_summary
from run_state import should_run_periodic


class TestPeriodicSteps(unittest.TestCase):
    def test_should_run_periodic(self) -> None:
        self.assertFalse(should_run_periodic(step=0, interval=0))
        self.assertFalse(should_run_periodic(step=0, interval=5))
        self.assertTrue(should_run_periodic(step=4, interval=5))
        self.assertTrue(should_run_periodic(step=0, interval=1))


class TestRolloutPersistence(unittest.TestCase):
    def test_persist_rollout_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "rollout"
            summary = {
                "model_url": "http://127.0.0.1:8000",
                "model_id": "test",
                "avg_reward": 0.5,
                "accuracy": 0.5,
                "results": [{"task_id": "t1", "status": "ok", "reward": 0.5}],
            }
            path = persist_rollout_summary(
                run_dir=run_dir,
                summary=summary,
                rollout_id=7,
                sandbox="local",
                task_ids=["t1"],
            )
            self.assertTrue(path.is_file())
            self.assertTrue((run_dir / "rollout_meta.json").is_file())
            self.assertTrue((run_dir / "tasks").exists() is False)


if __name__ == "__main__":
    unittest.main()
