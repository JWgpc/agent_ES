#!/usr/bin/env python3
"""Tests for inter-generation docker cleanup helpers."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from train_es_clawgym import cleanup_generation_docker_containers, generation_docker_rollout_ids


class TestDockerCleanup(unittest.TestCase):
    def test_generation_rollout_ids(self) -> None:
        self.assertEqual(generation_docker_rollout_ids(12, 4), {1200, 1201, 1202, 1203, 12})

    @patch("train_es_clawgym.subprocess.run")
    def test_cleanup_matches_train_and_eval_prefixes(self, mock_run) -> None:
        mock_run.side_effect = [
            type("R", (), {"returncode": 0, "stdout": "clawgym-rl-1200-3\nclawgym-rl-12-1\nother\n"})(),
            type("R", (), {"returncode": 0, "stdout": "", "stderr": ""})(),
            type("R", (), {"returncode": 0, "stdout": "", "stderr": ""})(),
        ]
        removed = cleanup_generation_docker_containers(12, 4)
        self.assertEqual(removed, ["clawgym-rl-1200-3", "clawgym-rl-12-1"])
        self.assertEqual(mock_run.call_count, 3)


if __name__ == "__main__":
    unittest.main()
