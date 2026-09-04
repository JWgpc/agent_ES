#!/usr/bin/env python3
"""Tests for agent loop length limits."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

RL_DIR = Path(__file__).resolve().parents[2] / "clawGym" / "ClawGym-Agents" / "RL"
if str(RL_DIR) not in sys.path:
    sys.path.insert(0, str(RL_DIR))

from clawgym_rollout_limits import cap_turn_max_tokens, estimate_messages_tokens  # noqa: E402


class TestRolloutLimits(unittest.TestCase):
    def test_estimate_messages_tokens(self) -> None:
        messages = [
            {"role": "system", "content": "hello" * 10},
            {"role": "user", "content": "world" * 10},
        ]
        self.assertGreater(estimate_messages_tokens(messages), 0)

    def test_cap_turn_respects_total_budget(self) -> None:
        self.assertEqual(
            cap_turn_max_tokens(prompt_tokens=65000, max_tokens=8192, max_total_tokens=65536),
            535,
        )
        self.assertEqual(
            cap_turn_max_tokens(prompt_tokens=65536, max_tokens=8192, max_total_tokens=65536),
            0,
        )

    def test_cap_turn_disabled_when_total_zero(self) -> None:
        self.assertEqual(
            cap_turn_max_tokens(prompt_tokens=100000, max_tokens=4096, max_total_tokens=0),
            4096,
        )


if __name__ == "__main__":
    unittest.main()
