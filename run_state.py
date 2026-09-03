"""Atomic history I/O and resume helpers for ES runs."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def atomic_write_json(path: Path, payload: Any) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def read_history(path: Path) -> list[dict[str, Any]]:
    path = path.expanduser().resolve()
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"history must be a JSON list: {path}")
    return data


def completed_update_records(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records = []
    for item in history:
        if not isinstance(item, dict):
            continue
        if "generation" in item and "seeds" in item and "weights" in item:
            records.append(item)
    return records
