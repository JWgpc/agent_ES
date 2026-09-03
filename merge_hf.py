"""Merge frozen base HF weights with a delta (noise_weight): out = base + delta."""

from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path

import torch
from safetensors.torch import save_file

from noise_weight import CONFIG_COPY_NAMES, get_tensor_from_dir, load_weight_map


def merge_base_delta(
    base_dir: Path,
    delta_dir: Path,
    output_dir: Path,
    *,
    config_from: Path | None = None,
    base_scale: float = 1.0,
    delta_scale: float = 1.0,
) -> Path:
    base_dir = base_dir.resolve()
    delta_dir = delta_dir.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    base_map = load_weight_map(base_dir)
    delta_map = load_weight_map(delta_dir)
    keys = sorted(base_map.keys())
    if set(keys) != set(delta_map.keys()):
        missing = set(base_map.keys()) ^ set(delta_map.keys())
        raise RuntimeError(f"weight key mismatch between base and delta: {sorted(missing)[:8]}")

    shard_tensors: dict[str, dict[str, torch.Tensor]] = defaultdict(dict)
    merged_map: dict[str, str] = {}

    for index, key in enumerate(keys, 1):
        base_tensor = get_tensor_from_dir(base_dir, base_map, key)
        delta_tensor = get_tensor_from_dir(delta_dir, delta_map, key)
        if torch.is_floating_point(base_tensor):
            merged = (base_scale * base_tensor.float() + delta_scale * delta_tensor.float()).to(base_tensor.dtype)
        else:
            merged = base_tensor
        shard_name = base_map[key]
        shard_tensors[shard_name][key] = merged.contiguous()
        merged_map[key] = shard_name
        if index % 100 == 0 or index == len(keys):
            print(f"merged {index}/{len(keys)} tensors", flush=True)

    total_size = 0
    for shard_name, tensors in sorted(shard_tensors.items()):
        out_path = output_dir / shard_name
        save_file(tensors, out_path)
        total_size += out_path.stat().st_size
        print(f"saved {out_path.name} ({out_path.stat().st_size / 1e9:.2f} GB)", flush=True)

    if len(shard_tensors) > 1 or (base_dir / "model.safetensors.index.json").exists():
        index = {"metadata": {"total_size": total_size}, "weight_map": merged_map}
        with (output_dir / "model.safetensors.index.json").open("w", encoding="utf-8") as handle:
            json.dump(index, handle, indent=2)

    config_src = (config_from or base_dir).resolve()
    for name in CONFIG_COPY_NAMES:
        src = config_src / name
        if src.is_file():
            shutil.copy2(src, output_dir / name)

    meta = {
        "merge_method": "add",
        "base_dir": str(base_dir),
        "delta_dir": str(delta_dir),
        "base_scale": base_scale,
        "delta_scale": delta_scale,
    }
    with (output_dir / "merge_config.json").open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)

    print(f"done -> {output_dir}", flush=True)
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge base HF checkpoint with delta: base + delta")
    parser.add_argument("base_dir", type=Path)
    parser.add_argument("delta_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--config-from", type=Path, default=None)
    parser.add_argument("--base-scale", type=float, default=1.0)
    parser.add_argument("--delta-scale", type=float, default=1.0)
    args = parser.parse_args()
    merge_base_delta(
        args.base_dir,
        args.delta_dir,
        args.output_dir,
        config_from=args.config_from,
        base_scale=args.base_scale,
        delta_scale=args.delta_scale,
    )


if __name__ == "__main__":
    main()
