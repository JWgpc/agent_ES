"""Delta weight store (noise_weight) for base + delta ES training."""

from __future__ import annotations

import json
import shutil
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from es_core import apply_seeded_noise_tensors, es_update_tensors, stable_tensor_id

CONFIG_COPY_NAMES = [
    "config.json",
    "configuration.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "chat_template.jinja",
    "preprocessor_config.json",
    "processor_config.json",
    "video_preprocessor_config.json",
    "merges.txt",
    "special_tokens_map.json",
    "added_tokens.json",
]


def load_weight_map(model_dir: Path) -> dict[str, str]:
    index = model_dir / "model.safetensors.index.json"
    if index.exists():
        with index.open(encoding="utf-8") as handle:
            return json.load(handle)["weight_map"]
    single = model_dir / "model.safetensors"
    with safe_open(single, framework="pt", device="cpu") as handle:
        return {key: "model.safetensors" for key in handle.keys()}


def get_tensor_from_dir(model_dir: Path, weight_map: dict[str, str], key: str) -> torch.Tensor:
    with safe_open(model_dir / weight_map[key], framework="pt", device="cpu") as handle:
        return handle.get_tensor(key)


@dataclass
class NoiseWeightStore:
    """In-memory delta weights aligned with a frozen base HF checkpoint."""

    base_dir: Path
    weight_map: dict[str, str]
    tensors: dict[str, torch.Tensor] = field(default_factory=dict)
    dtype: torch.dtype = torch.bfloat16

    @classmethod
    def zeros_from_base(cls, base_dir: Path, *, dtype: torch.dtype = torch.bfloat16) -> NoiseWeightStore:
        base_dir = base_dir.resolve()
        weight_map = load_weight_map(base_dir)
        tensors: dict[str, torch.Tensor] = {}
        for key in sorted(weight_map.keys()):
            ref = get_tensor_from_dir(base_dir, weight_map, key)
            if not torch.is_floating_point(ref):
                continue
            tensors[key] = torch.zeros(ref.shape, dtype=dtype)
        return cls(base_dir=base_dir, weight_map=weight_map, tensors=tensors, dtype=dtype)

    @classmethod
    def load(cls, delta_dir: Path, *, base_dir: Path | None = None) -> NoiseWeightStore:
        delta_dir = delta_dir.resolve()
        meta_path = delta_dir / "noise_weight_meta.json"
        if meta_path.is_file():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            resolved_base = Path(meta["base_dir"]).resolve()
        elif base_dir is not None:
            resolved_base = base_dir.resolve()
        else:
            raise ValueError("base_dir required when loading delta without noise_weight_meta.json")

        weight_map = load_weight_map(delta_dir)
        tensors: dict[str, torch.Tensor] = {}
        for key in sorted(weight_map.keys()):
            tensors[key] = get_tensor_from_dir(delta_dir, weight_map, key)
        dtype = next(iter(tensors.values())).dtype if tensors else torch.bfloat16
        return cls(base_dir=resolved_base, weight_map=weight_map, tensors=tensors, dtype=dtype)

    def tensor_infos(self) -> list[tuple[str, int, torch.Tensor]]:
        return [(name, stable_tensor_id(name), tensor) for name, tensor in self.tensors.items()]

    def copy(self) -> NoiseWeightStore:
        cloned = NoiseWeightStore(
            base_dir=self.base_dir,
            weight_map=dict(self.weight_map),
            dtype=self.dtype,
        )
        cloned.tensors = {key: tensor.clone() for key, tensor in self.tensors.items()}
        return cloned

    def add_perturbation(self, *, seed: int, sigma: float) -> None:
        apply_seeded_noise_tensors(self.tensor_infos(), seed=int(seed), sigma=float(sigma))

    def revert_perturbation(self, *, seed: int, sigma: float) -> None:
        apply_seeded_noise_tensors(self.tensor_infos(), seed=int(seed), sigma=-float(sigma))

    def apply_es_update(
        self,
        *,
        seeds: Sequence[int],
        weights: Sequence[float],
        alpha: float,
    ) -> None:
        es_update_tensors(
            self.tensor_infos(),
            seeds=[int(seed) for seed in seeds],
            weights=[float(weight) for weight in weights],
            alpha=float(alpha),
        )

    def save(self, output_dir: Path) -> None:
        output_dir = output_dir.resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        shard_tensors: dict[str, dict[str, torch.Tensor]] = defaultdict(dict)
        out_map: dict[str, str] = {}
        for key in sorted(self.tensors.keys()):
            shard_name = self.weight_map.get(key, "model.safetensors")
            shard_tensors[shard_name][key] = self.tensors[key].contiguous()
            out_map[key] = shard_name

        total_size = 0
        for shard_name, tensors in sorted(shard_tensors.items()):
            out_path = output_dir / shard_name
            save_file(tensors, out_path)
            total_size += out_path.stat().st_size

        if len(shard_tensors) == 1 and "model.safetensors.index.json" not in [
            p.name for p in output_dir.iterdir()
        ]:
            pass
        else:
            index = {"metadata": {"total_size": total_size}, "weight_map": out_map}
            with (output_dir / "model.safetensors.index.json").open("w", encoding="utf-8") as handle:
                json.dump(index, handle, indent=2)

        meta = {
            "base_dir": str(self.base_dir),
            "dtype": str(self.dtype).replace("torch.", ""),
            "n_tensors": len(self.tensors),
        }
        with (output_dir / "noise_weight_meta.json").open("w", encoding="utf-8") as handle:
            json.dump(meta, handle, indent=2)
