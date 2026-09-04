"""Seed-replay evolution strategy core (Agentic-ESOpt compatible)."""

from __future__ import annotations

import hashlib
import math
from typing import Iterable, Sequence

import torch

ParameterInfo = tuple[str, int, torch.nn.Parameter]
TensorInfo = tuple[str, int, torch.Tensor]

CHUNK_SIZE = 8_388_608


def stable_tensor_id(name: str) -> int:
    digest = hashlib.blake2b(str(name).encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little", signed=False)


def mix_seed(base_seed: int, tensor_id: int) -> int:
    return (int(base_seed) ^ int(tensor_id)) & 0xFFFFFFFFFFFFFFFF


def parse_csv_floats(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_csv_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def iter_flat_chunks(numel: int, chunk_size: int = CHUNK_SIZE):
    for start in range(0, int(numel), int(chunk_size)):
        yield start, min(start + int(chunk_size), int(numel))


def _advance_generator(generator: torch.Generator, *, steps: int) -> None:
    remaining = int(steps)
    while remaining > 0:
        step = min(remaining, CHUNK_SIZE)
        torch.randn((step,), generator=generator, dtype=torch.float32)
        remaining -= step


@torch.no_grad()
def apply_seeded_noise_tensors(
    tensors: Sequence[TensorInfo],
    *,
    seed: int,
    sigma: float,
    flat_offsets: dict[str, int] | None = None,
) -> None:
    if not math.isfinite(float(sigma)):
        raise ValueError("sigma must be finite")
    for name, tensor_id, tensor in tensors:
        flat = tensor.view(-1)
        flat_offset = int((flat_offsets or {}).get(name, 0))
        generator = torch.Generator(device=tensor.device)
        generator.manual_seed(mix_seed(seed, tensor_id))
        if flat_offset:
            _advance_generator(generator, steps=flat_offset)
        for start, end in iter_flat_chunks(flat.numel()):
            noise = torch.randn(
                (end - start,),
                generator=generator,
                dtype=torch.float32,
                device=tensor.device,
            )
            flat[start:end].add_(noise.to(dtype=tensor.dtype), alpha=float(sigma))


@torch.no_grad()
def apply_seeded_noise_params(
    params: Iterable[torch.nn.Parameter],
    *,
    seed: int,
    sigma: float,
    name_prefix: str = "",
) -> None:
    infos: list[TensorInfo] = []
    for name, param in _named_params(params, name_prefix):
        if not torch.is_floating_point(param):
            continue
        infos.append((name, stable_tensor_id(name), param))
    apply_seeded_noise_tensors(infos, seed=seed, sigma=sigma)


def _named_params(
    params: Iterable[torch.nn.Parameter],
    name_prefix: str,
) -> list[tuple[str, torch.nn.Parameter]]:
    if isinstance(params, dict):
        return [(str(k), v) for k, v in params.items()]
    out: list[tuple[str, torch.nn.Parameter]] = []
    for idx, param in enumerate(params):
        out.append((f"{name_prefix}{idx}", param))
    return out


@torch.no_grad()
def es_update_tensors(
    tensors: Sequence[TensorInfo],
    *,
    seeds: Sequence[int],
    weights: Sequence[float],
    alpha: float,
    flat_offsets: dict[str, int] | None = None,
) -> None:
    if not seeds:
        raise ValueError("ES update requires at least one seed")
    if len(seeds) != len(weights):
        raise ValueError("seeds and weights must have the same length")
    if not math.isfinite(float(alpha)) or float(alpha) < 0.0:
        raise ValueError("alpha must be finite and non-negative")

    scale = float(alpha) / float(len(seeds))
    for name, tensor_id, tensor in tensors:
        flat = tensor.view(-1)
        flat_offset = int((flat_offsets or {}).get(name, 0))
        generators: list[torch.Generator] = []
        coeffs: list[float] = []
        for seed, weight in zip(seeds, weights):
            generator = torch.Generator(device=tensor.device)
            generator.manual_seed(mix_seed(int(seed), tensor_id))
            if flat_offset:
                _advance_generator(generator, steps=flat_offset)
            generators.append(generator)
            coeffs.append(scale * float(weight))

        for start, end in iter_flat_chunks(flat.numel()):
            total_delta = torch.zeros((end - start,), dtype=torch.float32, device=tensor.device)
            for generator, coeff in zip(generators, coeffs):
                noise = torch.randn(
                    (end - start,),
                    generator=generator,
                    dtype=torch.float32,
                    device=tensor.device,
                )
                total_delta.add_(noise, alpha=coeff)
            flat[start:end].add_(total_delta.to(dtype=tensor.dtype))


def normalize_rewards(
    rewards: Sequence[float],
    mode: str = "zscore",
    *,
    ddof: int = 0,
    eps: float = 1e-8,
) -> list[float]:
    tensor = torch.tensor(list(rewards), dtype=torch.float32)
    normalized_mode = str(mode or "none").strip().lower()
    if normalized_mode in {"none", "identity", "off"}:
        return tensor.tolist()
    if normalized_mode == "zscore":
        if tensor.numel() <= int(ddof):
            return torch.zeros_like(tensor).tolist()
        std = torch.std(tensor, unbiased=bool(ddof))
        return ((tensor - torch.mean(tensor)) / (std + float(eps))).tolist()
    if normalized_mode == "centered_rank":
        order = torch.argsort(torch.argsort(tensor))
        if tensor.numel() == 1:
            return [0.0]
        return (order.float() / (tensor.numel() - 1) - 0.5).tolist()
    raise ValueError(f"Unsupported reward_normalization: {mode}")


def sigma_at_step(
    *,
    sigma_start: float,
    sigma_end: float,
    step: int,
    total_steps: int,
    schedule: str = "cosine",
    warmup_steps: int = 0,
) -> float:
    if step < warmup_steps:
        return float(sigma_start)
    if total_steps <= 1:
        return float(sigma_start)
    decay_steps = max(1, total_steps - 1 - warmup_steps)
    t = min(max((step - warmup_steps) / decay_steps, 0.0), 1.0)
    schedule_name = str(schedule or "constant").strip().lower()
    if schedule_name == "constant":
        return float(sigma_start)
    if schedule_name == "linear":
        return float(sigma_start + (sigma_end - sigma_start) * t)
    if schedule_name == "cosine":
        return float(sigma_end + (sigma_start - sigma_end) * (1.0 + math.cos(math.pi * t)) / 2.0)
    raise ValueError(f"Unsupported sigma schedule: {schedule}")
