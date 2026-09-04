"""Scheduler-side seeded ES operations on loaded GPU weights."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import torch
import torch.distributed as dist

from es_core import (
    apply_seeded_noise_tensors,
    es_update_tensors,
    parse_csv_floats,
    parse_csv_ints,
    stable_tensor_id,
)
from noise_weight import get_tensor_from_dir, load_weight_map
from sglang_es_io import (
    EsApplyReqInput,
    EsInitReqInput,
    EsLoadDeltaReqInput,
    EsOpReqOutput,
    EsRevertReqInput,
    EsStatusReqInput,
    EsUpdateReqInput,
)

logger = logging.getLogger(__name__)


@dataclass
class EsRuntime:
    tensor_infos: list[tuple[str, int, torch.Tensor]] = field(default_factory=list)
    flat_offsets: dict[str, int] = field(default_factory=dict)
    initialized: bool = False
    active_perturbations: list[tuple[int, float]] = field(default_factory=list)


def _collect_tensor_infos(model: torch.nn.Module) -> list[tuple[str, int, torch.Tensor]]:
    infos: list[tuple[str, int, torch.Tensor]] = []
    for name, param in model.named_parameters():
        if not torch.is_floating_point(param):
            continue
        infos.append((name, stable_tensor_id(name), param))
    return infos


def _compute_flat_offsets(
    tensor_infos: Sequence[tuple[str, int, torch.Tensor]],
    *,
    tp_rank: int,
    tp_size: int,
    device_group,
) -> dict[str, int]:
    if tp_size <= 1 or device_group is None:
        return {name: 0 for name, _, _ in tensor_infos}

    offsets: dict[str, int] = {}
    for name, _, tensor in tensor_infos:
        local = torch.tensor([tensor.numel()], device=tensor.device, dtype=torch.int64)
        gathered = [torch.zeros_like(local) for _ in range(tp_size)]
        dist.all_gather(gathered, local, group=device_group)
        offsets[name] = sum(int(item.item()) for item in gathered[:tp_rank])
    return offsets


def _runtime_for(weight_updater) -> EsRuntime:
    scheduler = weight_updater.scheduler
    if scheduler is None:
        raise RuntimeError("ES runtime requires scheduler reference on weight_updater")
    runtime = getattr(scheduler, "_agentic_es_runtime", None)
    if runtime is None:
        runtime = EsRuntime()
        scheduler._agentic_es_runtime = runtime
    return runtime


def _model_runner(weight_updater):
    return weight_updater.tp_worker.model_runner


def _model(weight_updater):
    runner = _model_runner(weight_updater)
    inner = getattr(runner, "weight_updater", None)
    if inner is not None and hasattr(inner, "get_model"):
        return inner.get_model()
    model = getattr(runner, "model", None)
    if model is None:
        raise RuntimeError("could not resolve SGLang model from model_runner")
    return model


def _barrier(weight_updater) -> None:
    scheduler = weight_updater.scheduler
    if scheduler is not None and getattr(scheduler, "tp_group", None) is not None:
        dist.barrier(group=scheduler.tp_group.device_group)


def es_init(weight_updater, recv_req: EsInitReqInput) -> EsOpReqOutput:
    runtime = _runtime_for(weight_updater)
    runner = _model_runner(weight_updater)
    model = _model(weight_updater)
    runtime.tensor_infos = _collect_tensor_infos(model)
    tp_rank = int(getattr(runner, "tp_rank", 0))
    tp_size = int(getattr(runner, "tp_size", 1))
    device_group = None
    scheduler = weight_updater.scheduler
    if scheduler is not None and getattr(scheduler, "tp_group", None) is not None:
        device_group = scheduler.tp_group.device_group
    runtime.flat_offsets = _compute_flat_offsets(
        runtime.tensor_infos,
        tp_rank=tp_rank,
        tp_size=tp_size,
        device_group=device_group,
    )
    runtime.initialized = True
    runtime.active_perturbations.clear()
    payload = {
        "n_tensors": len(runtime.tensor_infos),
        "tp_rank": tp_rank,
        "tp_size": tp_size,
    }
    return EsOpReqOutput(success=True, message=json.dumps(payload))


def es_apply(weight_updater, recv_req: EsApplyReqInput) -> EsOpReqOutput:
    runtime = _runtime_for(weight_updater)
    if not runtime.initialized:
        return EsOpReqOutput(success=False, message="ES runtime not initialized")
    apply_seeded_noise_tensors(
        runtime.tensor_infos,
        seed=int(recv_req.seed),
        sigma=float(recv_req.sigma),
        flat_offsets=runtime.flat_offsets,
    )
    runtime.active_perturbations.append((int(recv_req.seed), float(recv_req.sigma)))
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    _barrier(weight_updater)
    return EsOpReqOutput(success=True, message="applied")


def es_revert(weight_updater, recv_req: EsRevertReqInput) -> EsOpReqOutput:
    runtime = _runtime_for(weight_updater)
    if not runtime.initialized:
        return EsOpReqOutput(success=False, message="ES runtime not initialized")
    apply_seeded_noise_tensors(
        runtime.tensor_infos,
        seed=int(recv_req.seed),
        sigma=-float(recv_req.sigma),
        flat_offsets=runtime.flat_offsets,
    )
    key = (int(recv_req.seed), float(recv_req.sigma))
    if runtime.active_perturbations and runtime.active_perturbations[-1] == key:
        runtime.active_perturbations.pop()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    _barrier(weight_updater)
    return EsOpReqOutput(success=True, message="reverted")


def es_update(weight_updater, recv_req: EsUpdateReqInput) -> EsOpReqOutput:
    runtime = _runtime_for(weight_updater)
    if not runtime.initialized:
        return EsOpReqOutput(success=False, message="ES runtime not initialized")
    seeds = parse_csv_ints(recv_req.seeds_csv)
    weights = parse_csv_floats(recv_req.weights_csv)
    if len(seeds) != len(weights):
        return EsOpReqOutput(success=False, message="seeds/weights length mismatch")
    es_update_tensors(
        runtime.tensor_infos,
        seeds=seeds,
        weights=weights,
        alpha=float(recv_req.alpha),
        flat_offsets=runtime.flat_offsets,
    )
    runtime.active_perturbations.clear()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    _barrier(weight_updater)
    return EsOpReqOutput(success=True, message="updated")


def es_load_delta(weight_updater, recv_req: EsLoadDeltaReqInput) -> EsOpReqOutput:
    runtime = _runtime_for(weight_updater)
    if not runtime.initialized:
        return EsOpReqOutput(success=False, message="ES runtime not initialized")
    delta_dir = Path(recv_req.delta_dir).expanduser().resolve()
    if not delta_dir.is_dir():
        return EsOpReqOutput(success=False, message=f"delta dir not found: {delta_dir}")
    weight_map = load_weight_map(delta_dir)
    params = dict(_model(weight_updater).named_parameters())
    loaded = 0
    for key in sorted(weight_map.keys()):
        if key not in params:
            continue
        delta = get_tensor_from_dir(delta_dir, weight_map, key)
        if not torch.is_floating_point(delta):
            continue
        target = params[key]
        flat = target.view(-1)
        flat_offset = int(runtime.flat_offsets.get(key, 0))
        delta_flat = delta.reshape(-1)
        end = flat_offset + flat.numel()
        if end > delta_flat.numel():
            return EsOpReqOutput(
                success=False,
                message=(
                    f"delta shard out of range for {key}: "
                    f"offset={flat_offset} local={flat.numel()} full={delta_flat.numel()}"
                ),
            )
        shard = delta_flat[flat_offset:end]
        if shard.numel() != flat.numel():
            return EsOpReqOutput(
                success=False,
                message=(
                    f"delta shard size mismatch for {key}: "
                    f"expected {flat.numel()}, got {shard.numel()}"
                ),
            )
        flat.add_(shard.to(device=target.device, dtype=target.dtype))
        loaded += 1
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    _barrier(weight_updater)
    return EsOpReqOutput(success=True, message=json.dumps({"loaded_tensors": loaded}))


def es_status(weight_updater, recv_req: EsStatusReqInput) -> EsOpReqOutput:
    runtime = _runtime_for(weight_updater)
    runner = _model_runner(weight_updater)
    payload = {
        "initialized": runtime.initialized,
        "n_tensors": len(runtime.tensor_infos),
        "active_perturbations": runtime.active_perturbations,
        "tp_rank": int(getattr(runner, "tp_rank", 0)),
        "tp_size": int(getattr(runner, "tp_size", 1)),
    }
    return EsOpReqOutput(success=True, message=json.dumps(payload))


def es_request_handlers(weight_updater) -> list[tuple[type, Callable[[Any], EsOpReqOutput]]]:
    return [
        (EsInitReqInput, lambda req: es_init(weight_updater, req)),
        (EsApplyReqInput, lambda req: es_apply(weight_updater, req)),
        (EsRevertReqInput, lambda req: es_revert(weight_updater, req)),
        (EsUpdateReqInput, lambda req: es_update(weight_updater, req)),
        (EsLoadDeltaReqInput, lambda req: es_load_delta(weight_updater, req)),
        (EsStatusReqInput, lambda req: es_status(weight_updater, req)),
    ]
