"""Runtime hooks that wire Agentic ES into SGLang scheduler + HTTP server."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from sglang.srt.plugins.hook_registry import HookRegistry, HookType
from sglang.utils import TypeBasedDispatcher

if TYPE_CHECKING:
    from sglang.srt.managers.tokenizer_manager import TokenizerManager

logger = logging.getLogger(__name__)
_HOOKS_INSTALLED = False


def _after_init_request_dispatcher(_result, scheduler, *_args, **_kwargs):
    from sglang_es_worker import es_request_handlers

    scheduler._request_dispatcher += TypeBasedDispatcher(
        es_request_handlers(scheduler.weight_updater)
    )
    logger.info("[agentic_es] registered scheduler ES request handlers")
    return _result


def _ensure_es_communicator(tokenizer_manager: TokenizerManager) -> None:
    if getattr(tokenizer_manager, "es_op_communicator", None) is not None:
        return
    from sglang.srt.managers.communicator import FanOutCommunicator

    from sglang_es_io import EsOpReqOutput

    comm = FanOutCommunicator(
        tokenizer_manager._dispatch_to_scheduler,
        tokenizer_manager.server_args.dp_size,
        "queueing",
    )
    tokenizer_manager.es_op_communicator = comm
    tokenizer_manager._result_dispatcher += TypeBasedDispatcher(
        [(EsOpReqOutput, comm.handle_recv)]
    )
    logger.info("[agentic_es] lazily registered tokenizer ES communicator")


def _after_init_communicators(_result, tokenizer_manager: TokenizerManager, server_args):
    _ensure_es_communicator(tokenizer_manager)
    logger.info("[agentic_es] registered tokenizer ES communicator")
    return _result


def _attach_tokenizer_es_methods() -> None:
    from sglang.srt.managers.communicator import FanOutCommunicator
    from sglang.srt.managers.tokenizer_manager import TokenizerManager

    from sglang_es_io import (
        EsApplyReqInput,
        EsInitReqInput,
        EsLoadDeltaReqInput,
        EsRevertReqInput,
        EsStatusReqInput,
        EsUpdateReqInput,
    )

    async def _run_es_op(tokenizer_manager: TokenizerManager, obj):
        _ensure_es_communicator(tokenizer_manager)
        tokenizer_manager.auto_create_handle_loop()
        results = await tokenizer_manager.es_op_communicator(obj)
        return FanOutCommunicator.merge_results(results)

    async def es_init(self, request=None):
        return await _run_es_op(self, EsInitReqInput())

    async def es_apply(self, seed: int, sigma: float, request=None):
        return await _run_es_op(self, EsApplyReqInput(seed=int(seed), sigma=float(sigma)))

    async def es_revert(self, seed: int, sigma: float, request=None):
        return await _run_es_op(self, EsRevertReqInput(seed=int(seed), sigma=float(sigma)))

    async def es_update(self, seeds, weights, alpha: float, request=None):
        seeds_csv = ",".join(str(int(seed)) for seed in seeds)
        weights_csv = ",".join(str(float(weight)) for weight in weights)
        return await _run_es_op(
            self,
            EsUpdateReqInput(seeds_csv=seeds_csv, weights_csv=weights_csv, alpha=float(alpha)),
        )

    async def es_load_delta(self, delta_dir: str, request=None):
        return await _run_es_op(self, EsLoadDeltaReqInput(delta_dir=str(delta_dir)))

    async def es_status(self, request=None):
        return await _run_es_op(self, EsStatusReqInput())

    TokenizerManager.agentic_es_init = es_init
    TokenizerManager.agentic_es_apply = es_apply
    TokenizerManager.agentic_es_revert = es_revert
    TokenizerManager.agentic_es_update = es_update
    TokenizerManager.agentic_es_load_delta = es_load_delta
    TokenizerManager.agentic_es_status = es_status


def install_scheduler_hooks() -> None:
    global _HOOKS_INSTALLED
    HookRegistry.register(
        "sglang.srt.managers.scheduler.Scheduler.init_request_dispatcher",
        _after_init_request_dispatcher,
        HookType.AFTER,
    )
    _attach_tokenizer_es_methods()
    HookRegistry.register(
        "sglang.srt.managers.tokenizer_control_mixin.TokenizerControlMixin.init_communicators",
        _after_init_communicators,
        HookType.AFTER,
    )
    _HOOKS_INSTALLED = True


def run_scheduler_process_with_es(*args, **kwargs):
    install_scheduler_hooks()
    from sglang.srt.plugins.hook_registry import HookRegistry

    HookRegistry.apply_hooks()
    from sglang.srt.managers.scheduler import run_scheduler_process

    return run_scheduler_process(*args, **kwargs)
