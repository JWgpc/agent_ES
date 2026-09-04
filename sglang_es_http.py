"""HTTP /es/* routes for SGLang ES hook."""

from __future__ import annotations

import json
import logging
from typing import Any

from fastapi import Body, Request
from fastapi.responses import ORJSONResponse

logger = logging.getLogger(__name__)


def register_es_http_routes() -> None:
    from sglang.srt.entrypoints.http_server import app, get_global_state

    if getattr(app.state, "agentic_es_routes_registered", False):
        return

    @app.post("/es/init")
    async def es_init_route(request: Request):
        state = get_global_state()
        success, message = await state.tokenizer_manager.agentic_es_init(request=request)
        return ORJSONResponse({"success": success, "message": message, "data": _maybe_json(message)})

    @app.post("/es/apply")
    async def es_apply_route(payload: dict[str, Any] = Body(...), request: Request = None):
        state = get_global_state()
        success, message = await state.tokenizer_manager.agentic_es_apply(
            seed=int(payload["seed"]),
            sigma=float(payload["sigma"]),
            request=request,
        )
        return ORJSONResponse({"success": success, "message": message})

    @app.post("/es/revert")
    async def es_revert_route(payload: dict[str, Any] = Body(...), request: Request = None):
        state = get_global_state()
        success, message = await state.tokenizer_manager.agentic_es_revert(
            seed=int(payload["seed"]),
            sigma=float(payload["sigma"]),
            request=request,
        )
        return ORJSONResponse({"success": success, "message": message})

    @app.post("/es/update")
    async def es_update_route(payload: dict[str, Any] = Body(...), request: Request = None):
        state = get_global_state()
        success, message = await state.tokenizer_manager.agentic_es_update(
            seeds=[int(seed) for seed in payload["seeds"]],
            weights=[float(weight) for weight in payload["weights"]],
            alpha=float(payload["alpha"]),
            request=request,
        )
        return ORJSONResponse({"success": success, "message": message})

    @app.post("/es/load_delta")
    async def es_load_delta_route(payload: dict[str, Any] = Body(...), request: Request = None):
        state = get_global_state()
        success, message = await state.tokenizer_manager.agentic_es_load_delta(
            delta_dir=str(payload["delta_dir"]),
            request=request,
        )
        return ORJSONResponse({"success": success, "message": message, "data": _maybe_json(message)})

    @app.get("/es/status")
    async def es_status_route(request: Request):
        state = get_global_state()
        success, message = await state.tokenizer_manager.agentic_es_status(request=request)
        return ORJSONResponse({"success": success, "message": message, "data": _maybe_json(message)})

    app.state.agentic_es_routes_registered = True
    logger.info("[agentic_es] registered HTTP /es/* routes")


def _maybe_json(message: str) -> Any:
    try:
        return json.loads(message)
    except (TypeError, json.JSONDecodeError):
        return message
