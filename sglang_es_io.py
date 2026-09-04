"""IPC message types for in-place ES hooks on SGLang."""

from __future__ import annotations

from sglang.srt.managers.io_struct import BaseReq


class EsInitReqInput(BaseReq, kw_only=True):
    pass


class EsApplyReqInput(BaseReq, kw_only=True):
    seed: int
    sigma: float


class EsRevertReqInput(BaseReq, kw_only=True):
    seed: int
    sigma: float


class EsUpdateReqInput(BaseReq, kw_only=True):
    seeds_csv: str
    weights_csv: str
    alpha: float


class EsLoadDeltaReqInput(BaseReq, kw_only=True):
    delta_dir: str


class EsStatusReqInput(BaseReq, kw_only=True):
    pass


class EsOpReqOutput(BaseReq, kw_only=True):
    success: bool
    message: str
