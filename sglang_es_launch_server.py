#!/usr/bin/env python3
"""Launch SGLang with Agentic ES /es/* hooks enabled."""

from __future__ import annotations

import os
import sys
import warnings

from sglang.srt.server_args import prepare_server_args
from sglang.srt.utils import kill_process_tree
from sglang.srt.utils.common import suppress_noisy_warnings

from sglang_es_hooks import install_scheduler_hooks, run_scheduler_process_with_es
from sglang_es_http import register_es_http_routes

suppress_noisy_warnings()


def _launch_callback() -> None:
    register_es_http_routes()


def run_server(server_args):
    install_scheduler_hooks()
    from sglang.srt.plugins.hook_registry import HookRegistry

    HookRegistry.apply_hooks()
    register_es_http_routes()
    from sglang.srt.entrypoints.http_server import launch_server

    launch_server(
        server_args,
        run_scheduler_process_func=run_scheduler_process_with_es,
        launch_callback=_launch_callback,
    )


if __name__ == "__main__":
    warnings.warn(
        "agentic_es.sglang_es_launch_server wraps sglang.launch_server with /es/* hooks.",
        UserWarning,
        stacklevel=1,
    )
    from sglang.srt.plugins import load_plugins

    install_scheduler_hooks()
    load_plugins()
    server_args = prepare_server_args(sys.argv[1:])
    try:
        run_server(server_args)
    finally:
        kill_process_tree(os.getpid(), include_parent=False)
