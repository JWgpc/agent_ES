"""Manage multiple SGLang server processes for ES candidate evaluation."""

from __future__ import annotations

import os
import signal
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import requests


@dataclass
class SGLangInstance:
    model_path: Path
    port: int
    gpu_ids: Sequence[int]
    served_model_name: str
    tp_size: int = 1
    mem_fraction: float = 0.85
    context_length: int = 65536
    log_path: Path | None = None
    extra_args: list[str] = field(default_factory=list)
    _proc: subprocess.Popen | None = field(default=None, repr=False)

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def start(self, *, wait_timeout: float = 600.0) -> None:
        if self._proc is not None and self._proc.poll() is None:
            return
        gpu_csv = ",".join(str(g) for g in self.gpu_ids)
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu_csv
        cmd = [
            "python",
            "-m",
            "sglang.launch_server",
            "--model-path",
            str(self.model_path),
            "--served-model-name",
            self.served_model_name,
            "--port",
            str(self.port),
            "--tp-size",
            str(self.tp_size),
            "--mem-fraction-static",
            str(self.mem_fraction),
            "--context-length",
            str(self.context_length),
            "--tool-call-parser",
            "qwen3_coder",
            "--reasoning-parser",
            "qwen3",
            "--host",
            "127.0.0.1",
        ]
        cmd.extend(self.extra_args)
        if self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            log_handle = open(self.log_path, "w", encoding="utf-8")
        else:
            log_handle = subprocess.DEVNULL
        self._proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        self._wait_ready(timeout=wait_timeout)

    def _wait_ready(self, *, timeout: float) -> None:
        deadline = time.time() + timeout
        url = f"{self.base_url}/v1/models"
        last_error = ""
        while time.time() < deadline:
            if self._proc is not None and self._proc.poll() is not None:
                raise RuntimeError(f"SGLang exited early on port {self.port}")
            try:
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    return
                last_error = f"status={response.status_code}"
            except requests.RequestException as exc:
                last_error = str(exc)
            time.sleep(2.0)
        raise TimeoutError(f"SGLang not ready on port {self.port}: {last_error}")

    def stop(self, *, timeout: float = 30.0) -> None:
        if self._proc is None:
            return
        if self._proc.poll() is not None:
            self._proc = None
            return
        try:
            os.killpg(self._proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            self._proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(self._proc.pid, signal.SIGKILL)
            self._proc.wait(timeout=10)
        self._proc = None


class SGLangPool:
    def __init__(self, instances: Sequence[SGLangInstance]) -> None:
        self.instances = list(instances)

    def start_all(self, *, wait_timeout: float = 600.0) -> None:
        for instance in self.instances:
            instance.start(wait_timeout=wait_timeout)

    def stop_all(self) -> None:
        for instance in self.instances:
            instance.stop()

    def __enter__(self) -> SGLangPool:
        self.start_all()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop_all()
