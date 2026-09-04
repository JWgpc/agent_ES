"""Manage multiple SGLang server processes for ES candidate evaluation."""

from __future__ import annotations

import os
import re
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import requests


def wait_for_es_routes(base_url: str, *, timeout: float = 60.0) -> None:
    deadline = time.time() + timeout
    url = f"{base_url.rstrip('/')}/es/status"
    while time.time() < deadline:
        try:
            if requests.get(url, timeout=5).status_code != 404:
                return
        except requests.RequestException:
            pass
        time.sleep(1.0)
    raise TimeoutError(f"ES routes not ready on {base_url}")


def _sglang_python() -> str:
    return os.environ.get("SGLANG_PYTHON", sys.executable)


def _nvidia_lib_dirs(python: str) -> list[str]:
    """Discover pip-shipped NVIDIA libs (e.g. libnvrtc.so.13 for cu13 sgl_kernel)."""
    try:
        import site
        from pathlib import Path

        prefix = Path(python).resolve().parent.parent
        candidates: list[Path] = []
        try:
            candidates.extend(Path(p) for p in site.getsitepackages())
        except AttributeError:
            pass
        candidates.append(prefix / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages")
        seen: set[str] = set()
        dirs: list[str] = []
        for sp in candidates:
            for sub in ("nvidia/cu13/lib", "nvidia/cuda_nvrtc/lib", "nvidia/cudnn/lib"):
                lib_dir = sp / sub
                key = str(lib_dir)
                if lib_dir.is_dir() and key not in seen:
                    seen.add(key)
                    dirs.append(key)
        return dirs
    except Exception:
        return []


def _sglang_subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    extra = _nvidia_lib_dirs(_sglang_python())
    if extra:
        current = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = ":".join(extra + ([current] if current else []))
    return env


def _pids_on_tcp_port(port: int) -> set[int]:
    try:
        result = subprocess.run(
            ["ss", "-H", "-tlnp", f"sport = :{port}"],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return set()
    pids: set[int] = set()
    for line in result.stdout.splitlines():
        for match in re.finditer(r"pid=(\d+)", line):
            pids.add(int(match.group(1)))
    return pids


def _read_ppid(pid: int) -> int | None:
    try:
        with open(f"/proc/{pid}/stat", encoding="ascii") as handle:
            data = handle.read()
    except OSError:
        return None
    rparen = data.rfind(")")
    if rparen == -1:
        return None
    fields = data[rparen + 2 :].split()
    if len(fields) < 2:
        return None
    try:
        return int(fields[1])
    except ValueError:
        return None


def _proc_tree_pids(root_pid: int) -> set[int]:
    pids = {root_pid}
    changed = True
    while changed:
        changed = False
        try:
            for entry in os.listdir("/proc"):
                if not entry.isdigit():
                    continue
                pid = int(entry)
                if pid in pids:
                    continue
                ppid = _read_ppid(pid)
                if ppid is not None and ppid in pids:
                    pids.add(pid)
                    changed = True
        except OSError:
            break
    return pids


def _terminate_pids(pids: set[int], *, grace: float = 5.0) -> None:
    live = [pid for pid in sorted(pids) if pid > 1]
    if not live:
        return
    for pid in live:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.time() + grace
    while time.time() < deadline:
        if not any(os.path.exists(f"/proc/{pid}") for pid in live):
            return
        time.sleep(0.2)
    for pid in live:
        if not os.path.exists(f"/proc/{pid}"):
            continue
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def _free_port(port: int, *, exclude_pids: set[int] | None = None) -> None:
    exclude = exclude_pids or set()
    foreign = _pids_on_tcp_port(port) - exclude
    if foreign:
        _terminate_pids(foreign)


def _pool_start_delay() -> float:
    raw = os.environ.get("SGLANG_POOL_START_DELAY", "20")
    try:
        return max(0.0, float(raw))
    except ValueError:
        return 20.0


def _read_log_tail(log_path: Path | None, *, max_lines: int = 20) -> str:
    if log_path is None or not log_path.is_file():
        return ""
    try:
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    if not lines:
        return ""
    return "\n".join(lines[-max_lines:])


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
    use_es_hook: bool = True
    _proc: subprocess.Popen | None = field(default=None, repr=False)

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def start(self, *, wait_timeout: float = 600.0) -> None:
        if self._proc is not None and self._proc.poll() is None:
            return
        _free_port(self.port)
        gpu_csv = ",".join(str(g) for g in self.gpu_ids)
        env = _sglang_subprocess_env()
        env["CUDA_VISIBLE_DEVICES"] = gpu_csv
        launch_module = "sglang_es_launch_server" if self.use_es_hook else "sglang.launch_server"
        cmd = [
            _sglang_python(),
            "-m",
            launch_module,
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
                detail = _read_log_tail(self.log_path)
                msg = f"SGLang exited early on port {self.port}"
                if detail:
                    msg += f"\n--- log tail ({self.log_path}) ---\n{detail}"
                raise RuntimeError(msg)
            try:
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    if self._proc is None or self._proc.poll() is not None:
                        last_error = f"port {self.port} responded but child exited"
                        time.sleep(2.0)
                        continue
                    holders = _pids_on_tcp_port(self.port)
                    ours = _proc_tree_pids(self._proc.pid)
                    if holders and not (holders & ours):
                        last_error = (
                            f"port {self.port} held by foreign pid(s) "
                            f"{sorted(holders - ours)} (ours={self._proc.pid})"
                        )
                        time.sleep(2.0)
                        continue
                    if self.use_es_hook:
                        wait_for_es_routes(self.base_url, timeout=min(60.0, timeout / 4))
                    return
                last_error = f"status={response.status_code}"
            except requests.RequestException as exc:
                last_error = str(exc)
            time.sleep(2.0)
        raise TimeoutError(f"SGLang not ready on port {self.port}: {last_error}")

    def stop(self, *, timeout: float = 30.0) -> None:
        managed_pid = self._proc.pid if self._proc is not None else None
        if self._proc is not None and self._proc.poll() is None:
            try:
                os.killpg(self._proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                self._proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(self._proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                self._proc.wait(timeout=10)
        self._proc = None
        exclude = {managed_pid} if managed_pid is not None else None
        _free_port(self.port, exclude_pids=exclude)


class SGLangPool:
    def __init__(self, instances: Sequence[SGLangInstance]) -> None:
        self.instances = list(instances)

    def start_all(self, *, wait_timeout: float = 600.0, start_delay: float | None = None) -> None:
        delay = _pool_start_delay() if start_delay is None else max(0.0, start_delay)
        for index, instance in enumerate(self.instances):
            if index > 0 and delay > 0:
                time.sleep(delay)
            instance.start(wait_timeout=wait_timeout)

    def stop_all(self) -> None:
        for instance in self.instances:
            instance.stop()

    def __enter__(self) -> SGLangPool:
        self.start_all()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop_all()
