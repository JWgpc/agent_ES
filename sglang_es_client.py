"""HTTP client for SGLang in-place ES endpoints."""

from __future__ import annotations

from typing import Any, Sequence

import requests


class SGLangESClient:
    def __init__(self, base_url: str, *, timeout: float = 600.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _post(self, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        response = requests.post(
            f"{self.base_url}{path}",
            json=payload or {},
            timeout=self.timeout,
        )
        if response.status_code >= 400:
            detail = response.text[:500]
            raise RuntimeError(f"ES POST {path} failed: HTTP {response.status_code}: {detail}")
        body = response.json()
        if not body.get("success", False):
            raise RuntimeError(f"ES request failed ({path}): {body.get('message')}")
        return body

    def _get(self, path: str) -> dict[str, Any]:
        response = requests.get(f"{self.base_url}{path}", timeout=self.timeout)
        response.raise_for_status()
        body = response.json()
        if not body.get("success", False):
            raise RuntimeError(f"ES request failed ({path}): {body.get('message')}")
        return body

    def init(self) -> dict[str, Any]:
        return self._post("/es/init")

    def apply(self, *, seed: int, sigma: float) -> None:
        self._post("/es/apply", {"seed": int(seed), "sigma": float(sigma)})

    def revert(self, *, seed: int, sigma: float) -> None:
        self._post("/es/revert", {"seed": int(seed), "sigma": float(sigma)})

    def update(self, *, seeds: Sequence[int], weights: Sequence[float], alpha: float) -> None:
        self._post(
            "/es/update",
            {
                "seeds": [int(seed) for seed in seeds],
                "weights": [float(weight) for weight in weights],
                "alpha": float(alpha),
            },
        )

    def load_delta(self, delta_dir: str) -> dict[str, Any]:
        return self._post("/es/load_delta", {"delta_dir": str(delta_dir)})

    def status(self) -> dict[str, Any]:
        return self._get("/es/status")
