#!/usr/bin/env python3
"""Smoke-test SGLang /es/* hook on a running server."""

from __future__ import annotations

import argparse
import json
import sys
import time

import requests

from sglang_es_client import SGLangESClient


def wait_for_server(base_url: str, *, timeout: float = 900.0) -> None:
    deadline = time.time() + timeout
    url = f"{base_url.rstrip('/')}/v1/models"
    last_error = ""
    while time.time() < deadline:
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                return
            last_error = f"status={response.status_code}"
        except requests.RequestException as exc:
            last_error = str(exc)
        time.sleep(2.0)
    raise TimeoutError(f"SGLang not ready at {base_url}: {last_error}")


def wait_for_es_routes(base_url: str, *, timeout: float = 900.0) -> None:
    """Wait until /es/* routes are registered (after launch_callback)."""
    deadline = time.time() + timeout
    url = f"{base_url.rstrip('/')}/es/status"
    last_error = ""
    while time.time() < deadline:
        try:
            response = requests.get(url, timeout=5)
            if response.status_code != 404:
                return
            last_error = "route not registered yet"
        except requests.RequestException as exc:
            last_error = str(exc)
        time.sleep(2.0)
    raise TimeoutError(f"SGLang ES routes not ready at {base_url}: {last_error}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test SGLang ES hook endpoints")
    parser.add_argument("--base-url", default="http://127.0.0.1:12100")
    parser.add_argument("--wait-timeout", type=float, default=900.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sigma", type=float, default=1e-3)
    parser.add_argument("--skip-wait", action="store_true")
    args = parser.parse_args()

    if not args.skip_wait:
        print(f"[smoke] waiting for {args.base_url} ...", flush=True)
        wait_for_server(args.base_url, timeout=args.wait_timeout)
        print(f"[smoke] waiting for /es/* routes ...", flush=True)
        wait_for_es_routes(args.base_url, timeout=args.wait_timeout)

    client = SGLangESClient(args.base_url, timeout=120.0)
    print("[smoke] /es/init", flush=True)
    init_body = client.init()
    print(json.dumps(init_body, indent=2), flush=True)

    print("[smoke] /es/status (before apply)", flush=True)
    print(json.dumps(client.status(), indent=2), flush=True)

    print(f"[smoke] /es/apply seed={args.seed} sigma={args.sigma}", flush=True)
    client.apply(seed=args.seed, sigma=args.sigma)

    print("[smoke] /es/revert", flush=True)
    client.revert(seed=args.seed, sigma=args.sigma)

    print("[smoke] /es/update", flush=True)
    client.update(seeds=[args.seed, args.seed + 1], weights=[0.5, -0.5], alpha=1e-4)

    print("[smoke] /es/status (after update)", flush=True)
    print(json.dumps(client.status(), indent=2), flush=True)
    print("[smoke] OK", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[smoke] FAILED: {exc}", file=sys.stderr, flush=True)
        raise
