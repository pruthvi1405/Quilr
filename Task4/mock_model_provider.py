#!/usr/bin/env python3
"""
Mock model provider, standing in for a real completions backend. Used
as BOTH the primary and the backup in the harness -- which one is
"broken" for a given test is controlled independently per instance via
a fault-injection control endpoint, not by anything in the client's
actual completion request. That mirrors reality: a client shouldn't
need to know or say which backend is having trouble, and router.py's
job is to find that out itself by trying primary first.

Endpoints:
    POST /v1/completions   the actual completion endpoint
    POST /_set_fault       {"mode": "none"|"rate_limited"|"slow"|"error"}
    GET  /_calls           every /v1/completions request this process
                            has received, in order (for asserting which
                            backend actually got called)
    GET  /health

Usage:
    python3 mock_model_provider.py [--port 9201]
"""

import argparse
import asyncio
import json
import logging
import sys

from aiohttp import web

logging.basicConfig(
    level=logging.INFO,
    stream=sys.stderr,
    format="%(asctime)s %(levelname)s provider: %(message)s",
)
log = logging.getLogger("provider")

SLOW_DELAY_SECONDS = 4.0  # deliberately > router.py's 3000ms timeout


async def handle_completion(request: web.Request) -> web.Response:
    state = request.app["state"]
    try:
        payload = await request.json()
    except json.JSONDecodeError:
        payload = {}

    state["calls"].append(payload)

    mode = state["fault_mode"]

    if mode == "rate_limited":
        return web.json_response({"error": "rate limit exceeded"}, status=429)

    if mode == "error":
        # Deliberately traceback-shaped, to prove the gateway never
        # relays this verbatim to its own client.
        return web.json_response(
            {
                "error": "internal server error",
                "traceback": "Traceback (most recent call last):\n  File \"provider.py\", line 42, in handle\n    raise RuntimeError('db connection pool exhausted')\nRuntimeError: db connection pool exhausted",
            },
            status=500,
        )

    if mode == "slow":
        await asyncio.sleep(SLOW_DELAY_SECONDS)
        # falls through to the normal success response below, in case
        # a client's own timeout is longer than ours and actually
        # waits this out

    prompt = payload.get("prompt", "")
    return web.json_response(
        {
            "id": f"cmpl-{state['provider_name']}-{len(state['calls'])}",
            "provider": state["provider_name"],
            "text": f"[{state['provider_name']}] response to: {prompt}",
        }
    )


async def handle_set_fault(request: web.Request) -> web.Response:
    payload = await request.json()
    mode = payload.get("mode", "none")
    if mode not in ("none", "rate_limited", "slow", "error"):
        return web.json_response({"error": f"unknown mode: {mode!r}"}, status=400)
    request.app["state"]["fault_mode"] = mode
    log.info("fault mode set to %r", mode)
    return web.json_response({"fault_mode": mode})


async def handle_calls(request: web.Request) -> web.Response:
    return web.json_response({"calls": request.app["state"]["calls"]})


async def handle_health(_request: web.Request) -> web.Response:
    return web.json_response({"status": "ok"})


def build_app(provider_name: str) -> web.Application:
    app = web.Application()
    # A single mutable dict stored once at construction time, whose
    # CONTENTS get mutated per-request -- not reassigning app's own
    # keys after startup, which aiohttp deprecates.
    app["state"] = {"provider_name": provider_name, "fault_mode": "none", "calls": []}
    app.router.add_post("/v1/completions", handle_completion)
    app.router.add_post("/_set_fault", handle_set_fault)
    app.router.add_get("/_calls", handle_calls)
    app.router.add_get("/health", handle_health)
    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=9201)
    parser.add_argument("--name", default=None, help="Label used in responses/logs (default: derived from port)")
    args = parser.parse_args()
    name = args.name or f"provider-{args.port}"

    log.info("%s listening on 127.0.0.1:%d", name, args.port)
    web.run_app(build_app(name), host="127.0.0.1", port=args.port, print=None, access_log=None)


if __name__ == "__main__":
    main()
