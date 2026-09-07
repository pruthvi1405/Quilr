#!/usr/bin/env python3
"""
Model-fallback router for an LLM gateway: rate-limits each tenant by
token budget (see rate_limiter.py), then routes an admitted completion
request to a primary model provider, automatically failing over to a
secondary provider if the primary returns 429 or doesn't answer within
PRIMARY_TIMEOUT_SECONDS.

Timeout handling and the race it avoids: the primary call is made with
aiohttp's own ClientTimeout(total=PRIMARY_TIMEOUT_SECONDS) rather than
a manually spawned background task. That keeps control flow strictly
linear -- the await either returns a response or raises
asyncio.TimeoutError, and in the timeout case aiohttp has already torn
down that request's connection before the exception reaches this code.
There's no separately-running primary task left in flight that could
still complete and deliver a late response after the router has
already committed to (or returned) the secondary's answer, and no
double-accounting risk from a "finished after all, actually" primary
result showing up post-fallback.

Error sanitization: every path that returns an error to the client
goes through _error_payload, a single fixed shape
{"error": {"code", "message", "request_id"}}. Upstream response
bodies, exception messages, and stack traces are logged in full
server-side (see the log.warning/log.exception calls) but never placed
in a value returned to the client -- the client-facing message strings
are fixed, hand-written text, not str(exc) or an upstream body.

Usage:
    python3 router.py [--port 8100] [--primary URL] [--secondary URL] [--db PATH]
"""

import argparse
import asyncio
import json
import logging
import sys
import time
import uuid

from aiohttp import ClientError, ClientSession, ClientTimeout, web

from rate_limiter import SqliteSlidingWindowLimiter

logging.basicConfig(
    level=logging.INFO,
    stream=sys.stderr,
    format="%(asctime)s %(levelname)s router: %(message)s",
)
log = logging.getLogger("router")

PRIMARY_TIMEOUT_SECONDS = 3.0
RATE_LIMIT_TOKENS_PER_MINUTE = 50_000
RATE_LIMIT_WINDOW_SECONDS = 60.0
PURGE_INTERVAL_SECONDS = 30.0

PRIMARY_URL = "http://127.0.0.1:9201/v1/completions"
SECONDARY_URL = "http://127.0.0.1:9202/v1/completions"


def _error_payload(code: str, message: str, request_id: str) -> dict:
    return {"error": {"code": code, "message": message, "request_id": request_id}}


def _tenant_key_from_header(auth_header: str | None) -> str | None:
    if not auth_header:
        return None
    scheme, _, token = auth_header.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        return None
    return token.strip()


async def _call_provider(session: ClientSession, url: str, payload: dict, timeout_seconds: float) -> tuple[int, dict]:
    async with session.post(url, json=payload, timeout=ClientTimeout(total=timeout_seconds)) as resp:
        try:
            body = await resp.json()
        except Exception:
            body = {"_raw": await resp.text()}
        return resp.status, body


async def _route(session: ClientSession, payload: dict, request_id: str) -> tuple[int, dict | None]:
    """Try primary, fail over to secondary on any non-success status,
    timeout, or connection error. Returns (status, body); body is None
    if both providers failed, signalling the caller to answer with a
    sanitized error -- this check applies identically to secondary as
    to primary. Treating a bad secondary response as "the answer" just
    because there's nowhere left to fail over to would relay whatever
    secondary sent, unsanitized, straight to the client."""
    try:
        status, body = await _call_provider(session, PRIMARY_URL, payload, PRIMARY_TIMEOUT_SECONDS)
        if status == 200:
            return status, body
        log.warning("[%s] primary returned %s; failing over to secondary", request_id, status)
    except asyncio.TimeoutError:
        log.warning("[%s] primary timed out after %.1fs; failing over to secondary", request_id, PRIMARY_TIMEOUT_SECONDS)
    except (ClientError, OSError) as exc:
        log.warning("[%s] primary request failed (%s); failing over to secondary", request_id, exc)

    try:
        status, body = await _call_provider(session, SECONDARY_URL, payload, PRIMARY_TIMEOUT_SECONDS)
        if status == 200:
            return status, body
        log.error("[%s] secondary also returned %s", request_id, status)
    except asyncio.TimeoutError:
        log.error("[%s] secondary also timed out after %.1fs", request_id, PRIMARY_TIMEOUT_SECONDS)
    except (ClientError, OSError) as exc:
        log.error("[%s] secondary also failed (%s)", request_id, exc)

    return 502, None


async def handle_completion(request: web.Request) -> web.Response:
    request_id = str(uuid.uuid4())

    try:
        payload = await request.json()
    except json.JSONDecodeError:
        return web.json_response(
            _error_payload("invalid_request", "Request body must be valid JSON.", request_id), status=400
        )

    tenant_key = _tenant_key_from_header(request.headers.get("Authorization"))
    if tenant_key is None:
        return web.json_response(
            _error_payload("invalid_request", "Missing or malformed Authorization header.", request_id), status=401
        )

    tokens_requested = payload.get("tokens")
    if not isinstance(tokens_requested, int) or isinstance(tokens_requested, bool) or tokens_requested <= 0:
        return web.json_response(
            _error_payload("invalid_request", '"tokens" must be a positive integer.', request_id), status=400
        )

    limiter: SqliteSlidingWindowLimiter = request.app["limiter"]
    result = await limiter.check_and_consume(tenant_key, tokens_requested)
    if not result.allowed:
        resp = web.json_response(
            _error_payload("rate_limited", "Token rate limit exceeded for this API key.", request_id), status=429
        )
        resp.headers["Retry-After"] = str(max(1, int(result.retry_after_seconds) + 1))
        return resp

    try:
        session: ClientSession = request.app["client_session"]
        status, body = await _route(session, payload, request_id)
    except Exception:
        # Catch-all: whatever went wrong, the client only ever sees
        # the fixed sanitized shape below. Full detail (exception
        # type, message, traceback) is in this log line, server-side
        # only.
        log.exception("[%s] unhandled error while routing request", request_id)
        return web.json_response(
            _error_payload("internal_error", "An internal error occurred.", request_id), status=500
        )

    if body is None:
        return web.json_response(
            _error_payload("upstream_unavailable", "Both the primary and backup model providers are unavailable.", request_id),
            status=502,
        )

    return web.json_response(body, status=status)


async def handle_health(_request: web.Request) -> web.Response:
    return web.json_response({"status": "ok"})


async def handle_usage(request: web.Request) -> web.Response:
    """Test/ops introspection: current token usage for a tenant."""
    tenant_key = request.query.get("tenant")
    if not tenant_key:
        return web.json_response({"error": "missing ?tenant="}, status=400)
    limiter: SqliteSlidingWindowLimiter = request.app["limiter"]
    usage = await limiter.current_usage(tenant_key)
    return web.json_response({"tenant": tenant_key, "usage_tokens": usage, "limit_tokens": limiter.limit_tokens})


async def _purge_loop(app: web.Application) -> None:
    limiter: SqliteSlidingWindowLimiter = app["limiter"]
    try:
        while True:
            await asyncio.sleep(PURGE_INTERVAL_SECONDS)
            deleted = await limiter.purge_expired()
            if deleted:
                log.info("background purge evicted %d expired usage row(s)", deleted)
    except asyncio.CancelledError:
        pass


async def _on_startup(app: web.Application) -> None:
    app["client_session"] = ClientSession()
    app["purge_task"] = asyncio.create_task(_purge_loop(app))


async def _on_cleanup(app: web.Application) -> None:
    app["purge_task"].cancel()
    await app["client_session"].close()


def build_app(db_path: str) -> web.Application:
    app = web.Application()
    app["limiter"] = SqliteSlidingWindowLimiter(
        db_path, limit_tokens=RATE_LIMIT_TOKENS_PER_MINUTE, window_seconds=RATE_LIMIT_WINDOW_SECONDS
    )
    app.router.add_post("/v1/completions", handle_completion)
    app.router.add_get("/health", handle_health)
    app.router.add_get("/_usage", handle_usage)
    app.on_startup.append(_on_startup)
    app.on_cleanup.append(_on_cleanup)
    return app


def main() -> None:
    global PRIMARY_URL, SECONDARY_URL

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8100)
    parser.add_argument("--primary", default=PRIMARY_URL)
    parser.add_argument("--secondary", default=SECONDARY_URL)
    parser.add_argument("--db", default="rate_limit.db")
    args = parser.parse_args()
    PRIMARY_URL = args.primary
    SECONDARY_URL = args.secondary

    log.info("listening on 127.0.0.1:%d, primary=%s secondary=%s db=%s", args.port, PRIMARY_URL, SECONDARY_URL, args.db)
    web.run_app(build_app(args.db), host="127.0.0.1", port=args.port, print=None, access_log=None)


if __name__ == "__main__":
    main()
