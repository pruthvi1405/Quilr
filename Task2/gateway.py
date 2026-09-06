#!/usr/bin/env python3
"""
MCP Security Gateway: a reverse proxy that sits between an MCP client
and a downstream MCP server speaking the real Streamable HTTP
transport (not a simplified stand-in) -- so any standard MCP client
(client.py, or the official SDK directly) can point at the gateway
exactly as it would point at the real server, and the gateway is free
to inspect and gate individual tool calls in between.

- Extracts the caller's role from an `Authorization: Bearer <token>`
  header (a static lookup table stands in for real token verification
  -- JWT validation, OAuth introspection -- swapping that in is the
  only change needed to make this production-shaped).
- tools/list is forwarded to downstream untouched.
- tools/call is inspected: if params.name starts with admin_, the
  caller's role must be "admin" or the call is intercepted and
  answered locally with a JSON-RPC -32001 error -- downstream is never
  contacted for a blocked call. Everything else is forwarded.
- Every other HTTP method (GET for the server-push stream, DELETE for
  session termination) and every other JSON-RPC method is forwarded
  through unchanged, byte for byte, including a streaming
  text/event-stream response -- the gateway only ever inspects and
  acts on a tools/call POST body, nothing else.

This does not forward the client's Authorization header downstream --
downstream never sees a client-controlled credential.

Usage:
    python3 gateway.py [--port 8080] [--downstream http://127.0.0.1:9000/mcp]
"""

import argparse
import json
import logging
import sys
from contextlib import asynccontextmanager

import httpx
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

logging.basicConfig(
    level=logging.INFO,
    stream=sys.stderr,
    format="%(asctime)s %(levelname)s gateway: %(message)s",
)
log = logging.getLogger("gateway")

# Stand-in for real token verification. Swapping this function for a
# real verifier is the only change needed to make the rest of the
# gateway production-shaped.
TOKEN_ROLES = {
    "admin-token-abc123": "admin",
    "viewer-token-xyz789": "viewer",
}

ADMIN_TOOL_PREFIX = "admin_"
UNAUTHORIZED_TOOL_CALL = -32001

DOWNSTREAM_URL = "http://127.0.0.1:9000/mcp"  # set from --downstream in main()

# Headers that describe the PREVIOUS hop's connection, not the
# message body -- copying these through to the new connection is
# wrong regardless of which direction they're being relayed.
_HOP_BY_HOP = {"content-length", "transfer-encoding", "connection", "keep-alive"}


def _role_for_header(auth_header: str | None) -> str | None:
    if not auth_header:
        return None
    scheme, _, token = auth_header.partition(" ")
    if scheme.lower() != "bearer" or not token:
        return None
    return TOKEN_ROLES.get(token.strip())


def _error_response(request_id, code: int, message: str) -> dict:
    return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}


def _filtered_headers(headers, *, drop: set[str]) -> dict[str, str]:
    return {k: v for k, v in headers.items() if k.lower() not in _HOP_BY_HOP | drop}


async def _proxy_streaming(request: Request, client: httpx.AsyncClient, *, method: str, content: bytes | None) -> Response:
    """
    Transparent passthrough for anything the gateway doesn't need to
    inspect: GET's long-lived server push stream, DELETE's session
    termination, and any POST that isn't a gated tools/call. Streams
    the downstream response back chunk by chunk rather than buffering
    it -- required for a text/event-stream response, which is
    long-lived by design.
    """
    upstream_req = client.build_request(
        method,
        DOWNSTREAM_URL,
        content=content,
        headers=_filtered_headers(request.headers, drop={"host", "authorization"}),
    )
    upstream_resp = await client.send(upstream_req, stream=True)

    async def body_iterator():
        try:
            async for chunk in upstream_resp.aiter_raw():
                yield chunk
        finally:
            await upstream_resp.aclose()

    return StreamingResponse(
        body_iterator(),
        status_code=upstream_resp.status_code,
        headers=_filtered_headers(upstream_resp.headers, drop=set()),
    )


async def handle_mcp(request: Request) -> Response:
    client: httpx.AsyncClient = request.app.state.client

    if request.method in ("GET", "DELETE"):
        return await _proxy_streaming(request, client, method=request.method, content=None)

    # POST: read the body once so it can both be inspected (to decide
    # whether to gate it) and, if not gated, forwarded unchanged.
    raw_body = await request.body()

    try:
        message = json.loads(raw_body)
    except json.JSONDecodeError:
        # Not our call to police JSON-RPC framing here -- pass it
        # through and let downstream's own transport reject it exactly
        # as it would if the gateway weren't in the path.
        return await _proxy_streaming(request, client, method="POST", content=raw_body)

    if isinstance(message, dict) and message.get("method") == "tools/call":
        params = message.get("params")
        tool_name = params.get("name") if isinstance(params, dict) else None

        if isinstance(tool_name, str) and tool_name.startswith(ADMIN_TOOL_PREFIX):
            role = _role_for_header(request.headers.get("authorization"))
            if role != "admin":
                request_id = message.get("id")
                log.warning("blocked tools/call name=%r role=%r", tool_name, role)
                return JSONResponse(_error_response(request_id, UNAUTHORIZED_TOOL_CALL, "Unauthorized Tool Call"))

    return await _proxy_streaming(request, client, method="POST", content=raw_body)


async def handle_health(_request: Request) -> Response:
    return JSONResponse({"status": "ok"})


@asynccontextmanager
async def _lifespan(app: Starlette):
    app.state.client = httpx.AsyncClient(timeout=httpx.Timeout(30.0, read=300.0))
    try:
        yield
    finally:
        await app.state.client.aclose()


def build_app() -> Starlette:
    return Starlette(
        routes=[
            Route("/mcp", handle_mcp, methods=["GET", "POST", "DELETE"]),
            Route("/health", handle_health, methods=["GET"]),
        ],
        lifespan=_lifespan,
    )


def main() -> None:
    global DOWNSTREAM_URL

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--downstream", default=DOWNSTREAM_URL)
    args = parser.parse_args()
    DOWNSTREAM_URL = args.downstream

    import uvicorn

    log.info("listening on 127.0.0.1:%d/mcp, forwarding to %s", args.port, DOWNSTREAM_URL)
    uvicorn.run(build_app(), host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
