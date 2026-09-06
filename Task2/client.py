#!/usr/bin/env python3
"""
MCP client for exercising gateway.py, built on the official SDK's own
HTTP client transport (mcp.client.streamable_http.streamablehttp_client
+ mcp.ClientSession) -- not a raw HTTP request script. Every call this
script makes is a real MCP protocol exchange; the only thing under
test is what sits between this client and downstream_server.py.

Points at the GATEWAY's URL, never at downstream_server.py directly --
that's the whole point: everything here passes through the gateway on
its way to the real server, and the assertions confirm what the
gateway did to each call along the way.

Usage:
    python3 client.py [--gateway http://127.0.0.1:8080/mcp]
"""

import argparse
import asyncio
import sys

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

ADMIN_TOKEN = "admin-token-abc123"
VIEWER_TOKEN = "viewer-token-xyz789"

failures = 0


def check(label: str, condition: bool, detail: str = "") -> None:
    global failures
    status = "PASS" if condition else "FAIL"
    print(f"{status}  {label}" + (f"    ({detail})" if detail and not condition else ""))
    if not condition:
        failures += 1


async def _call(gateway_url: str, token: str | None, name: str, arguments: dict):
    """Returns (result, exception). The exception, if any, must be
    caught HERE -- inside the nested streamablehttp_client/ClientSession
    context managers -- rather than letting it propagate out through
    them uncaught. anyio's task-group teardown (both context managers
    are backed by one) re-wraps an exception that's still propagating
    when it exits into an ExceptionGroup; catching it before that
    unwind starts keeps the original McpError intact and inspectable.
    """
    headers = {"Authorization": f"Bearer {token}"} if token else None
    async with streamablehttp_client(gateway_url, headers=headers) as (read, write, _get_session_id):
        async with ClientSession(read, write) as session:
            await session.initialize()
            try:
                return await session.call_tool(name, arguments), None
            except Exception as exc:
                return None, exc


async def _list_tools(gateway_url: str, token: str | None):
    headers = {"Authorization": f"Bearer {token}"} if token else None
    async with streamablehttp_client(gateway_url, headers=headers) as (read, write, _get_session_id):
        async with ClientSession(read, write) as session:
            await session.initialize()
            return await session.list_tools()


def _text_of(result) -> str:
    return "".join(block.text for block in result.content if block.type == "text")


async def run(gateway_url: str) -> int:
    print(f"Connecting through gateway at {gateway_url}\n")

    tools = await _list_tools(gateway_url, token=VIEWER_TOKEN)
    names = {t.name for t in tools.tools}
    check("tools/list forwarded through the gateway", names == {"get_weather", "admin_reset_key", "admin_delete_user"}, str(names))

    result, exc = await _call(gateway_url, VIEWER_TOKEN, "get_weather", {"city": "Boston"})
    check("public tool call succeeds for viewer token", exc is None and not result.isError, str(exc) or _text_of(result))

    result, exc = await _call(gateway_url, None, "get_weather", {"city": "Boston"})
    check("public tool call succeeds with no token at all", exc is None and not result.isError, str(exc) or _text_of(result))

    result, exc = await _call(gateway_url, ADMIN_TOKEN, "admin_reset_key", {"customer_id": "CUST-00042"})
    check("admin tool call succeeds for admin token", exc is None and not result.isError, str(exc) or _text_of(result))

    # The gateway intercepts this one itself -- it should come back as
    # a protocol-level JSON-RPC error (-32001), which the SDK surfaces
    # as a McpError raised out of call_tool, not as an isError result.
    result, exc = await _call(gateway_url, VIEWER_TOKEN, "admin_reset_key", {"customer_id": "CUST-00042"})
    code = getattr(getattr(exc, "error", None), "code", None)
    check("admin tool call is rejected for viewer token", code == -32001, f"result={result} exc={exc}")

    result, exc = await _call(gateway_url, None, "admin_delete_user", {"user_id": "u-1"})
    code = getattr(getattr(exc, "error", None), "code", None)
    check("admin tool call is rejected with no token at all", code == -32001, f"result={result} exc={exc}")

    print()
    print("ALL PASS" if failures == 0 else f"{failures} FAILURE(S)")
    return 0 if failures == 0 else 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gateway", default="http://127.0.0.1:8080/mcp")
    args = parser.parse_args()
    sys.exit(asyncio.run(run(args.gateway)))


if __name__ == "__main__":
    main()
