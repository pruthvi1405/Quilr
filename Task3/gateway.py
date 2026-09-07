#!/usr/bin/env python3
"""
LLM Gateway streaming guardrail, built entirely on high-level MCP: a
FastMCP server that is, for the duration of each call, also an MCP
client to the upstream provider.

The gateway exposes its own `generate` tool. When a client calls it:
  1. The gateway opens an MCP client session to llm_provider.py and
     calls ITS `generate` tool, passing a progress_callback.
  2. Each raw delta the provider streams back (via its own
     ctx.report_progress) arrives at that callback in real time --
     before the provider's tool call has finished.
  3. Each delta is fed into a StreamRedactor (redactor.py) the moment
     it arrives. Whatever comes back as "safe to emit" is immediately
     forwarded to the GATEWAY's own caller via ctx.report_progress --
     never buffered until the upstream call completes.
  4. Once upstream finishes, the redactor is flushed (nothing more is
     coming, so whatever's left in its holdback buffer is now safe)
     and the accumulated, fully-redacted text is returned as the
     gateway tool's own result -- so a caller that only reads the
     final return value also gets clean text, not just one that
     streams via progress.

This keeps the two concerns doing exactly what redactor.py's own
docstring describes -- feed() never buffers more than a small, bounded
window regardless of total response length -- while the actual
transport in both directions (provider -> gateway, gateway -> client)
is genuine MCP, not a hand-rolled protocol.

Usage:
    python3 gateway.py [--port 8090] [--upstream http://127.0.0.1:9100/mcp]
"""

import argparse
import logging
import sys

import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.server.fastmcp import Context, FastMCP

from redactor import StreamRedactor

logging.basicConfig(
    level=logging.INFO,
    stream=sys.stderr,
    format="%(asctime)s %(levelname)s gateway: %(message)s",
)
log = logging.getLogger("gateway")

UPSTREAM_URL = "http://127.0.0.1:9100/mcp"  # set from --upstream in main()

mcp = FastMCP("llm-gateway", host="127.0.0.1", port=8090)


@mcp.tool(description="Generate text for a prompt; PII in the streamed response is redacted in real time.")
async def generate(prompt: str, scenario: str, ctx: Context) -> str:
    redactor = StreamRedactor()
    collected: list[str] = []
    step = 0

    async def on_upstream_progress(progress: float, total: float | None, message: str | None) -> None:
        nonlocal step
        if not message:
            return
        safe_text = redactor.feed(message)
        if safe_text:
            collected.append(safe_text)
            step += 1
            await ctx.report_progress(progress=step, total=total, message=safe_text)

    async with httpx.AsyncClient() as http_client:
        async with streamable_http_client(UPSTREAM_URL, http_client=http_client) as (read, write, _get_session_id):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(
                    "generate",
                    {"prompt": prompt, "scenario": scenario},
                    progress_callback=on_upstream_progress,
                )

    if result.isError:
        detail = "".join(block.text for block in result.content if block.type == "text")
        raise ValueError(f"upstream generation failed: {detail}")

    tail = redactor.flush()
    if tail:
        collected.append(tail)
        step += 1
        await ctx.report_progress(progress=step, total=None, message=tail)

    return "".join(collected)


def main() -> None:
    global UPSTREAM_URL

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--upstream", default=UPSTREAM_URL)
    args = parser.parse_args()
    UPSTREAM_URL = args.upstream
    mcp.settings.port = args.port

    log.info("listening on 127.0.0.1:%d/mcp, upstream %s", args.port, UPSTREAM_URL)
    mcp.run(transport="streamable-http")


if __name__ == "__main__":
    main()
