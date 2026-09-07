#!/usr/bin/env python3
"""
Mock LLM provider, exposed as an MCP tool (FastMCP, the SDK's
high-level API) rather than a raw HTTP/SSE endpoint. Standing in for a
real completions API.

MCP's mechanism for streaming intermediate output during a single
tool call is progress notifications: a tool can call
ctx.report_progress(progress, total, message) repeatedly while it
runs, and a client that passed a progress_callback to call_tool()
receives each one in real time, before the tool's final result comes
back. generate() below uses `message` to carry each raw (unredacted)
text delta, simulating token-by-token generation with a short delay
between deltas -- gateway.py is what actually turns this into a
redacted stream for ITS OWN client.

The scenarios exist to stress gateway.py's streaming redactor against
chunk boundaries a real token-by-token LLM stream would produce,
including deliberately awkward ones -- PII split across two chunks,
three chunks, and (worst case) one character at a time.

Usage:
    python3 llm_provider.py [--port 9100]
"""

import argparse
import asyncio
import logging
import sys

from mcp.server.fastmcp import Context, FastMCP

logging.basicConfig(
    level=logging.INFO,
    stream=sys.stderr,
    format="%(asctime)s %(levelname)s provider: %(message)s",
)
log = logging.getLogger("provider")

# Each scenario is a list of delta strings streamed in order.
# Concatenating a scenario's deltas reproduces the full response text.
SCENARIOS: dict[str, list[str]] = {
    "plain": [
        "The ",
        "weather ",
        "today ",
        "is ",
        "sunny ",
        "with ",
        "a ",
        "high ",
        "of ",
        "72",
        " degrees.",
    ],
    # Email split across 2 chunks, SSN split across 3, credit card
    # split across 2 -- interleaved with plain text, matching how a
    # real streamed response mixes generated prose with the PII
    # incidentally contained in it.
    "pii_mixed": [
        "Thanks for reaching out. Please contact ",
        "supp",
        "ort@ex",
        "ample.co",
        "m",
        " for help. Your records show SSN ",
        "123-",
        "45-",
        "6789",
        " on file. The card ending was ",
        "4111-1111-",
        "1111-1111",
        ". Let us know if you need anything else.",
    ],
    # Worst case: an email fed one character at a time, proving the
    # holdback logic doesn't depend on PII landing on any particular
    # chunk granularity.
    "pii_char_by_char": [
        "Reply to: ",
        *list("zzuser@worstcase.io"),
        " when ready.",
    ],
    # Forces a multi-second stall before the first chunk, to make TTFT
    # and streamed-vs-buffered behavior directly observable in client.py.
    "slow_start": [
        "__SLEEP:2.0__",
        "Finally, ",
        "here's ",
        "your ",
        "answer.",
    ],
}

CHUNK_DELAY_SECONDS = 0.03

mcp = FastMCP("llm-provider", host="127.0.0.1", port=9100)


@mcp.tool(description="Generate text for a prompt, streaming deltas via progress notifications.")
async def generate(prompt: str, scenario: str, ctx: Context) -> str:
    deltas = SCENARIOS.get(scenario)
    if deltas is None:
        raise ValueError(f"unknown scenario: {scenario!r}")

    full_text_parts: list[str] = []
    step = 0
    total = len(deltas)

    for delta in deltas:
        if delta.startswith("__SLEEP:"):
            await asyncio.sleep(float(delta.split(":")[1].rstrip("_")))
            continue
        step += 1
        full_text_parts.append(delta)
        await ctx.report_progress(progress=step, total=total, message=delta)
        await asyncio.sleep(CHUNK_DELAY_SECONDS)

    return "".join(full_text_parts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=9100)
    args = parser.parse_args()
    mcp.settings.port = args.port

    log.info("listening on 127.0.0.1:%d/mcp", args.port)
    mcp.run(transport="streamable-http")


if __name__ == "__main__":
    main()
