#!/usr/bin/env python3
"""
MCP client for exercising gateway.py, built on the official SDK's
client transport (mcp.client.streamable_http.streamable_http_client +
mcp.ClientSession) with a progress_callback -- the high-level,
spec-compliant way to receive a tool's intermediate output in real
time, as opposed to only seeing its final return value.

Two layers:
  1. Unit-level checks against StreamRedactor directly (redactor.py,
     no network) -- including feeding PII one character at a time,
     the worst-case chunk fragmentation a real stream could produce.
  2. End-to-end checks over real MCP: calls gateway.py's `generate`
     tool (which itself calls llm_provider.py's `generate` tool and
     redacts each delta as it arrives) and checks both correctness
     (PII redacted, ordinary text untouched, no raw PII ever visible
     in a single streamed chunk) and the streaming properties the task
     asks for: a prompt first chunk (low TTFT), chunks spread out over
     time rather than delivered in one burst, and a deliberately
     slow-starting upstream response staying visibly slow to the
     client rather than being hidden behind full buffering.

Points at the GATEWAY's URL, never at llm_provider.py directly.

Usage:
    python3 client.py [--gateway http://127.0.0.1:8090/mcp]
"""

import argparse
import asyncio
import sys
import time
from contextlib import asynccontextmanager

import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

from redactor import PII_PATTERN, StreamRedactor

failures = 0
rows: list[tuple[str, bool, str]] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    global failures
    rows.append((label, condition, detail))
    if not condition:
        failures += 1


# ---------------------------------------------------------------
# Unit-level: StreamRedactor in isolation
# ---------------------------------------------------------------


def run_unit_checks() -> None:
    r = StreamRedactor()
    out = r.feed("Contact us at ") + r.feed("supp") + r.feed("ort@ex") + r.feed("ample.co") + r.feed("m") + r.feed(" soon.")
    out += r.flush()
    check("unit: email split across 5 feed() calls is redacted", out == "Contact us at [REDACTED] soon.", repr(out))

    r = StreamRedactor()
    out = ""
    for ch in "SSN: 123-45-6789 done.":
        out += r.feed(ch)
    out += r.flush()
    check("unit: SSN fed one character at a time is redacted", out == "SSN: [REDACTED] done.", repr(out))

    r = StreamRedactor()
    out = ""
    for ch in "Card 4111-1111-1111-1111 charged.":
        out += r.feed(ch)
    out += r.flush()
    check("unit: credit card fed one character at a time is redacted", out == "Card [REDACTED] charged.", repr(out))

    r = StreamRedactor()
    out = r.feed("Nothing sensitive here, just ") + r.feed("plain conversational text.")
    out += r.flush()
    check(
        "unit: plain text passes through unchanged",
        out == "Nothing sensitive here, just plain conversational text.",
        repr(out),
    )

    r = StreamRedactor()
    partials = []
    for ch in "email me: someone@example.org thanks":
        piece = r.feed(ch)
        if piece:
            partials.append(piece)
    partials.append(r.flush())
    assembled = "".join(partials)
    check(
        "unit: incrementally-assembled output matches flush-based redaction",
        assembled == "email me: [REDACTED] thanks",
        repr(assembled),
    )
    check(
        "unit: no individual feed() piece contains a raw PII match",
        all(not PII_PATTERN.search(p) for p in partials if p),
        repr(partials),
    )

    r = StreamRedactor()
    for _ in range(5000):
        r.feed("word ")
    check(
        "unit: internal buffer stays bounded on a long plain stream",
        len(r._buffer) <= StreamRedactor.HOLDBACK + len("word "),
        f"buffer length = {len(r._buffer)}",
    )


# ---------------------------------------------------------------
# End-to-end: real MCP calls against gateway.py -> llm_provider.py
# ---------------------------------------------------------------


@asynccontextmanager
async def _connect(url: str):
    async with httpx.AsyncClient() as http_client:
        async with streamable_http_client(url, http_client=http_client) as (read, write, _get_session_id):
            async with ClientSession(read, write) as session:
                await session.initialize()
                yield session


async def _generate(gateway_url: str, scenario: str, prompt: str = "hi") -> tuple[str, list[tuple[float, str]]]:
    """Calls the gateway's `generate` tool, collecting every streamed
    delta with its arrival time via a progress_callback. Returns
    (final_text, [(elapsed_seconds, delta), ...])."""
    start = time.monotonic()
    events: list[tuple[float, str]] = []

    async def on_progress(progress: float, total: float | None, message: str | None) -> None:
        if message:
            events.append((time.monotonic() - start, message))

    async with _connect(gateway_url) as session:
        result = await session.call_tool(
            "generate", {"prompt": prompt, "scenario": scenario}, progress_callback=on_progress
        )

    final_text = "".join(block.text for block in result.content if block.type == "text")
    return final_text, events


async def run_e2e_checks(gateway_url: str) -> None:
    text, events = await _generate(gateway_url, "plain")
    check(
        "e2e: plain-text scenario passes through unmodified",
        text == "The weather today is sunny with a high of 72 degrees.",
        repr(text),
    )
    check("e2e: plain scenario streamed more than one chunk", len(events) > 1, f"{len(events)} chunk(s)")

    text, events = await _generate(gateway_url, "pii_mixed")
    expected = (
        "Thanks for reaching out. Please contact [REDACTED] for help. "
        "Your records show SSN [REDACTED] on file. The card ending was "
        "[REDACTED]. Let us know if you need anything else."
    )
    check("e2e: mixed PII scenario fully redacted", text == expected, repr(text))
    check(
        "e2e: no output chunk in mixed scenario ever carried raw PII",
        all(not PII_PATTERN.search(d) for _, d in events),
        repr([d for _, d in events]),
    )

    text, events = await _generate(gateway_url, "pii_char_by_char")
    check("e2e: char-by-char email split fully redacted", text == "Reply to: [REDACTED] when ready.", repr(text))
    check(
        "e2e: no output chunk in char-by-char scenario ever carried raw PII",
        all(not PII_PATTERN.search(d) for _, d in events),
        repr([d for _, d in events]),
    )

    # Streaming responsiveness: the provider paces "plain" at ~30ms
    # between 11 deltas (~300ms total). If the gateway buffered the
    # whole response before responding, every progress notification
    # (or the single final result) would arrive at ~the same instant,
    # near the end. Streamed correctly, the first chunk arrives
    # promptly and later chunks are spread out, not clustered together.
    text, events = await _generate(gateway_url, "plain")
    first_chunk_time = events[0][0] if events else None
    last_chunk_time = events[-1][0] if events else None
    span = (last_chunk_time - first_chunk_time) if events else 0
    check(
        "e2e: first chunk arrives promptly (low TTFT)",
        first_chunk_time is not None and first_chunk_time < 0.2,
        f"first chunk at {first_chunk_time}",
    )
    check(
        "e2e: chunks are spread over time, not delivered all at once",
        span > 0.15,
        f"span between first and last chunk = {span:.3f}s over {len(events)} chunks",
    )

    # slow_start: provider stalls ~2s before its first delta. If the
    # gateway buffered the whole response, this would be
    # indistinguishable from a fast response timing-wise; streamed
    # correctly, the client sees the same ~2s gap the provider itself
    # introduces before anything arrives.
    text, events = await _generate(gateway_url, "slow_start")
    check(
        "e2e: slow-start scenario's delay is visible to the client, not hidden by buffering",
        bool(events) and events[0][0] > 1.5,
        f"first chunk at {events[0][0] if events else None}",
    )
    check("e2e: slow-start scenario still delivers correct text", text == "Finally, here's your answer.", repr(text))


def run(gateway_url: str) -> int:
    run_unit_checks()
    asyncio.run(run_e2e_checks(gateway_url))

    print("=" * 95)
    print("REDACTION GATEWAY TEST RESULTS")
    print("=" * 95)
    for label, ok, detail in rows:
        print(f"{'PASS' if ok else 'FAIL':<5} {label:<65} {detail if not ok else ''}")

    print()
    print(f"RESULT: {'ALL PASS' if failures == 0 else f'{failures} FAILURE(S)'}")
    return 0 if failures == 0 else 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gateway", default="http://127.0.0.1:8090/mcp")
    args = parser.parse_args()
    sys.exit(run(args.gateway))


if __name__ == "__main__":
    main()
