# Task 3: LLM Gateway Streaming Guardrail (PII Redaction)

An LLM gateway that streams a text-generation response back to its
caller in real time, redacting PII (emails, SSNs, credit card numbers)
from the stream as it passes through — without ever buffering the
full response in memory. Built entirely on high-level MCP: the
provider, the gateway, and the client all speak the real MCP
Streamable HTTP transport, and "streaming" is MCP's own mechanism for
it — progress notifications during a tool call — not a hand-rolled
SSE proxy.

```
┌─────────────┐   MCP tools/call    ┌──────────────┐   MCP tools/call    ┌────────────────────┐
│  client.py  │ ──────────────────▶ │  gateway.py  │ ──────────────────▶ │  llm_provider.py    │
│ (ClientSession│  progress_callback │ (FastMCP srv  │  progress_callback  │  (FastMCP server)    │
│  + progress  │ ◀────────────────── │  + MCP client │ ◀────────────────── │  simulates token-by- │
│  callback)   │  redacted deltas,   │  to upstream) │  raw deltas via     │  token generation     │
└─────────────┘  streamed live      └──────────────┘  report_progress    └────────────────────┘
                                        :8090                                    :9100
```

## Why progress notifications, not SSE

The original task statement describes an HTTP/SSE proxy. This version
was rebuilt to use high-level MCP only. MCP's actual mechanism for a
single tool call to stream intermediate output to its caller, in real
time, before the call finishes, is `ctx.report_progress(progress,
total, message)` on the server side paired with a `progress_callback`
passed to `ClientSession.call_tool()` on the client side — a
notification for every delta, dispatched to the callback the instant
it arrives, well before the tool's final return value comes back. That
is what "streaming" means here, and it's genuinely real-time: nothing
in this design waits for the tool call to finish before delivering the
first chunk.

## The three pieces

- **`llm_provider.py`** — a plain FastMCP server with one tool,
  `generate(prompt, scenario)`. It has no idea PII redaction exists;
  it just simulates token-by-token generation by emitting one
  `report_progress` call per delta (with a short delay between them,
  to mimic real generation latency) and returning the full text as its
  final result. The `scenario` argument selects a canned response —
  including ones that deliberately split an email, SSN, and credit
  card number across multiple deltas, down to one character per delta
  in the worst case — used to stress-test the gateway's redaction
  against exactly the chunk boundaries a real streaming LLM would
  produce.
- **`gateway.py`** — the actual guardrail, and the only piece with any
  redaction logic. Its own `generate` tool is, for the duration of
  each call, simultaneously an MCP *client* to `llm_provider.py`: it
  calls the provider's `generate` with a `progress_callback`, and the
  moment a raw delta arrives, feeds it into `StreamRedactor`
  (`redactor.py`) and forwards whatever comes back as safe-to-emit to
  its *own* caller via its *own* `ctx.report_progress` — before the
  upstream call has finished. Once upstream completes, the redactor is
  flushed and the fully-redacted, complete text is returned as the
  gateway tool's own result, so a caller that only reads the final
  return value also gets clean text.
- **`redactor.py`** — the actual streaming-redaction algorithm,
  transport-agnostic (it doesn't know or care whether deltas arrive
  via MCP progress notifications, SSE, or anything else). See its
  module docstring for the full reasoning; summarized below.
- **`client.py`** — a real MCP client (`streamable_http_client` +
  `ClientSession.call_tool(..., progress_callback=...)`), pointed at
  the gateway. Runs fast unit-level checks against `StreamRedactor`
  directly, then drives the whole stack end to end and checks both
  correctness and the streaming properties the task asks for.

## How the redactor actually works

Two distinct problems, solved together:

1. **PII can straddle a chunk boundary.** `"user@example.com"`
   arriving as `"user@exam"` then `"ple.com"` means neither chunk
   alone contains a complete match. Fix: don't emit the trailing part
   of the buffer that might still be a forming match — hold it back,
   append the next chunk to it, and re-scan.

2. **A greedy, open-ended pattern can commit to a match too early.**
   The email TLD pattern (`[A-Za-z]{2,}`) has no upper bound. If the
   buffer happens to end right after `"...@example.co"`, the regex
   engine has no way to know a trailing `"m"` is one chunk away — it
   matches `"example.co"` as a complete, valid email *right now*.
   Redacting on that basis fires one character too early, and the
   `"m"` leaks out as plain text on the next chunk. (The same applies
   to `\b` word-boundary anchors on the SSN/card patterns: Python's
   `\b` is satisfied by the end of the search string, so a digit run
   that happens to end at the buffer's edge looks "bounded" even
   though more digits — which would break the boundary — might be one
   chunk away.) Fix: a match is only trusted if it ends strictly
   *before* the end of the current buffer — proof the regex engine had
   the chance to extend it and didn't, not just that input ran out. A
   match sitting exactly at the buffer's edge is left raw until either
   more data arrives to settle it, or the stream ends.

Responsiveness matters just as much as correctness here, so the
holdback is **not** a flat "always keep the last N characters." For
ordinary prose, almost everything is emitted immediately — there's
nothing at the tail that could plausibly become PII. Only the trailing
run of characters that could still be part of an in-progress match
(letters, digits, and the punctuation these three pattern types use)
is actually held back, capped at 128 characters as a hard upper bound.
A short response with no PII near the end streams out essentially as
fast as it arrives.

Both of the above were real bugs caught while building this, not
hypotheticals — an earlier version genuinely leaked a trailing
character past a redaction, and separately, a flat 128-character
holdback meant any response shorter than that never streamed at all
(it all came out in one shot at the very end). `client.py`'s
`e2e: first chunk arrives promptly` and `chunks are spread over time`
checks exist specifically to catch a regression back to the second
one.

## Run it

Three processes, in order.

**1. Install dependencies** (from the repo root, shared venv):

```bash
source .venv/bin/activate
uv pip install -r Task3/requirements.txt
```

**2. Start the LLM provider** (terminal 1):

```bash
cd Task3
python3 llm_provider.py --port 9100
```

**3. Start the gateway** (terminal 2):

```bash
cd Task3
python3 gateway.py --port 8090 --upstream http://127.0.0.1:9100/mcp
```

**4. Run the client** (terminal 3 — this exercises everything):

```bash
cd Task3
python3 client.py --gateway http://127.0.0.1:8090/mcp
```

Expected output: 17 `PASS` lines and `ALL PASS` — 7 unit-level checks
against `StreamRedactor` in isolation, and 10 end-to-end checks
(correctness of redaction across all three PII types and chunk
splits, no raw PII ever visible in a single streamed chunk, low
time-to-first-chunk, chunks arriving spread out over time rather than
in one burst, and a deliberately slow-starting response staying
visibly slow to the client rather than being hidden by buffering).

## Scope note

The regexes are practical, not exhaustive: the email pattern isn't
fully RFC 5322-compliant, the credit-card pattern matches major
network BIN prefixes (Visa/Mastercard/Amex/Discover) rather than
Luhn-validating every 13–19-digit run, and the SSN pattern only
matches the dashed `XXX-XX-XXXX` form. None of that changes the
streaming design, which is what this task is actually about.
