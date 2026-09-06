# Task 2: MCP Security Gateway Proxy

A reverse proxy that sits between an MCP client and a downstream MCP
server, enforcing role-based access on tool calls. Built on the real
MCP Streamable HTTP transport throughout — the client, the gateway,
and the downstream server all speak the actual protocol, not a
simplified stand-in.

```
┌─────────────┐  Streamable HTTP   ┌──────────────┐  Streamable HTTP   ┌──────────────────────┐
│  client.py  │ ─────────────────▶ │  gateway.py  │ ─────────────────▶ │ downstream_server.py │
│ (MCP SDK    │  Authorization:    │  (Starlette  │  no Authorization  │  (FastMCP server)     │
│  client)    │  Bearer <token>    │   + httpx)   │  header forwarded  │                       │
└─────────────┘ ◀───────────────── └──────────────┘ ◀───────────────── └──────────────────────┘
                  JSON-RPC over          :8080              JSON-RPC over          :9000
                  POST/GET/DELETE                           POST/GET/DELETE
                  to /mcp                                   to /mcp
```

- **`downstream_server.py`** — a plain FastMCP server with no concept
  of roles or auth; it trusts every caller. Exposes `get_weather`
  (public) and `admin_reset_key` / `admin_delete_user` (admin-only by
  naming convention).
- **`gateway.py`** — the actual security boundary. Reads the caller's
  role from `Authorization: Bearer <token>`. `tools/list` and any
  non-admin `tools/call` pass through untouched, including a streamed
  `text/event-stream` response. A `tools/call` naming an `admin_*`
  tool is checked against the caller's role; if unauthorized, the
  gateway answers with a JSON-RPC `-32001` error itself and the
  downstream server is never contacted. The client's own
  `Authorization` header is never forwarded downstream.
- **`client.py`** — a real MCP client (the official SDK's
  `streamablehttp_client` + `ClientSession`), pointed at the gateway.
  Exercises the whole flow and asserts on the outcomes.

## Run it

Three processes, in order.

**1. Install dependencies** (from the repo root, shared venv):

```bash
source .venv/bin/activate
uv pip install -r Task2/requirements.txt
```

**2. Start the downstream MCP server** (terminal 1):

```bash
cd Task2
python3 downstream_server.py --port 9000
```

**3. Start the gateway** (terminal 2):

```bash
cd Task2
python3 gateway.py --port 8080 --downstream http://127.0.0.1:9000/mcp
```

**4. Run the client** (terminal 3 — this is what exercises everything):

```bash
cd Task2
python3 client.py --gateway http://127.0.0.1:8080/mcp
```

Expected output: six `PASS` lines and `ALL PASS`. The client connects
through the gateway (never directly to the downstream server), lists
tools, calls the public `get_weather` tool, calls `admin_reset_key`
with an admin token, then tries the same admin call with a viewer
token and with no token at all — both must come back as a genuine
JSON-RPC `-32001` error (surfaced by the SDK as `McpError`), not a
tool-level failure.

## Scope note

This implements the literal spec: bearer token → role, `admin_`
prefix check, `tools/list` passthrough, `-32001` on rejection. An
earlier iteration of this gateway (built without a real downstream MCP
server, over plain HTTP/JSON) also handled a catalog-based allowlist
(rather than a per-request prefix check), JSON-RPC notifications,
request body size caps, and a timing-safe token comparison. None of
that hardening is in this version.
