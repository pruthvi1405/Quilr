#!/usr/bin/env python3
"""
Downstream MCP server that gateway.py sits in front of. A real MCP
server (built on FastMCP, the SDK's high-level API), speaking the
actual Streamable HTTP transport -- not a hand-rolled JSON mock. This
is what makes it possible to test the gateway with a real MCP client
(client.py): the client only ever talks MCP, whether it's pointed
directly at this server or through the gateway.

Exposes:
  - get_weather        public tool, anyone may call it.
  - admin_reset_key     admin-only by naming convention (admin_ prefix).
  - admin_delete_user   admin-only by naming convention.

The admin_ prefix carries no special meaning to this server itself --
it's just a naming convention gateway.py's authorization logic keys
off of. This server has no concept of roles or tokens at all; it
trusts whoever is calling it, which is exactly why it must sit behind
the gateway rather than be exposed directly.

Usage:
    python3 downstream_server.py [--port 9000]
"""

import argparse
import logging
import sys

from mcp.server.fastmcp import FastMCP

logging.basicConfig(
    level=logging.INFO,
    stream=sys.stderr,
    format="%(asctime)s %(levelname)s downstream: %(message)s",
)
log = logging.getLogger("downstream")


def build_server(port: int) -> FastMCP:
    mcp = FastMCP("downstream", host="127.0.0.1", port=port)

    @mcp.tool(description="Get the current weather for a city.")
    async def get_weather(city: str) -> dict:
        return {"city": city, "forecast": "sunny", "high_f": 72}

    @mcp.tool(description="Rotate a customer's API key.")
    async def admin_reset_key(customer_id: str) -> dict:
        return {"customer_id": customer_id, "status": "key_rotated", "new_key": f"key-{customer_id}-rotated"}

    @mcp.tool(description="Permanently delete a user account.")
    async def admin_delete_user(user_id: str) -> dict:
        return {"user_id": user_id, "status": "deleted"}

    return mcp


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=9000)
    args = parser.parse_args()

    log.info("listening on 127.0.0.1:%d/mcp", args.port)
    build_server(args.port).run(transport="streamable-http")


if __name__ == "__main__":
    main()
