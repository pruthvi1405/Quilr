#!/usr/bin/env python3
"""
Simple MCP client for exercising server.py, built on the official SDK's
client API (mcp.client.stdio.stdio_client + mcp.ClientSession) -- the
intended way to talk to an MCP server, as opposed to harness.py's raw
JSON-RPC-over-stdio probing.

Spawns server.py as a subprocess, does the initialize handshake, lists
the published tools, then calls each tool with a valid input and a
couple of invalid ones, printing what comes back.

Usage:
    python3 client.py
"""

import asyncio
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


def _print_result(label: str, result) -> None:
    marker = "ERROR" if result.isError else "OK"
    text = "".join(block.text for block in result.content if block.type == "text")
    print(f"[{marker}] {label}\n    {text}\n")


async def main() -> None:
    params = StdioServerParameters(command=sys.executable, args=["server.py"])

    async with stdio_client(params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            tools = await session.list_tools()
            print("Tools published by the server:")
            for tool in tools.tools:
                print(f"  - {tool.name}: {tool.description}")
            print()

            result = await session.call_tool("get_customer_record", {"customer_id": "CUST-00042"})
            _print_result("get_customer_record, valid customer", result)

            result = await session.call_tool("get_customer_record", {"customer_id": "CUST-99999"})
            _print_result("get_customer_record, unknown customer", result)

            result = await session.call_tool("get_customer_record", {"customer_id": "not-a-valid-id"})
            _print_result("get_customer_record, malformed customer_id", result)

            result = await session.call_tool(
                "trigger_refund",
                {"customer_id": "CUST-00042", "amount": 100.0, "reason": "duplicate charge on invoice 8891"},
            )
            _print_result("trigger_refund, valid", result)

            result = await session.call_tool(
                "trigger_refund",
                {"customer_id": "CUST-00042", "amount": 100.0, "reason": "oops"},
            )
            _print_result("trigger_refund, reason too short", result)

            result = await session.call_tool(
                "trigger_refund",
                {"customer_id": "CUST-00042", "amount": -5.0, "reason": "duplicate charge on invoice 8891"},
            )
            _print_result("trigger_refund, negative amount", result)

            result = await session.call_tool(
                "trigger_refund",
                {"customer_id": "CUST-00001", "amount": 999999.0, "reason": "duplicate charge on invoice 8891"},
            )
            _print_result("trigger_refund, exceeds balance", result)

            # Idempotency: same key + same arguments replays the first
            # result instead of refunding twice.
            idem_args = {
                "customer_id": "CUST-00042",
                "amount": 50.0,
                "reason": "idempotency check via client.py",
                "idempotency_key": "client-demo-1",
            }
            first = await session.call_tool("trigger_refund", idem_args)
            replay = await session.call_tool("trigger_refund", idem_args)
            _print_result("trigger_refund, idempotent replay (first call)", first)
            _print_result("trigger_refund, idempotent replay (second call, same key+args)", replay)


if __name__ == "__main__":
    asyncio.run(main())
