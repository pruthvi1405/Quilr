#!/usr/bin/env python3
"""
Customer support MCP server, built on FastMCP (mcp.server.fastmcp) --
the SDK's high-level API. Each tool is a plain async function with
type-hinted parameters; FastMCP derives both the published JSON Schema
and the Pydantic validation from the function signature automatically,
and mcp.run(transport="stdio") handles the stdio transport and
lifecycle -- no manual protocol/transport wiring.

Validation failures come back as CallToolResult(isError=True) (an
in-band tool result), not a top-level JSON-RPC error. That's FastMCP's
own behavior: its tools/call dispatch wraps every exception raised
while handling a call -- including a Pydantic validation failure --
into an in-band error result rather than a protocol-level one. Left as
the SDK's default here rather than overridden.

Usage:
    python3 server.py
"""

import logging
import sys
from decimal import Decimal, InvalidOperation
from typing import Annotated, Any

from mcp.server.fastmcp import FastMCP
from pydantic import BeforeValidator, Field

logging.basicConfig(
    level=logging.INFO,
    stream=sys.stderr,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger("task1")

mcp = FastMCP("customer-support")


# ---------------------------------------------------------
# Shared field types
# ---------------------------------------------------------

# [0-9] rather than \d: \d also matches non-ASCII Unicode decimal
# digits (e.g. Arabic-Indic), which would otherwise slip past the
# pattern check.
CustomerId = Annotated[
    str,
    Field(
        pattern=r"^CUST-[0-9]{5}$",
        description="Customer ID in the form CUST-XXXXX (exactly 5 digits), e.g. CUST-00042",
    ),
]


def _coerce_amount(v: object) -> Decimal:
    # Decimal, not float: an accepted refund amount must be exactly
    # reflected in the balance, not rounded. Accepts int/float (JSON
    # numbers) but not strings -- no silent str -> number coercion --
    # and explicitly rejects bool (Python's bool is an int subclass).
    if isinstance(v, bool):
        raise ValueError("amount must be a number, not a boolean")
    if isinstance(v, Decimal):
        d = v
    elif isinstance(v, (int, float)):
        try:
            d = Decimal(str(v))
        except InvalidOperation as exc:
            raise ValueError(f"amount is not a valid number: {v!r}") from exc
    else:
        raise ValueError("amount must be a number")
    if not d.is_finite():
        raise ValueError("amount must be finite")
    return d


RefundAmount = Annotated[
    Decimal,
    BeforeValidator(_coerce_amount),
    Field(gt=0, description="Refund amount in USD, must be greater than 0"),
]

RefundReason = Annotated[
    str,
    BeforeValidator(lambda v: v.strip() if isinstance(v, str) else v),
    Field(min_length=10, max_length=500, description="Why the refund is being issued, at least 10 characters"),
]

IdempotencyKey = Annotated[
    str | None,
    Field(
        min_length=1,
        max_length=200,
        description="Optional. Repeating a call with the same key returns the original result instead of issuing a second refund.",
    ),
]


# ---------------------------------------------------------
# Mock customer database
# ---------------------------------------------------------

CUSTOMERS: dict[str, dict[str, Any]] = {
    "CUST-00001": {
        "customer_id": "CUST-00001",
        "name": "Ada Lovelace",
        "plan": "pro",
        "balance": Decimal("250.00"),
    },
    "CUST-00042": {
        "customer_id": "CUST-00042",
        "name": "Grace Hopper",
        "plan": "enterprise",
        "balance": Decimal("1200.50"),
    },
}

_refund_seq = 0
# idempotency_key -> ((customer_id, amount, reason), result_dict)
_refund_idempotency_cache: dict[str, tuple[tuple[Any, ...], dict]] = {}

_CENTS = Decimal("0.01")


def _for_display(amount: Decimal) -> Decimal:
    # Pad a short-precision amount up to 2 decimal places for display
    # (Decimal("17.0") -> Decimal("17.00")) without ever truncating a
    # genuinely higher-precision (sub-cent) amount.
    if amount.as_tuple().exponent <= _CENTS.as_tuple().exponent:
        return amount
    return amount.quantize(_CENTS)


# ---------------------------------------------------------
# Tools
# ---------------------------------------------------------


@mcp.tool(description="Fetch the full record for one customer.")
async def get_customer_record(customer_id: CustomerId) -> dict:
    record = CUSTOMERS.get(customer_id)
    if record is None:
        raise ValueError(f"No customer found with id {customer_id}")
    return {**record, "balance": str(record["balance"])}


@mcp.tool(description="Issue a refund to a customer.")
async def trigger_refund(
    customer_id: CustomerId,
    amount: RefundAmount,
    reason: RefundReason,
    idempotency_key: IdempotencyKey = None,
) -> dict:
    global _refund_seq

    signature = (customer_id, amount, reason)

    if idempotency_key is not None:
        cached = _refund_idempotency_cache.get(idempotency_key)
        if cached is not None:
            cached_signature, cached_result = cached
            if cached_signature != signature:
                raise ValueError(f"Idempotency key {idempotency_key!r} was already used with different arguments")
            return cached_result

    record = CUSTOMERS.get(customer_id)
    if record is None:
        raise ValueError(f"No customer found with id {customer_id}")

    if amount > record["balance"]:
        raise ValueError(f"Refund {amount:.2f} exceeds balance {record['balance']:.2f}")

    _refund_seq += 1
    record["balance"] = record["balance"] - amount

    result = {
        "status": "refund_issued",
        "refund_id": f"REF-{_refund_seq:05d}",
        "amount": str(_for_display(amount)),
        "remaining_balance": str(record["balance"]),
    }

    if idempotency_key is not None:
        _refund_idempotency_cache[idempotency_key] = (signature, result)

    return result


if __name__ == "__main__":
    log.info("Starting customer-support MCP server")
    mcp.run(transport="stdio")
