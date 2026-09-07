#!/usr/bin/env python3
"""
Test harness for the Task 4 rate limiter + fallback router.

Two layers:
  1. Unit-level checks against SqliteSlidingWindowLimiter directly (a
     temp on-disk db, no HTTP), including firing many concurrent
     admission checks at once to catch the classic check-then-act race
     a naive "read usage, then decide, then write" implementation
     would be vulnerable to.
  2. End-to-end checks over real HTTP: starts two mock_model_provider.py
     instances (primary + secondary) and router.py as actual
     subprocesses, and exercises normal routing, 429-triggered
     failover, timeout-triggered failover (with a timing assertion
     that the primary's 3000ms timeout actually cuts it off rather
     than waiting out a slower failure), total-failure error
     sanitization, and the gateway's own rate limit -- including a
     concurrent-requests version of the race check, this time through
     the full HTTP + asyncio stack.

Usage:
    python3 harness.py
"""

import asyncio
import json
import os
import subprocess
import sys
import tempfile
import time

import aiohttp

from rate_limiter import SqliteSlidingWindowLimiter

PRIMARY_PORT = 9201
SECONDARY_PORT = 9202
ROUTER_PORT = 8100

PRIMARY_URL = f"http://127.0.0.1:{PRIMARY_PORT}"
SECONDARY_URL = f"http://127.0.0.1:{SECONDARY_PORT}"
ROUTER_URL = f"http://127.0.0.1:{ROUTER_PORT}"

failures = 0
rows: list[tuple[str, bool, str]] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    global failures
    rows.append((label, condition, detail))
    if not condition:
        failures += 1


# ---------------------------------------------------------------
# Unit-level: SqliteSlidingWindowLimiter in isolation
# ---------------------------------------------------------------


async def run_limiter_unit_checks() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        db_path = os.path.join(tmp, "unit.db")
        limiter = SqliteSlidingWindowLimiter(db_path, limit_tokens=1000, window_seconds=60.0)

        r1 = await limiter.check_and_consume("tenant-a", 400, now=1000.0)
        r2 = await limiter.check_and_consume("tenant-a", 400, now=1001.0)
        check("unit: sequential admits within budget succeed", r1.allowed and r2.allowed, f"{r1} {r2}")

        r3 = await limiter.check_and_consume("tenant-a", 300, now=1002.0)
        check(
            "unit: request that would exceed the limit is denied",
            not r3.allowed,
            f"{r3}",
        )
        check("unit: denied request reports a positive retry_after", r3.retry_after_seconds > 0, f"{r3.retry_after_seconds}")

        r4 = await limiter.check_and_consume("tenant-a", 200, now=1002.0)
        check(
            "unit: a smaller request that fits remaining budget still succeeds",
            r4.allowed and r4.remaining_tokens == 0,
            f"{r4}",
        )

        r5 = await limiter.check_and_consume("tenant-b", 1000, now=1002.0)
        check("unit: a different tenant has an independent budget", r5.allowed, f"{r5}")

        # Sliding window eviction: once the earliest admitted request
        # (at now=1000.0, 400 tokens) is more than window_seconds in
        # the past, its tokens must no longer count -- not "wait for a
        # fixed window boundary to reset everyone at once".
        usage_before = await limiter.current_usage("tenant-a", now=1002.0)
        later = 1000.0 + 60.0 + 0.001
        # At `later`, the now-60s-old first entry (400 tokens @
        # t=1000.0) has aged out of the window, leaving 600 (400 @
        # t=1001.0 + 200 @ t=1002.0). That frees enough room for this
        # 400-token request to be admitted even though the tenant was
        # previously at the limit -- proof eviction actually happened,
        # not just that admission math tolerates going over.
        r6 = await limiter.check_and_consume("tenant-a", 400, now=later)
        usage_after = await limiter.current_usage("tenant-a", now=later)
        check(
            "unit: the oldest entry ages out of the window, freeing room for a new admit",
            usage_before == 1000 and r6.allowed and usage_after == 600 + 400,
            f"before={usage_before} r6={r6} after={usage_after}",
        )

        # Background purge sweeps expired rows for tenants that have
        # gone quiet, not just the tenant in the current request path.
        deleted = await limiter.purge_expired(now=1000.0 + 60.0 + 1000.0)
        check("unit: purge_expired evicts rows across all tenants", deleted >= 2, f"deleted={deleted}")

        # --- concurrency: the actual race-condition check -----------
        limiter2 = SqliteSlidingWindowLimiter(
            os.path.join(tmp, "race.db"), limit_tokens=50_000, window_seconds=60.0
        )
        results = await asyncio.gather(
            *(limiter2.check_and_consume("tenant-race", 1_000, now=2000.0) for _ in range(60))
        )
        admitted = [r for r in results if r.allowed]
        denied = [r for r in results if not r.allowed]
        total_admitted_tokens = len(admitted) * 1_000
        check(
            "unit: 60 concurrent 1000-token requests against a 50000 limit admit exactly 50",
            len(admitted) == 50 and len(denied) == 10,
            f"admitted={len(admitted)} denied={len(denied)}",
        )
        check(
            "unit: concurrent admission never exceeds the configured limit (no over-admission race)",
            total_admitted_tokens <= 50_000,
            f"total_admitted_tokens={total_admitted_tokens}",
        )
        final_usage = await limiter2.current_usage("tenant-race", now=2000.0)
        check(
            "unit: final recorded usage matches exactly what was admitted",
            final_usage == total_admitted_tokens,
            f"final_usage={final_usage} expected={total_admitted_tokens}",
        )


# ---------------------------------------------------------------
# End-to-end: real HTTP against router.py + two mock providers
# ---------------------------------------------------------------


async def _wait_healthy(session: aiohttp.ClientSession, url: str, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    last_error = None
    while time.monotonic() < deadline:
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=0.5)) as resp:
                await resp.read()
                return
        except Exception as exc:  # noqa: BLE001 - just polling for readiness
            last_error = exc
            await asyncio.sleep(0.1)
    raise RuntimeError(f"{url} never became healthy: {last_error}")


async def _set_fault(session: aiohttp.ClientSession, base_url: str, mode: str) -> None:
    async with session.post(f"{base_url}/_set_fault", json={"mode": mode}) as resp:
        await resp.json()


async def _call_count(session: aiohttp.ClientSession, base_url: str) -> int:
    async with session.get(f"{base_url}/_calls") as resp:
        return len((await resp.json())["calls"])


async def _completion(session: aiohttp.ClientSession, tenant: str, tokens: int, prompt: str = "hi") -> tuple[int, dict]:
    async with session.post(
        f"{ROUTER_URL}/v1/completions",
        json={"prompt": prompt, "tokens": tokens},
        headers={"Authorization": f"Bearer {tenant}"},
    ) as resp:
        return resp.status, await resp.json()


async def run_e2e_checks() -> None:
    async with aiohttp.ClientSession() as session:
        await _wait_healthy(session, f"{PRIMARY_URL}/health")
        await _wait_healthy(session, f"{SECONDARY_URL}/health")
        await _wait_healthy(session, f"{ROUTER_URL}/health")

        # --- normal path: primary handles it, secondary untouched ---
        await _set_fault(session, PRIMARY_URL, "none")
        await _set_fault(session, SECONDARY_URL, "none")
        primary_before, secondary_before = await _call_count(session, PRIMARY_URL), await _call_count(session, SECONDARY_URL)
        status, body = await _completion(session, "tenant-normal", tokens=100)
        check("e2e: normal request succeeds via primary", status == 200 and body.get("provider", "").startswith("primary"), f"status={status} body={body}")
        primary_after, secondary_after = await _call_count(session, PRIMARY_URL), await _call_count(session, SECONDARY_URL)
        check(
            "e2e: normal request only hits primary, never secondary",
            primary_after - primary_before == 1 and secondary_after == secondary_before,
            f"primary +{primary_after - primary_before} secondary +{secondary_after - secondary_before}",
        )

        # --- primary 429 triggers failover to secondary --------------
        await _set_fault(session, PRIMARY_URL, "rate_limited")
        primary_before, secondary_before = await _call_count(session, PRIMARY_URL), await _call_count(session, SECONDARY_URL)
        status, body = await _completion(session, "tenant-429", tokens=100)
        check(
            "e2e: primary 429 fails over to secondary, client still gets 200",
            status == 200 and body.get("provider", "").startswith("secondary"),
            f"status={status} body={body}",
        )
        primary_after, secondary_after = await _call_count(session, PRIMARY_URL), await _call_count(session, SECONDARY_URL)
        check(
            "e2e: both primary (attempted, rejected) and secondary (served it) were called exactly once",
            primary_after - primary_before == 1 and secondary_after - secondary_before == 1,
            f"primary +{primary_after - primary_before} secondary +{secondary_after - secondary_before}",
        )
        await _set_fault(session, PRIMARY_URL, "none")

        # --- primary timeout triggers failover, and does so PROMPTLY -
        # slow = 4s sleep, well past router.py's 3.0s timeout. If the
        # timeout weren't actually enforced, this would take ~4s+
        # (waiting for primary) instead of ~3s (timeout) + secondary's
        # near-instant response.
        await _set_fault(session, PRIMARY_URL, "slow")
        start = time.monotonic()
        status, body = await _completion(session, "tenant-timeout", tokens=100)
        elapsed = time.monotonic() - start
        check(
            "e2e: primary timeout fails over to secondary, client still gets 200",
            status == 200 and body.get("provider", "").startswith("secondary"),
            f"status={status} body={body}",
        )
        check(
            "e2e: failover happens around the 3s timeout, not after primary's full 4s delay",
            2.7 < elapsed < 3.8,
            f"elapsed={elapsed:.2f}s",
        )
        await _set_fault(session, PRIMARY_URL, "none")

        # --- both providers down: standardized, sanitized error ------
        await _set_fault(session, PRIMARY_URL, "error")
        await _set_fault(session, SECONDARY_URL, "error")
        status, body = await _completion(session, "tenant-bothdown", tokens=100)
        raw = json.dumps(body)
        check("e2e: both providers failing returns 502", status == 502, f"status={status} body={body}")
        check(
            "e2e: error body has the standardized {error: {code, message, request_id}} shape",
            isinstance(body.get("error"), dict)
            and {"code", "message", "request_id"} <= body["error"].keys()
            and body["error"]["code"] == "upstream_unavailable",
            repr(body),
        )
        check(
            "e2e: sanitized error never leaks the upstream's raw traceback/exception text",
            "Traceback" not in raw and "RuntimeError" not in raw and "db connection pool" not in raw,
            repr(body),
        )
        await _set_fault(session, PRIMARY_URL, "none")
        await _set_fault(session, SECONDARY_URL, "none")

        # --- gateway's own rate limit (50000 tokens/60s) --------------
        tenant = "tenant-ratelimit"
        status1, body1 = await _completion(session, tenant, tokens=40_000)
        status2, body2 = await _completion(session, tenant, tokens=15_000)
        check("e2e: first request within budget succeeds", status1 == 200, f"status={status1} body={body1}")
        check(
            "e2e: second request pushing total over budget is denied with the gateway's own 429",
            status2 == 429 and body2.get("error", {}).get("code") == "rate_limited",
            f"status={status2} body={body2}",
        )

        # The denied request above was never recorded (it was
        # rejected, not partially admitted), so the tenant still has
        # 10000 tokens of headroom (40000 used / 50000 limit). A
        # request within that headroom should still succeed...
        status3, body3 = await _completion(session, tenant, tokens=1_000)
        check("e2e: a request that fits remaining headroom after a prior denial still succeeds", status3 == 200, f"status={status3} body={body3}")

        # ...but one that exceeds it is denied, and specifically
        # without ever reaching a provider -- the limiter check must
        # happen strictly before routing, not after a wasted upstream
        # call.
        primary_before = await _call_count(session, PRIMARY_URL)
        secondary_before = await _call_count(session, SECONDARY_URL)
        status4, body4 = await _completion(session, tenant, tokens=20_000)
        primary_after = await _call_count(session, PRIMARY_URL)
        secondary_after = await _call_count(session, SECONDARY_URL)
        check(
            "e2e: a rate-limited request never reaches either provider",
            status4 == 429 and primary_after == primary_before and secondary_after == secondary_before,
            f"status={status4} primary +{primary_after - primary_before} secondary +{secondary_after - secondary_before}",
        )

        # --- missing/malformed auth and body validation ---------------
        async with session.post(f"{ROUTER_URL}/v1/completions", json={"prompt": "hi", "tokens": 10}) as resp:
            no_auth_status = resp.status
            no_auth_body = await resp.json()
        check(
            "e2e: missing Authorization header is rejected with 401",
            no_auth_status == 401 and no_auth_body.get("error", {}).get("code") == "invalid_request",
            f"status={no_auth_status} body={no_auth_body}",
        )

        status, body = await _completion(session, "tenant-badtokens", tokens=-5)
        check(
            "e2e: a non-positive tokens value is rejected with 400",
            status == 400 and body.get("error", {}).get("code") == "invalid_request",
            f"status={status} body={body}",
        )

        # --- concurrency through the full HTTP stack -------------------
        tenant = "tenant-concurrent"
        results = await asyncio.gather(
            *(_completion(session, tenant, tokens=1_000, prompt=f"req-{i}") for i in range(60))
        )
        succeeded = [r for r in results if r[0] == 200]
        denied = [r for r in results if r[0] == 429]
        check(
            "e2e: 60 concurrent 1000-token HTTP requests against the 50000 limit admit exactly 50",
            len(succeeded) == 50 and len(denied) == 10,
            f"succeeded={len(succeeded)} denied={len(denied)}",
        )


def run() -> int:
    db_path = os.path.abspath(os.path.join(tempfile.gettempdir(), "task4_e2e_rate_limit.db"))
    if os.path.exists(db_path):
        os.remove(db_path)
    for suffix in ("-wal", "-shm"):
        if os.path.exists(db_path + suffix):
            os.remove(db_path + suffix)

    primary = subprocess.Popen(
        [sys.executable, "mock_model_provider.py", "--port", str(PRIMARY_PORT), "--name", "primary"],
        stdout=subprocess.DEVNULL,
    )
    secondary = subprocess.Popen(
        [sys.executable, "mock_model_provider.py", "--port", str(SECONDARY_PORT), "--name", "secondary"],
        stdout=subprocess.DEVNULL,
    )
    router = subprocess.Popen(
        [
            sys.executable,
            "router.py",
            "--port",
            str(ROUTER_PORT),
            "--primary",
            f"{PRIMARY_URL}/v1/completions",
            "--secondary",
            f"{SECONDARY_URL}/v1/completions",
            "--db",
            db_path,
        ],
        stdout=subprocess.DEVNULL,
    )

    try:
        asyncio.run(run_limiter_unit_checks())
        asyncio.run(run_e2e_checks())
    finally:
        router.terminate()
        primary.terminate()
        secondary.terminate()
        router.wait(timeout=3)
        primary.wait(timeout=3)
        secondary.wait(timeout=3)

    print("=" * 100)
    print("RATE LIMITER + FALLBACK ROUTER TEST RESULTS")
    print("=" * 100)
    for label, ok, detail in rows:
        print(f"{'PASS' if ok else 'FAIL':<5} {label:<75} {detail if not ok else ''}")

    print()
    print(f"RESULT: {'ALL PASS' if failures == 0 else f'{failures} FAILURE(S)'}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(run())
