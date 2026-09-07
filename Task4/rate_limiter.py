"""
Token-aware sliding-window rate limiter, backed by on-disk SQLite.

Sliding window (a log of individual admitted requests, not a fixed
bucket that resets on a clock boundary): each admitted request records
(tenant_key, timestamp, tokens) as its own row. A check sums the
tokens recorded for that tenant within the trailing `window_seconds`
and admits the new request only if adding its own token count would
stay within `limit`. Expired rows (older than the window) are deleted
as part of the very same check -- eviction isn't a separate sweep that
might lag behind, it's inline with every read.

Concurrency and correctness
----------------------------
sqlite3 is blocking, so every operation here runs in a worker thread
via asyncio.to_thread -- the event loop is never blocked waiting on
disk I/O. Correctness across CONCURRENT callers (the actual hard part
of "check current usage, then decide, then record" under concurrency
-- a classic check-then-act race) comes from SQLite's own transaction
locking, not from anything in Python: every check opens a BEGIN
IMMEDIATE transaction, which acquires SQLite's write lock up front,
before the read even happens. Two concurrent callers for the same
tenant can't both see the same "not yet over limit" snapshot and both
insert -- the second one blocks (up to busy_timeout) until the first
commits or rolls back, then re-reads current (now already updated)
usage. This holds even across multiple processes sharing the same
database file, which an in-process asyncio.Lock alone could not
guarantee.

A short-lived connection is opened per call rather than pooled or
shared across threads -- the simplest way to avoid sqlite3's
not-thread-safe-by-default connection semantics entirely. That's a
deliberate simplicity-over-throughput tradeoff for this scope; a
high-QPS deployment would more likely use a single writer thread with
a queue, or a connection pool, in front of the same schema and locking
strategy.
"""

import asyncio
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RateLimitResult:
    allowed: bool
    remaining_tokens: int
    retry_after_seconds: float


class SqliteSlidingWindowLimiter:
    def __init__(self, db_path: str | Path, limit_tokens: int, window_seconds: float = 60.0) -> None:
        self.db_path = str(db_path)
        self.limit_tokens = limit_tokens
        self.window_seconds = window_seconds
        # Same-process optimization only, not the correctness
        # mechanism -- see module docstring.
        self._tenant_locks: dict[str, asyncio.Lock] = {}
        self._locks_guard = asyncio.Lock()
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, isolation_level=None, timeout=5.0)
        conn.execute("PRAGMA busy_timeout = 5000")
        conn.execute("PRAGMA journal_mode = WAL")
        return conn

    def _init_db(self) -> None:
        conn = self._connect()
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS token_usage (
                    tenant_key TEXT NOT NULL,
                    ts REAL NOT NULL,
                    tokens INTEGER NOT NULL
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_token_usage_tenant_ts ON token_usage(tenant_key, ts)")
        finally:
            conn.close()

    async def _tenant_lock(self, tenant_key: str) -> asyncio.Lock:
        async with self._locks_guard:
            lock = self._tenant_locks.get(tenant_key)
            if lock is None:
                lock = asyncio.Lock()
                self._tenant_locks[tenant_key] = lock
            return lock

    def _check_and_consume_sync(self, tenant_key: str, tokens_requested: int, now: float) -> RateLimitResult:
        cutoff = now - self.window_seconds
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute("DELETE FROM token_usage WHERE tenant_key = ? AND ts <= ?", (tenant_key, cutoff))

            current_usage = conn.execute(
                "SELECT COALESCE(SUM(tokens), 0) FROM token_usage WHERE tenant_key = ?", (tenant_key,)
            ).fetchone()[0]

            if current_usage + tokens_requested > self.limit_tokens:
                oldest = conn.execute(
                    "SELECT MIN(ts) FROM token_usage WHERE tenant_key = ?", (tenant_key,)
                ).fetchone()[0]
                # Approximation: time until the single oldest entry
                # ages out of the window. Not a precise "time until
                # exactly enough capacity frees up for THIS request"
                # (that would need walking the window in age order
                # accumulating freed tokens) -- a reasonable,
                # documented simplification for a Retry-After hint.
                retry_after = max(0.0, (oldest + self.window_seconds) - now) if oldest is not None else self.window_seconds
                conn.execute("ROLLBACK")
                return RateLimitResult(
                    allowed=False,
                    remaining_tokens=max(0, self.limit_tokens - current_usage),
                    retry_after_seconds=retry_after,
                )

            conn.execute(
                "INSERT INTO token_usage (tenant_key, ts, tokens) VALUES (?, ?, ?)",
                (tenant_key, now, tokens_requested),
            )
            conn.execute("COMMIT")
            return RateLimitResult(
                allowed=True,
                remaining_tokens=self.limit_tokens - (current_usage + tokens_requested),
                retry_after_seconds=0.0,
            )
        except Exception:
            conn.execute("ROLLBACK")
            raise
        finally:
            conn.close()

    async def check_and_consume(
        self, tenant_key: str, tokens_requested: int, *, now: float | None = None
    ) -> RateLimitResult:
        if now is None:
            now = time.time()
        lock = await self._tenant_lock(tenant_key)
        async with lock:
            return await asyncio.to_thread(self._check_and_consume_sync, tenant_key, tokens_requested, now)

    def _purge_all_expired_sync(self, now: float) -> int:
        cutoff = now - self.window_seconds
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            cur = conn.execute("DELETE FROM token_usage WHERE ts <= ?", (cutoff,))
            deleted = cur.rowcount
            conn.execute("COMMIT")
            return deleted
        except Exception:
            conn.execute("ROLLBACK")
            raise
        finally:
            conn.close()

    async def purge_expired(self, *, now: float | None = None) -> int:
        """
        Sweep EVERY tenant's expired rows, not just the one in the
        current request path. check_and_consume() only ever evicts the
        tenant it's currently handling, as a side effect of that
        tenant's own traffic -- correct for that tenant's own future
        admission decisions, but a tenant that goes quiet would
        otherwise leave stale rows sitting in the table forever (they
        don't count against anyone's limit -- the per-tenant eviction
        in check_and_consume already guarantees that -- but the table
        itself would grow unbounded). Meant to be called periodically
        from a background task, not per-request.
        """
        if now is None:
            now = time.time()
        return await asyncio.to_thread(self._purge_all_expired_sync, now)

    async def current_usage(self, tenant_key: str, *, now: float | None = None) -> int:
        """Read-only: tokens currently counted against this tenant's
        window. Doesn't evict; used for introspection/testing."""
        if now is None:
            now = time.time()

        def _query() -> int:
            cutoff = now - self.window_seconds
            conn = self._connect()
            try:
                row = conn.execute(
                    "SELECT COALESCE(SUM(tokens), 0) FROM token_usage WHERE tenant_key = ? AND ts > ?",
                    (tenant_key, cutoff),
                ).fetchone()
                return row[0]
            finally:
                conn.close()

        return await asyncio.to_thread(_query)
