"""
Streaming PII redactor: feed it text deltas as they arrive from an LLM
stream, get back the text that's safe to forward to the client right
now, already redacted.

Two distinct problems, not one
-------------------------------
1. A PII pattern can straddle a chunk boundary -- "user@example.com"
   arriving as "user@exam" then "ple.com" means neither delta alone
   contains a complete match. The fix is a holdback buffer: don't
   emit the trailing part of the buffer that might still be a
   forming match; carry it into the next feed() call instead.

2. A *greedy, open-ended* pattern can commit to a match too early.
   The email TLD is `[A-Za-z]{2,}` -- unbounded on the high end. If
   the buffer happens to end right after "...@example.co" (before a
   trailing "m" arrives to make it "...co" -> "...com"), the regex
   engine has no way to know more letters are coming: it matches
   "example.co" as a complete, valid email *right now*. Redact on
   that basis and the redaction fires one character too early, and
   the "m" that was actually part of the address leaks out as plain
   text on the next call. The same applies to `\\b` word-boundary
   anchors on the SSN/card patterns: Python's `\\b` is satisfied by
   the end of the search string, so a digit run that just happens to
   end at the current buffer's edge looks "bounded" even though more
   digits (which would break the boundary) might be one delta away.

   The fix: a match is only trusted if it ends strictly *before* the
   end of the current buffer, i.e. there's at least one more
   character already in hand that the regex engine had the chance to
   extend the match into and chose not to -- proof the match is
   genuinely finished, not just that we've run out of input to look
   at. A match sitting exactly at the buffer's edge is left alone
   (raw) until either more data arrives to settle it, or flush() is
   called (nothing more is ever coming, so every match is final).

Responsiveness: holdback is NOT a flat "always keep the last 128
characters" -- for ordinary prose most of the buffer is emitted
immediately, since there's nothing there that could plausibly become
PII. Only the trailing run of characters that could still be part of
an in-progress match (letters, digits, and the punctuation that
appears inside these three pattern types) is held back, capped at
HOLDBACK as a hard upper bound. A short response with no PII anywhere
near the end streams out essentially as fast as it arrives; a holdback
only actually happens while the tail of the buffer looks like it could
be building toward an email, SSN, or card number.
"""

import re
import string

# Practical, not RFC-exhaustive: good enough for catching PII in prose
# without chasing every legal-but-unrealistic edge case of each spec.
_EMAIL = r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"
_SSN = r"\b\d{3}-\d{2}-\d{4}\b"
# Major-network BIN prefixes (Visa/Mastercard/Amex/Discover) grouped
# in 4s with optional space/dash separators. Deliberately not "any
# 13-19 digit run" -- that alone would flag phone numbers, order IDs,
# zip+4s, etc. Not Luhn-validated either; that's a reasonable next
# step for a production redactor but doesn't change the streaming
# design this task is actually about.
_CREDIT_CARD = r"\b(?:4\d{3}|5[1-5]\d{2}|3[47]\d{2}|6011)[ -]?\d{4}[ -]?\d{4}[ -]?\d{3,4}\b"

PII_PATTERN = re.compile("|".join(f"(?:{p})" for p in (_EMAIL, _SSN, _CREDIT_CARD)))

REDACTED = "[REDACTED]"

# Every character that can legally appear inside one of the patterns
# above. A trailing run made up only of these characters is what gets
# held back -- anything else (a space, a comma, ordinary punctuation)
# can't be part of a growing match, so text in front of it is safe to
# emit right away.
_RISKY_CHARS = frozenset(string.ascii_letters + string.digits + "@._%+-")


class StreamRedactor:
    # Hard upper bound on the holdback, regardless of how long a
    # single unbroken "risky" run gets (e.g. a long word with no
    # spaces that isn't actually PII). Comfortably covers any
    # realistic email/SSN/card number; an adversarially long local
    # part or domain beyond this could in principle still slip past --
    # a documented tradeoff, not a silent one.
    HOLDBACK = 128

    def __init__(self) -> None:
        self._buffer = ""

    @staticmethod
    def _redact(text: str, *, is_final: bool) -> str:
        pieces = []
        last_end = 0
        for m in PII_PATTERN.finditer(text):
            if not is_final and m.end() == len(text):
                # Ends exactly at the edge of what we have so far --
                # could still grow (e.g. "...@example.co" before a
                # trailing "m" arrives). Leave it raw; re-evaluated
                # next call once more text is appended after it.
                continue
            pieces.append(text[last_end : m.start()])
            pieces.append(REDACTED)
            last_end = m.end()
        pieces.append(text[last_end:])
        return "".join(pieces)

    @classmethod
    def _risky_tail_length(cls, text: str) -> int:
        n = 0
        for ch in reversed(text):
            if ch not in _RISKY_CHARS:
                break
            n += 1
            if n >= cls.HOLDBACK:
                break
        return n

    def feed(self, delta: str) -> str:
        """Feed one incoming text delta. Returns the text that's now
        safe to emit (already redacted) -- may be empty if the whole
        buffer still looks like it could be forming a match."""
        if not delta:
            return ""

        self._buffer = self._redact(self._buffer + delta, is_final=False)

        holdback = self._risky_tail_length(self._buffer)
        if holdback == 0:
            safe, self._buffer = self._buffer, ""
        else:
            safe, self._buffer = self._buffer[:-holdback], self._buffer[-holdback:]
        return safe

    def flush(self) -> str:
        """Call once, at the end of the stream: no more data is
        coming, so every match still in the buffer is final."""
        out = self._redact(self._buffer, is_final=True)
        self._buffer = ""
        return out
