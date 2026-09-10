"""Behavioural outcomes (spec 5.1). Primary results; no logprob dependency."""
from __future__ import annotations
import hashlib
import json
from dataclasses import dataclass


@dataclass(frozen=True)
class ToolCall:
    name: str
    args_hash: str

    @classmethod
    def from_payload(cls, name: str, args: dict) -> "ToolCall":
        # sort_keys makes the hash insensitive to serialisation order, so a
        # reordered-but-identical call is not scored as a disagreement.
        blob = json.dumps(args, sort_keys=True, separators=(",", ":"))
        return cls(name=name, args_hash=hashlib.sha256(blob.encode()).hexdigest()[:16])


def exact_match(a, b) -> bool:
    return list(a) == list(b)


def first_divergence(a, b, cap: int):
    """Return (position, censored). Censored means they never diverged."""
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i, False
    if len(a) != len(b):
        return n, False
    return cap, True


def tool_call_agree(a, b) -> bool:
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    return a == b
