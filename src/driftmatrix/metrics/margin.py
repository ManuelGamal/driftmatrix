"""Decision margin: how close a completion's decisions came to a tie (spec 5.1).

The margin is the smallest gap between the top-1 and top-2 logprobs at any
generated position through the end of the first complete tool call - which
covers both the choice of tool and the choice of arguments.

It is measured on the recompute arm only, never on the treatment, so results
can be stratified by it without selecting on the outcome under test.
Pre-registered bins: near_tie < 0.5 nats <= contested < 2.0 nats <= confident.
"""
from __future__ import annotations

MARGIN_BINS = (0.5, 2.0)
BIN_LABELS = ("near_tie", "contested", "confident")
TOOL_CALL_END = "</tool_call>"


def tool_call_span_end(pieces):
    """Index of the generated position that completes the first </tool_call>."""
    text = ""
    for i, piece in enumerate(pieces):
        text += piece or ""
        if TOOL_CALL_END in text:
            return i
    return None


def _top_two_gap(candidates):
    logprobs = sorted((lp for _, lp in candidates), reverse=True)
    return logprobs[0] - logprobs[1] if len(logprobs) >= 2 else None


def decision_margin(top, pieces):
    """Return (margin, position), or (None, None) without a complete tool call."""
    end = tool_call_span_end(pieces)
    if end is None:
        return None, None
    best_margin, best_pos = None, None
    for i in range(min(end + 1, len(top))):
        gap = _top_two_gap(top[i])
        if gap is not None and (best_margin is None or gap < best_margin):
            best_margin, best_pos = gap, i
    return best_margin, best_pos


def margin_bin(margin):
    if margin is None:
        return None
    if margin < MARGIN_BINS[0]:
        return BIN_LABELS[0]
    if margin < MARGIN_BINS[1]:
        return BIN_LABELS[1]
    return BIN_LABELS[2]
