"""Action margin: how close a completion's action came to a tie (spec 5.1, v4.3).

An agent's action is either a tool call (a tool plus its arguments) or no call:
a clarifying question or a text answer. The action is decided at two kinds of
generated position:

  - the first position, where the model starts either a tool call or prose -
    the act-or-not choice;
  - every position of the first complete tool call, if there is one - which
    tool, which arguments.

The action margin is the smallest top-1 minus top-2 logprob gap over those
positions. Prose after the first position cannot change the action, so it is
excluded, as is any preamble before a tool call. Every turn therefore has a
margin, whether or not it called a tool - without that, comparing strata would
silently drop every clarifying question.

Measured on the recompute arm only, so stratifying by it never selects on the
treatment's outcome. Pre-registered bins:
near_tie < 0.5 nats <= contested < 2.0 nats <= confident.
"""
from __future__ import annotations

MARGIN_BINS = (0.5, 2.0)
BIN_LABELS = ("near_tie", "contested", "confident")
TOOL_CALL_START = "<tool_call>"
TOOL_CALL_END = "</tool_call>"


def _top_two_gap(candidates):
    logprobs = sorted((lp for _, lp in candidates), reverse=True)
    return logprobs[0] - logprobs[1] if len(logprobs) >= 2 else None


def position_gaps(top):
    """Top-1 minus top-2 gap at every generated position; None under two candidates."""
    return [_top_two_gap(c) for c in top]


def tool_call_span(pieces):
    """(first, last) positions of the first complete tool call, or None.

    Tags may be split across pieces; positions are located by character offset.
    """
    ends, text = [], ""
    for piece in pieces:
        text += piece or ""
        ends.append(len(text))
    start = text.find(TOOL_CALL_START)
    end = text.find(TOOL_CALL_END, start + len(TOOL_CALL_START)) if start >= 0 else -1
    if start < 0 or end < 0:
        return None
    first = next(i for i, e in enumerate(ends) if e > start)
    last = next(i for i, e in enumerate(ends) if e >= end + len(TOOL_CALL_END))
    return first, last


def decision_margin(top, pieces):
    """Action margin as (margin, position), or (None, None) if nothing is scoreable."""
    if not top:
        return None, None
    positions = {0}
    span = tool_call_span(pieces)
    if span is not None:
        positions.update(range(span[0], span[1] + 1))
    best_margin, best_pos = None, None
    for i in sorted(positions):
        if i >= len(top):
            continue
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
