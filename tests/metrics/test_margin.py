import pytest

from driftmatrix.metrics.margin import (
    MARGIN_BINS, decision_margin, margin_bin, position_gaps, tool_call_span,
)


def gaps_top(*gaps):
    """Per-position candidate lists whose top-two gaps are exactly `gaps`."""
    return [[(1, -0.1), (2, -0.1 - g)] for g in gaps]


def test_span_found_when_both_tags_are_split_across_pieces():
    pieces = ["Sure. <tool", "_call>", '{"name": "x"}', "</tool", "_call>", " trailing"]
    assert tool_call_span(pieces) == (0, 4)       # piece 0 carries the start of '<tool_call>'


def test_span_starts_at_the_tag_not_at_a_preamble():
    pieces = ["Let me check.", " ", "<tool_call>", "x", "</tool_call>"]
    assert tool_call_span(pieces) == (2, 4)


def test_unterminated_or_missing_call_has_no_span():
    assert tool_call_span(["<tool_call>", '{"name": "x"']) is None
    assert tool_call_span(["plain text"]) is None


def test_call_at_position_zero_uses_the_smallest_gap_in_the_span():
    top = gaps_top(2.9, 0.7, 1.4)
    assert decision_margin(top, ["<tool_call>", "lookup", "</tool_call>"]) == (pytest.approx(0.7), 1)


def test_no_call_turn_is_scored_at_the_first_position_only():
    top = gaps_top(1.2, 0.01, 0.02)               # prose near-ties after position 0 are ignored
    assert decision_margin(top, ["I", " need", " clarification"]) == (pytest.approx(1.2), 0)


def test_preamble_positions_are_excluded_but_the_first_position_counts():
    top = gaps_top(3.0, 0.05, 0.05, 1.5, 2.5, 4.0)
    pieces = ["Let", " me", " check.", "<tool_call>", "x", "</tool_call>"]
    assert decision_margin(top, pieces) == (pytest.approx(1.5), 3)


def test_first_position_can_be_the_closest_call():
    top = gaps_top(0.2, 3.0, 3.0)
    assert decision_margin(top, ["<tool_call>", "x", "</tool_call>"]) == (pytest.approx(0.2), 0)


def test_positions_after_the_span_are_ignored():
    top = gaps_top(1.9, 1.9, 0.01)
    assert decision_margin(top, ["<tool_call>x", "</tool_call>", " and more"])[0] == pytest.approx(1.9)


def test_unsorted_candidates_and_single_candidates():
    assert decision_margin([[(2, -1.2), (1, -0.2), (3, -4.0)]], ["hi"])[0] == pytest.approx(1.0)
    assert decision_margin([[(1, -0.1)]], ["hi"]) == (None, None)
    assert decision_margin([], []) == (None, None)


def test_position_gaps():
    assert position_gaps([[(1, -0.1), (2, -0.6)], [(3, -0.1)]]) == [pytest.approx(0.5), None]


def test_bins_and_their_edges():
    assert MARGIN_BINS == (0.5, 2.0)
    assert margin_bin(0.49) == "near_tie"
    assert margin_bin(0.5) == "contested"
    assert margin_bin(1.99) == "contested"
    assert margin_bin(2.0) == "confident"
    assert margin_bin(None) is None
