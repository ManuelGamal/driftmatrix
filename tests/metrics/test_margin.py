import pytest

from driftmatrix.metrics.margin import (
    MARGIN_BINS, decision_margin, margin_bin, tool_call_span_end,
)


def test_span_end_found_when_the_closing_tag_is_split_across_pieces():
    pieces = ["<tool_call>", '{"name": "x"}', "</tool", "_call>", " trailing"]
    assert tool_call_span_end(pieces) == 3


def test_no_complete_tool_call_means_no_span():
    assert tool_call_span_end(["<tool_call>", '{"name": "x"'] ) is None
    assert decision_margin([[(1, -0.1), (2, -0.2)]], ["plain text"]) == (None, None)


def test_margin_is_the_smallest_top_two_gap_within_the_span():
    top = [[(1, -0.1), (2, -3.0)],      # gap 2.9
           [(3, -0.2), (4, -0.9)],      # gap 0.7  <- smallest inside the span
           [(5, -0.1), (6, -1.5)]]      # gap 1.4
    pieces = ["<tool_call>", "lookup", "</tool_call>"]
    margin, pos = decision_margin(top, pieces)
    assert margin == pytest.approx(0.7) and pos == 1


def test_positions_after_the_span_are_ignored():
    top = [[(1, -0.1), (2, -2.0)],
           [(3, -0.1), (4, -2.0)],
           [(5, -0.1), (6, -0.11)]]     # near-tie, but after </tool_call>
    pieces = ["<tool_call>x", "</tool_call>", " and more"]
    margin, pos = decision_margin(top, pieces)
    assert margin == pytest.approx(1.9) and pos in (0, 1)


def test_unsorted_candidate_lists_are_handled():
    top = [[(2, -1.2), (1, -0.2), (3, -4.0)]]
    assert decision_margin(top, ["<tool_call></tool_call>"])[0] == pytest.approx(1.0)


def test_positions_with_a_single_candidate_are_skipped():
    top = [[(1, -0.1)], [(2, -0.1), (3, -0.6)]]
    assert decision_margin(top, ["<tool_call>", "</tool_call>"]) == (pytest.approx(0.5), 1)


def test_bins_and_their_edges():
    assert MARGIN_BINS == (0.5, 2.0)
    assert margin_bin(0.49) == "near_tie"
    assert margin_bin(0.5) == "contested"
    assert margin_bin(1.99) == "contested"
    assert margin_bin(2.0) == "confident"
    assert margin_bin(None) is None
