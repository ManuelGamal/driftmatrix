from driftmatrix.metrics.behavioral import (
    ToolCall, exact_match, first_divergence, tool_call_agree,
)


def test_exact_match():
    assert exact_match([1, 2, 3], [1, 2, 3]) is True
    assert exact_match([1, 2, 3], [1, 2, 4]) is False
    assert exact_match([1, 2], [1, 2, 3]) is False


def test_first_divergence_reports_position_and_not_censored():
    assert first_divergence([1, 2, 3, 4], [1, 2, 9, 4], cap=32) == (2, False)


def test_identical_sequences_are_censored_at_cap():
    assert first_divergence([1, 2, 3], [1, 2, 3], cap=32) == (32, True)


def test_length_mismatch_diverges_at_shorter_length():
    assert first_divergence([1, 2], [1, 2, 3], cap=32) == (2, False)


def test_tool_call_args_hash_is_order_insensitive():
    a = ToolCall.from_payload("search", {"q": "x", "n": 3})
    b = ToolCall.from_payload("search", {"n": 3, "q": "x"})
    assert a == b and tool_call_agree(a, b) is True


def test_tool_call_disagreement_on_name_and_on_args():
    base = ToolCall.from_payload("search", {"q": "x"})
    assert tool_call_agree(base, ToolCall.from_payload("lookup", {"q": "x"})) is False
    assert tool_call_agree(base, ToolCall.from_payload("search", {"q": "y"})) is False


def test_both_none_counts_as_agreement_but_one_none_does_not():
    assert tool_call_agree(None, None) is True
    assert tool_call_agree(ToolCall.from_payload("s", {}), None) is False
