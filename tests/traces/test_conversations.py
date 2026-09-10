import json
from collections import Counter

import pytest

from driftmatrix.traces.conversations import (
    AMBIGUOUS_TYPES, STRATA, SYSTEM, generate_conversations, openai_tools, to_oai,
)
from driftmatrix.traces.tools import TOOL_SCHEMAS, call_tool

NAMES = {s["name"] for s in TOOL_SCHEMAS}
AMBIGUOUS_PREFIXES = ("Pull up that record", "Give me something on record", "Can you total up records",
                      "Add up the numbers", "Write that up")


def test_generation_is_deterministic_per_seed():
    assert generate_conversations(5, 2, seed=3) == generate_conversations(5, 2, seed=3)
    assert generate_conversations(5, 2, seed=3) != generate_conversations(5, 2, seed=4)


def test_structure_system_history_final_user():
    conv = generate_conversations(1, 3, seed=1)[0]
    m = conv.messages
    assert m[0] == {"role": "system", "content": SYSTEM}
    assert len(m) == 1 + 4 * 3 + 1
    assert m[-1]["role"] == "user"
    for t in range(3):
        user, call, tool, summary = m[1 + 4 * t: 5 + 4 * t]
        assert user["role"] == "user"
        assert call["role"] == "assistant" and isinstance(call["tool_calls"][0]["function"]["arguments"], dict)
        assert tool["role"] == "tool" and tool["tool_call_id"] == call["tool_calls"][0]["id"]
        fn = call["tool_calls"][0]["function"]
        assert tool["content"] == call_tool(fn["name"], fn["arguments"])
        assert summary["role"] == "assistant"


def test_expected_call_is_valid_for_the_mock_tools():
    for conv in generate_conversations(20, 1, seed=7):
        assert conv.expected_tool in NAMES
        call_tool(conv.expected_tool, conv.expected_args)   # must not raise


def test_all_three_tools_appear_as_expected_calls():
    assert {c.expected_tool for c in generate_conversations(30, 1, seed=11)} == NAMES


def test_explicit_generation_unchanged_from_the_g2_run():
    # pins the conversations behind the committed G2 reports (seed 2026, 3 round trips)
    convs = generate_conversations(2, 3, seed=2026)
    assert convs[0].conv_id == "c2026_000" and convs[0].stratum == "explicit"
    assert convs[0].messages[-1]["content"] == "Look up record 153 and tell me its field."
    assert convs[1].messages[-1]["content"] == "Look up record 0 and tell me its field."


def test_ambiguous_twin_shares_history_and_replaces_only_the_final_request():
    explicit = generate_conversations(10, 3, seed=5)
    ambiguous = generate_conversations(10, 3, seed=5, stratum="ambiguous")
    for e, a in zip(explicit, ambiguous):
        assert a.messages[:-1] == e.messages[:-1]
        assert a.messages[-1]["role"] == "user"
        assert a.messages[-1]["content"] != e.messages[-1]["content"]
        assert a.stratum == "ambiguous"
        assert a.expected_tool is None and a.expected_args is None
        assert a.conv_id == "a" + e.conv_id[1:]


def test_ambiguous_request_types_rotate_evenly():
    ambiguous = generate_conversations(30, 3, seed=2026, stratum="ambiguous")
    counts = Counter(next(p for p in AMBIGUOUS_PREFIXES if c.messages[-1]["content"].startswith(p))
                     for c in ambiguous)
    assert len(AMBIGUOUS_TYPES) == len(AMBIGUOUS_PREFIXES)
    assert counts == {p: 6 for p in AMBIGUOUS_PREFIXES}


def test_ambiguous_generation_is_deterministic_and_varied():
    a1 = generate_conversations(30, 3, seed=9, stratum="ambiguous")
    assert a1 == generate_conversations(30, 3, seed=9, stratum="ambiguous")
    assert len({c.messages[-1]["content"] for c in a1}) >= 5


def test_ambiguous_generation_does_not_mutate_the_explicit_twin():
    before = generate_conversations(5, 2, seed=13)
    generate_conversations(5, 2, seed=13, stratum="ambiguous")
    assert generate_conversations(5, 2, seed=13) == before


def test_unknown_stratum_raises():
    assert STRATA == ("explicit", "ambiguous")
    with pytest.raises(ValueError, match="unknown stratum"):
        generate_conversations(1, 1, seed=1, stratum="hard")


def test_openai_tools_wrap_each_schema():
    tools = openai_tools()
    assert [t["type"] for t in tools] == ["function"] * len(TOOL_SCHEMAS)
    assert [t["function"]["name"] for t in tools] == [s["name"] for s in TOOL_SCHEMAS]


def test_to_oai_stringifies_arguments_without_mutating_the_original():
    conv = generate_conversations(1, 2, seed=5)[0]
    oai = to_oai(conv.messages)
    for orig, wire in zip(conv.messages, oai):
        for tc_o, tc_w in zip(orig.get("tool_calls") or [], wire.get("tool_calls") or []):
            assert isinstance(tc_o["function"]["arguments"], dict)
            assert tc_w["function"]["arguments"] == json.dumps(tc_o["function"]["arguments"],
                                                               ensure_ascii=False)
            assert tc_w["id"] == tc_o["id"]
