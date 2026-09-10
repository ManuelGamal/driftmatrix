import json

from driftmatrix.traces.conversations import (
    SYSTEM, generate_conversations, openai_tools, to_oai,
)
from driftmatrix.traces.tools import TOOL_SCHEMAS, call_tool

NAMES = {s["name"] for s in TOOL_SCHEMAS}


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
