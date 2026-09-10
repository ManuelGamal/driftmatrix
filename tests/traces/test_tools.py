import pytest
from driftmatrix.traces.tools import TOOL_SCHEMAS, call_tool, ToolError


def test_schema_order_is_stable():
    assert [s["name"] for s in TOOL_SCHEMAS] == [
        "lookup_record", "compute_sum", "format_report"]


def test_tools_are_pure_across_repeated_calls():
    for _ in range(50):
        assert call_tool("lookup_record", {"record_id": 7}) == \
               call_tool("lookup_record", {"record_id": 7})


def test_known_outputs():
    assert call_tool("compute_sum", {"values": [1, 2, 3]}) == "6"
    assert call_tool("lookup_record", {"record_id": 0}).startswith("record 0:")
    assert call_tool("format_report", {"title": "T", "body": "B"}) == "# T\n\nB"


def test_unknown_tool_raises():
    with pytest.raises(ToolError, match="unknown tool"):
        call_tool("nope", {})


def test_missing_argument_raises():
    with pytest.raises(ToolError, match="missing argument"):
        call_tool("compute_sum", {})
