import pytest

from driftmatrix.metrics.behavioral import ToolCall
from driftmatrix.traces.toolcalls import FORMAT_BY_MODEL, parse_tool_call

EXPECTED = ToolCall.from_payload("lookup_record", {"record_id": 7})


def test_parse_ornith_qwen_xml():
    txt = "<tool_call>\n<function=lookup_record>\n<parameter=record_id>\n7\n</parameter>\n</function>\n</tool_call>"
    assert parse_tool_call(txt, "qwen_xml") == EXPECTED


def test_parse_ornith_allows_reasoning_before_the_call():
    txt = ("I will look it up.\n\n<tool_call>\n<function=lookup_record>\n"
           "<parameter=record_id>\n7\n</parameter>\n</function>\n</tool_call>")
    assert parse_tool_call(txt, "qwen_xml") == EXPECTED


def test_parse_ornith_multiline_string_parameter():
    txt = ("<tool_call>\n<function=format_report>\n<parameter=title>\nBatch 5\n</parameter>\n"
           "<parameter=body>\nline one\nline two\n</parameter>\n</function>\n</tool_call>")
    assert parse_tool_call(txt, "qwen_xml") == ToolCall.from_payload(
        "format_report", {"title": "Batch 5", "body": "line one\nline two"})


def test_parse_qwen3_json():
    txt = '<tool_call>\n{"name": "lookup_record", "arguments": {"record_id": 7}}\n</tool_call>'
    assert parse_tool_call(txt, "qwen_json") == EXPECTED


def test_parse_qwen3_json_with_string_encoded_arguments():
    txt = '<tool_call>\n{"name": "lookup_record", "arguments": "{\\"record_id\\": 7}"}\n</tool_call>'
    assert parse_tool_call(txt, "qwen_json") == EXPECTED


def test_parse_spark_argkv():
    txt = "<tool_call>lookup_record<arg_key>record_id</arg_key><arg_value>7</arg_value></tool_call>"
    assert parse_tool_call(txt, "spark_argkv") == EXPECTED


def test_string_values_survive_canonicalisation():
    txt = "<tool_call>format_report<arg_key>title</arg_key><arg_value>Batch 5</arg_value></tool_call>"
    assert parse_tool_call(txt, "spark_argkv") == ToolCall.from_payload(
        "format_report", {"title": "Batch 5"})


def test_same_call_in_all_three_formats_hashes_identically():
    xml = "<tool_call>\n<function=compute_sum>\n<parameter=values>\n[1, 2, 3]\n</parameter>\n</function>\n</tool_call>"
    js = '<tool_call>\n{"name": "compute_sum", "arguments": {"values": [1, 2, 3]}}\n</tool_call>'
    kv = "<tool_call>compute_sum<arg_key>values</arg_key><arg_value>[1, 2, 3]</arg_value></tool_call>"
    assert parse_tool_call(xml, "qwen_xml") == parse_tool_call(js, "qwen_json") == \
        parse_tool_call(kv, "spark_argkv")


def test_only_the_first_complete_call_is_parsed():
    txt = ('<tool_call>\n{"name": "lookup_record", "arguments": {"record_id": 7}}\n</tool_call>\n'
           '<tool_call>\n{"name": "compute_sum", "arguments": {"values": [1]}}\n</tool_call>')
    assert parse_tool_call(txt, "qwen_json") == EXPECTED


def test_plain_text_and_truncated_calls_are_none():
    assert parse_tool_call("The record says alpha.", "qwen_json") is None
    assert parse_tool_call('<tool_call>\n{"name": "lookup_rec', "qwen_json") is None
    assert parse_tool_call("", "qwen_xml") is None


def test_malformed_bodies_are_none():
    assert parse_tool_call("<tool_call>not json</tool_call>", "qwen_json") is None
    assert parse_tool_call("<tool_call>no function block</tool_call>", "qwen_xml") is None
    assert parse_tool_call('<tool_call>{"arguments": {}}</tool_call>', "qwen_json") is None


def test_unknown_format_raises():
    with pytest.raises(ValueError, match="unknown tool-call format"):
        parse_tool_call("<tool_call>x</tool_call>", "nope")


def test_every_model_maps_to_a_parser():
    for fmt in FORMAT_BY_MODEL.values():
        assert parse_tool_call("plain", fmt) is None
