"""G2 tool-call logic against a fake server exposing both paths."""
import copy

import pytest

from driftmatrix.g2 import (
    adjudicate_rows, g2_verdict, run_g2, server_prompt_tokens, server_tool_call,
    stratum_check, summarise,
)
from driftmatrix.metrics.behavioral import ToolCall
from driftmatrix.runner.llamaserver import Completion, ServerError, parse_completion
from driftmatrix.traces.conversations import generate_conversations

# Renders history and serialises assistant tool calls exactly as the fake model emits them.
TEMPLATE = ('{% for m in messages %}{{ m.role }}: {{ m.content }} '
            '{% for tc in (m.tool_calls or []) %}<tool_call>\n{"name": "{{ tc.function.name }}", '
            '"arguments": {{ tc.function.arguments | tojson }}}\n</tool_call>{% endfor %}{% endfor %}'
            '{% if tools %}tools: {{ tools | length }}{% endif %}')
# Serialises arguments with no space after the colon, so the model's text cannot round-trip.
COMPACT_TEMPLATE = TEMPLATE.replace('"arguments": {{ tc.function.arguments | tojson }}',
                                    '"arguments": {{ tc.function.arguments | tojson | replace(": ", ":") }}')

CONVS = generate_conversations(10, 2, seed=1)
CALL_TEXT = '<tool_call>\n{"name": "lookup_record", "arguments": {"record_id": 7}}\n</tool_call>'
CALL_PIECES = ("<tool_call>", '\n{"name": "lookup_record", ', '"arguments": {"record_id": 7}}\n</tool', "_call>")
LOOKUP_7 = ToolCall.from_payload("lookup_record", {"record_id": 7})


class FakeServer:
    def __init__(self, emits=None, prompt_offset=0, server_parses=True, chat_raises=False,
                 flat_shape=False, margins=None, server_args='{"record_id": 7}',
                 server_invents_call=False):
        self.emits = emits or [True]
        self.margins = margins or [3.0]
        self.offset, self.server_parses = prompt_offset, server_parses
        self.chat_raises, self.flat_shape = chat_raises, flat_shape
        self.server_args, self.server_invents_call = server_args, server_invents_call
        self.i, self.last_len, self.last_emit = 0, 0, True

    def tokenize(self, text, add_special=True):
        toks = text.split()
        self.last_len = len(toks)
        return toks

    def slot_erase(self, slot=0):
        return {}

    def complete(self, toks, n_predict, n_probs=0, cache_prompt=False, slot=0, seed=1234):
        self.last_emit = self.emits[self.i % len(self.emits)]
        margin = self.margins[self.i % len(self.margins)]
        self.i += 1
        if self.last_emit:
            content, pieces = CALL_TEXT, CALL_PIECES
        else:
            content, pieces = "Record 7 is alpha.", ("Record 7 is alpha.",)
        top = [[(1, -0.05), (2, -0.05 - margin)] for _ in pieces] if n_probs else []
        return Completion([], top, len(toks), 0, {"content": content}, pieces if n_probs else ())

    def chat(self, messages, tools=None, *, max_tokens=192, chat_template_kwargs=None, seed=1234):
        if self.chat_raises:
            raise ServerError(400, "unsupported message shape", "/v1/chat/completions")
        assert chat_template_kwargs == {"enable_thinking": False}
        assert all(isinstance(tc["function"]["arguments"], str)
                   for m in messages for tc in (m.get("tool_calls") or []))
        calls = []
        if (self.last_emit and self.server_parses) or self.server_invents_call:
            fn = {"name": "lookup_record", "arguments": self.server_args}
            calls = [fn] if self.flat_shape else [{"id": "x", "type": "function", "function": fn}]
        return {"choices": [{"message": {"content": None, "tool_calls": calls}}],
                "usage": {"prompt_tokens": self.last_len + self.offset}}


def test_passes_when_prompts_match_parsers_agree_and_calls_are_made():
    s = run_g2(FakeServer(), CONVS, TEMPLATE, "qwen_json")["summary"]
    assert s["verdict"] == "G2_TOOLCALLS_PASS"
    assert s["prompt_match_rate"] == s["parser_agreement_rate"] == s["tool_call_rate"] == 1.0
    assert s["our_roundtrip_rate"] == 1.0
    assert s["chat_errors"] == 0 and s["strata"] == ["explicit"]


def test_flat_tool_call_shape_from_llama_cpp_docs_is_understood():
    s = run_g2(FakeServer(flat_shape=True), CONVS, TEMPLATE, "qwen_json")["summary"]
    assert s["verdict"] == "G2_TOOLCALLS_PASS"


def test_chat_path_error_is_recorded_not_raised():
    r = run_g2(FakeServer(chat_raises=True), CONVS, TEMPLATE, "qwen_json")
    assert r["summary"]["verdict"] == "CHAT_PATH_ERROR"
    assert r["summary"]["chat_errors"] == len(CONVS)
    assert "unsupported message shape" in r["rows"][0]["chat_error"]
    assert r["summary"]["tool_call_rate"] == 1.0          # the study path still ran


def test_prompt_token_mismatch_is_caught_before_parser_checks():
    s = run_g2(FakeServer(prompt_offset=1, server_parses=False), CONVS, TEMPLATE, "qwen_json")["summary"]
    assert s["verdict"] == "RENDER_WITH_TOOLS_MISMATCH"
    assert s["parser_agreement_rate"] is None     # not judged when prompts differ


def test_lossy_server_parse_is_adjudicated_by_round_trip_and_passes():
    r = run_g2(FakeServer(server_args='{"record_id": 8}'), CONVS, TEMPLATE, "qwen_json")
    s = r["summary"]
    assert s["parser_disagreements"] == len(CONVS) and s["parser_agreement_rate"] == 0.0
    assert s["server_parser_lossy"] == len(CONVS) and s["unresolved_disagreements"] == 0
    assert r["rows"][0]["adjudication"] == "server_parser_lossy"
    assert s["verdict"] == "G2_TOOLCALLS_PASS"


def test_server_missing_a_call_is_adjudicated_as_its_loss():
    r = run_g2(FakeServer(server_parses=False), CONVS, TEMPLATE, "qwen_json")
    assert r["rows"][0]["adjudication"] == "server_parser_lossy"
    assert r["summary"]["verdict"] == "G2_TOOLCALLS_PASS"


def test_call_found_only_by_the_server_is_an_unresolved_disagreement():
    r = run_g2(FakeServer(emits=[False], server_invents_call=True), CONVS, TEMPLATE, "qwen_json")
    assert r["rows"][0]["adjudication"] == "unresolved"
    assert r["summary"]["verdict"] == "PARSER_DISAGREES"


def test_our_parse_must_round_trip_through_the_template():
    s = run_g2(FakeServer(), CONVS, COMPACT_TEMPLATE, "qwen_json")["summary"]
    assert s["our_roundtrip_rate"] == 0.0
    assert s["verdict"] == "OUR_PARSE_NOT_FAITHFUL"


def test_tool_call_rate_below_gate():
    emits = [True] * 8 + [False] * 2
    s = run_g2(FakeServer(emits=emits), CONVS, TEMPLATE, "qwen_json")["summary"]
    assert s["tool_call_rate"] == 0.8
    assert s["verdict"] == "TOOL_CALL_RATE_BELOW_GATE"


def test_correctness_rates_are_reported_but_do_not_gate():
    r = run_g2(FakeServer(), CONVS, TEMPLATE, "qwen_json")
    lookups = sum(1 for c in CONVS if c.expected_tool == "lookup_record")
    assert r["summary"]["expected_tool_rate"] == lookups / len(CONVS)
    assert r["summary"]["verdict"] == "G2_TOOLCALLS_PASS"


def test_decision_margin_recorded_and_summarised():
    r = run_g2(FakeServer(margins=[0.3, 1.0, 3.0]), CONVS, TEMPLATE, "qwen_json")
    s = r["summary"]
    assert s["median_margin"] == pytest.approx(1.0)
    assert s["margin_bins"] == {"near_tie": 4, "contested": 3, "confident": 3, "unavailable": 0}
    assert r["rows"][0]["margin_bin"] == "near_tie" and r["rows"][0]["margin_position"] == 0


def test_no_call_turns_get_an_action_margin_from_the_first_position():
    r = run_g2(FakeServer(emits=[False], margins=[0.3]), CONVS, TEMPLATE, "qwen_json")
    s = r["summary"]
    assert s["median_margin"] == pytest.approx(0.3)
    assert s["margin_bins"]["near_tie"] == len(CONVS) and s["margin_bins"]["unavailable"] == 0
    assert all(row["action"] == "no_call" for row in r["rows"])
    assert s["our_roundtrip_rate"] is None        # no calls to round-trip


def test_ambiguous_conversations_have_no_correctness_scores():
    ambiguous = generate_conversations(10, 2, seed=1, stratum="ambiguous")
    r = run_g2(FakeServer(), ambiguous, TEMPLATE, "qwen_json")
    s = r["summary"]
    assert s["strata"] == ["ambiguous"]
    assert s["expected_tool_rate"] is None and s["expected_call_exact_rate"] is None
    assert s["verdict"] == "G2_TOOLCALLS_PASS"


def test_stratum_check_verdicts():
    assert stratum_check({"median_margin": 3.0}, {"median_margin": 0.8})["verdict"] == "STRATUM_LOWERS_MARGIN"
    assert stratum_check({"median_margin": 0.8}, {"median_margin": 0.8})["verdict"] == "STRATUM_DOES_NOT_LOWER_MARGIN"
    assert stratum_check({"median_margin": None}, {"median_margin": 0.8})["verdict"] == "MARGIN_UNAVAILABLE"


def test_parse_completion_keeps_each_positions_token_text():
    resp = {"tokens": [7, 8],
            "completion_probabilities": [
                {"id": 7, "token": "<tool_call>", "logprob": -0.1, "top_logprobs": [{"id": 7, "logprob": -0.1}]},
                {"id": 8, "token": "</tool_call>", "logprob": -0.2, "top_logprobs": [{"id": 8, "logprob": -0.2}]}]}
    assert parse_completion(resp).pieces == ("<tool_call>", "</tool_call>")


def test_server_tool_call_accepts_string_and_object_arguments_in_both_shapes():
    for args in ('{"record_id": 7}', {"record_id": 7}):
        nested = {"choices": [{"message": {"tool_calls": [
            {"type": "function", "function": {"name": "lookup_record", "arguments": args}}]}}]}
        flat = {"choices": [{"message": {"tool_calls": [
            {"name": "lookup_record", "arguments": args}]}}]}
        assert server_tool_call(nested) == LOOKUP_7
        assert server_tool_call(flat) == LOOKUP_7


def test_server_tool_call_none_cases():
    assert server_tool_call({"choices": [{"message": {"content": "hi"}}]}) is None
    assert server_tool_call({}) is None
    bad = {"choices": [{"message": {"tool_calls": [{"function": {"name": "x", "arguments": "{oops"}}]}}]}
    assert server_tool_call(bad) is None


def test_server_prompt_tokens_prefers_usage_then_timings():
    assert server_prompt_tokens({"usage": {"prompt_tokens": 12}, "timings": {"prompt_n": 3}}) == 12
    assert server_prompt_tokens({"timings": {"prompt_n": 3}}) == 3
    assert server_prompt_tokens({}) == -1


def test_verdict_order():
    base = {"chat_errors": 0, "prompt_match_rate": 1.0, "parser_agreement_rate": 1.0,
            "tool_call_rate": 1.0}
    assert g2_verdict(base) == "G2_TOOLCALLS_PASS"
    assert g2_verdict({**base, "tool_call_rate": 0.89}) == "TOOL_CALL_RATE_BELOW_GATE"
    assert g2_verdict({**base, "parser_agreement_rate": 0.9, "tool_call_rate": 0.1}) == "PARSER_DISAGREES"
    assert g2_verdict({**base, "parser_agreement_rate": 0.9, "unresolved_disagreements": 0}) == "G2_TOOLCALLS_PASS"
    assert g2_verdict({**base, "our_roundtrip_rate": 0.5, "unresolved_disagreements": 3}) == "OUR_PARSE_NOT_FAITHFUL"
    assert g2_verdict({**base, "prompt_match_rate": 0.5, "parser_agreement_rate": None}) == \
        "RENDER_WITH_TOOLS_MISMATCH"
    assert g2_verdict({**base, "chat_errors": 1, "prompt_match_rate": None}) == "CHAT_PATH_ERROR"


def test_rows_record_both_parsers_arguments_for_diagnosis():
    row = run_g2(FakeServer(), CONVS, TEMPLATE, "qwen_json")["rows"][0]
    assert row["ours"]["args"] == {"record_id": 7}
    assert row["theirs"]["args"] == {"record_id": 7}


def test_ambiguous_stratum_is_held_to_the_call_floor_not_the_explicit_gate():
    ambiguous = generate_conversations(10, 2, seed=1, stratum="ambiguous")
    emits = [True] * 6 + [False] * 4
    amb = run_g2(FakeServer(emits=emits), ambiguous, TEMPLATE, "qwen_json")["summary"]
    exp = run_g2(FakeServer(emits=emits), CONVS, TEMPLATE, "qwen_json")["summary"]
    assert amb["tool_call_rate"] == exp["tool_call_rate"] == 0.6
    assert amb["verdict"] == "G2_TOOLCALLS_PASS" and amb["call_threshold"] == 0.5
    assert exp["verdict"] == "TOOL_CALL_RATE_BELOW_GATE" and exp["call_threshold"] == 0.9


def test_ambiguous_stratum_below_the_call_floor_fails():
    ambiguous = generate_conversations(10, 2, seed=1, stratum="ambiguous")
    s = run_g2(FakeServer(emits=[True] * 4 + [False] * 6), ambiguous, TEMPLATE, "qwen_json")["summary"]
    assert s["verdict"] == "TOOL_CALL_RATE_BELOW_GATE"


def test_rows_record_the_action_and_every_positions_gap():
    r = run_g2(FakeServer(emits=[True, False], margins=[1.0]), CONVS, TEMPLATE, "qwen_json")
    call_row, no_call_row = r["rows"][0], r["rows"][1]
    assert call_row["action"] == "call" and no_call_row["action"] == "no_call"
    assert len(call_row["gaps"]) == len(CALL_PIECES) and call_row["gaps"][0] == pytest.approx(1.0)
    assert no_call_row["parser_agrees"] is True          # both parsers agree there is no call
    assert r["summary"]["no_call_rate"] == 0.5


def test_offline_rescoring_reproduces_the_live_summary():
    live = run_g2(FakeServer(emits=[True, True, False], server_args='{"record_id": 8}'),
                  CONVS, TEMPLATE, "qwen_json")
    saved = copy.deepcopy(live["rows"])
    for row in saved:
        row.pop("our_roundtrip"), row.pop("adjudication")
    adjudicate_rows(saved, TEMPLATE, "qwen_json")
    assert summarise(saved) == live["summary"]
