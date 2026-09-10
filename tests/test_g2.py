"""G2 tool-call logic against a fake server exposing both paths."""
from driftmatrix.g2 import g2_verdict, run_g2, server_prompt_tokens, server_tool_call
from driftmatrix.metrics.behavioral import ToolCall
from driftmatrix.runner.llamaserver import Completion, ServerError
from driftmatrix.traces.conversations import generate_conversations

TEMPLATE = ("{% for m in messages %}{{ m.role }}: {{ m.content }} {% endfor %}"
            "{% if tools %}tools: {{ tools | length }}{% endif %}")
CONVS = generate_conversations(10, 2, seed=1)
CALL_TEXT = '<tool_call>\n{"name": "lookup_record", "arguments": {"record_id": 7}}\n</tool_call>'
LOOKUP_7 = ToolCall.from_payload("lookup_record", {"record_id": 7})


class FakeServer:
    def __init__(self, emits=None, prompt_offset=0, server_parses=True, chat_raises=False,
                 flat_shape=False):
        self.emits = emits or [True]
        self.offset, self.server_parses = prompt_offset, server_parses
        self.chat_raises, self.flat_shape = chat_raises, flat_shape
        self.i, self.last_len, self.last_emit = 0, 0, True

    def tokenize(self, text, add_special=True):
        toks = text.split()
        self.last_len = len(toks)
        return toks

    def slot_erase(self, slot=0):
        return {}

    def complete(self, toks, n_predict, n_probs=0, cache_prompt=False, slot=0, seed=1234):
        self.last_emit = self.emits[self.i % len(self.emits)]
        self.i += 1
        content = CALL_TEXT if self.last_emit else "Record 7 is alpha."
        return Completion([], [], len(toks), 0, {"content": content})

    def chat(self, messages, tools=None, *, max_tokens=192, chat_template_kwargs=None, seed=1234):
        if self.chat_raises:
            raise ServerError(400, "unsupported message shape", "/v1/chat/completions")
        assert chat_template_kwargs == {"enable_thinking": False}
        assert all(isinstance(tc["function"]["arguments"], str)
                   for m in messages for tc in (m.get("tool_calls") or []))
        calls = []
        if self.last_emit and self.server_parses:
            fn = {"name": "lookup_record", "arguments": '{"record_id": 7}'}
            calls = [fn] if self.flat_shape else [{"id": "x", "type": "function", "function": fn}]
        return {"choices": [{"message": {"content": None, "tool_calls": calls}}],
                "usage": {"prompt_tokens": self.last_len + self.offset}}


def test_passes_when_prompts_match_parsers_agree_and_calls_are_made():
    s = run_g2(FakeServer(), CONVS, TEMPLATE, "qwen_json")["summary"]
    assert s["verdict"] == "G2_TOOLCALLS_PASS"
    assert s["prompt_match_rate"] == s["parser_agreement_rate"] == s["tool_call_rate"] == 1.0
    assert s["chat_errors"] == 0


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


def test_parser_disagreement_is_caught():
    s = run_g2(FakeServer(server_parses=False), CONVS, TEMPLATE, "qwen_json")["summary"]
    assert s["verdict"] == "PARSER_DISAGREES"


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
    assert g2_verdict({**base, "prompt_match_rate": 0.5, "parser_agreement_rate": None}) == \
        "RENDER_WITH_TOOLS_MISMATCH"
    assert g2_verdict({**base, "chat_errors": 1, "prompt_match_rate": None}) == "CHAT_PATH_ERROR"
