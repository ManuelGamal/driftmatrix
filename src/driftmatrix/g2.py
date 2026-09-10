"""Gate G2, tool calls: are the study's prompts the server's, and do they call tools?

For each conversation, with tools on and thinking off:

  1. render offline, tokenize with the server, complete through /completion -
     the study's own path - and parse the output with the model's parser;
  2. send the same conversation to /v1/chat/completions, where the server
     renders it itself and parses tool calls with llama.cpp's parser.

Checks, in verdict order:

  chat errors    the server rejected the OpenAI-format conversation. llama.cpp's
                 docs do not specify how assistant tool_calls and tool messages
                 must be shaped in requests, so a rejection is recorded, not raised.
  prompt_match   the server's prompt token count equals our offline render's.
                 /apply-template cannot render tools, so this is the only
                 end-to-end check that prompts carrying tools match the
                 server's - including its conversion of string arguments to
                 objects for templates that require them.
  parser_agrees  our parse of the /completion output equals llama.cpp's parsed
                 tool_calls from the chat path. Same prompt, greedy decoding,
                 so the model output is the same text in both paths.
  tool-call rate share of outputs that parse as a tool call; gate >= 0.90.

Correct-tool and correct-arguments rates are reported but do not gate: the
study measures whether restore changes the call, not whether the call is right.
"""
from __future__ import annotations

import json

from .metrics.behavioral import ToolCall
from .runner.llamaserver import ServerError
from .traces.conversations import to_oai
from .traces.render import render
from .traces.toolcalls import parse_tool_call

TOOL_CALL_GATE = 0.90


def server_tool_call(resp):
    """First tool call from a chat completion, as a ToolCall.

    Accepts both the OpenAI shape ({"function": {"name", "arguments"}}) and the
    flat shape shown in llama.cpp's function-calling docs ({"name", "arguments"}).
    """
    try:
        calls = resp["choices"][0]["message"].get("tool_calls") or []
    except (KeyError, IndexError, TypeError, AttributeError):
        return None
    if not calls or not isinstance(calls[0], dict):
        return None
    fn = calls[0]["function"] if isinstance(calls[0].get("function"), dict) else calls[0]
    name, args = fn.get("name"), fn.get("arguments", {})
    if isinstance(args, str):
        try:
            args = json.loads(args) if args.strip() else {}
        except ValueError:
            return None
    if not isinstance(name, str) or not isinstance(args, dict):
        return None
    return ToolCall.from_payload(name, args)


def server_prompt_tokens(resp) -> int:
    usage = resp.get("usage") or {}
    if isinstance(usage.get("prompt_tokens"), int):
        return usage["prompt_tokens"]
    timings = resp.get("timings") or {}
    n = timings.get("prompt_n")
    return n if isinstance(n, int) else -1


def _rate(values):
    vals = [v for v in values if v is not None]
    return sum(1 for v in vals if v) / len(vals) if vals else None


def g2_verdict(summary: dict) -> str:
    if summary.get("chat_errors", 0):
        return "CHAT_PATH_ERROR"
    if summary["prompt_match_rate"] != 1.0:
        return "RENDER_WITH_TOOLS_MISMATCH"
    if summary["parser_agreement_rate"] != 1.0:
        return "PARSER_DISAGREES"
    if summary["tool_call_rate"] < TOOL_CALL_GATE:
        return "TOOL_CALL_RATE_BELOW_GATE"
    return "G2_TOOLCALLS_PASS"


def run_g2(client, conversations, template: str, fmt: str, specials=None,
           n_predict: int = 192) -> dict:
    specials = specials or {}
    rows = []
    for conv in conversations:
        text = render(template, conv.messages, conv.tools, add_generation_prompt=True,
                      enable_thinking=False, **specials)
        toks = client.tokenize(text)
        client.slot_erase()
        comp = client.complete(toks, n_predict, 0, cache_prompt=False)
        output = comp.raw.get("content") or ""
        ours = parse_tool_call(output, fmt)

        client.slot_erase()
        try:
            chat = client.chat(to_oai(conv.messages), conv.tools, max_tokens=n_predict,
                               chat_template_kwargs={"enable_thinking": False})
            chat_error = None
        except ServerError as e:
            chat, chat_error = {}, str(e)
        theirs = server_tool_call(chat)
        n_server = server_prompt_tokens(chat)
        prompt_match = None if chat_error else n_server == len(toks)
        expected = ToolCall.from_payload(conv.expected_tool, conv.expected_args)

        rows.append({
            "conv_id": conv.conv_id,
            "offline_prompt_tokens": len(toks),
            "server_prompt_tokens": n_server,
            "chat_error": chat_error,
            "prompt_match": prompt_match,
            "tool_call_made": ours is not None,
            "parser_agrees": (ours == theirs) if prompt_match else None,
            "ours": None if ours is None else {"name": ours.name, "args_hash": ours.args_hash},
            "theirs": None if theirs is None else {"name": theirs.name, "args_hash": theirs.args_hash},
            "expected_tool_called": ours is not None and ours.name == conv.expected_tool,
            "expected_call_exact": ours == expected,
            "output": output[:400],
        })

    summary = {
        "n": len(rows),
        "chat_errors": sum(1 for r in rows if r["chat_error"]),
        "prompt_match_rate": _rate(r["prompt_match"] for r in rows),
        "parser_agreement_rate": _rate(r["parser_agrees"] for r in rows),
        "tool_call_rate": _rate(r["tool_call_made"] for r in rows),
        "expected_tool_rate": _rate(r["expected_tool_called"] for r in rows),
        "expected_call_exact_rate": _rate(r["expected_call_exact"] for r in rows),
    }
    summary["verdict"] = g2_verdict(summary)
    return {"summary": summary, "rows": rows}
