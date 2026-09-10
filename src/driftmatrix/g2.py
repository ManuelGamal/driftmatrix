"""Gate G2, tool calls: are the study's prompts the server's, and do they act as intended?

For each conversation, with tools on and thinking off:

  1. render offline, tokenize with the server, complete through /completion -
     the study's own path, with no restore involved - and parse the output with
     the model's parser. Top-k logprobs give the action margin (spec 5.1);
  2. send the same conversation to /v1/chat/completions, where the server
     renders it itself and parses tool calls with llama.cpp's parser.

The agent's action is a tool call with its arguments, or no call (a clarifying
question or text answer). Checks, in verdict order:

  chat errors    the server rejected the OpenAI-format conversation. llama.cpp's
                 docs do not specify how assistant tool_calls and tool messages
                 must be shaped in requests, so a rejection is recorded, not raised.
  prompt_match   the server's prompt token count equals our offline render's.
                 /apply-template cannot render tools, so this is the only
                 end-to-end check that prompts carrying tools match the server's.
  parser_agrees  our parse of the /completion output equals llama.cpp's parsed
                 tool_calls from the chat path, including agreeing that there is
                 no call. Same prompt, greedy decoding, same output text.
  call rate      share of outputs that are tool calls. Explicit requests must
                 reach 90%: they validate the tool-call pipeline. Ambiguous
                 requests are held to a 50% floor instead - the requests that
                 create near-tie decisions also sometimes end in a clarifying
                 question, and that outcome is part of what is measured (v4.3).

Correct-call rates are reported for explicit requests only; ambiguous requests
have no single correct action. stratum_check compares two strata's reports: the
ambiguous stratum must lower the median action margin.
"""
from __future__ import annotations

import json
import statistics

from .metrics.behavioral import ToolCall
from .metrics.margin import BIN_LABELS, decision_margin, margin_bin, position_gaps
from .runner.llamaserver import ServerError
from .traces.conversations import to_oai
from .traces.render import render
from .traces.toolcalls import parse_tool_call

TOOL_CALL_GATE = 0.90
AMBIGUOUS_CALL_FLOOR = 0.50


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


def _median(values):
    vals = [v for v in values if v is not None]
    return statistics.median(vals) if vals else None


def call_threshold(strata) -> float:
    return AMBIGUOUS_CALL_FLOOR if "ambiguous" in (strata or []) else TOOL_CALL_GATE


def g2_verdict(summary: dict) -> str:
    if summary.get("chat_errors", 0):
        return "CHAT_PATH_ERROR"
    if summary["prompt_match_rate"] != 1.0:
        return "RENDER_WITH_TOOLS_MISMATCH"
    if summary["parser_agreement_rate"] != 1.0:
        return "PARSER_DISAGREES"
    if summary["tool_call_rate"] < call_threshold(summary.get("strata")):
        return "TOOL_CALL_RATE_BELOW_GATE"
    return "G2_TOOLCALLS_PASS"


def stratum_check(explicit: dict, ambiguous: dict) -> dict:
    """Does the ambiguous stratum lower the median action margin?"""
    e, a = explicit.get("median_margin"), ambiguous.get("median_margin")
    out = {"explicit_median_margin": e, "ambiguous_median_margin": a,
           "explicit_tool_call_rate": explicit.get("tool_call_rate"),
           "ambiguous_tool_call_rate": ambiguous.get("tool_call_rate")}
    if e is None or a is None:
        out["verdict"] = "MARGIN_UNAVAILABLE"
    elif a < e:
        out["verdict"] = "STRATUM_LOWERS_MARGIN"
    else:
        out["verdict"] = "STRATUM_DOES_NOT_LOWER_MARGIN"
    return out


def _call_dict(call):
    return None if call is None else {"name": call.name, "args_hash": call.args_hash,
                                      "args": call.args}


def run_g2(client, conversations, template: str, fmt: str, specials=None,
           n_predict: int = 192, n_probs: int = 10) -> dict:
    specials = specials or {}
    rows = []
    for conv in conversations:
        text = render(template, conv.messages, conv.tools, add_generation_prompt=True,
                      enable_thinking=False, **specials)
        toks = client.tokenize(text)
        client.slot_erase()
        comp = client.complete(toks, n_predict, n_probs, cache_prompt=False)
        output = comp.raw.get("content") or ""
        ours = parse_tool_call(output, fmt)
        margin, margin_pos = decision_margin(comp.top, comp.pieces)

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
        expected = (ToolCall.from_payload(conv.expected_tool, conv.expected_args)
                    if conv.expected_tool else None)

        rows.append({
            "conv_id": conv.conv_id,
            "stratum": getattr(conv, "stratum", "explicit"),
            "offline_prompt_tokens": len(toks),
            "server_prompt_tokens": n_server,
            "chat_error": chat_error,
            "prompt_match": prompt_match,
            "action": "call" if ours is not None else "no_call",
            "tool_call_made": ours is not None,
            "parser_agrees": (ours == theirs) if prompt_match else None,
            "ours": _call_dict(ours),
            "theirs": _call_dict(theirs),
            "expected_tool_called": (None if expected is None
                                     else ours is not None and ours.name == conv.expected_tool),
            "expected_call_exact": None if expected is None else ours == expected,
            "decision_margin": margin,
            "margin_position": margin_pos,
            "margin_bin": margin_bin(margin),
            "gaps": [None if g is None else round(g, 4) for g in position_gaps(comp.top)],
            "output": output[:400],
        })

    strata = sorted({r["stratum"] for r in rows})
    bins = {label: sum(1 for r in rows if r["margin_bin"] == label) for label in BIN_LABELS}
    bins["unavailable"] = sum(1 for r in rows if r["margin_bin"] is None)
    summary = {
        "n": len(rows),
        "strata": strata,
        "chat_errors": sum(1 for r in rows if r["chat_error"]),
        "prompt_match_rate": _rate(r["prompt_match"] for r in rows),
        "parser_agreement_rate": _rate(r["parser_agrees"] for r in rows),
        "tool_call_rate": _rate(r["tool_call_made"] for r in rows),
        "no_call_rate": _rate(not r["tool_call_made"] for r in rows),
        "call_threshold": call_threshold(strata),
        "expected_tool_rate": _rate(r["expected_tool_called"] for r in rows),
        "expected_call_exact_rate": _rate(r["expected_call_exact"] for r in rows),
        "median_margin": _median(r["decision_margin"] for r in rows),
        "margin_bins": bins,
    }
    summary["verdict"] = g2_verdict(summary)
    return {"summary": summary, "rows": rows}
