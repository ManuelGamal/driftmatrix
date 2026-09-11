"""Gate G2, tool calls: are the study's prompts the server's, and do they act as intended?

For each conversation, with tools on and thinking off:

  1. render offline, tokenize with the server, complete through /completion -
     the study's own path, with no restore involved - and parse the output with
     the model's parser. Top-k logprobs give the action margin (spec 5.1);
  2. send the same conversation to /v1/chat/completions, where the server
     renders it itself, constrains generation with its tool-call grammar, and
     parses tool calls with llama.cpp's parser.

The agent's action is a tool call with its arguments, or no call (a clarifying
question or text answer). Checks, in verdict order:

  chat errors    the server rejected the OpenAI-format conversation. llama.cpp's
                 docs do not specify how assistant tool_calls and tool messages
                 must be shaped in requests, so a rejection is recorded, not raised.
  prompt_match   the server's prompt token count equals our offline render's.
                 /apply-template cannot render tools, so this is the only
                 end-to-end check that prompts carrying tools match the server's.
  round trip     every call we parse must reproduce the model's own <tool_call>
                 text when serialised back through its chat template (v4.4).
                 This validates our parser against the model's format itself.
  parsers        disagreements with the server's parsed call are adjudicated by
                 the same round trip. If our reading reproduces the model's text
                 and the server's does not, the difference is on the server's side
                 (labelled server_parser_lossy) and does not fail the gate;
                 anything else does. The label predates the diagnosis: in G2 the
                 cause was the chat endpoint's tool grammar forcing schema argument
                 order, which changed the generation, not a parsing error.
  call rate      share of outputs that are tool calls. Explicit requests must
                 reach 90%: they validate the tool-call pipeline. Ambiguous
                 requests are held to a 50% floor: the requests that create
                 near-tie decisions also sometimes end in a clarifying question,
                 and that outcome is part of what is measured (v4.3).

adjudicate_rows and summarise work on saved report rows as well as fresh ones,
so reports can be re-scored offline (scripts/g2_rescore.py).
"""
from __future__ import annotations

import json
import statistics

from .metrics.behavioral import ToolCall
from .metrics.margin import BIN_LABELS, decision_margin, margin_bin, position_gaps
from .runner.llamaserver import ServerError
from .traces.conversations import to_oai
from .traces.render import render
from .traces.roundtrip import in_emitted_order, round_trips
from .traces.toolcalls import parse_tool_call, parse_tool_call_ordered

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
    roundtrip = summary.get("our_roundtrip_rate")
    if roundtrip is not None and roundtrip < 1.0:
        return "OUR_PARSE_NOT_FAITHFUL"
    if "unresolved_disagreements" in summary:
        if summary["unresolved_disagreements"]:
            return "PARSER_DISAGREES"
    elif summary["parser_agreement_rate"] != 1.0:
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


def adjudicate_rows(rows, template: str, fmt: str, specials=None):
    """Round-trip every parsed call and adjudicate parser disagreements, in place."""
    specials = specials or {}
    for row in rows:
        parsed = parse_tool_call_ordered(row["output"], fmt)
        row["our_roundtrip"] = None if parsed is None else round_trips(
            template, parsed[2], parsed[0], parsed[1], **specials)
        row["adjudication"] = None
        if row["parser_agrees"] is not False:
            continue
        theirs = row["theirs"]
        if parsed is None or (theirs is not None and theirs.get("args") is None):
            row["adjudication"] = "unresolved"       # a call we did not find, or no arguments to test
            continue
        name, args, block = parsed
        theirs_roundtrip = theirs is not None and round_trips(
            template, block, theirs["name"], in_emitted_order(theirs["args"], list(args)), **specials)
        if row["our_roundtrip"] and not theirs_roundtrip:
            row["adjudication"] = "server_parser_lossy"
        elif theirs_roundtrip and not row["our_roundtrip"]:
            row["adjudication"] = "our_parser_lossy"
        else:
            row["adjudication"] = "unresolved"
    return rows


def summarise(rows) -> dict:
    strata = sorted({r["stratum"] for r in rows})
    calls = [r for r in rows if r["tool_call_made"]]
    bins = {label: sum(1 for r in rows if r["margin_bin"] == label) for label in BIN_LABELS}
    bins["unavailable"] = sum(1 for r in rows if r["margin_bin"] is None)
    summary = {
        "n": len(rows),
        "strata": strata,
        "chat_errors": sum(1 for r in rows if r["chat_error"]),
        "prompt_match_rate": _rate(r["prompt_match"] for r in rows),
        "parser_agreement_rate": _rate(r["parser_agrees"] for r in rows),
        "our_roundtrip_rate": _rate(r.get("our_roundtrip") for r in calls),
        "parser_disagreements": sum(1 for r in rows if r["parser_agrees"] is False),
        "server_parser_lossy": sum(1 for r in rows if r.get("adjudication") == "server_parser_lossy"),
        "unresolved_disagreements": sum(1 for r in rows
                                        if r.get("adjudication") in ("unresolved", "our_parser_lossy")),
        "tool_call_rate": _rate(r["tool_call_made"] for r in rows),
        "no_call_rate": _rate(not r["tool_call_made"] for r in rows),
        "call_threshold": call_threshold(strata),
        "expected_tool_rate": _rate(r["expected_tool_called"] for r in rows),
        "expected_call_exact_rate": _rate(r["expected_call_exact"] for r in rows),
        "median_margin": _median(r["decision_margin"] for r in rows),
        "margin_bins": bins,
    }
    summary["verdict"] = g2_verdict(summary)
    return summary


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
            "output": output,
        })

    adjudicate_rows(rows, template, fmt, specials)
    return {"summary": summarise(rows), "rows": rows}
