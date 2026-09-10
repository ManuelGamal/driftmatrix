"""Agent conversations with real tool round trips (spec 4.4).

Each conversation is a system prompt, a history of completed tool round trips
(user request, assistant tool call, tool result, assistant summary), and a
final user request.

Two wire formats exist for the same conversation:
- template form (the default): tool-call arguments are dicts, which is what
  chat templates need - Spark's raises on anything else;
- OpenAI form (to_oai): arguments are JSON strings, as /v1/chat/completions
  expects. The string is json.dumps(args, ensure_ascii=False), which produces
  the same bytes as the renderer's tojson.

Two strata (spec v4.2):
- explicit: the final request names one tool and its arguments. Generation is
  unchanged from the G2 run, so its committed reports stay reproducible.
- ambiguous: the explicit twin with only the final request replaced by one that
  is still answerable with a tool call but leaves a real choice - more than one
  plausible tool, or an argument to infer from history. Both strata therefore
  share the same history, which pairs the stratum contrast. There is no single
  correct call, so expected_tool and expected_args are None.

The five ambiguous request types rotate evenly across conversations, so no
single phrasing dominates the stratum. How ambiguous any one instance is varies
with its history; that variation is what the recompute-arm decision margin
measures (spec 5.1), not something the generator tries to control.
"""
from __future__ import annotations

import copy
import json
import random
from dataclasses import dataclass

from .tools import TOOL_SCHEMAS, call_tool

SYSTEM = ("You are an operations assistant with access to tools. When a request needs "
          "data or a computation, call the appropriate tool instead of answering from "
          "memory.")

STRATA = ("explicit", "ambiguous")


def openai_tools():
    return [{"type": "function", "function": copy.deepcopy(s)} for s in TOOL_SCHEMAS]


@dataclass(frozen=True)
class Conversation:
    conv_id: str
    messages: list
    tools: list
    expected_tool: str | None
    expected_args: dict | None
    stratum: str = "explicit"


# ---------------------------------------------------------------- explicit requests
def _lookup(rng):
    args = {"record_id": rng.randrange(0, 200)}
    return "lookup_record", args, f"Look up record {args['record_id']} and tell me its field."


def _sum(rng):
    args = {"values": [rng.randrange(1, 50) for _ in range(3)]}
    return "compute_sum", args, f"What is the sum of {', '.join(map(str, args['values']))}?"


def _report(rng):
    args = {"title": f"Batch {rng.randrange(1, 99)}", "body": "status nominal"}
    return "format_report", args, f"Format a report titled '{args['title']}' with body '{args['body']}'."


_REQUESTS = (_lookup, _sum, _report)


def _explicit(n: int, history_turns: int, seed: int):
    rng = random.Random(seed)
    tools = openai_tools()
    out = []
    for c in range(n):
        msgs = [{"role": "system", "content": SYSTEM}]
        for t in range(history_turns):
            name, args, request = rng.choice(_REQUESTS)(rng)
            call_id = f"call_{seed}_{c}_{t}"
            result = call_tool(name, args)
            msgs.append({"role": "user", "content": request})
            msgs.append({"role": "assistant", "content": "",
                         "tool_calls": [{"id": call_id, "type": "function",
                                         "function": {"name": name, "arguments": args}}]})
            msgs.append({"role": "tool", "tool_call_id": call_id, "content": result})
            msgs.append({"role": "assistant", "content": f"Done. {result}"})
        name, args, request = rng.choice(_REQUESTS)(rng)
        msgs.append({"role": "user", "content": request})
        out.append(Conversation(f"c{seed}_{c:03d}", msgs, tools, name, args))
    return out


# --------------------------------------------------------------- ambiguous requests
def _amb_that_record(rng):
    return "Pull up that record again, the one from before."           # which one?


def _amb_something_on_record(rng):
    return f"Give me something on record {rng.randrange(0, 200)}."     # look up, or report?


def _amb_total_records(rng):
    a, b = rng.randrange(0, 200), rng.randrange(0, 200)
    return f"Can you total up records {a} and {b}?"                   # look up first, or sum?


def _amb_add_earlier(rng):
    return "Add up the numbers from earlier."                          # which numbers?


def _amb_write_up(rng):
    return "Write that up for me."                                     # what title and body?


AMBIGUOUS_TYPES = (_amb_that_record, _amb_something_on_record, _amb_total_records,
                   _amb_add_earlier, _amb_write_up)


def generate_conversations(n: int, history_turns: int, seed: int, stratum: str = "explicit"):
    if stratum not in STRATA:
        raise ValueError(f"unknown stratum: {stratum}")
    explicit = _explicit(n, history_turns, seed)
    if stratum == "explicit":
        return explicit
    out = []
    for i, conv in enumerate(explicit):
        rng = random.Random(f"{seed}:ambiguous:{conv.conv_id}")
        request = AMBIGUOUS_TYPES[i % len(AMBIGUOUS_TYPES)](rng)
        msgs = copy.deepcopy(conv.messages[:-1]) + [{"role": "user", "content": request}]
        out.append(Conversation("a" + conv.conv_id[1:], msgs, conv.tools, None, None, "ambiguous"))
    return out


def to_oai(messages):
    """Copy of messages with tool-call arguments as JSON strings (OpenAI wire format)."""
    out = copy.deepcopy(messages)
    for m in out:
        for tc in m.get("tool_calls") or []:
            fn = tc.get("function", {})
            if not isinstance(fn.get("arguments"), str):
                fn["arguments"] = json.dumps(fn.get("arguments", {}), ensure_ascii=False)
    return out
