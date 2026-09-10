"""Agent conversations with real tool round trips (spec 4.4).

Each conversation is a system prompt, a history of completed tool round trips
(user request, assistant tool call, tool result, assistant summary), and a
final user request that needs a tool. The expected tool and arguments are kept
so a completion can be scored for whether it called the right thing.

Two wire formats exist for the same conversation:
- template form (the default): tool-call arguments are dicts, which is what
  chat templates need - Spark's raises on anything else;
- OpenAI form (to_oai): arguments are JSON strings, as /v1/chat/completions
  expects. The string is json.dumps(args, ensure_ascii=False), which produces
  the same bytes as the renderer's tojson, so both paths can render the same
  prompt if the server converts faithfully. G2 checks that it does.
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


def openai_tools():
    return [{"type": "function", "function": copy.deepcopy(s)} for s in TOOL_SCHEMAS]


@dataclass(frozen=True)
class Conversation:
    conv_id: str
    messages: list
    tools: list
    expected_tool: str
    expected_args: dict


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


def generate_conversations(n: int, history_turns: int, seed: int):
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


def to_oai(messages):
    """Copy of messages with tool-call arguments as JSON strings (OpenAI wire format)."""
    out = copy.deepcopy(messages)
    for m in out:
        for tc in m.get("tool_calls") or []:
            fn = tc.get("function", {})
            if not isinstance(fn.get("arguments"), str):
                fn["arguments"] = json.dumps(fn.get("arguments", {}), ensure_ascii=False)
    return out
