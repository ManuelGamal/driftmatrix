"""Agent-shaped trace generation (spec 4.3).

Stable system prompt + stable tool schemas + monotonically growing history is
the shape that makes state reuse meaningful. The framework is the traffic
generator, never the contribution.
"""
from __future__ import annotations
import json
import random
from dataclasses import dataclass
from .tools import TOOL_SCHEMAS, call_tool

SYSTEM_PROMPT = (
    "You are an operations assistant. Use the provided tools when they help. "
    "Answer concisely and cite any record ids you used.\n\n"
    "Available tools:\n" + json.dumps(TOOL_SCHEMAS, sort_keys=True, indent=2) + "\n"
)

_QUESTIONS = [
    "Look up record {rid} and summarise it.",
    "Sum the values {vals} and report the total.",
    "Format a report titled 'Batch {rid}' describing record {rid}.",
    "Compare record {rid} against the running total of {vals}.",
]


@dataclass(frozen=True)
class Turn:
    session_id: str
    turn_idx: int
    prefix_text: str
    suffix_text: str
    expected_tool: str


def _tool_for(question: str):
    if "Look up" in question:
        return "lookup_record"
    if "Sum the" in question:
        return "compute_sum"
    if "Format a" in question:
        return "format_report"
    return None


def _args_for(tool, rid, vals):
    if tool == "lookup_record":
        return {"record_id": rid}
    if tool == "compute_sum":
        return {"values": vals}
    return {"title": f"Batch {rid}", "body": f"record {rid}"}


def generate_sessions(n_sessions: int, turns_per_session: int, seed: int):
    rng = random.Random(seed)
    turns = []
    for s in range(n_sessions):
        sid = f"s{seed}_{s:03d}"
        history = SYSTEM_PROMPT
        for t in range(turns_per_session):
            rid = rng.randrange(0, 200)
            vals = [rng.randrange(1, 50) for _ in range(3)]
            q = rng.choice(_QUESTIONS).format(rid=rid, vals=vals)
            tool = _tool_for(q)
            turns.append(Turn(session_id=sid, turn_idx=t, prefix_text=history,
                              suffix_text=f"\nUser: {q}\nAssistant:",
                              expected_tool=tool))
            observed = call_tool(tool, _args_for(tool, rid, vals)) if tool else "(no tool)"
            history += f"\nUser: {q}\nAssistant: [{tool}] {observed}\n"
    return turns
