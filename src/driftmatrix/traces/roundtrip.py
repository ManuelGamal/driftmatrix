"""Template round trip: does a parsed tool call reproduce the model's own text? (spec 4.4)

A tool call is faithfully parsed when serialising it back through the model's
chat template - arguments in the order the model emitted them - reproduces the
<tool_call> block the model generated, byte for byte.

This validates a parser against the model's own format rather than against
another parser, and it adjudicates disagreements with the server's parsed calls:
in G2's ambiguous stratum, one Spark call returned by llama-server did not
reproduce the model's text while our reading did. The cause was llama-server's
tool-call grammar forcing schema argument order during generation, not its
parser (docs/findings/2026-09-11-spark-argument-order.md).
"""
from __future__ import annotations

from .render import render


def serialise_call(template: str, name: str, args: dict, **specials) -> str:
    """Render a minimal conversation whose last assistant turn is exactly this call."""
    messages = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "u"},
        {"role": "assistant", "content": "",
         "tool_calls": [{"id": "roundtrip", "type": "function",
                         "function": {"name": name, "arguments": args}}]},
    ]
    return render(template, messages, None, add_generation_prompt=False,
                  enable_thinking=False, **specials)


def round_trips(template: str, block: str, name: str, args: dict, **specials) -> bool:
    try:
        return block in serialise_call(template, name, args, **specials)
    except Exception:  # noqa: BLE001 - a template that rejects the call cannot reproduce it
        return False


def in_emitted_order(args: dict, order) -> dict:
    """Reorder parsed arguments to follow the model's emitted key order."""
    ordered = {k: args[k] for k in order if k in args}
    ordered.update({k: v for k, v in args.items() if k not in ordered})
    return ordered
