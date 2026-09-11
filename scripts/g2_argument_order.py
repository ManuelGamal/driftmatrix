"""Does llama-server's tool grammar force schema argument order? (docs/findings/2026-09-11-spark-argument-order.md)

Start llama-server for Spark as in G2 (with --jinja), then:

    python scripts/g2_argument_order.py --gguf "<Spark .gguf path>"

Sends G2 conversation a2026_024 three ways, greedy:
  1. /completion on the offline render (no grammar): the model's unconstrained text
  2. /v1/chat/completions with the original schema (title before body)
  3. /v1/chat/completions with the schema's properties reversed (body before title)

If the grammar forces required arguments into schema order, run 2 comes back
title first with a body that lacks "record 165: ", and run 3 keeps the model's
own body-first call with the full body.

Both request paths disable prompt caching, so the slot erase between runs is only
hygiene and is skipped when the server was started without --slot-save-path.
"""
import argparse
import copy
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from driftmatrix.runner.llamaserver import ServerClient, ServerError  # noqa: E402
from driftmatrix.traces.conversations import generate_conversations, to_oai  # noqa: E402
from driftmatrix.traces.render import load_specials, load_template, render  # noqa: E402


def erase_if_supported(client):
    try:
        client.slot_erase()
    except ServerError as e:
        if e.status != 501:
            raise


def reversed_properties(tools):
    out = copy.deepcopy(tools)
    for t in out:
        props = t["function"]["parameters"].get("properties", {})
        t["function"]["parameters"]["properties"] = dict(reversed(list(props.items())))
        if "required" in t["function"]["parameters"]:
            t["function"]["parameters"]["required"] = list(reversed(t["function"]["parameters"]["required"]))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8090")
    ap.add_argument("--gguf", required=True)
    args = ap.parse_args()

    conv = next(c for c in generate_conversations(30, 3, seed=2026, stratum="ambiguous") if c.conv_id == "a2026_024")
    template, specials = load_template(args.gguf), load_specials(args.gguf)
    client = ServerClient(args.url)
    client.wait_healthy()

    text = render(template, conv.messages, conv.tools, add_generation_prompt=True, enable_thinking=False, **specials)
    erase_if_supported(client)
    raw = client.complete(client.tokenize(text), 192, 0, cache_prompt=False)
    print("1. /completion, no grammar:")
    print("   " + (raw.raw.get("content") or ""))

    for label, tools in (("2. chat, schema order title->body", conv.tools),
                         ("3. chat, schema order body->title", reversed_properties(conv.tools))):
        erase_if_supported(client)
        resp = client.chat(to_oai(conv.messages), tools, max_tokens=192, chat_template_kwargs={"enable_thinking": False})
        msg = resp["choices"][0]["message"]
        print(f"\n{label}:")
        for call in msg.get("tool_calls") or []:
            print("   " + json.dumps(call.get("function", call), ensure_ascii=False))
        if msg.get("content"):
            print("   content: " + msg["content"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
