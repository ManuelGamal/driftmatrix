"""G2 render check: offline chat-template rendering must equal the server's own.

The main study renders prompts offline because /apply-template accepts no
tools. That is only safe if the offline renderer reproduces the server's
rendering exactly, so this compares them byte for byte on tool-less
conversations of increasing length.

Start llama-server for one model (see G2 instructions), then:

    python scripts/g2_render_check.py --model ornith --gguf <path to that model's .gguf>

Writes data/g2_render_<model>.json.
"""
import argparse
import itertools
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from driftmatrix.runner.llamaserver import ServerClient  # noqa: E402
from driftmatrix.traces.render import load_specials, load_template, render  # noqa: E402
from driftmatrix.traces.tools import call_tool  # noqa: E402

VARIANTS = list(itertools.product((True, False), (True, None, False)))  # (add_generation_prompt, enable_thinking)


def conversations(n: int):
    """Tool-less conversations of 1..n exchanges, each ending on a user turn."""
    out = []
    for size in range(1, n + 1):
        msgs = [{"role": "system", "content": "You are an operations assistant. Answer concisely."}]
        for i in range(size):
            rid = (i * 37) % 200
            msgs.append({"role": "user", "content": f"Look up record {rid} and summarise it."})
            msgs.append({"role": "assistant",
                         "content": f"Summary: {call_tool('lookup_record', {'record_id': rid})}."})
        msgs.append({"role": "user", "content": "Thanks. What should I check next?"})
        out.append(msgs)
    return out


def first_difference(a: str, b: str):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8090")
    ap.add_argument("--model", required=True, help="label, e.g. ornith / qwen3 / spark")
    ap.add_argument("--gguf", required=True, help="the same .gguf the server loaded")
    ap.add_argument("--n", type=int, default=12)
    args = ap.parse_args()

    specials = load_specials(args.gguf)
    template = load_template(args.gguf)

    client = ServerClient(args.url)
    print(f"waiting for {args.url} ...", flush=True)
    client.wait_healthy()

    convs = conversations(args.n)
    server_prompts = [client.apply_template(m) for m in convs]

    results = {}
    for add_gen, thinking in VARIANTS:
        key = f"add_generation_prompt={add_gen}, enable_thinking={thinking}"
        matches, first_bad = 0, None
        for msgs, served in zip(convs, server_prompts):
            offline = render(template, msgs, None, add_generation_prompt=add_gen,
                             enable_thinking=thinking, **specials)
            if offline == served:
                matches += 1
            elif first_bad is None:
                i = first_difference(offline, served)
                first_bad = {"index": i,
                             "offline": offline[max(0, i - 60): i + 60],
                             "server": served[max(0, i - 60): i + 60]}
        results[key] = {"matches": matches, "of": len(convs), "first_mismatch": first_bad}

    exact = [k for k, v in results.items() if v["matches"] == v["of"]]
    verdict = "RENDER_MATCHES" if exact else "RENDER_MISMATCH"

    report = {"model": args.model, "gguf": Path(args.gguf).name,
              "build": client.props().get("build_info"), "n": len(convs),
              "verdict": verdict, "matching_variants": exact, "variants": results}
    out = REPO / "data" / f"g2_render_{args.model}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n{args.model}: {len(convs)} tool-less conversations compared with /apply-template\n")
    for k, v in results.items():
        print(f"  {v['matches']:>3}/{v['of']}  {k}")
    print(f"\nVERDICT: {verdict}")
    if exact:
        print(f"  byte-exact under: {exact}")
        print("  The main study renders with enable_thinking=False on purpose; a match under any")
        print("  thinking setting proves the renderer's mechanics (escaping, whitespace, specials).")
    else:
        best = max(results.items(), key=lambda kv_: kv_[1]["matches"])
        print(f"  closest variant: {best[0]}  first mismatch:")
        print(f"    offline: {best[1]['first_mismatch']['offline']!r}")
        print(f"    server:  {best[1]['first_mismatch']['server']!r}")
    print(f"\nreport: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
