"""Gate G2, tool calls: prompts with tools match the server's, and produce tool calls.

Start llama-server for one model with --jinja (see G2 instructions), then, for
each stratum:

    python scripts/g2_toolcalls.py --model ornith --gguf <path> --stratum explicit
    python scripts/g2_toolcalls.py --model ornith --gguf <path> --stratum ambiguous

--model is one of: ornith, qwen3, spark. Writes data/g2_toolcalls_<model>_<stratum>.json.
Compare strata afterwards with scripts/g2_strata.py.
"""
import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from driftmatrix.g2 import TOOL_CALL_GATE, run_g2  # noqa: E402
from driftmatrix.runner.llamaserver import ServerClient  # noqa: E402
from driftmatrix.traces.conversations import STRATA, generate_conversations  # noqa: E402
from driftmatrix.traces.render import load_specials, load_template  # noqa: E402
from driftmatrix.traces.toolcalls import FORMAT_BY_MODEL  # noqa: E402

EXPLAIN = {
    "CHAT_PATH_ERROR": "The server rejected the OpenAI-format conversation; see chat_error in the report. The study path still ran, but prompts with tools are unverified.",
    "G2_TOOLCALLS_PASS": "Prompts with tools match the server's, parsers agree with llama.cpp, tool-call rate meets the gate.",
    "RENDER_WITH_TOOLS_MISMATCH": "Offline prompts with tools do not tokenize to the server's own prompt length. Do not use offline rendering until explained.",
    "PARSER_DISAGREES": "Our parser and llama.cpp's parser read the same output differently.",
    "TOOL_CALL_RATE_BELOW_GATE": f"Fewer than {TOOL_CALL_GATE:.0%} of turns produced a parseable tool call.",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8090")
    ap.add_argument("--model", required=True, choices=sorted(FORMAT_BY_MODEL))
    ap.add_argument("--gguf", required=True, help="the same .gguf the server loaded")
    ap.add_argument("--stratum", choices=STRATA, default="explicit")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--history", type=int, default=3, help="completed tool round trips per conversation")
    ap.add_argument("--n-predict", type=int, default=192)
    args = ap.parse_args()

    template = load_template(args.gguf)
    specials = load_specials(args.gguf)
    fmt = FORMAT_BY_MODEL[args.model]
    convs = generate_conversations(args.n, args.history, seed=2026, stratum=args.stratum)

    client = ServerClient(args.url)
    print(f"waiting for {args.url} ...", flush=True)
    client.wait_healthy()
    report = run_g2(client, convs, template, fmt, specials, n_predict=args.n_predict)
    report.update({"model": args.model, "gguf": Path(args.gguf).name, "format": fmt,
                   "stratum": args.stratum, "build": client.props().get("build_info")})

    out = REPO / "data" / f"g2_toolcalls_{args.model}_{args.stratum}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    s = report["summary"]

    def pct(x):
        return "n/a" if x is None else f"{x:.0%}"

    print(f"\n{args.model} ({fmt}), {args.stratum} stratum, {s['n']} conversations, "
          f"{args.history} prior tool round trips each\n")
    print(f"  prompt tokens match server   {pct(s['prompt_match_rate'])}")
    print(f"  parser agrees with llama.cpp {pct(s['parser_agreement_rate'])}")
    print(f"  tool-call rate               {pct(s['tool_call_rate'])}   (gate {TOOL_CALL_GATE:.0%})")
    print(f"  called the expected tool     {pct(s['expected_tool_rate'])}")
    print(f"  exact expected call          {pct(s['expected_call_exact_rate'])}")
    median = s["median_margin"]
    print(f"  median decision margin       {'n/a' if median is None else f'{median:.3f} nats'}")
    print(f"  margin bins                  {s['margin_bins']}")

    problems = [r for r in report["rows"]
                if not r["prompt_match"] or r["parser_agrees"] is False or not r["tool_call_made"]]
    for r in problems[:3]:
        print(f"\n  {r['conv_id']}: prompt {r['offline_prompt_tokens']} offline vs {r['server_prompt_tokens']} server; "
              f"ours={r['ours']} theirs={r['theirs']}")
        print(f"    output: {r['output'][:200]!r}")

    print(f"\nVERDICT: {s['verdict']}\n  {EXPLAIN[s['verdict']]}")
    print(f"report: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
