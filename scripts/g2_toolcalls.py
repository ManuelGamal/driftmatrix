"""Gate G2, tool calls: prompts with tools match the server's, and act as intended.

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

from driftmatrix.g2 import run_g2  # noqa: E402
from driftmatrix.runner.llamaserver import ServerClient  # noqa: E402
from driftmatrix.traces.conversations import STRATA, generate_conversations  # noqa: E402
from driftmatrix.traces.render import load_specials, load_template  # noqa: E402
from driftmatrix.traces.toolcalls import FORMAT_BY_MODEL  # noqa: E402

EXPLAIN = {
    "CHAT_PATH_ERROR": "The server rejected the OpenAI-format conversation; see chat_error in the report. The study path still ran, but prompts with tools are unverified.",
    "G2_TOOLCALLS_PASS": "Prompts with tools match the server's, every parsed call round-trips through the template, parser disagreements are attributed to llama.cpp, and the call rate meets this stratum's threshold.",
    "RENDER_WITH_TOOLS_MISMATCH": "Offline prompts with tools do not tokenize to the server's own prompt length. Do not use offline rendering until explained.",
    "OUR_PARSE_NOT_FAITHFUL": "A call we parsed does not reproduce the model's own tool-call text when serialised through its template. The row is listed first above.",
    "PARSER_DISAGREES": "Our parser and llama.cpp's read the same output differently, and the template round trip does not show that llama.cpp's reading is the lossy one. The row is listed first above.",
    "TOOL_CALL_RATE_BELOW_GATE": "Fewer than {threshold:.0%} of turns produced a parseable tool call.",
}


def severity(row):
    """Unfaithful parses, then unresolved and adjudicated disagreements, prompt mismatches, turns without a call."""
    if row.get("our_roundtrip") is False:
        return 0
    if row["parser_agrees"] is False:
        return 2 if row.get("adjudication") == "server_parser_lossy" else 1
    if not row["prompt_match"]:
        return 3
    return 4


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
    print(f"  our calls round-trip         {pct(s['our_roundtrip_rate'])}")
    print(f"  parser agrees with llama.cpp {pct(s['parser_agreement_rate'])}   "
          f"({s['parser_disagreements']} disagreements: {s['server_parser_lossy']} llama.cpp lossy, "
          f"{s['unresolved_disagreements']} unresolved)")
    print(f"  tool-call rate               {pct(s['tool_call_rate'])}   (threshold for this stratum {s['call_threshold']:.0%})")
    print(f"  no-call rate                 {pct(s['no_call_rate'])}")
    print(f"  called the expected tool     {pct(s['expected_tool_rate'])}")
    print(f"  exact expected call          {pct(s['expected_call_exact_rate'])}")
    median = s["median_margin"]
    print(f"  median action margin         {'n/a' if median is None else f'{median:.3f} nats'}")
    print(f"  margin bins                  {s['margin_bins']}")

    problems = sorted((r for r in report["rows"] if severity(r) < 4 or not r["tool_call_made"]),
                      key=severity)
    labels = {0: "PARSE NOT FAITHFUL", 1: "PARSER DISAGREEMENT", 2: "llama.cpp parse lossy",
              3: "PROMPT MISMATCH", 4: "no call"}
    for r in problems[:3]:
        print(f"\n  [{labels[severity(r)]}] {r['conv_id']}: prompt {r['offline_prompt_tokens']} offline vs "
              f"{r['server_prompt_tokens']} server")
        print(f"    ours:   {r['ours']}")
        print(f"    theirs: {r['theirs']}")
        print(f"    output: {r['output'][:200]!r}")

    print(f"\nVERDICT: {s['verdict']}\n  {EXPLAIN[s['verdict']].format(threshold=s['call_threshold'])}")
    print(f"report: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
