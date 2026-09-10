"""Freeze the main-study trace set for one model: token lengths and reuse boundaries.

Start llama-server for the model (same flags as G2), then:

    python scripts/study_manifest.py --model ornith --gguf <path>

Tokenizes every main-study turn with the server's tokenizer. Nothing is
generated and no restore is run, so no treatment effect is examined. Writes
data/study_manifest_<model>.json and exits 1 if the server has a different model
loaded, or if any turn breaks the pre-registered budget: prompt plus 192 new
tokens must fit in the 8192-token context, and every turn needs a reuse boundary
of at least one micro-batch so the snapped boundary is not zero.
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from driftmatrix.config import (  # noqa: E402
    MAX_NEW_TOKENS, ROUND_TRIPS, STUDY_BASES, STUDY_CONTEXT, STUDY_SEED, STUDY_UBATCH,
)
from driftmatrix.runner.llamaserver import ServerClient  # noqa: E402
from driftmatrix.servergrid import loaded_model_matches  # noqa: E402
from driftmatrix.traces.conversations import generate_study_turns, trace_hash  # noqa: E402
from driftmatrix.traces.render import (  # noqa: E402
    _last_user_index, load_specials, load_template, render, reuse_boundary, snap,
)
from driftmatrix.traces.toolcalls import FORMAT_BY_MODEL  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8090")
    ap.add_argument("--model", required=True, choices=sorted(FORMAT_BY_MODEL))
    ap.add_argument("--gguf", required=True, help="the same .gguf the server loaded")
    args = ap.parse_args()

    template, specials = load_template(args.gguf), load_specials(args.gguf)
    turns = generate_study_turns(STUDY_BASES, STUDY_SEED, ROUND_TRIPS)
    client = ServerClient(args.url)
    print(f"waiting for {args.url} ...", flush=True)
    client.wait_healthy()
    props = client.props()
    if not loaded_model_matches(props, args.gguf):
        print(f"MANIFEST: FAIL  (server has {props.get('model_path')!r} loaded, not {Path(args.gguf).name})")
        return 1

    rows = []
    for t in turns:
        k, full = reuse_boundary(client.tokenize, template, t.messages, t.tools,
                                 enable_thinking=False, **specials)
        head = render(template, t.messages[:_last_user_index(t.messages)], t.tools,
                      add_generation_prompt=False, enable_thinking=False, **specials)
        history_tokens = len(client.tokenize(head))
        rows.append({"conv_id": t.conv_id, "base_id": t.base_id, "prefix_band": t.prefix_band,
                     "stratum": t.stratum, "prompt_tokens": len(full), "history_tokens": history_tokens,
                     "k_natural": k, "k_mod_ubatch": k % STUDY_UBATCH, "k_snapped": snap(k, STUDY_UBATCH),
                     "fits": len(full) + MAX_NEW_TOKENS <= STUDY_CONTEXT,
                     "snapped_nonzero": snap(k, STUDY_UBATCH) > 0})

    report = {"model": args.model, "gguf": Path(args.gguf).name, "build": props.get("build_info"),
              "seed": STUDY_SEED, "bases": STUDY_BASES, "round_trips": ROUND_TRIPS,
              "ubatch": STUDY_UBATCH, "trace_hash": trace_hash(turns), "rows": rows}
    out = REPO / "data" / f"study_manifest_{args.model}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\n{args.model}: {len(rows)} turns, trace {report['trace_hash'][:16]}, model {report['gguf']}\n")
    print(f"  {'band':<6}{'prompt tokens (min/median/max)':>32}{'reuse boundary (min/median/max)':>34}"
          f"{'aligned':>9}{'history not fully reused':>26}")
    for band in ("short", "long"):
        rs = [r for r in rows if r["prefix_band"] == band]
        p, k = [r["prompt_tokens"] for r in rs], [r["k_natural"] for r in rs]
        partial = sum(r["k_natural"] < r["history_tokens"] for r in rs)
        print(f"  {band:<6}{min(p):>14}/{statistics.median(p):>7.0f}/{max(p):>7}"
              f"{min(k):>18}/{statistics.median(k):>7.0f}/{max(k):>7}"
              f"{sum(r['k_mod_ubatch'] == 0 for r in rs):>9}{partial:>22}/{len(rs)}")
    bad = [r for r in rows if not (r["fits"] and r["snapped_nonzero"])]
    for r in bad[:5]:
        print(f"  OVER BUDGET: {r}")
    print(f"\nMANIFEST: {'FAIL' if bad else 'PASS'}  ({len(bad)} turns over budget)\nreport: {out}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
