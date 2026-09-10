"""Gate G0 via llama-server. Standard library only - no pip install needed.

Start llama-server first (see README / G0 instructions), then:

    python scripts/g0_server.py --url http://127.0.0.1:8090 --ubatch 512 --label ornith

--ubatch MUST match the server's -ub value, or the alignment reading is wrong.
Writes data/g0_<label>_report.json and prints the verdict.
"""
import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from driftmatrix.g0 import run_g0  # noqa: E402
from driftmatrix.runner.llamaserver import ServerClient  # noqa: E402
from driftmatrix.traces.generator import generate_sessions  # noqa: E402

EXPLAIN = {
    "STATE_API_UNSUPPORTED": "Slot save/restore refused. Was the server started with --slot-save-path?",
    "RECOMPUTE_NOT_CLEAN": "A recompute arm reused cached state it never computed. Restart the server with --cache-ram 0 and --ctx-checkpoints 0, then rerun.",
    "NONDETERMINISTIC_FLOOR": "The engine differs from itself with no restore involved. Nothing else is interpretable until this is fixed.",
    "SAVE_MISMATCH": "The saved slot did not contain exactly the prefix. The restore arm is not testing what it claims.",
    "RESTORE_NOT_USED": "The server reprocessed the whole prompt instead of reusing restored state (llama.cpp #20225). Identical output here proves nothing.",
    "DIVERGENCE_ON_DAY_ONE": "Restored state was genuinely reused AND output changed at one or more splits.",
    "FAITHFUL_AT_TESTED_SPLITS": "Restored state was reused and output was identical at every split. Proceed to G1.",
}

TILING = {
    "CONSISTENT_WITH_UBATCH_TILING": "Only splits misaligned with the micro-batch diverged. The prompt-segmentation mechanism predicts exactly this.",
    "NOT_EXPLAINED_BY_UBATCH_TILING": "Micro-batch-aligned splits diverged too. Tiling alone does not explain it - something else changes on restore.",
    "MIXED": "Some misaligned splits stayed identical. Partial support only; inspect the margins.",
    "NO_DIVERGENCE": "No split diverged.",
    "UNTESTED_NO_UBATCH_ALIGNED_SPLIT": "Prompt too short to include a micro-batch-aligned split.",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8090")
    ap.add_argument("--label", default="ornith", help="names the output file")
    ap.add_argument("--ubatch", type=int, default=512, help="must equal the server's -ub")
    ap.add_argument("--turns", type=int, default=30, help="agent turns folded into the prompt")
    ap.add_argument("--n-predict", type=int, default=32)
    ap.add_argument("--n-probs", type=int, default=10)
    args = ap.parse_args()

    turns = generate_sessions(n_sessions=1, turns_per_session=args.turns, seed=2026)
    prompt = turns[-1].prefix_text + turns[-1].suffix_text

    client = ServerClient(args.url)
    print(f"waiting for {args.url} ...", flush=True)
    client.wait_healthy()

    report = run_g0(client, prompt, n_predict=args.n_predict, n_probs=args.n_probs,
                    ubatch=args.ubatch)

    out = REPO / "data" / f"g0_{args.label}_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    fl = report["floor"]
    print(f"\nmodel: {report.get('props', {}).get('model_path', '?')}")
    print(f"prompt tokens: {report['n_tokens']}   ubatch: {args.ubatch}")
    print(f"floor (recompute vs recompute): identical={fl['identical']}  "
          f"gap={fl['max_top1_logprob_gap']:.3g}")
    print(f"\n  {'split':<12}{'k':>6}  {'ub-aligned':<11}{'used':<6}{'identical':<10}"
          f"{'first_div':<10}{'gap':<10}{'margin@div (restore/recompute)'}")
    for name, c in report["cases"].items():
        if "state_api_error" in c:
            print(f"  {name:<12}{c['split_at']:>6}  STATE API ERROR: {c['state_api_error']}")
            continue
        rm, cm = c["restore_margin_at_divergence"], c["recompute_margin_at_divergence"]
        margin = "-" if rm is None and cm is None else f"{rm if rm is None else round(rm, 4)} / {cm if cm is None else round(cm, 4)}"
        print(f"  {name:<12}{c['split_at']:>6}  {str(c['aligned_to_ubatch']):<11}"
              f"{str(c['restore_used']):<6}{str(c['identical']):<10}"
              f"{str(c['first_divergence']):<10}{c['max_top1_logprob_gap']:<10.3g}{margin}")
    print(f"\nVERDICT: {report['verdict']}\n  {EXPLAIN[report['verdict']]}")
    print(f"TILING:  {report['tiling_reading']}\n  {TILING[report['tiling_reading']]}")
    print(f"\nfull report: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
