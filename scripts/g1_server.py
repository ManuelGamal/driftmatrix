"""Gate G1, one server run: exactness at each reuse boundary for one n_ubatch.

Start llama-server with -ub <U> (see the G1 instructions), then:

    python scripts/g1_server.py --ubatch <U> --label ornith_q4km

--ubatch MUST equal the server's -ub. Writes data/g1_<label>_ub<U>.json.
Run once per n_ubatch, then combine with scripts/g1_analyze.py.
"""
import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from driftmatrix.g1 import run_g1  # noqa: E402
from driftmatrix.runner.llamaserver import ServerClient  # noqa: E402
from driftmatrix.traces.generator import generate_sessions  # noqa: E402

EXPLAIN = {
    "PREDICTION_HOLDS": "Exact exactly at multiples of n_ubatch. Run the next n_ubatch.",
    "ALIGNED_NOT_ALWAYS_EXACT": "Some micro-batch-aligned boundaries were not exact. Tiling is not the whole story.",
    "MISALIGNED_SOMETIMES_EXACT": "Some misaligned boundaries were exact. Tiling matters less than predicted here.",
    "PREDICTION_FAILS": "Mismatches in both directions. Tiling does not explain exactness.",
    "SPLITS_DO_NOT_DISCRIMINATE": "Every tested boundary was in the same alignment class.",
    "RESTORE_NOT_VALID": "A restore arm reprocessed the prompt or saved the wrong slot. Results void.",
    "STATE_API_ERROR": "Slot save/restore refused. Check --slot-save-path.",
    "NONDETERMINISTIC_FLOOR": "Recompute differs from itself. Nothing else is interpretable.",
    "RECOMPUTE_NOT_CLEAN": "A recompute arm reused cached state. Check --cache-ram 0 --ctx-checkpoints 0.",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8090")
    ap.add_argument("--ubatch", type=int, required=True, help="must equal the server's -ub")
    ap.add_argument("--label", required=True, help="model label, e.g. ornith_q4km")
    ap.add_argument("--turns", type=int, default=45)
    ap.add_argument("--n-predict", type=int, default=16)
    args = ap.parse_args()

    turns = generate_sessions(n_sessions=1, turns_per_session=args.turns, seed=2026)
    prompt = turns[-1].prefix_text + turns[-1].suffix_text

    client = ServerClient(args.url)
    print(f"waiting for {args.url} ...", flush=True)
    client.wait_healthy()
    report = run_g1(client, prompt, ubatch=args.ubatch, n_predict=args.n_predict)

    out = REPO / "data" / f"g1_{args.label}_ub{args.ubatch}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    print(f"\n{args.label}  n_ubatch={args.ubatch}  prompt tokens={report['n_tokens']}  "
          f"floor exact={report['floor']['exact']}")
    if report["splits_skipped"]:
        print(f"skipped (no room for a suffix): {report['splits_skipped']}")
    print(f"\n  {'k':>6}  {'aligned':<8}{'restore ok':<11}{'exact':<7}{'first_div':<10}{'max TV':>8}")
    for c in report["cases"]:
        if "error" in c:
            print(f"  {c['k']:>6}  ERROR {c['error']}")
            continue
        ok = c["restore_used"] and c["saved_exact"]
        flag = "" if c["exact"] == c["aligned"] else "   <-- contradicts prediction"
        print(f"  {c['k']:>6}  {str(c['aligned']):<8}{str(ok):<11}{str(c['exact']):<7}"
              f"{str(c['first_divergence']):<10}{c['max_tv']:>8.4f}{flag}")
    print(f"\nVERDICT: {report['verdict']}\n  {EXPLAIN[report['verdict']]}")
    print(f"report: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
