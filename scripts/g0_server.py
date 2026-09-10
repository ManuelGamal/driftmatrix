"""Gate G0 via llama-server. Standard library only - no pip install needed.

Start llama-server first (see README / G0 instructions), then:

    python3 scripts/g0_server.py --url http://127.0.0.1:8090

Writes data/g0_server_report.json and prints the verdict.
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
    "NONDETERMINISTIC_FLOOR":"The engine differs from itself with no restore involved. Nothing else is interpretable until this is fixed.",
    "SAVE_MISMATCH": "The saved slot did not contain exactly the prefix. The restore arm is not testing what it claims.",
    "RESTORE_NOT_USED": "The server reprocessed the whole prompt instead of reusing restored state (llama.cpp #20225). Identical output here proves nothing.",
    "DIVERGENCE_ON_DAY_ONE": "Restored state was genuinely reused AND output changed. This is the effect under study.",
    "FAITHFUL_AT_TESTED_SPLITS": "Restored state was reused and output was identical at both splits. Proceed to G1.",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8090")
    ap.add_argument("--turns", type=int, default=30, help="agent turns folded into the prompt")
    ap.add_argument("--n-predict", type=int, default=32)
    ap.add_argument("--out", default=str(REPO / "data" / "g0_server_report.json"))
    args = ap.parse_args()

    turns = generate_sessions(n_sessions=1, turns_per_session=args.turns, seed=2026)
    prompt = turns[-1].prefix_text + turns[-1].suffix_text

    client = ServerClient(args.url)
    print(f"waiting for {args.url} ...", flush=True)
    client.wait_healthy()

    report = run_g0(client, prompt, n_predict=args.n_predict)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    fl = report["floor"]
    print(f"\nprompt tokens: {report['n_tokens']}")
    print(f"floor (recompute vs recompute): identical={fl['identical']}  "
          f"max top-1 logprob gap={fl['max_top1_logprob_gap']:.3g}")
    for name, c in report["cases"].items():
        if "state_api_error" in c:
            print(f"  {name:<11} split={c['split_at']:<5} STATE API ERROR: {c['state_api_error']}")
            continue
        print(f"  {name:<11} split={c['split_at']:<5} saved={c['n_saved']} "
              f"restore prompt_n={c['restore_prompt_n']} (suffix={c['n_suffix']}) "
              f"used={c['restore_used']}  identical={c['identical']}  "
              f"first_div={c['first_divergence']}  gap={c['max_top1_logprob_gap']:.3g}")
    print(f"\nVERDICT: {report['verdict']}\n  {EXPLAIN[report['verdict']]}")
    print(f"\nfull report: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
