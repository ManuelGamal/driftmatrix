"""Run the G3 pilot or the main study for one model against llama-server.

Start llama-server with the pre-registered flags, including --slot-save-path, then:

    python scripts/run_study.py --pilot --model ornith --gguf <path> --slot-dir <the --slot-save-path dir>
    python scripts/run_study.py --model ornith --gguf <path> --slot-dir <the --slot-save-path dir>

--pilot runs the G3 pilot (seed 20260911, 10 bases, short band only) into
data/pilot/<model>.jsonl and prints the G3 check. Without it, the 360 main-study
turns go to data/study/<model>.jsonl and only arm validity is printed; no
treatment outcome is computed until all three models have finished
(pre-registration section 11).

--slot-dir enables the state-byte comparison. It must be the directory the
server writes slot files to. Interrupt and rerun the same command to resume.
"""
import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from driftmatrix.config import (  # noqa: E402
    PILOT_BASES, PILOT_SEED, ROUND_TRIPS, STUDY_BASES, STUDY_SEED,
)
from driftmatrix.g3 import g3_check  # noqa: E402
from driftmatrix.runner.llamaserver import ServerClient  # noqa: E402
from driftmatrix.servergrid import load_records, run_study  # noqa: E402
from driftmatrix.traces.conversations import generate_study_turns, trace_hash  # noqa: E402
from driftmatrix.traces.render import load_specials, load_template  # noqa: E402
from driftmatrix.traces.toolcalls import FORMAT_BY_MODEL  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8090")
    ap.add_argument("--model", required=True, choices=sorted(FORMAT_BY_MODEL))
    ap.add_argument("--gguf", required=True, help="the same .gguf the server loaded")
    ap.add_argument("--slot-dir", help="the server's --slot-save-path; enables state-byte comparison")
    ap.add_argument("--pilot", action="store_true", help="run the G3 pilot instead of the main study")
    args = ap.parse_args()

    if args.pilot:
        generated = generate_study_turns(PILOT_BASES, PILOT_SEED, ROUND_TRIPS)
        turns = [t for t in generated if t.prefix_band == "short"]
        out = REPO / "data" / "pilot" / f"{args.model}.jsonl"
    else:
        generated = turns = generate_study_turns(STUDY_BASES, STUDY_SEED, ROUND_TRIPS)
        out = REPO / "data" / "study" / f"{args.model}.jsonl"

    client = ServerClient(args.url)
    print(f"waiting for {args.url} ...", flush=True)
    client.wait_healthy()
    summary = run_study(client, turns, load_template(args.gguf), load_specials(args.gguf), out,
                        model=args.model, gguf=args.gguf, trace_hash_value=trace_hash(generated),
                        state_dir=Path(args.slot_dir) if args.slot_dir else None,
                        progress=lambda m: print(m, flush=True))
    print(f"\n{args.model}: {summary['turns_run']} turns run, {summary['turns_skipped']} already done; "
          f"arm outcomes {summary['outcomes']}\nrecords: {out}")

    if not args.pilot:
        return 0
    result = g3_check(*load_records(out))
    print(json.dumps({k: v for k, v in result.items() if k != "verdict"}, indent=2))
    print(f"\nG3 ({args.model}): {result['verdict']}")
    return 0 if result["verdict"] == "G3_PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
