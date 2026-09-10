"""Gate G3 across all three models, from pilot records.

    python scripts/g3_check.py
    python scripts/g3_check.py --dir docs/findings/g3

Reads <dir>/<model>.jsonl for ornith, qwen3 and spark and applies the
pre-registered G3 checks only (pre-registration 11.2): arm proofs, snapped
exactness, and the positive control on the hybrid. Exits 0 only if every model
passes.
"""
import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from driftmatrix.g3 import g3_check  # noqa: E402
from driftmatrix.servergrid import load_records  # noqa: E402

MODELS = ("ornith", "qwen3", "spark")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(REPO / "data" / "pilot"))
    ap.add_argument("--json", action="store_true", help="print the full check per model")
    args = ap.parse_args()

    results = []
    for model in MODELS:
        path = Path(args.dir) / f"{model}.jsonl"
        if not path.exists():
            print(f"  {model:<8}missing {path.name}")
            results.append({"model": model, "verdict": "MISSING"})
            continue
        meta, turns, arms = load_records(path)
        r = g3_check(meta, turns, arms)
        results.append(r)
        control = (f"{r['natural_turns_with_drift']}/{r['turns']} natural turns drift"
                   if r["positive_control_required"] else "positive control not required")
        print(f"  {model:<8}{r['verdict']:<26}{r['turns']:>3} turns | invalid arms {len(r['invalid_arms'])}"
              f" | snapped inexact {len(r['snapped_inexact'])} | {control} | build {meta['build']}")
        if args.json:
            print(json.dumps(r, indent=2))

    ok = all(r["verdict"] == "G3_PASS" for r in results)
    print(f"\nG3: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
