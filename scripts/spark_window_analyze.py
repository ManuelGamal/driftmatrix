"""Combine the Spark window follow-up runs: exact grid, rules scored, diagnostics compared.

    python scripts/spark_window_analyze.py
    python scripts/spark_window_analyze.py --dir docs/findings/followup

Default runs (flash attention auto, --swa-full) are scored against the rules in
src/driftmatrix/followup.py. Diagnostic runs are compared with the default run
at the same n_ubatch.
"""
import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from driftmatrix.followup import HYPOTHESES, SPLITS, compare, evaluate, exact_set, valid_cases  # noqa: E402


def is_default(r):
    return r["config"]["flash_attn"] == "auto" and r["config"]["swa_full"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(REPO / "data" / "followup"))
    args = ap.parse_args()
    reports = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(Path(args.dir).glob("spark_*.json"))]
    if not reports:
        print(f"no spark_*.json in {args.dir}")
        return 1
    reports.sort(key=lambda r: (not is_default(r), r["ubatch"], r["label"]))

    print("Runs:")
    for r in reports:
        ok, bad = valid_cases(r)
        print(f"  {r['label']:<20} n_ubatch {r['ubatch']:>4}  flash-attn {r['config']['flash_attn']:<4} "
              f"swa-full {str(r['config']['swa_full']):<5}  floor exact {str(r['floor']['exact']):<5}  "
              f"proven {len(ok)}/{len(r['cases'])}  exact at {exact_set(r)}")

    print("\nExact grid (E exact, d inexact, ! restore not proven, - not tested):")
    print("  " + f"{'k':>5}  " + "  ".join(f"{r['label']:>18}" for r in reports))
    for k in SPLITS:
        cells = []
        for r in reports:
            case = next((c for c in r["cases"] if c["k"] == k), None)
            if case is None:
                cells.append("-")
            elif case in valid_cases(r)[0]:
                cells.append("E" if case["exact"] else "d")
            else:
                cells.append("!")
        print("  " + f"{k:>5}  " + "  ".join(f"{c:>18}" for c in cells))

    defaults = [r for r in reports if is_default(r)]
    print(f"\nRules scored on the {len(defaults)} default runs:")
    for name, result in evaluate(defaults).items():
        bad = result["contradictions"]
        detail = "" if not bad else "  e.g. " + ", ".join(
            f"ub{c['ubatch']} k={c['k']} exact={c['exact']}" for c in bad[:4])
        print(f"  {name:<32} {len(bad):>2} contradictions in {result['tested']} cases{detail}")

    for r in reports:
        if is_default(r):
            continue
        base = next((d for d in defaults if d["ubatch"] == r["ubatch"]), None)
        if base is None:
            print(f"\n{r['label']}: no default run at n_ubatch {r['ubatch']} to compare with")
            continue
        print(f"\n{r['label']} against {base['label']}: {compare(r, base)}")
    print("\nRule definitions:", ", ".join(HYPOTHESES))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
