"""Compare the explicit and ambiguous strata of G2 for every model that has both.

    python scripts/g2_strata.py
    python scripts/g2_strata.py --dir docs/findings/g2

The ambiguous stratum earns its place only if, on every model, both strata pass
their own G2 checks (explicit: 90% tool calls; ambiguous: a 50% call floor;
both: prompts match, every parsed call round-trips, disagreements adjudicated)
AND the ambiguous stratum's median action margin is below the explicit one's.
"""
import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from driftmatrix.g2 import AMBIGUOUS_CALL_FLOOR, TOOL_CALL_GATE, stratum_check  # noqa: E402

MODELS = ("ornith", "qwen3", "spark")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(REPO / "data"))
    args = ap.parse_args()

    rows, all_pass = [], True
    for model in MODELS:
        paths = {st: Path(args.dir) / f"g2_toolcalls_{model}_{st}.json"
                 for st in ("explicit", "ambiguous")}
        if not all(p.exists() for p in paths.values()):
            print(f"{model}: missing {[p.name for p in paths.values() if not p.exists()]}")
            all_pass = False
            continue
        e, a = (json.loads(paths[st].read_text(encoding="utf-8"))["summary"]
                for st in ("explicit", "ambiguous"))
        chk = stratum_check(e, a)
        ok = (e["verdict"] == "G2_TOOLCALLS_PASS" and a["verdict"] == "G2_TOOLCALLS_PASS"
              and chk["verdict"] == "STRATUM_LOWERS_MARGIN")
        all_pass &= ok
        rows.append((model, e, a, chk, ok))

    def num(x, fmt):
        return "n/a" if x is None else format(x, fmt)

    print(f"\n  {'model':<8}{'stratum':<11}{'verdict':<22}{'tool calls':>11}{'median margin':>15}  margin bins")
    for model, e, a, chk, ok in rows:
        for name, s in (("explicit", e), ("ambiguous", a)):
            print(f"  {model:<8}{name:<11}{s['verdict']:<22}{num(s['tool_call_rate'], '.0%'):>11}"
                  f"{num(s['median_margin'], '.3f'):>15}  {s['margin_bins']}")
        print(f"  {'':<8}{'-> ' + chk['verdict']:<33}{'PASS' if ok else 'FAIL':>11}\n")

    print(f"G2 STRATA: {'PASS' if all_pass and rows else 'FAIL'}"
          f"  (explicit >= {TOOL_CALL_GATE:.0%} and ambiguous >= {AMBIGUOUS_CALL_FLOOR:.0%} tool calls, "
          f"parsers validated, ambiguous median margin below explicit, every model)")
    return 0 if all_pass and rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
