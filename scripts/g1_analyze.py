"""Combine G1 runs across n_ubatch values into the exact-set grid.

    python scripts/g1_analyze.py --label ornith_q4km
    python scripts/g1_analyze.py --label ornith_q4km --dir docs/findings/g1

Reads g1_<label>_ub*.json and prints, for every reuse boundary k, whether
restore was exact at each n_ubatch alongside what tiling predicts, then the
size of divergence at misaligned boundaries and how often a greedy token flipped.
"""
import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from driftmatrix.g1 import cross_ubatch_summary, magnitude_summary  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--dir", default=str(REPO / "data"),
                    help="directory holding g1_<label>_ub*.json")
    args = ap.parse_args()

    paths = sorted(Path(args.dir).glob(f"g1_{args.label}_ub*.json"))
    if not paths:
        print(f"no reports matching {args.dir}/g1_{args.label}_ub*.json")
        return 1
    reports = [json.loads(p.read_text(encoding="utf-8")) for p in paths]
    s = cross_ubatch_summary(reports)

    ubs = s["ubatches"]
    print(f"{args.label}: n_ubatch runs {ubs}")
    print("cell = observed / predicted   (E = exact, d = diverged, ! = contradiction)\n")
    print(f"  {'k':>6}  " + "".join(f"{'ub ' + str(u):>12}" for u in ubs))
    for k, row in s["grid"].items():
        cells = []
        for u in ubs:
            obs = row.get(u)
            if obs is None:
                cells.append(f"{'-':>12}")
                continue
            pred = (k % u == 0)
            mark = "" if obs == pred else "!"
            cells.append(f"{('E' if obs else 'd') + '/' + ('E' if pred else 'd') + mark:>12}")
        print(f"  {k:>6}  " + "".join(cells))

    print(f"\nper-run verdicts: {s['verdicts']}")
    print(f"discriminating boundaries: {s['discriminating_splits']}")
    print(f"\nEXACT SET TRACKS n_ubatch: {s['exact_set_tracks_ubatch']}")

    m = magnitude_summary(reports)
    print("\nmisaligned boundaries: size of divergence (worst-position TV) and greedy token flips")
    print(f"  {'n_ubatch':>8}  {'cases':>5}  {'median TV':>9}  {'min TV':>7}  {'max TV':>7}  {'flips':>5}  flip splits")
    for ub, v in m["per_ubatch"].items():
        print(f"  {ub:>8}  {v['n_misaligned']:>5}  {v['median_max_tv']:>9.4f}  {v['min_max_tv']:>7.4f}  "
              f"{v['max_max_tv']:>7.4f}  {v['token_flips']:>5}  {v['flip_splits']}")
    al = m["all"]
    print(f"  {'all':>8}  {al['n_misaligned']:>5}  {al['median_max_tv']:>9.4f}  {'':>7}  {'':>7}  {al['token_flips']:>5}")
    print("  (splits share one prompt per model: descriptive, not a population estimate)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
